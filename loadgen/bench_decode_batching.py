"""Decode batching gain vs context length, against one or more running SGLang servers.

Question: does the throughput gain from batching decode shrink as the context
length N grows?

For each context length N: flush the radix cache, warm max_bs distinct random
prompts of N + 1 tokens (max_new_tokens=1) so their prefix is cached, then for
every batch size B (1..max_bs, or --bs-list) submit the first B prompts together with
--out output tokens (ignore_eos). Prefill of the measured requests is ~1 token.
TPOT is measured only over the window in which all B requests are decoding
(after the last first token, before the first last token), so prefill and
arrival skew are excluded. Aggregate decode throughput is y(B, N) = B / TPOT;
the plot normalizes it by y(1, N).

With several --urls, the context lengths are split across the servers (greedy
by estimated prefill cost) and run concurrently; each N runs on one server only.

Validity per trial: every request returned exactly --out tokens, every request
hit >= N - 64 cached tokens, the full-batch window covers >= 32 tokens, and sglang:num_retracted_requests_total /
sglang:evicted_tokens_total did not change during the trial. Before each N the
script skips every B with B * (N + 1 + out) larger than the server's KV pool.

Writes into --outdir: trials.csv, requests.csv, events.jsonl, config.json.

Usage:
    python loadgen/bench_decode_batching.py --urls http://127.0.0.1:30000 --outdir logs/x
    python loadgen/bench_decode_batching.py --urls U0,U1 --outdir ... --lens 1024 98304 --reps 1
"""

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time

import aiohttp
from bench_long_kv import COUNTERS, Stream, _flush, _ids, _metrics

DEFAULT_LENS = [1024, 8192, 16384, 32768, 65536, 98304]


def _cost(n: int) -> float:
    """Rough relative prefill cost of max_bs prompts of length n (linear + attention)."""
    return n * (1 + n / 65536)


def assign(lens: list, n_urls: int) -> list:
    """Greedy longest-first split of lens across n_urls servers."""
    buckets = [[] for _ in range(n_urls)]
    load = [0.0] * n_urls
    for n in sorted(lens, reverse=True):
        i = load.index(min(load))
        buckets[i].append(n)
        load[i] += _cost(n)
    return buckets


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, key)
            if r is not None:
                return r
    return None


async def _pool_tokens(http, url) -> int:
    async with http.get(f"{url}/server_info") as resp:
        info = await resp.json()
    v = _find(info, "max_total_num_tokens")
    if v is None:
        raise RuntimeError(f"{url}/server_info has no max_total_num_tokens")
    return int(v)


class Writer:
    def __init__(self, outdir):
        self.f_tr = open(os.path.join(outdir, "trials.csv"), "w", newline="")
        self.f_rq = open(os.path.join(outdir, "requests.csv"), "w", newline="")
        self.f_ev = open(os.path.join(outdir, "events.jsonl"), "w")
        self.w_tr = self.w_rq = None

    def write(self, trial, rows, events):
        if self.w_tr is None:
            self.w_tr = csv.DictWriter(self.f_tr, list(trial))
            self.w_tr.writeheader()
            self.w_rq = csv.DictWriter(self.f_rq, list(rows[0]))
            self.w_rq.writeheader()
        self.w_tr.writerow(trial)
        self.w_rq.writerows(rows)
        for e in events:
            self.f_ev.write(json.dumps(e) + "\n")
        for f in (self.f_tr, self.f_rq, self.f_ev):
            f.flush()

    def close(self):
        for f in (self.f_tr, self.f_rq, self.f_ev):
            f.close()


def _steady_tpot(events_per_req, min_tokens=32):
    """Mean TPOT over [max first-token time, min last-token time], when all B decode at once.

    Requests reach the server up to ~0.5 s apart (each carries N input ids), so a
    request's own first-to-last TPOT also covers steps at a smaller batch size.
    """
    if not all(len(ev) >= 2 for ev in events_per_req):
        return None, None
    t_lo = max(ev[0][0] for ev in events_per_req)
    t_hi = min(ev[-1][0] for ev in events_per_req)
    tpots = []
    for ev in events_per_req:
        inside = [(t, n) for t, n in ev if t_lo <= t <= t_hi]
        if len(inside) < 2 or inside[-1][1] - inside[0][1] < min_tokens:
            return None, t_hi - t_lo
        tpots.append((inside[-1][0] - inside[0][0]) / (inside[-1][1] - inside[0][1]))
    return statistics.mean(tpots), t_hi - t_lo


async def run_trial(http, url, args, *, server, trial_id, rep, n_ctx, bs, prompts):
    c_before = await _metrics(http, url)
    streams = [Stream(kind="req", idx=i, input_ids=p, max_new=args.out)
               for i, p in enumerate(prompts[:bs])]
    t0 = time.monotonic()
    await asyncio.gather(*(s.run(http, url) for s in streams))
    t_end = time.monotonic()
    c_after = await _metrics(http, url)

    reasons = []
    for s in streams:
        if s.error:
            reasons.append(f"req{s.idx} error")
        elif s.n_out != args.out:
            reasons.append(f"req{s.idx} out={s.n_out}")
    cached_min = min((s.meta.get("cached_tokens") or 0) for s in streams)
    # One mamba cache chunk (64 tokens) of slack; observed hits are N - 15 .. N.
    if cached_min < n_ctx - 64:
        reasons.append(f"prefix hit {cached_min} < {n_ctx - 64}")
    deltas = {name: c_after[name] - c_before[name] for name in COUNTERS}
    if deltas[COUNTERS[0]]:
        reasons.append(f"retracted {deltas[COUNTERS[0]]:g}")
    if deltas[COUNTERS[1]]:
        reasons.append(f"evicted {deltas[COUNTERS[1]]:g} tokens")

    rows = []
    for s in streams:
        ev = s.events
        ok = len(ev) >= 2
        rows.append({
            "server": server, "trial_id": trial_id, "rep": rep, "n_ctx": n_ctx, "bs": bs,
            "idx": s.idx,
            "ttft": ev[0][0] - s.t_submit if ev else None,
            "t_first": ev[0][0] - t0 if ev else None,
            "t_last": ev[-1][0] - t0 if ev else None,
            "tpot": (ev[-1][0] - ev[0][0]) / (ev[-1][1] - ev[0][1]) if ok else None,
            "prompt_tokens": s.meta.get("prompt_tokens"),
            "cached_tokens": s.meta.get("cached_tokens"),
            "completion_tokens": s.n_out, "n_events": len(ev),
        })
    tpots = [r["tpot"] for r in rows if r["tpot"] is not None]
    firsts = [r["t_first"] for r in rows if r["t_first"] is not None]
    mean_tpot = statistics.mean(tpots) if tpots else None
    steady_tpot, steady_s = _steady_tpot([s.events for s in streams])
    if steady_tpot is None:
        reasons.append("no full-batch decode window")
    trial = {
        "server": server, "trial_id": trial_id, "rep": rep, "n_ctx": n_ctx, "bs": bs,
        "valid": int(not reasons),
        "invalid_reason": "; ".join(reasons),
        "mean_tpot": mean_tpot,
        "max_tpot": max(tpots) if tpots else None,
        "min_tpot": min(tpots) if tpots else None,
        "steady_tpot": steady_tpot, "steady_window_s": steady_s,
        "throughput": bs / steady_tpot if steady_tpot else None,
        "first_token_skew": max(firsts) - min(firsts) if firsts else None,
        "mean_ttft": statistics.mean(r["ttft"] for r in rows if r["ttft"] is not None)
        if firsts else None,
        "cached_min": cached_min, "window_s": t_end - t0,
        "retracted": deltas[COUNTERS[0]], "evicted_tokens": deltas[COUNTERS[1]],
    }
    events = [dict(trial_id=trial_id, server=server, **s.record(t0)) for s in streams]
    return trial, rows, events


def _cache_miss(trial) -> bool:
    return "prefix hit" in trial["invalid_reason"] or "evicted" in trial["invalid_reason"]


async def _warm(http, url, prompts):
    warm = [Stream(kind="warm", idx=i, input_ids=p, max_new=1) for i, p in enumerate(prompts)]
    await asyncio.gather(*(w.run(http, url) for w in warm))
    return next((w.error for w in warm if w.error), None)


async def _measure_point(http, url, args, writer, *, server, counter, rep, n_ctx, bs, prompts):
    """Run one (N, B) trial; on a prefix miss or eviction, re-warm and measure again."""
    for attempt in range(1 + (args.retries if rep >= 0 else 0)):
        if attempt:
            # Re-cache the evicted prompts outside the measured window.
            await _warm(http, url, prompts[:bs])
        k = counter[0]
        counter[0] += 1
        trial, rows, events = await run_trial(
            http, url, args, server=server, trial_id=k, rep=rep, n_ctx=n_ctx, bs=bs,
            prompts=prompts)
        trial["attempt"] = attempt
        writer.write(trial, rows, events)
        tp = trial["steady_tpot"]
        print(f"[{server}] N={n_ctx:>6} B={bs:>2} rep={rep:>2} try={attempt} "
              f"valid={trial['valid']} tpot={'nan' if tp is None else f'{tp * 1e3:.2f}'}ms "
              f"thr={trial['throughput'] or 0:.1f}tok/s "
              f"skew={(trial['first_token_skew'] or 0) * 1e3:.0f}ms "
              f"{trial['invalid_reason']}", flush=True)
        if trial["valid"] or not _cache_miss(trial):
            return trial
    return trial


async def run_server(http, args, writer, server, url, lens, counter):
    pool = await _pool_tokens(http, url)
    print(f"[{server}] {url} KV pool {pool} tokens, lens {lens}", flush=True)
    order_rng = random.Random(args.seed + server)
    wanted = args.bs_list or list(range(1, args.max_bs + 1))
    for n_ctx in lens:
        bss = [b for b in wanted if b * (n_ctx + 1 + args.out) <= pool]
        skipped = sorted(set(wanted) - set(bss))
        if skipped:
            print(f"[{server}] SKIP N={n_ctx} B={skipped}: exceeds KV pool {pool}", flush=True)
        if not bss:
            continue
        max_bs = max(bss)
        await _flush(http, url)
        # N + 1 tokens: the hybrid prefix cache keeps a state on its 64-token chunk
        # grid and a hit must leave >= 1 token to prefill, so the hit is exactly N.
        prompts = [_ids(args.seed * 10**6 + n_ctx * 100 + i, n_ctx + 1)
                   for i in range(max_bs)]
        t = time.monotonic()
        err = await _warm(http, url, prompts)
        if err:
            print(f"[{server}] N={n_ctx} warm failed: {err}", flush=True)
            continue
        print(f"[{server}] N={n_ctx} warmed {max_bs} prompts in "
              f"{time.monotonic() - t:.1f}s", flush=True)
        # One discarded trial at max_bs (rep=-1), then shuffled B per rep.
        plan = [(-1, max_bs)]
        for rep in range(args.reps):
            order = bss[:]
            order_rng.shuffle(order)
            plan += [(rep, b) for b in order]
        for rep, bs in plan:
            trial = await _measure_point(http, url, args, writer, server=server, counter=counter,
                                         rep=rep, n_ctx=n_ctx, bs=bs, prompts=prompts)
            if rep < 0 and _cache_miss(trial):
                # Every later trial would re-prefill too; give the walltime to the next N.
                print(f"[{server}] ABORT N={n_ctx}: warmed prefixes not cached", flush=True)
                break


async def main_async(args):
    os.makedirs(args.outdir, exist_ok=True)
    urls = args.urls.split(",")
    buckets = assign(args.lens, len(urls))
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({**vars(args), "assignment": dict(zip(urls, buckets))}, f, indent=1)
    writer = Writer(args.outdir)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    counter = [0]
    try:
        async with aiohttp.ClientSession(timeout=timeout) as http:
            await asyncio.gather(*(
                run_server(http, args, writer, i, u, b, counter)
                for i, (u, b) in enumerate(zip(urls, buckets)) if b))
    finally:
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--urls", required=True, help="comma-separated server URLs")
    p.add_argument("--outdir", required=True)
    p.add_argument("--lens", type=int, nargs="+", default=DEFAULT_LENS,
                   help="context lengths N; multiples of 64 (mamba cache chunk grid)")
    p.add_argument("--max-bs", type=int, default=16,
                   help="sweep B = 1..max_bs unless --bs-list is given")
    p.add_argument("--bs-list", type=int, nargs="+", default=None,
                   help="explicit batch sizes; ones that do not fit the KV pool are skipped")
    p.add_argument("--out", type=int, default=128, help="output tokens per request")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--retries", type=int, default=2,
                   help="re-warm and re-measure a trial invalidated by a prefix miss or eviction")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    bad = [n for n in args.lens if n % 64]
    if bad:
        p.error(f"--lens must be multiples of 64: {bad}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
