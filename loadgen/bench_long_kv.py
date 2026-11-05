"""Long-KV decode interference micro-benchmark against a running SGLang server.

Measures how N_short short requests (fixed context, prefix-cached) decode when
they share an instance with N_stress stressors of context length L.

  A     : the short requests alone
  C(L)  : the same short requests, joined after N_stress stressors of length L
          have finished prefill and entered decode

Mix sweep (--stress-counts): hold shorts + stressors at --concurrency and vary the
number of stressors k, with a single stressor length; k = 0 is the reference.

Per trial: flush the radix cache, warm the short prompts (identical input,
max_new_tokens=1), [start stressors and wait until each has streamed
>= --stress-min-tokens tokens], submit the shorts concurrently, wait for them,
abort the stressors. Retractions and radix evictions are read from /metrics
before and after the measured window; a trial with either is marked invalid.

Writes into --outdir:
  events.jsonl   one record per request: submit time and (t, completion_tokens)
                 for every stream event that advanced completion_tokens
  requests.csv   one row per short request
  trials.csv     one row per trial (means over the shorts, validity checks)
  config.json    the CLI arguments and derived prompt lengths

Usage:
    python loadgen/bench_long_kv.py --url http://127.0.0.1:30000 --outdir logs/x
    python loadgen/bench_long_kv.py --url ... --outdir ... --stress-lens 128 2048 --reps 1
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import statistics
import time
import uuid

import aiohttp

# Arbitrary id range inside the Qwen3.8 vocab that avoids special tokens
# (same range as bench_prefill.py). Prompts are random ids: no two requests
# share a prefix, so no physical KV is shared.
TOKEN_LO, TOKEN_HI = 1000, 150000
DEFAULT_STRESS_LENS = [128, 512, 2048, 8192, 16384, 32768, 65536, 98304]
COUNTERS = ("sglang:num_retracted_requests_total", "sglang:evicted_tokens_total")


def _ids(seed: int, n: int) -> list:
    rng = random.Random(seed)
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


class Stream:
    """One streaming /generate request; records every token-advancing event."""

    def __init__(self, *, kind: str, idx: int, input_ids: list, max_new: int):
        self.kind, self.idx = kind, idx
        self.rid = f"{kind}-{idx}-{uuid.uuid4().hex[:8]}"
        self.input_ids, self.max_new = input_ids, max_new
        self.t_submit = None
        self.events = []  # (t_monotonic, completion_tokens)
        self.meta = {}
        self.finish_reason = None
        self.error = None
        self.first_token = asyncio.Event()
        self.reached = asyncio.Event()  # set once completion_tokens >= reach_target
        self.reach_target = 1

    @property
    def n_out(self) -> int:
        return self.events[-1][1] if self.events else 0

    async def run(self, http: aiohttp.ClientSession, url: str):
        payload = {
            "rid": self.rid,
            "input_ids": self.input_ids,
            "sampling_params": {
                "max_new_tokens": self.max_new, "temperature": 0, "ignore_eos": True,
            },
            "stream": True,
        }
        self.t_submit = time.monotonic()
        try:
            async with http.post(f"{url}/generate", json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    t = time.monotonic()
                    chunk = json.loads(data)
                    meta = chunk.get("meta_info", {})
                    self.meta = meta
                    n = meta.get("completion_tokens", 0)
                    if n > self.n_out:  # ignore events that carry no new token
                        self.events.append((t, n))
                        self.first_token.set()
                        if n >= self.reach_target:
                            self.reached.set()
                    fr = meta.get("finish_reason")
                    if fr:
                        self.finish_reason = fr.get("type") if isinstance(fr, dict) else fr
        except Exception as e:  # recorded, the trial is marked invalid
            self.error = repr(e)
        finally:
            self.first_token.set()
            self.reached.set()

    def tokens_at(self, t: float) -> int:
        n = 0
        for te, ne in self.events:
            if te > t:
                break
            n = ne
        return n

    def record(self, t0: float) -> dict:
        return {
            "kind": self.kind, "idx": self.idx, "rid": self.rid,
            "t_submit": self.t_submit - t0,
            "events": [(round(t - t0, 6), n) for t, n in self.events],
            "prompt_tokens": self.meta.get("prompt_tokens"),
            "cached_tokens": self.meta.get("cached_tokens"),
            "completion_tokens": self.meta.get("completion_tokens"),
            "finish_reason": self.finish_reason, "error": self.error,
        }


async def _metrics(http, url) -> dict:
    async with http.get(f"{url}/metrics") as resp:
        text = await resp.text()
    out = {}
    for name in COUNTERS:
        vals = [float(m.group(1)) for m in re.finditer(
            rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", text, re.M)]
        # Labelled counters export no sample (not even # TYPE under multiprocess
        # metrics) until their first increment, so absent means 0 since startup.
        out[name] = sum(vals) if vals else 0.0
    return out


async def _flush(http, url):
    for _ in range(120):
        async with http.post(f"{url}/flush_cache") as resp:
            if resp.status == 200:
                return
        await asyncio.sleep(1)
    raise RuntimeError("flush_cache kept failing (requests still active?)")


async def _abort(http, url, rid):
    async with http.post(f"{url}/abort_request", json={"rid": rid}) as resp:
        await resp.read()


async def run_trial(http, args, *, trial_id, rep, point, short_prompts, stress_seed0):
    n_short, n_stress, stress_len = point
    short_prompts = short_prompts[:n_short]
    url = args.url
    await _flush(http, url)
    # Warm the short prefixes with the exact measured input, 1 output token.
    warm = [Stream(kind="warm", idx=i, input_ids=p, max_new=1) for i, p in enumerate(short_prompts)]
    await asyncio.gather(*(w.run(http, url) for w in warm))

    stressors = []
    tasks = []
    t0 = time.monotonic()
    if n_stress:
        for j in range(n_stress):
            # Fresh content per trial: a stressor never hits a stale prefix.
            s = Stream(kind="stress", idx=j,
                       input_ids=_ids(stress_seed0 + j, stress_len), max_new=args.stress_max_new)
            s.reach_target = args.stress_min_tokens
            stressors.append(s)
            tasks.append(asyncio.create_task(s.run(http, url)))
        await asyncio.gather(*(s.reached.wait() for s in stressors))

    c_before = await _metrics(http, url)
    shorts = [Stream(kind="short", idx=i, input_ids=p, max_new=args.short_out)
              for i, p in enumerate(short_prompts)]
    t_short_start = time.monotonic()
    stress_at_start = [s.n_out for s in stressors]
    await asyncio.gather(*(s.run(http, url) for s in shorts))
    t_short_end = time.monotonic()
    stress_at_end = [s.n_out for s in stressors]
    stress_alive = [s.finish_reason is None and s.error is None for s in stressors]
    c_after = await _metrics(http, url)

    for s in stressors:
        await _abort(http, url, s.rid)
    if tasks:
        await asyncio.wait(tasks, timeout=60)

    # ----- validity -----
    reasons = []
    for s in shorts:
        if s.error:
            reasons.append(f"short{s.idx} error")
        elif s.n_out != args.short_out:
            reasons.append(f"short{s.idx} out={s.n_out}")
    min_cached = min((s.meta.get("cached_tokens") or 0) for s in shorts)
    if min_cached < args.short_ctx:
        reasons.append(f"short prefix hit {min_cached} < {args.short_ctx}")
    if stressors and not all(stress_alive):
        reasons.append("stressor ended before shorts finished")
    if any(s.error for s in stressors):
        reasons.append("stressor error")
    deltas = {}
    for name in COUNTERS:
        b, a = c_before.get(name), c_after.get(name)
        deltas[name] = None if a is None or b is None else a - b
    if deltas[COUNTERS[0]]:
        reasons.append(f"retracted {deltas[COUNTERS[0]]:g}")
    if deltas[COUNTERS[1]]:
        reasons.append(f"evicted {deltas[COUNTERS[1]]:g} tokens")

    # ----- per-request metrics -----
    rows = []
    for s in shorts:
        ev = s.events
        ok = len(ev) >= 2 and s.n_out >= 2
        t_first, t_last = (ev[0][0], ev[-1][0]) if ev else (None, None)
        rows.append({
            "trial_id": trial_id, "rep": rep,
            "mode": "A" if not n_stress else "C", "n_short": n_short, "n_stress": n_stress,
            "stress_len": stress_len if n_stress else 0,
            "idx": s.idx,
            "t_submit": s.t_submit - t0,
            "ttft": t_first - s.t_submit if ev else None,
            "tpot": (t_last - t_first) / (s.n_out - 1) if ok else None,
            "e2e": t_last - s.t_submit if ev else None,
            "decode_s": t_last - t_first if ev else None,
            "prompt_tokens": s.meta.get("prompt_tokens"),
            "cached_tokens": s.meta.get("cached_tokens"),
            "completion_tokens": s.n_out,
            "n_events": len(ev),
        })

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else None

    stress_prompt = [s.meta.get("prompt_tokens") for s in stressors]
    trial = {
        "trial_id": trial_id, "rep": rep,
        "mode": "A" if not n_stress else "C", "n_short": n_short, "n_stress": n_stress,
        "stress_len": stress_len if n_stress else 0,
        "valid": int(not reasons), "invalid_reason": "; ".join(reasons),
        "mean_tpot": mean("tpot"), "mean_e2e": mean("e2e"), "mean_ttft": mean("ttft"),
        "mean_decode_s": mean("decode_s"),
        "short_cached_min": min((r["cached_tokens"] or 0) for r in rows),
        "short_prompt_tokens": rows[0]["prompt_tokens"],
        "stress_prompt_tokens": stress_prompt[0] if stress_prompt else None,
        "stress_wait_s": t_short_start - t0 if stressors else 0.0,
        "stress_tokens_at_short_start": ",".join(map(str, stress_at_start)),
        "stress_tokens_at_short_end": ",".join(map(str, stress_at_end)),
        "short_window_s": t_short_end - t_short_start,
        "retracted": deltas[COUNTERS[0]], "evicted_tokens": deltas[COUNTERS[1]],
    }
    events = [dict(trial_id=trial_id, **x.record(t0)) for x in (*stressors, *shorts)]
    return trial, rows, events


def _points(args) -> list:
    """(n_short, n_stress, stress_len) per sweep point; n_stress == 0 is the A point."""
    if args.stress_counts is None:
        L_points = [(args.n_short, args.n_stress, L) for L in args.stress_lens]
        return [(args.n_short, 0, 0)] + L_points
    L = args.stress_lens[0]
    return [(args.concurrency - k, k, L) for k in args.stress_counts]


async def main_async(args):
    os.makedirs(args.outdir, exist_ok=True)
    # One token past short_ctx: hybrid (mamba) prefix cache keeps a state only on its
    # chunk grid, and a hit must leave >= 1 token to prefill, so the warm request
    # leaves a state at short_ctx and the measured request hits exactly short_ctx.
    points = _points(args)
    max_short = max(p[0] for p in points)
    short_prompts = [_ids(args.seed * 1000 + i, args.short_ctx + 1) for i in range(max_short)]
    order_rng = random.Random(args.seed)
    # Discarded first-touch trials (JIT, allocator growth); logged with rep=-1.
    plan = [(-1, p) for p in points[:2]] * args.warmup_trials
    for rep in range(args.reps):
        pts = points[:]
        if args.shuffle:
            order_rng.shuffle(pts)
        plan += [(rep, p) for p in pts]
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({**vars(args), "plan": plan}, f, indent=1)

    trial_fields = request_fields = None
    f_tr = open(os.path.join(args.outdir, "trials.csv"), "w", newline="")
    f_rq = open(os.path.join(args.outdir, "requests.csv"), "w", newline="")
    f_ev = open(os.path.join(args.outdir, "events.jsonl"), "w")
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        async with http.get(f"{args.url}/metrics") as resp:
            with open(os.path.join(args.outdir, "metrics_start.txt"), "w") as f:
                f.write(await resp.text())
        for k, (rep, point) in enumerate(plan):
            n_short, n_stress, L = point
            label = f"{n_short}:{n_stress}" + ("" if not n_stress else f" C({L})")
            label += " warmup" if rep < 0 else ""
            trial, rows, events = await run_trial(
                http, args, trial_id=k, rep=rep, point=point,
                short_prompts=short_prompts, stress_seed0=10**6 + 1000 * k)
            if trial_fields is None:
                trial_fields = list(trial)
                w_tr = csv.DictWriter(f_tr, trial_fields)
                w_tr.writeheader()
                w_rq = csv.DictWriter(f_rq, list(rows[0]))
                w_rq.writeheader()
            w_tr.writerow(trial)
            w_rq.writerows(rows)
            for e in events:
                f_ev.write(json.dumps(e) + "\n")
            for f in (f_tr, f_rq, f_ev):
                f.flush()
            fmt = lambda v, s=1e3: "nan" if v is None else f"{v * s:.2f}"
            print(f"[{k + 1}/{len(plan)}] rep={rep} {label:>14} valid={trial['valid']} "
                  f"tpot={fmt(trial['mean_tpot'])}ms e2e={fmt(trial['mean_e2e'])}ms "
                  f"ttft={fmt(trial['mean_ttft'])}ms cached_min={trial['short_cached_min']} "
                  f"stress_wait={trial['stress_wait_s']:.1f}s "
                  f"stress_tok@start={trial['stress_tokens_at_short_start']} "
                  f"{trial['invalid_reason']}", flush=True)
    for f in (f_tr, f_rq, f_ev):
        f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-short", type=int, default=8)
    p.add_argument("--n-stress", type=int, default=8)
    p.add_argument("--short-ctx", type=int, default=512,
                   help="prefix-cached context of each short request; must lie on the "
                   "mamba cache chunk grid (64); the prompt is this plus 1 token")
    p.add_argument("--short-out", type=int, default=128)
    p.add_argument("--stress-lens", type=int, nargs="+", default=DEFAULT_STRESS_LENS)
    p.add_argument("--stress-max-new", type=int, default=2048,
                   help="stressor output budget; aborted once the shorts finish")
    p.add_argument("--stress-min-tokens", type=int, default=4,
                   help="each stressor must stream this many tokens before shorts join")
    p.add_argument("--stress-counts", type=int, nargs="+", default=None,
                   help="mix sweep: run --concurrency - k shorts with k stressors of "
                   "length --stress-lens[0] for each k, instead of the L sweep")
    p.add_argument("--concurrency", type=int, default=16,
                   help="shorts + stressors per point in the mix sweep")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-trials", type=int, default=1,
                   help="rounds of the first two sweep points run before measuring, "
                   "logged with rep=-1")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                   help="run points in sweep order instead of shuffling per rep")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
