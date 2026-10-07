"""Agentic session replay load generator for SGLang (see .claude/REPLAY.md).

Each arrival in the schedule replays one trajectory from the corpus as a
separate replica: every turn is a native /generate request with input_ids, the
dummy output stays in the context, and the next turn is sent
turn_toolcall_latency_ms after the previous one finishes. All turns of a
replica go to the same endpoint (round-robin by arrival order).

The servers must run with --enable-metrics; otherwise queue / prefill / decode
latencies are missing from meta_info and those columns stay empty.

One CSV row is appended (and flushed) per turn finished within --duration.
All latencies are in seconds; t_send / t_end are unix timestamps.

Usage:
    python loadgen.py --corpus hermes_clean.jsonl \
        --arrivals session_arrival.jsonl \
        --endpoints http://10.0.0.1:30000,http://10.0.0.2:30000 \
        --output results.csv
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from typing import NamedTuple, Optional

import aiohttp

IM_END = 248046
USER_TAG = [248045, 846, 198]  # <|im_start|>user\n

CSV_COLUMNS = [
    "replica_id",
    "session_id",
    "turn_id",
    "endpoint",
    "t_send",
    "t_end",
    "turn_e2e_latency",
    "turn_queuing_latency",
    "turn_prefill_latency",
    "turn_decode_latency",
    "turn_toolcall_latency",
    "kvc_hit_ratio",
    "turn_prefill_token_num",
    "turn_decode_token_num",
    "server_e2e_latency",
    "input_len",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "expected_decode_tokens",
    "fidelity_ok",
    "status",
    "error",
]


class Arrival(NamedTuple):
    replica_id: int
    session_id: str
    arrival_offset: float
    endpoint: str


class TurnResult(NamedTuple):
    output_ids: list
    meta: dict
    t_send: float
    t_end: float


# ===== Inputs =====


def _load_corpus(path: str) -> dict:
    with open(path) as f:
        sessions = [json.loads(line) for line in f if line.strip()]
    return {s["session_id"]: s for s in sessions}


def _load_arrivals(*, path: str, corpus: dict, endpoints: list) -> list:
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs.sort(key=lambda r: r["arrival_offset"])
    arrivals = []
    seen = set()
    for i, r in enumerate(recs):
        replica_id = int(r.get("replica_id", i))
        if replica_id in seen:
            sys.exit(f"duplicate replica_id {replica_id} in {path}")
        seen.add(replica_id)
        if r["session_id"] not in corpus:
            sys.exit(f"session {r['session_id']} in {path} is not in the corpus")
        arrivals.append(
            Arrival(
                replica_id=replica_id,
                session_id=r["session_id"],
                arrival_offset=float(r["arrival_offset"]),
                endpoint=endpoints[i % len(endpoints)],
            )
        )
    return arrivals


# ===== Salt (REPLAY.md 3.1) =====


async def _fetch_salts(
    *, http: aiohttp.ClientSession, endpoint: str, replica_ids: list
) -> dict:
    prompts = [f"replica={rid}\n" for rid in replica_ids]
    async with http.post(
        f"{endpoint}/tokenize",
        json={"prompt": prompts, "add_special_tokens": False},
    ) as resp:
        resp.raise_for_status()
        tokens = (await resp.json())["tokens"]
    return dict(zip(replica_ids, tokens))


def _insert_salt(*, ids: list, salt: list) -> list:
    """Insert salt right after the first <|im_start|>user\\n."""
    n = len(USER_TAG)
    for i in range(len(ids) - n + 1):
        if ids[i : i + n] == USER_TAG:
            return ids[: i + n] + salt + ids[i + n :]
    raise ValueError("reset row has no <|im_start|>user\\n to salt")


# ===== Requests =====


async def _generate(
    *, http: aiohttp.ClientSession, endpoint: str, rid: str, ctx: list, max_new: int
) -> TurnResult:
    payload = {
        "rid": rid,
        "input_ids": ctx,
        "sampling_params": {
            "max_new_tokens": max_new,
            "ignore_eos": True,
            "temperature": 0,
        },
        "stream": False,
    }
    t_send = time.time()
    async with http.post(f"{endpoint}/generate", json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:500]}")
        body = await resp.json()
    t_end = time.time()
    return TurnResult(
        output_ids=body["output_ids"], meta=body["meta_info"], t_send=t_send, t_end=t_end
    )


def _span(meta: dict, start: str, end: str) -> Optional[float]:
    if start in meta and end in meta:
        return meta[end] - meta[start]
    return None


def _turn_row(
    *, arrival: Arrival, turn: dict, input_len: int, res: TurnResult
) -> dict:
    meta = res.meta
    prompt_tokens = meta.get("prompt_tokens")
    cached = meta.get("cached_tokens", 0)
    completion = meta.get("completion_tokens")
    expected = turn["decode_num_tokens"]
    toolcall_ms = turn["turn_toolcall_latency_ms"]
    return {
        "replica_id": arrival.replica_id,
        "session_id": arrival.session_id,
        "turn_id": turn["turn_index"],
        "endpoint": arrival.endpoint,
        "t_send": f"{res.t_send:.6f}",
        "t_end": f"{res.t_end:.6f}",
        "turn_e2e_latency": res.t_end - res.t_send,
        "turn_queuing_latency": meta.get("queue_time"),
        "turn_prefill_latency": _span(meta, "forward_entry_time", "prefill_finished_time"),
        "turn_decode_latency": _span(meta, "prefill_finished_time", "request_finished_ts"),
        "turn_toolcall_latency": None if toolcall_ms is None else toolcall_ms / 1000,
        "kvc_hit_ratio": cached / prompt_tokens if prompt_tokens else None,
        "turn_prefill_token_num": prompt_tokens - cached if prompt_tokens else None,
        "turn_decode_token_num": completion,
        "server_e2e_latency": meta.get("e2e_latency"),
        "input_len": input_len,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "expected_decode_tokens": expected,
        "fidelity_ok": int(prompt_tokens == input_len and completion == expected),
        "status": "ok",
        "error": "",
    }


# ===== Replay =====


class Recorder:
    """Appends one CSV row per turn and keeps run-level counters."""

    def __init__(self, path: str):
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS)
        if new_file:
            self._w.writeheader()
            self._f.flush()
        self.turns_ok = 0
        self.turns_mismatch = 0
        self.turns_failed = 0
        self.sessions_active = 0
        self.sessions_done = 0
        self.sessions_failed = 0
        self.requests_inflight = 0
        self.turns_late = 0
        # Unix time after which finished turns are dropped (--duration).
        self.deadline = float("inf")
        self.warned_no_metrics = False

    def write(self, row: dict):
        if float(row["t_end"]) > self.deadline:
            self.turns_late += 1
            return
        self._w.writerow({k: "" if v is None else v for k, v in row.items()})
        self._f.flush()
        if row["status"] != "ok":
            self.turns_failed += 1
            return
        self.turns_ok += 1
        if not row["fidelity_ok"]:
            self.turns_mismatch += 1
        if row["turn_queuing_latency"] is None and not self.warned_no_metrics:
            self.warned_no_metrics = True
            print(
                "WARNING: meta_info has no queue_time; "
                "launch the servers with --enable-metrics",
                file=sys.stderr,
            )

    def close(self):
        self._f.close()


def _next_ctx(*, turn: dict, ctx_before, ctx_after, salt: list) -> list:
    if turn["prompt_reset"]:
        return _insert_salt(ids=turn["prefill_token_ids"], salt=salt)
    base = ctx_after if turn["prev_response_in_history"] else ctx_before
    return base + turn["prefill_token_ids"]


async def _replay_session(
    *,
    http: aiohttp.ClientSession,
    arrival: Arrival,
    session: dict,
    salt: list,
    t0: float,
    rec: Recorder,
):
    await asyncio.sleep(max(0.0, t0 + arrival.arrival_offset - time.monotonic()))
    rec.sessions_active += 1
    ctx_before = ctx_after = None
    turns = session["turns"]
    try:
        for i, turn in enumerate(turns):
            ctx = _next_ctx(
                turn=turn, ctx_before=ctx_before, ctx_after=ctx_after, salt=salt
            )
            rec.requests_inflight += 1
            try:
                res = await _generate(
                    http=http,
                    endpoint=arrival.endpoint,
                    rid=f"r{arrival.replica_id}-t{turn['turn_index']}-{os.getpid()}",
                    ctx=ctx,
                    max_new=turn["decode_num_tokens"],
                )
            except Exception as e:
                # The context cannot be continued without this turn's output.
                rec.write(_error_row(arrival=arrival, turn=turn, input_len=len(ctx), err=e))
                rec.sessions_failed += 1
                return
            finally:
                rec.requests_inflight -= 1
            rec.write(_turn_row(arrival=arrival, turn=turn, input_len=len(ctx), res=res))
            ctx_before, ctx_after = ctx, ctx + res.output_ids + [IM_END]
            if i + 1 < len(turns):
                await asyncio.sleep(turn["turn_toolcall_latency_ms"] / 1000)
        rec.sessions_done += 1
    finally:
        rec.sessions_active -= 1


def _error_row(*, arrival: Arrival, turn: dict, input_len: int, err: Exception) -> dict:
    row = dict.fromkeys(CSV_COLUMNS)
    row.update(
        replica_id=arrival.replica_id,
        session_id=arrival.session_id,
        turn_id=turn["turn_index"],
        endpoint=arrival.endpoint,
        t_end=f"{time.time():.6f}",
        input_len=input_len,
        expected_decode_tokens=turn["decode_num_tokens"],
        status="error",
        error=f"{type(err).__name__}: {err}"[:500],
    )
    return row


async def _report_progress(*, rec: Recorder, total: int, t0: float, interval: float):
    while True:
        await asyncio.sleep(interval)
        print(
            f"[{(time.monotonic() - t0) / 60:7.1f} min] "
            f"active={rec.sessions_active} done={rec.sessions_done} "
            f"failed={rec.sessions_failed} pending="
            f"{total - rec.sessions_done - rec.sessions_failed - rec.sessions_active} "
            f"inflight={rec.requests_inflight} "
            f"turns_ok={rec.turns_ok} mismatch={rec.turns_mismatch}",
            flush=True,
        )


async def run(args):
    endpoints = [e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]
    corpus = _load_corpus(args.corpus)
    arrivals = _load_arrivals(path=args.arrivals, corpus=corpus, endpoints=endpoints)
    print(
        f"{len(arrivals)} arrivals of {len(corpus)} trajectories over "
        f"{arrivals[-1].arrival_offset / 60:.1f} min -> {len(endpoints)} endpoints"
    )

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    rec = Recorder(args.output)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        salts = await _fetch_salts(
            http=http, endpoint=endpoints[0], replica_ids=[a.replica_id for a in arrivals]
        )
        t0 = time.monotonic()
        progress = asyncio.create_task(
            _report_progress(
                rec=rec, total=len(arrivals), t0=t0, interval=args.progress_interval
            )
        )
        if args.duration is not None:
            rec.deadline = time.time() + args.duration
        replay = asyncio.gather(
            *(
                _replay_session(
                    http=http,
                    arrival=a,
                    session=corpus[a.session_id],
                    salt=salts[a.replica_id],
                    t0=t0,
                    rec=rec,
                )
                for a in arrivals
            )
        )
        done, _ = await asyncio.wait({replay}, timeout=args.duration)
        if not done:
            print(
                f"duration {args.duration:.0f}s reached: cutting "
                f"{rec.sessions_active} active sessions, "
                f"{rec.requests_inflight} in-flight requests"
            )
            replay.cancel()
            await asyncio.gather(replay, return_exceptions=True)
        progress.cancel()
    rec.close()

    mismatch_ratio = rec.turns_mismatch / max(rec.turns_ok, 1)
    print(
        f"finished: sessions done={rec.sessions_done} failed={rec.sessions_failed}, "
        f"turns ok={rec.turns_ok} failed={rec.turns_failed} "
        f"dropped_after_deadline={rec.turns_late}, "
        f"fidelity mismatch={rec.turns_mismatch} ({mismatch_ratio:.2%})"
    )
    if mismatch_ratio > 0.01:
        print(
            "WARNING: >1% of turns have prompt_tokens != len(input_ids) or "
            "completion_tokens != decode_num_tokens; this run is invalid "
            "(REPLAY.md 7.2)",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", required=True, help="output of prepare_corpus.py")
    parser.add_argument("--arrivals", required=True, help="output of make_arrivals.py")
    parser.add_argument(
        "--endpoints", required=True, help="comma-separated SGLang base URLs"
    )
    parser.add_argument("--output", required=True, help="per-turn CSV (appended)")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="per-request timeout in seconds (default: none)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds from start: pending arrivals are "
        "skipped, in-flight requests are aborted, and turns finishing later are "
        "not recorded (default: run until every session finishes)",
    )
    parser.add_argument("--progress-interval", type=float, default=30.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
