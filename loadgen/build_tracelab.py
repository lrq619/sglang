"""Build trace_lab_single_round.csv / trace_lab_multi_round.csv (the hermes.csv
replay format, .claude/REPLAY.md) from the TraceLab coding-agent trace
(github.com/uw-syfi/TraceLab, CC BY 4.0).

TraceLab keeps token counts only, so every token is random (REPLAY.md 2.2,
shape corpus): the load matches the recording, the content does not.

Terms: session -> round (one user input and the agent loop it triggers) ->
turn (one LLM call and its tools). A turn's turn_toolcall_latency_ms is its
gap: the wait before the next turn's request, either a tool gap (inside a
round: the slowest tool of this turn) or a user gap (the next turn opens a new
round: user message time - this turn's end). turn_toolcall_latency_source
says which ("tool" / "user").

Per turn k, with in_k = input_tokens_total and out_k = output_tokens:
  * k = 0: prompt_reset, in_0 random tokens;
  * in_k >= in_{k-1} + out_{k-1}: the reply stays in history, append the rest;
  * in_k >= in_{k-1}: the reply left history (reasoning is not resent), drop it
    (prev_response_in_history False) and append in_k - in_{k-1};
  * otherwise the history was rewritten (compaction): prompt_reset, in_k tokens.
Every request is then exactly in_k tokens long and decodes out_k tokens.

A fixed-seed random sample of --sessions-per-class sessions per class is
written (the full set is ~7 GB of token ids). Sessions are kept when they have
>= 2 turns, no user gap > 1h, total gap <= 2h, every turn fits the context
window (in_k + out_k <= --context-length), and fewer than half of their gaps
are untracked (no tool, no user message, yet the context grows: a batch of
Codex sessions whose tools went unrecorded, ~14M prefill tokens each).
single_round = no user gap; multi_round = at least one.

Usage:
    python build_tracelab.py ~/scratch/tracelab/syfi_coding_trace.jsonl.gz -o ../data
    python build_tracelab.py ~/scratch/tracelab/syfi_coding_trace.jsonl.gz --dry-run
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

from build_hermes_v1 import CSV_COLUMNS

# Plain-text vocabulary range, clear of the special tokens above 248000.
VOCAB = (1000, 150000)
MAX_USER_GAP_S = 3600
MAX_TOTAL_GAP_S = 7200
MAX_UNTRACKED_GAP_SHARE = 0.5


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def read_sessions(path: str) -> dict:
    sessions = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            sessions[r["session_id"]].append(r)
    for turns in sessions.values():
        turns.sort(key=lambda r: r["round_index"])
    return sessions


def turn_times(r: dict):
    """(user message time or None, LLM start, LLM end) of one turn."""
    ev = [e for e in r["timing_events"] if e.get("timestamp")]
    inputs = [ts(e["timestamp"]) for e in ev if e["event_type"] in ("user_message", "tool_result")]
    outputs = [ts(e["timestamp"]) for e in ev if e["event_type"] not in ("user_message", "tool_result")]
    users = [ts(e["timestamp"]) for e in ev if e["event_type"] == "user_message"]
    start = max(inputs) if inputs else (min(outputs) if outputs else None)
    end = max(outputs) if outputs else start
    return (users[0] if users else None), start, end


def gaps(turns: list) -> list:
    """[(gap_s, kind)] for every turn but the last."""
    out = []
    for prev, cur in zip(turns, turns[1:]):
        user, _, _ = turn_times(cur)
        _, _, prev_end = turn_times(prev)
        if user is not None and prev_end is not None:
            out.append((max(0.0, user - prev_end), "user"))
        else:
            lat = [t["tool_wall_latency_ms"] for t in prev["tools"] if t.get("tool_wall_latency_ms") is not None]
            out.append((max(0.0, max(lat) / 1e3) if lat else 0.0, "tool"))
    return out


def classify(turns: list, g: list, context_length: int):
    """'single' / 'multi', or the reason the session is dropped."""
    if len(turns) < 2:
        return "drop: < 2 turns"
    if any(r["input_tokens_total"] + r["output_tokens"] > context_length for r in turns):
        return "drop: exceeds context"
    user = [s for s, k in g if k == "user"]
    if user and max(user) > MAX_USER_GAP_S:
        return "drop: user gap > 1h"
    if sum(s for s, _ in g) > MAX_TOTAL_GAP_S:
        return "drop: total gap > 2h"
    untracked = sum(1 for r, (_, k) in zip(turns, g) if k == "tool" and not r["tools"])
    if untracked >= MAX_UNTRACKED_GAP_SHARE * len(g):
        return "drop: untracked gaps"
    return "multi" if user else "single"


def prefill_plan(turns: list) -> list:
    """[(prompt_reset, prev_response_in_history, new prefill tokens)] per turn."""
    plan = []
    for k, r in enumerate(turns):
        n_in = r["input_tokens_total"]
        if k == 0 or n_in < turns[k - 1]["input_tokens_total"]:
            plan.append((True, True, n_in))
            continue
        prev_in, prev_out = turns[k - 1]["input_tokens_total"], turns[k - 1]["output_tokens"]
        if n_in >= prev_in + prev_out:
            plan.append((False, True, n_in - prev_in - prev_out))
        else:
            plan.append((False, False, n_in - prev_in))
    return plan


def build_rows(sid: str, turns: list, g: list, rng: random.Random) -> list:
    rows = []
    for k, (r, (reset, in_hist, n_new)) in enumerate(zip(turns, prefill_plan(turns))):
        n_in, n_out = r["input_tokens_total"], r["output_tokens"]
        _, start, end = turn_times(r)
        gap = g[k] if k < len(g) else None
        lat = [t.get("tool_wall_latency_ms") for t in r["tools"]]
        rows.append(
            {
                "session_id": sid,
                "benchmark": f"tracelab_{r['provider']}",
                "task": r.get("project", ""),
                "turn_index": k,
                "prompt_reset": reset,
                "prev_response_in_history": in_hist,
                "prompt_delta": "",
                "response": "",
                "prefill_token_ids": json.dumps([rng.randint(*VOCAB) for _ in range(n_new)]),
                "decode_token_ids": "",
                "prefill_num_tokens": n_new,
                "decode_num_tokens": n_out,
                "turn_toolcall_latency_ms": round(gap[0] * 1e3, 3) if gap else "",
                "turn_toolcall_latency_source": gap[1] if gap else "",
                "per_tool_latency_ms": json.dumps(lat),
                "tool_names": json.dumps([t["tool_name"] for t in r["tools"]]),
                "per_tool_latency_reliable": json.dumps([x is not None for x in lat]),
                "llm_start_ns": int(start * 1e9) if start is not None else "",
                "llm_end_ns": int(end * 1e9) if end is not None else "",
                "recorded_context_tokens": n_in,
                "recorded_decode_tokens": n_out,
                "prompt_available": True,
                "response_source": "random",
            }
        )
    return rows


def session_summary(turns: list, g: list) -> dict:
    return {
        "turns": len(turns),
        "mean_gap_s": sum(s for s, _ in g) / len(g),
        "prefill_tokens": sum(n for _, _, n in prefill_plan(turns)),
    }


def quantiles(xs: list) -> str:
    xs = sorted(xs)
    return " / ".join(f"{xs[int(q * (len(xs) - 1))]:.3g}" for q in (0.1, 0.5, 0.9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace", help="syfi_coding_trace.jsonl.gz")
    p.add_argument("-o", "--output-dir", default="data")
    p.add_argument("--context-length", type=int, default=262144)
    p.add_argument("--sessions-per-class", type=int, default=300, help="random sample size; 0 keeps all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="classify and sample only")
    args = p.parse_args()

    sessions = read_sessions(args.trace)
    stats = Counter()
    pool = defaultdict(list)
    for sid, turns in sessions.items():
        g = gaps(turns)
        kind = classify(turns, g, args.context_length)
        stats[kind] += 1
        if not kind.startswith("drop"):
            pool[kind].append((sid, session_summary(turns, g)))
    print(json.dumps(dict(stats), indent=1))

    sample_rng = random.Random(args.seed)
    for kind in ("single", "multi"):
        everyone = sorted(pool[kind])
        n = args.sessions_per_class or len(everyone)
        picked = sorted(sample_rng.sample(everyone, min(n, len(everyone))))
        for name, group in (("all", everyone), ("sample", picked)):
            print(
                f"{kind:6s} {name:6s} n={len(group):5d}"
                f"  turns {quantiles([s['turns'] for _, s in group])}"
                f"  mean gap s {quantiles([s['mean_gap_s'] for _, s in group])}"
                f"  prefill {sum(s['prefill_tokens'] for _, s in group) / 1e6:.1f}M tokens"
            )
        if not args.dry_run:
            write_csv(
                path=os.path.join(args.output_dir, f"trace_lab_{kind}_round.csv"),
                sessions={sid: sessions[sid] for sid, _ in picked},
                seed=args.seed,
            )


def write_csv(*, path: str, sessions: dict, seed: int) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for sid, turns in sessions.items():
            # Per-session seed: resampling never shifts another session's tokens.
            digest = hashlib.sha1(sid.encode()).hexdigest()
            rng = random.Random(f"{seed}:{digest}")
            w.writerows(build_rows(digest[:16], turns, gaps(turns), rng))
    print(f"wrote {len(sessions)} sessions -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")

if __name__ == "__main__":
    main()
