"""Filter a replay CSV (see .claude/REPLAY.md) down to its clean sessions.

A session is clean when every row has prompt_available = True, every row except
the last has turn_toolcall_latency_ms, and it is not on the exclusion list.
For hermes.csv this keeps 36 sessions (DR 20, TB 16).

Output is JSONL, one session per line, holding only the fields replay needs:
    {"session_id", "benchmark", "task", "turns": [
        {"turn_index", "prompt_reset", "prev_response_in_history",
         "prefill_token_ids", "decode_num_tokens", "turn_toolcall_latency_ms"}, ...]}

Usage:
    python prepare_corpus.py hermes.csv -o hermes_clean.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

# Sessions whose transcript does not match the recording (REPLAY.md 2.1).
DEFAULT_EXCLUDE = ["854fa26a036b557a"]


def parse_bool(s: str) -> bool:
    return s.strip().lower() == "true"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_path")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=DEFAULT_EXCLUDE,
        help="session ids (or unique prefixes) to drop",
    )
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    sessions = defaultdict(list)
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            sessions[row["session_id"]].append(row)

    kept, dropped = [], []
    for sid, rows in sessions.items():
        rows.sort(key=lambda r: int(r["turn_index"]))
        if any(sid.startswith(p) for p in args.exclude):
            dropped.append((sid, "excluded"))
            continue
        if not all(parse_bool(r["prompt_available"]) for r in rows):
            dropped.append((sid, "prompt_available=False"))
            continue
        if any(r["turn_toolcall_latency_ms"].strip() == "" for r in rows[:-1]):
            dropped.append((sid, "missing turn_toolcall_latency_ms"))
            continue
        if not parse_bool(rows[0]["prompt_reset"]):
            dropped.append((sid, "first row is not prompt_reset"))
            continue
        kept.append((sid, rows))

    with open(args.output, "w") as f:
        for sid, rows in kept:
            turns = []
            for i, r in enumerate(rows):
                lat = r["turn_toolcall_latency_ms"].strip()
                turns.append(
                    {
                        "turn_index": int(r["turn_index"]),
                        "prompt_reset": parse_bool(r["prompt_reset"]),
                        "prev_response_in_history": parse_bool(
                            r["prev_response_in_history"]
                        ),
                        "prefill_token_ids": json.loads(r["prefill_token_ids"]),
                        "decode_num_tokens": int(r["decode_num_tokens"]),
                        # The last turn has no follow-up request, so no wait.
                        "turn_toolcall_latency_ms": (
                            float(lat) if lat and i + 1 < len(rows) else None
                        ),
                    }
                )
            rec = {
                "session_id": sid,
                "benchmark": rows[0]["benchmark"],
                "task": rows[0]["task"],
                "turns": turns,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    by_bench = defaultdict(int)
    for _, rows in kept:
        by_bench[rows[0]["benchmark"]] += 1
    n_turns = sum(len(rows) for _, rows in kept)
    print(f"kept {len(kept)} sessions / {n_turns} turns -> {args.output}")
    for b, n in sorted(by_bench.items()):
        print(f"  {b}: {n}")
    reasons = defaultdict(int)
    for _, why in dropped:
        reasons[why] += 1
    print(f"dropped {len(dropped)} sessions: {dict(reasons)}")


if __name__ == "__main__":
    main()
