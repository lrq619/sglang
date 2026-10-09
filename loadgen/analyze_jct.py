"""Censored session JCT per run, for comparing cache policies on the same trace.

For every session that arrived within the window:
    T_i = min(t_end, window_end) - t_arrival
where a session that is cut at the window end or failed counts as window_end
(failed sessions are censored, never dropped, so errors cannot flatter a
policy). The LLM-attributable variant subtracts the tool wait the session
actually spent, which is identical across policies for the same progress.

Post-pressure variant: drops sessions that finished before memory pressure
first appeared on their endpoint (either pool's free space reaching ~0 in
metrics.csv). Before that point every cache policy behaves identically, so
those sessions only dilute the comparison.

Runs with sessions.csv use it directly; older runs are reconstructed from
arrivals.jsonl + loadgen.csv + the corpus (source=reconstructed: the tool wait
after a cut session's last finished turn is unknown and left out).

Usage:
    python analyze_jct.py logs/sweep_X_r6_* logs/sweep_X_r6.5_* -o jct.csv
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Optional

# A pool counts as exhausted below this fraction of its idle free space.
PRESSURE_FREE_FRACTION = 0.01

SUMMARY_COLUMNS = [
    "run",
    "source",
    "duration_s",
    "arrived",
    "done",
    "cut",
    "failed",
    "not_arrived",
    "jct_mean",
    "jct_p50",
    "jct_p90",
    "jct_p99",
    "llm_mean",
    "llm_p50",
    "llm_p90",
    "t_pressure_s",
    "pp_sessions",
    "pp_jct_mean",
    "pp_llm_mean",
]


# ===== Session table =====


def _read_run_info(run_dir: str) -> dict:
    info = {}
    path = os.path.join(run_dir, "run_info.txt")
    if os.path.exists(path):
        for line in open(path):
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
    return info


def _load_sessions(*, run_dir: str, corpus_override: Optional[str]) -> tuple:
    path = os.path.join(run_dir, "sessions.csv")
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        return [_parse_session_row(r) for r in rows], "sessions.csv"
    return _reconstruct_sessions(run_dir=run_dir, corpus_override=corpus_override), "reconstructed"


def _parse_session_row(r: dict) -> dict:
    f = lambda k: float(r[k]) if r[k] != "" else None
    return {
        "replica_id": r["replica_id"],
        "endpoint": r["endpoint"],
        "status": r["status"],
        "t_arrival": f("t_arrival"),
        "t_end": f("t_end"),
        "tool_wait_s": f("tool_wait_s") or 0.0,
        "window_start": f("window_start"),
        "window_end": f("window_end"),
    }


def _reconstruct_sessions(*, run_dir: str, corpus_override: Optional[str]) -> list:
    info = _read_run_info(run_dir)
    duration = float(info["duration_s"])
    corpus_path = corpus_override or info.get("corpus")
    if not corpus_path or not os.path.exists(corpus_path):
        sys.exit(f"{run_dir}: no sessions.csv and corpus not found (pass --corpus)")
    turns_total = {
        s["session_id"]: len(s["turns"])
        for s in map(json.loads, open(corpus_path))
    }
    arrivals = [json.loads(l) for l in open(os.path.join(run_dir, "arrivals.jsonl")) if l.strip()]
    by_replica = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run_dir, "loadgen.csv"))):
        by_replica[r["replica_id"]].append(r)

    offset = {str(a["replica_id"]): float(a["arrival_offset"]) for a in arrivals}
    # The first turn of each session is sent the moment it arrives.
    window_start = min(
        min(float(r["t_send"]) for r in rows if r["t_send"]) - offset[rid]
        for rid, rows in by_replica.items()
        if any(r["t_send"] for r in rows)
    )
    window_end = window_start + duration
    return [
        _reconstruct_one(
            arrival=a,
            rows=by_replica.get(str(a["replica_id"]), []),
            turns_total=turns_total[a["session_id"]],
            window_start=window_start,
            window_end=window_end,
        )
        for a in arrivals
    ]


def _reconstruct_one(
    *, arrival: dict, rows: list, turns_total: int, window_start: float, window_end: float
) -> dict:
    ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: float(r["t_end"]))
    errors = [r for r in rows if r["status"] != "ok"]
    t_arrival = window_start + float(arrival["arrival_offset"])
    if t_arrival >= window_end:
        status, t_end = "not_arrived", None
    elif len(ok) == turns_total:
        status, t_end = "done", float(ok[-1]["t_end"])
    elif errors:
        status, t_end = "failed", float(errors[0]["t_end"])
    else:
        status, t_end = "cut", window_end
    # Count waits known to have fully elapsed: a failed session sent the turn
    # after its last finished one; for a cut session that last wait is unknown.
    waited = ok if status == "failed" else ok[:-1]
    tool_wait = sum(float(r["turn_toolcall_latency"] or 0) for r in waited)
    return {
        "replica_id": str(arrival["replica_id"]),
        "endpoint": rows[0]["endpoint"] if rows else "",
        "status": status,
        "t_arrival": None if status == "not_arrived" else t_arrival,
        "t_end": t_end,
        "tool_wait_s": tool_wait,
        "window_start": window_start,
        "window_end": window_end,
    }


# ===== Censored JCT =====


def _censored_jct(s: dict) -> float:
    end = s["window_end"]
    if s["status"] == "done" and (end is None or s["t_end"] <= end):
        end = s["t_end"]
    return end - s["t_arrival"]


def _pressure_onset(*, run_dir: str) -> dict:
    """First unix time either pool's free space hits ~0, per endpoint."""
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return {}
    by_ep = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_ep[r["endpoint"]].append(r)
    onset = {}
    for ep, rows in by_ep.items():
        if "kv_available_tokens" not in rows[0]:
            return {}
        for pool in ("kv_available_tokens", "mamba_available_tokens"):
            vals = [(float(r["t"]), float(r[pool])) for r in rows if r.get(pool)]
            if not vals:
                continue
            idle_free = max(v for _, v in vals)
            hit = [t for t, v in vals if v <= PRESSURE_FREE_FRACTION * idle_free]
            if hit:
                onset[ep] = min(onset.get(ep, hit[0]), hit[0])
    return onset


def _stats(values: list) -> tuple:
    if not values:
        return (None, None, None)
    p90 = statistics.quantiles(values, n=10)[-1] if len(values) > 1 else values[0]
    return (statistics.mean(values), statistics.median(values), p90)


def summarize(*, run_dir: str, corpus_override: Optional[str]) -> dict:
    sessions, source = _load_sessions(run_dir=run_dir, corpus_override=corpus_override)
    arrived = [s for s in sessions if s["status"] != "not_arrived"]
    jct = [_censored_jct(s) for s in arrived]
    llm = [_censored_jct(s) - s["tool_wait_s"] for s in arrived]

    onset = _pressure_onset(run_dir=run_dir)
    pp = [
        s for s in arrived
        if s["endpoint"] in onset
        and not (s["status"] == "done" and s["t_end"] < onset[s["endpoint"]])
    ]
    window_start = arrived[0]["window_start"] if arrived else None
    count = lambda st: sum(s["status"] == st for s in sessions)
    row = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "source": source,
        "duration_s": _read_run_info(run_dir).get("duration_s", ""),
        "arrived": len(arrived),
        "done": count("done"),
        "cut": count("cut"),
        "failed": count("failed"),
        "not_arrived": count("not_arrived"),
        "t_pressure_s": (min(onset.values()) - window_start) if onset and window_start else None,
        "pp_sessions": len(pp) if onset else None,
        "pp_jct_mean": _stats([_censored_jct(s) for s in pp])[0] if onset else None,
        "pp_llm_mean": _stats([_censored_jct(s) - s["tool_wait_s"] for s in pp])[0] if onset else None,
    }
    row.update(zip(["jct_mean", "jct_p50", "jct_p90"], _stats(jct)))
    row["jct_p99"] = statistics.quantiles(jct, n=100)[-1] if len(jct) > 1 else None
    row.update(zip(["llm_mean", "llm_p50", "llm_p90"], _stats(llm)))
    return row


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("-o", "--output", help="summary CSV (one row per run)")
    parser.add_argument("--corpus", help="corpus for reconstructing runs without sessions.csv")
    args = parser.parse_args()

    rows = [summarize(run_dir=d, corpus_override=args.corpus) for d in args.run_dirs]
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows({k: _fmt(r[k]) for k in SUMMARY_COLUMNS} for r in rows)
    for r in rows:
        print(
            f"{r['run']}: [{r['source']}] arrived={r['arrived']} done={r['done']} "
            f"cut={r['cut']} failed={r['failed']} | censored JCT mean/p50/p90 = "
            f"{_fmt(r['jct_mean'])}/{_fmt(r['jct_p50'])}/{_fmt(r['jct_p90'])}s, "
            f"LLM-only mean = {_fmt(r['llm_mean'])}s"
            + (
                f" | pressure at {_fmt(r['t_pressure_s'])}s, post-pressure "
                f"n={r['pp_sessions']} JCT mean = {_fmt(r['pp_jct_mean'])}s"
                if r["t_pressure_s"] is not None else ""
            )
        )


if __name__ == "__main__":
    main()
