"""Scan a loadgen run for signs that a server stopped making progress.

Reports, per endpoint, every interval of at least --stall-s where requests
were queued but none was running (metrics.csv), and every gap of at least
--no-progress-s with requests queued and no turn finishing (loadgen.csv). An
empty report means the run never stalled that way. Prefix-pin force releases
are summarized alongside, since they are how the server gets out of such a state.

Usage:
    python check_deadlock.py logs/<run_dir> [--stall-s 30] [--no-progress-s 120]
"""

import argparse
import bisect
import csv
import os
from collections import defaultdict


def _metrics_by_endpoint(run_dir: str) -> dict:
    by_ep = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))):
        if r["error"] or not r["num_queue_reqs"]:
            continue
        by_ep[r["endpoint"]].append(r)
    return dict(by_ep)


def _intervals(rows: list, predicate, min_s: float) -> list:
    """Maximal runs of consecutive samples satisfying predicate, lasting >= min_s."""
    found, start = [], None
    for prev, r in zip([None] + rows[:-1], rows):
        if predicate(r):
            start = start if start is not None else float(r["t"])
        elif start is not None:
            if float(prev["t"]) - start >= min_s:
                found.append((start, float(prev["t"])))
            start = None
    if start is not None and float(rows[-1]["t"]) - start >= min_s:
        found.append((start, float(rows[-1]["t"])))
    return found


def _no_progress_gaps(*, rows: list, ends: list, min_s: float) -> list:
    """Gaps between finished turns lasting >= min_s while the queue was non-empty."""
    gaps = []
    t_lo, t_hi = float(rows[0]["t"]), float(rows[-1]["t"])
    marks = [t_lo] + [t for t in ends if t_lo <= t <= t_hi] + [t_hi]
    queued_t = [float(r["t"]) for r in rows if float(r["num_queue_reqs"]) > 0]
    for a, b in zip(marks, marks[1:]):
        if b - a < min_s:
            continue
        i = bisect.bisect_left(queued_t, a)
        if i < len(queued_t) and queued_t[i] <= b:
            gaps.append((a, b))
    return gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    # Arbitrary: long enough to rule out a step boundary or a scrape gap.
    parser.add_argument("--stall-s", type=float, default=30.0)
    parser.add_argument("--no-progress-s", type=float, default=120.0)
    args = parser.parse_args()

    metrics = _metrics_by_endpoint(args.run_dir)
    ends_by_ep = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(args.run_dir, "loadgen.csv"))):
        if r["status"] == "ok":
            ends_by_ep[r["endpoint"]].append(float(r["t_end"]))
    t0 = min(float(rows[0]["t"]) for rows in metrics.values())

    stalled = False
    for ep, rows in sorted(metrics.items()):
        ends = sorted(ends_by_ep.get(ep, []))
        stalls = _intervals(
            rows,
            lambda r: (
                float(r["num_queue_reqs"]) > 0 and float(r["num_running_reqs"]) == 0
            ),
            args.stall_s,
        )
        gaps = _no_progress_gaps(rows=rows, ends=ends, min_s=args.no_progress_s)
        last = rows[-1]
        releases = {
            k.removeprefix("pin_force_released_"): last.get(k) or "0"
            for k in last
            if k.startswith("pin_force_released_")
        }
        print(
            f"{ep}: {len(ends)} turns finished, max queue "
            f"{max(float(r['num_queue_reqs']) for r in rows):.0f}, "
            f"pin force releases {releases}, retracted "
            f"{last.get('num_retracted_requests_total') or '0'}"
        )
        for a, b in stalls:
            print(
                f"  STALL queued-but-idle {a - t0:7.0f}s .. {b - t0:7.0f}s ({b - a:.0f}s)"
            )
        for a, b in gaps:
            print(
                f"  NO PROGRESS            {a - t0:7.0f}s .. {b - t0:7.0f}s ({b - a:.0f}s)"
            )
        stalled |= bool(stalls or gaps)
    print("RESULT:", "stalls found" if stalled else "no stall found")


if __name__ == "__main__":
    main()
