"""Monte Carlo sample count P vs a point estimate, for queueing-latency prediction.

Same setting as qp_online.py (online decode-length pool, first 20% of each run's
turns as warm-up, v2 simulator). Per target:
  * point: every request's remaining decode length is the conditional mean
    E[D - g | D > g] of the online pool; one simulation;
  * point_median: same with the conditional median of D - g given D > g;
  * mc_P: mean wait over the first P of 32 Monte Carlo simulations, each with
    lengths sampled from D | D > g; P in {1, 2, 4, 8, 16, 32};
  * oracle: true lengths, one simulation.

Usage:
    python qp_psweep.py logs/sweep_*_r* -o work_dirs/queueing_predict_psweep.csv
"""

import argparse
import bisect
import csv
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_online import WARMUP_FRAC, OnlinePool, _select_targets  # noqa: E402
from qp_predict import (  # noqa: E402
    DecodeModel, Snapshot, _rate_of, fit_calibration, predict_continuum, predict_fluid, predict_oracle_v2,
)
from qp_sim2 import fit_step_costs, simulate_wait_v2, split_state  # noqa: E402
from qp_state import InstanceView, collapse_onset, load_turns  # noqa: E402

P_VALUES = [1, 2, 4, 8, 16, 32]
# Past every observed length: same arbitrary short tail as sample_remaining.
TAIL_REMAINING = 64


class ConditionalMean:
    """E[D - g | D > g] and median(D - g | D > g) over a sorted pool; suffix sums
    are rebuilt when the pool grows."""

    def __init__(self):
        self._size = -1
        self._arr = None
        self._suffix = None

    def refresh(self, pool: list):
        if len(pool) != self._size:
            self._arr = np.asarray(pool, dtype=np.float64)
            self._suffix = np.concatenate([np.cumsum(self._arr[::-1])[::-1], [0.0]])
            self._size = len(pool)

    def remaining(self, g: int) -> int:
        i = bisect.bisect_right(self._arr, g)
        n = self._size - i
        if n <= 0:
            return TAIL_REMAINING
        return max(1, int(round(self._suffix[i] / n - g)))

    def median_remaining(self, g: int) -> int:
        i = bisect.bisect_right(self._arr, g)
        n = self._size - i
        if n <= 0:
            return TAIL_REMAINING
        return max(1, int(self._arr[i + n // 2]) - g)


def evaluate_run(*, run_dir: str, costs, cal, rng: random.Random, n_uniform: int, n_queued: int) -> list:
    turns = load_turns(os.path.join(run_dir, "loadgen.csv"))
    views = {ep: InstanceView(ts) for ep, ts in turns.items()}
    all_turns = sorted((x for ts in turns.values() for x in ts), key=lambda x: x.t_send)
    t_start = all_turns[0].t_send
    t_warm = all_turns[int(WARMUP_FRAC * len(all_turns))].t_send
    info = dict(l.split(": ", 1) for l in open(os.path.join(run_dir, "run_info.txt")).read().splitlines() if ": " in l)
    t_tail = t_start + float(info["duration_s"]) - 3600
    onsets = {ep: collapse_onset(v.turns) for ep, v in views.items()}
    targets = _select_targets(views=views, t_lo=t_warm, t_hi={ep: t_tail for ep in views},
                              rng=rng, n_uniform=n_uniform, n_queued=n_queued)
    pool = OnlinePool(all_turns)
    cmean = ConditionalMean()
    rows = []
    for ep, x, sample in targets:
        pool.advance(x.t_send)
        cmean.refresh(pool.lengths)
        snap = Snapshot(view=views[ep], x=x, t0=x.t_send)
        running, inflight, queue, busy = split_state(snap=snap, decode_of=lambda t, g: cmean.remaining(g))
        point = simulate_wait_v2(running=running, inflight=inflight, queue=queue, costs=costs, busy_s=busy)
        running, inflight, queue, busy = split_state(snap=snap, decode_of=lambda t, g: cmean.median_remaining(g))
        point_median = simulate_wait_v2(running=running, inflight=inflight, queue=queue, costs=costs, busy_s=busy)
        sims = []
        for _ in range(max(P_VALUES)):
            running, inflight, queue, busy = split_state(
                snap=snap, decode_of=lambda t, g: DecodeModel.sample_remaining(pool.lengths, g, rng))
            sims.append(simulate_wait_v2(running=running, inflight=inflight, queue=queue, costs=costs, busy_s=busy))
        row = {
            "run": os.path.basename(os.path.normpath(run_dir)),
            "rate": _rate_of(run_dir),
            "sample": sample,
            "endpoint": ep,
            "t_offset_s": round(x.t_send - t_start, 1),
            "pre_collapse": int(x.t_send < onsets[ep] - 300),
            "running": len(snap.running),
            "queue_ahead": len(snap.queue_ahead),
            "actual_s": round(x.t_admit - x.t_send, 3),
            "point": round(point, 3),
            "point_median": round(point_median, 3),
            "oracle": round(predict_oracle_v2(snap=snap, costs=costs), 3),
            "continuum": round(predict_continuum(view=views[ep], snap=snap), 3),
            "fluid": round(predict_fluid(view=views[ep], snap=snap, cal=cal), 3),
        }
        for p in P_VALUES:
            row[f"mc_{p}"] = round(float(np.mean(sims[:p])), 3)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--n-uniform", type=int, default=500)
    parser.add_argument("--n-queued", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    run_dirs = sorted({d.rstrip("/") for p in args.run_dirs for d in glob.glob(p) if os.path.isdir(d)}, key=_rate_of)
    all_views = [InstanceView(ts) for d in run_dirs for ts in load_turns(os.path.join(d, "loadgen.csv")).values()]
    costs = fit_step_costs(all_views, rng)
    # Separate RNG so adding the fluid baseline leaves target selection and MC draws unchanged.
    cal = fit_calibration(all_views, random.Random(args.seed + 1))
    rows = []
    for d in run_dirs:
        r = evaluate_run(run_dir=d, costs=costs, cal=cal, rng=rng, n_uniform=args.n_uniform, n_queued=args.n_queued)
        print(f"{os.path.basename(d)}: {len(r)} targets", flush=True)
        rows += r
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
