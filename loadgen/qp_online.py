"""Queueing-latency prediction with an online decode-length distribution.

Same v2 Monte Carlo predictor as qp_predict.py, but the decode-length pool is
what an online server could know at the target's arrival: the decode lengths
of every turn of the run (both instances) that finished before that moment.
The first WARMUP_FRAC of the run's turns, by send time, only feed the pool and
are never evaluated. The offline-profiled step costs are unchanged.

For comparison, each target is also predicted with the global
leave-one-session-out pool (mc_v2_global) and with true lengths (oracle_v2).

Usage:
    python qp_online.py logs/sweep_*_r* --corpus data/hermes_clean.jsonl \
        -o work_dirs/queueing_predict_online.csv
"""

import argparse
import bisect
import csv
import glob
import os
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_predict import (  # noqa: E402
    MC_SAMPLES, DecodeModel, Snapshot, _rate_of, predict_mc_v2, predict_oracle_v2,
)
from qp_sim2 import fit_step_costs, simulate_wait_v2, split_state  # noqa: E402
from qp_state import InstanceView, collapse_onset, load_turns  # noqa: E402

WARMUP_FRAC = 0.2


class OnlinePool:
    """Sorted decode lengths of turns finished so far; advance() in time order."""

    def __init__(self, turns: list):
        self._pending = sorted(turns, key=lambda t: t.t_end)
        self._next = 0
        self.lengths = []

    def advance(self, t: float):
        while self._next < len(self._pending) and self._pending[self._next].t_end < t:
            bisect.insort(self.lengths, self._pending[self._next].decode_len)
            self._next += 1


def predict_mc_online(*, snap: Snapshot, costs, pool: list, rng: random.Random) -> list:
    waits = []
    for _ in range(MC_SAMPLES):
        running, inflight, queue, busy = split_state(
            snap=snap, decode_of=lambda t, g: DecodeModel.sample_remaining(pool, g, rng))
        waits.append(simulate_wait_v2(running=running, inflight=inflight, queue=queue, costs=costs, busy_s=busy))
    return waits


def _select_targets(*, views: dict, t_lo: float, t_hi: dict, rng: random.Random, n_uniform: int, n_queued: int) -> list:
    cand = [(ep, x) for ep, v in views.items() for x in v.turns if t_lo <= x.t_send <= t_hi[ep]]
    queued = [p for p in cand if p[1].t_admit - p[1].t_send > 1.0]
    picked = [(ep, x, "uniform") for ep, x in rng.sample(cand, min(n_uniform, len(cand)))]
    picked += [(ep, x, "queued") for ep, x in rng.sample(queued, min(n_queued, len(queued)))]
    return sorted(picked, key=lambda p: p[1].t_send)


def evaluate_run(*, run_dir: str, model: DecodeModel, costs, rng: random.Random,
                 n_uniform: int, n_queued: int) -> list:
    turns = load_turns(os.path.join(run_dir, "loadgen.csv"))
    views = {ep: InstanceView(ts) for ep, ts in turns.items()}
    all_turns = sorted((x for ts in turns.values() for x in ts), key=lambda x: x.t_send)
    t_start = all_turns[0].t_send
    t_warm = all_turns[int(WARMUP_FRAC * len(all_turns))].t_send
    info = dict(l.split(": ", 1) for l in open(os.path.join(run_dir, "run_info.txt")).read().splitlines() if ": " in l)
    # The window tail is cut because turns unfinished at the end are missing from loadgen.csv.
    t_tail = t_start + float(info["duration_s"]) - 3600
    t_hi = {ep: t_tail for ep in views}
    onsets = {ep: collapse_onset(v.turns) for ep, v in views.items()}
    targets = _select_targets(views=views, t_lo=t_warm, t_hi=t_hi, rng=rng, n_uniform=n_uniform, n_queued=n_queued)
    pool = OnlinePool(all_turns)
    rows = []
    for ep, x, sample in targets:
        pool.advance(x.t_send)
        snap = Snapshot(view=views[ep], x=x, t0=x.t_send)
        online = predict_mc_online(snap=snap, costs=costs, pool=pool.lengths, rng=rng)
        glob_ = predict_mc_v2(snap=snap, costs=costs, model=model, rng=rng)
        rows.append({
            "run": os.path.basename(os.path.normpath(run_dir)),
            "rate": _rate_of(run_dir),
            "sample": sample,
            "endpoint": ep,
            "t_offset_s": round(x.t_send - t_start, 1),
            "pre_collapse": int(x.t_send < onsets[ep] - 300),
            "pool_size": len(pool.lengths),
            "actual_s": round(x.t_admit - x.t_send, 3),
            "mc_v2_online_mean": round(statistics.mean(online), 3),
            "mc_v2_online_p90": round(float(np.percentile(online, 90)), 3),
            "mc_v2_global_mean": round(statistics.mean(glob_), 3),
            "mc_v2_global_p90": round(float(np.percentile(glob_, 90)), 3),
            "oracle_v2": round(predict_oracle_v2(snap=snap, costs=costs), 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--n-uniform", type=int, default=500)
    parser.add_argument("--n-queued", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    run_dirs = sorted({d.rstrip("/") for p in args.run_dirs for d in glob.glob(p) if os.path.isdir(d)}, key=_rate_of)
    all_views = [InstanceView(ts) for d in run_dirs for ts in load_turns(os.path.join(d, "loadgen.csv")).values()]
    costs = fit_step_costs(all_views, rng)
    model = DecodeModel(args.corpus)
    rows = []
    for d in run_dirs:
        r = evaluate_run(run_dir=d, model=model, costs=costs, rng=rng,
                         n_uniform=args.n_uniform, n_queued=args.n_queued)
        print(f"{os.path.basename(d)}: {len(r)} targets, pool size at first target {r[0]['pool_size']}", flush=True)
        rows += r
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
