"""Score the sidecar's first-sight predictions (predictions.csv) against the queue
times loadgen measured (loadgen.csv), joined by rid r{replica}-t{turn}-{pid}.

predicted queue time = wait_age_s + pred_mean_s (the prediction covers only the
time left after the poll saw the request); actual = turn_queuing_latency.
Acceptance (.claude/queuing_predictor.md section 4): on actual > 1 s, median
relative error <= 15% and Spearman >= 0.95.

Usage:
    python -m controller.queuing_predictor.eval_shadow logs/<run dir> [--min-wait 1.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def load_joined(run_dir: str) -> list:
    actual = {}
    with open(os.path.join(run_dir, "loadgen.csv")) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok" or not r["turn_queuing_latency"]:
                continue
            actual[(r["replica_id"], r["turn_id"])] = r
    rows = []
    with open(os.path.join(run_dir, "predictions.csv")) as f:
        for p in csv.DictReader(f):
            replica, turn, _ = p["rid"].split("-")
            a = actual.get((replica[1:], turn[1:]))
            if a is None:
                continue
            rows.append({**p, "actual_s": float(a["turn_queuing_latency"]),
                         "actual_prefill_s": a["turn_prefill_latency"]})
    return rows


def summarize(rows: list, *, min_wait: float) -> dict:
    act = np.array([r["actual_s"] for r in rows])
    age = np.array([float(r["wait_age_s"] or 0) for r in rows])
    mean = age + np.array([float(r["pred_mean_s"]) for r in rows])
    p50 = age + np.array([float(r["pred_p50_s"]) for r in rows])
    p90 = age + np.array([float(r["pred_p90_s"]) for r in rows])
    sel = act > min_wait
    out = {"matched": len(rows), f"actual_gt_{min_wait}s": int(sel.sum())}
    if sel.sum() == 0:
        return out
    a, m = act[sel], mean[sel]
    rel = np.abs(m - a) / a
    out.update({
        "median_rel_err": float(np.median(rel)),
        "p90_rel_err": float(np.quantile(rel, 0.9)),
        "median_signed_rel_err": float(np.median((m - a) / a)),
        "spearman": spearman(m, a),
        "median_rel_err_p50": float(np.median(np.abs(p50[sel] - a) / a)),
        "coverage_le_p90": float(np.mean(a <= p90[sel])),
        "actual_median_s": float(np.median(a)),
        "pred_median_s": float(np.median(m)),
        "median_wait_age_at_pred_s": float(np.median(age[sel])),
        "prefix_known_frac": float(np.mean([r["prefix_known"] == "1" for r, s in zip(rows, sel) if s])),
        "predict_ms_median": float(np.median([float(r["predict_ms"]) for r in rows])),
    })
    for lo, hi in [(1, 10), (10, 60), (60, 1e9)]:
        b = (a > lo) & (a <= hi)
        if b.sum():
            out[f"bucket_{lo}-{int(hi) if hi < 1e9 else 'inf'}s"] = {
                "n": int(b.sum()), "median_rel_err": float(np.median(rel[b])),
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--min-wait", type=float, default=1.0)
    p.add_argument(
        "--max-age", type=float, default=1.0,
        help="first-sight subset: the request had waited at most this long when predicted",
    )
    args = p.parse_args()
    rows = load_joined(args.run_dir)
    fresh = [r for r in rows if float(r["wait_age_s"] or 0) <= args.max_age]
    res = {
        "all": summarize(rows, min_wait=args.min_wait),
        f"first_sight_age_le_{args.max_age}s": summarize(fresh, min_wait=args.min_wait),
    }
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.run_dir, "predictions_eval.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
