"""Re-predict a finished run offline from the sidecar's dumps, with decode lengths
from the pool (as online) and with the true lengths from loadgen.csv (oracle).

Inputs in the run dir: snapshots.jsonl.gz (--dump-snapshots), predict_log.jsonl
(--log-predicts), loadgen.csv. Targets:
  send    every /predict call, from the snapshot it used (the loadgen column);
  shadow  every request the first time a poll saw it waiting.
The pool is rebuilt by replaying the finished ring in poll order, as online.

Usage:
    python -m controller.queuing_predictor.replay logs/<run dir> --costs <costs.json>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from controller.queuing_predictor.costs import StepCosts
from controller.queuing_predictor.eval_shadow import spearman
from controller.queuing_predictor.pool import DecodePool
from controller.queuing_predictor.predictor import predict
from controller.queuing_predictor.simulator import build_scenario, simulate

BUCKETS = [(1, 10), (10, 60), (60, float("inf"))]


def _key(rid: str) -> Tuple[str, str]:
    """(replica_id, turn_id) from loadgen's rid r{replica}-t{turn}-{pid}."""
    replica, turn, _ = rid.split("-")
    return replica[1:], turn[1:]


def load_turns(run_dir: str) -> Dict[Tuple[str, str], dict]:
    with open(os.path.join(run_dir, "loadgen.csv")) as f:
        return {
            (r["replica_id"], r["turn_id"]): r
            for r in csv.DictReader(f)
            if r["status"] == "ok"
        }


def oracle_wait(
    *, snap: dict, costs: StepCosts, input_len: int, prefix_len: int,
    max_new: int, prefix_path: Optional[list], ratio: float,
    true_len: Callable[[str], Optional[int]], pool: DecodePool, rng: np.random.Generator,
) -> Tuple[float, int]:
    """One simulation with every decode length known; a request missing from
    loadgen.csv (cut by the run deadline) falls back to a pool draw."""
    sc = build_scenario(
        snap=snap, target_input_len=input_len, target_prefix_len=prefix_len,
        target_max_new=max_new, prefix_ratio=ratio, costs=costs,
        target_prefix_path=prefix_path,
    )
    missing = 0

    def remaining(rid: str, generated: int) -> int:
        nonlocal missing
        d = true_len(rid)
        if d is None:
            missing += 1
            return int(pool.sample_remaining(np.array([generated]), rng)[0])
        return max(1, d - generated)

    running = np.array(
        [remaining(r["rid"], r["output_len"]) for r in snap["running"]], dtype=float
    )
    waiting = snap["waiting"]
    queue = [remaining(w["rid"], w["output_len"]) for w in waiting]
    # A queue longer than the reply was padded by cycling the head (build_scenario).
    queue = [queue[i % len(queue)] for i in range(len(sc.queue))] if queue else []
    chunked = snap["chunked"]
    chunked_len = remaining(chunked["rid"], 0) if chunked else 0
    wait, _ = simulate(
        sc=sc, running_remaining=running, chunked_decode=chunked_len,
        queue_decode=np.array(queue + [0]), costs=costs,
    )
    return wait, missing


def replay(args) -> list:
    turns = load_turns(args.run_dir)

    def true_by_rid(rid: str) -> Optional[int]:
        turn = turns.get(_key(rid))
        return None if turn is None else int(turn["completion_tokens"])

    costs = StepCosts.load(args.costs)
    rng = np.random.default_rng(args.seed)

    predicts = defaultdict(list)
    log_path = os.path.join(args.run_dir, "predict_log.jsonl")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                if "queue_mean_s" in rec and rec.get("rid"):
                    predicts[(rec["instance"], rec["snapshot_unix"])].append(rec)

    pool = DecodePool(cap=args.pool_cap, min_size=args.min_pool)
    seen_finished, seen_waiting = set(), set()
    rows = []
    with gzip.open(os.path.join(args.run_dir, "snapshots.jsonl.gz"), "rt") as f:
        for line in f:
            dump = json.loads(line)
            for snap in dump["snaps"]:
                for rec in snap["finished"]:
                    k = (dump["url"], snap["dp_rank"], rec["seq"])
                    if k not in seen_finished and rec["finish_reason"] != "abort":
                        seen_finished.add(k)
                        pool.add(rec["output_len"])
                if not pool.ready:
                    continue
                for rec in predicts.pop((dump["url"], snap["snapshot_unix"]), []):
                    rows.append(_send_row(rec, snap, costs, pool, rng, true_by_rid, turns, args))
                for i, w in enumerate(snap["waiting"]):
                    if w["rid"] in seen_waiting:
                        continue
                    seen_waiting.add(w["rid"])
                    if not w["is_retracted"]:
                        rows.append(
                            _shadow_row(w, i, snap, costs, pool, rng, true_by_rid, turns, args)
                        )
    return [r for r in rows if r is not None]


def _predictions(*, snap, age, costs, pool, rng, true_len, input_len, prefix_len,
                 max_new, prefix_path, ratio, args) -> dict:
    out = {}
    for name, samples in (("mc8", 8), ("mc64", 64)):
        p = predict(
            snap=snap, snapshot_age_s=age, pool=pool, costs=costs, input_len=input_len,
            prefix_len=prefix_len, max_new_tokens=max_new, prefix_ratio=ratio,
            samples=samples, rng=rng, prefix_path=prefix_path,
        )
        out[name] = p.queue_mean_s
        out[name + "_p90"] = p.queue_p90_s
    wait, missing = oracle_wait(
        snap=snap, costs=costs, input_len=input_len, prefix_len=prefix_len,
        max_new=max_new, prefix_path=prefix_path, ratio=ratio, true_len=true_len,
        pool=pool, rng=rng,
    )
    out["oracle"] = max(0.0, wait - age)
    out["oracle_missing"] = missing
    return out


def _send_row(rec, snap, costs, pool, rng, true_len, turns, args) -> Optional[dict]:
    turn = turns.get(_key(rec["rid"]))
    if turn is None or not turn["turn_queuing_latency"]:
        return None
    age = max(0.0, rec["t_send_unix"] - snap["snapshot_unix"])
    preds = _predictions(
        snap=snap, age=age, costs=costs, pool=pool, rng=rng, true_len=true_len,
        input_len=rec["input_len"], prefix_len=rec["prefix_len"],
        max_new=rec["max_new_tokens"], prefix_path=rec["prefix_path"],
        ratio=rec["prefix_ratio"], args=args,
    )
    return {"kind": "send", "rid": rec["rid"], "t": rec["t_send_unix"],
            "actual": float(turn["turn_queuing_latency"]), "live": rec["queue_mean_s"],
            **preds}


def _shadow_row(w, index, snap, costs, pool, rng, true_len, turns, args) -> Optional[dict]:
    turn = turns.get(_key(w["rid"]))
    if turn is None or not turn["turn_queuing_latency"]:
        return None
    ahead = dict(snap, waiting=snap["waiting"][:index], num_waiting=index)
    age = w["wait_queue_entry_age_s"] or 0.0
    preds = _predictions(
        snap=ahead, age=0.0, costs=costs, pool=pool, rng=rng, true_len=true_len,
        input_len=w["input_len"], prefix_len=w["prefix_len"] or 0,
        max_new=w["max_new_tokens"], prefix_path=w.get("prefix_path"), ratio=0.0,
        args=args,
    )
    # Shadow predictions cover the time left after the poll saw the request.
    for k in ("mc8", "mc8_p90", "mc64", "mc64_p90", "oracle"):
        preds[k] += age
    return {"kind": "shadow", "rid": w["rid"], "t": snap["snapshot_unix"] - age,
            "actual": float(turn["turn_queuing_latency"]), "live": None, **preds}


def summarize(rows: list) -> dict:
    out = {}
    for kind in ("send", "shadow"):
        rs = [r for r in rows if r["kind"] == kind and r["actual"] >= 1]
        if not rs:
            continue
        a = np.array([r["actual"] for r in rs])
        out[kind] = {"n": len(rs)}
        for name in ("live", "mc8", "mc64", "oracle"):
            if rs[0][name] is None:
                continue
            p = np.array([r[name] for r in rs])
            e = (p - a) / a
            res = {
                "median_abs_rel_err": float(np.median(np.abs(e))),
                "median_signed_rel_err": float(np.median(e)),
                "spearman": spearman(p, a),
            }
            for lo, hi in BUCKETS:
                b = (a >= lo) & (a < hi)
                if b.sum():
                    res[f"{lo}-{hi}s"] = {
                        "n": int(b.sum()),
                        "median_abs_rel_err": float(np.median(np.abs(e[b]))),
                        "median_signed_rel_err": float(np.median(e[b])),
                    }
            if name + "_p90" in rs[0]:
                res["actual_le_p90"] = float(np.mean(a <= np.array([r[name + "_p90"] for r in rs])))
            out[kind][name] = res
        out[kind]["oracle_missing_lengths"] = int(sum(r["oracle_missing"] for r in rs))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--costs", required=True)
    p.add_argument("--min-pool", type=int, default=500)
    p.add_argument("--pool-cap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = replay(args)
    with open(os.path.join(args.run_dir, "replay.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = summarize(rows)
    with open(os.path.join(args.run_dir, "replay_eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
