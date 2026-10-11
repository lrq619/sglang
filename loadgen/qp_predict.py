"""Offline queueing-latency predictors evaluated on loadgen runs.

For every evaluated turn x, the instance state at x's arrival is rebuilt from
loadgen.csv (qp_state), and each predictor estimates how long x waits before
SGLang admits it. The Monte Carlo predictors sample each running request's
remaining decode length from an empirical distribution conditioned on what it
has generated so far, then replay SGLang's admission rule event by event:

    admit queue head  iff  running < max_running_requests  and
        ctx_head + min(dec_head, 4096) < capacity - sum_running(ctx + ratio * min(rem, 4096))

(PrefillAdder: available + evictable minus decode headroom; schedule_policy=fcfs.)
Decode-length pools are leave-one-session-out, since the replay reuses the
corpus sessions.

Usage:
    python qp_predict.py logs/sweep_*_r* --corpus data/hermes_clean.jsonl \
        -o work_dirs/queueing_predict.csv --summary work_dirs/queueing_predict_summary.csv
"""

import argparse
import bisect
import csv
import glob
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_sim2 import fit_step_costs, simulate_wait_v2, split_state  # noqa: E402
from qp_state import InstanceView, collapse_onset, load_turns  # noqa: E402

# Server constants of the evaluated runs (server log: max_total_num_tokens,
# max_running_requests; environ.py: SGLANG_INIT_NEW_TOKEN_RATIO * MIN_FACTOR).
KV_CAPACITY = 957105
MAX_RUNNING = 52
NEW_TOKEN_RATIO = 0.7 * 0.14
CLIP_MAX_NEW = 4096
HORIZON_S = 1800.0
# Arbitrary; long enough to average over several completions per instance.
WINDOW_S = 300.0
MC_SAMPLES = 32
# Pools smaller than this fall back to the session-excluded global pool.
MIN_GROUP_POOL = 30


# ===== Calibration: decode step time and prefill time =====


class Calibration:
    """Median decode step time per batch size and prefill time per extend size."""

    def __init__(self, *, step_by_b: np.ndarray, prefill_edges: np.ndarray, prefill_med: np.ndarray):
        self.step_by_b = step_by_b
        self.prefill_edges = prefill_edges
        self.prefill_med = prefill_med

    def step(self, b: int) -> float:
        return float(self.step_by_b[min(max(b, 1), len(self.step_by_b) - 1)])

    def scaled(self, factor: float) -> "Calibration":
        return Calibration(step_by_b=self.step_by_b * factor, prefill_edges=self.prefill_edges,
                           prefill_med=self.prefill_med)

    def prefill(self, extend: int) -> float:
        i = min(bisect.bisect_right(self.prefill_edges, extend), len(self.prefill_med) - 1)
        return float(self.prefill_med[i])


def fit_calibration(views: list, rng: random.Random) -> Calibration:
    by_b = defaultdict(list)
    prefill = []
    for v in views:
        for x in rng.sample(v.turns, min(1500, len(v.turns))):
            if x.decode_len >= 32 and x.decode_s > 0:
                b = len(v.running_at((x.t_decode_start + x.t_end) / 2))
                by_b[b].append(x.decode_s / x.decode_len)
            prefill.append((max(0, x.input_len - x.cached), x.prefill_s))
    step = np.zeros(MAX_RUNNING + 2)
    known = sorted(b for b, ys in by_b.items() if len(ys) >= 20)
    for b in range(len(step)):
        # Nearest batch size with enough samples; batch sizes are dense here.
        nb = min(known, key=lambda k: abs(k - b))
        step[b] = statistics.median(by_b[nb])
    prefill.sort()
    edges = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    med = []
    for lo, hi in zip(edges, edges[1:] + [10**9]):
        ys = [p for e, p in prefill if lo <= e < hi]
        med.append(statistics.median(ys) if ys else (med[-1] if med else 0.1))
    return Calibration(step_by_b=step, prefill_edges=np.array(edges[1:]), prefill_med=np.array(med))


# ===== Decode-length model =====


def _turn_bucket(turn_id: int) -> int:
    return 0 if turn_id == 0 else 1 if turn_id < 5 else 2 if turn_id < 15 else 3


def _prev_bucket(prev_decode) -> int:
    if prev_decode is None:
        return 0
    return 1 + bisect.bisect_right([100, 300, 1000], prev_decode)


class DecodeModel:
    """Leave-one-session-out empirical decode-length pools, global and grouped."""

    def __init__(self, corpus_path: str):
        self.decode = {}
        rows = []
        for s in map(json.loads, open(corpus_path)):
            prev = None
            for t in s["turns"]:
                d = t["decode_num_tokens"]
                key = (s["session_id"], t["turn_index"])
                self.decode[key] = d
                rows.append((s["session_id"], (_turn_bucket(t["turn_index"]), _prev_bucket(prev)), d))
                prev = d
        sessions = {r[0] for r in rows}
        self._global = {s: sorted(d for ss, _, d in rows if ss != s) for s in sessions}
        groups = {g for _, g, _ in rows}
        self._group = {
            (s, g): sorted(d for ss, gg, d in rows if ss != s and gg == g)
            for s in sessions for g in groups
        }

    def group_of(self, session_id: str, turn_id: int) -> tuple:
        return (_turn_bucket(turn_id), _prev_bucket(self.decode.get((session_id, turn_id - 1))))

    def pool(self, *, session_id: str, turn_id: int, grouped: bool) -> list:
        if grouped:
            p = self._group.get((session_id, self.group_of(session_id, turn_id)), [])
            if len(p) >= MIN_GROUP_POOL:
                return p
        return self._global[session_id]

    @staticmethod
    def sample_remaining(pool: list, generated: int, rng: random.Random) -> int:
        """Remaining decode tokens given `generated` so far: D ~ pool | D > generated."""
        i = bisect.bisect_right(pool, generated)
        if i >= len(pool):
            # Already past every observed length; arbitrary short tail.
            return 64
        return pool[rng.randrange(i, len(pool))] - generated


# ===== Admission simulator =====


def simulate_wait(*, running: list, queue: list, cal: Calibration) -> float:
    """Seconds until the last queue entry is admitted.

    running: [ctx, remaining_decode_tokens]; queue: (ctx, decode, extend) in FCFS
    order ending with the target request.
    """
    ctx = np.array([r[0] for r in running], dtype=np.float64)
    rem = np.array([r[1] for r in running], dtype=np.float64)
    t = 0.0
    qi = 0
    while True:
        reserved = ctx.sum() + NEW_TOKEN_RATIO * np.minimum(rem, CLIP_MAX_NEW).sum()
        while qi < len(queue) and len(rem) < MAX_RUNNING:
            q_ctx, q_dec, q_ext = queue[qi]
            if q_ctx + min(q_dec, CLIP_MAX_NEW) >= KV_CAPACITY - reserved:
                break
            if qi == len(queue) - 1:
                return t
            # Prefill delays the start of decoding; fold it into remaining steps.
            extra = cal.prefill(q_ext) / cal.step(len(rem) + 1)
            ctx = np.append(ctx, q_ctx)
            rem = np.append(rem, q_dec + extra)
            reserved += q_ctx + NEW_TOKEN_RATIO * min(q_dec, CLIP_MAX_NEW)
            qi += 1
        if len(rem) == 0:
            return HORIZON_S
        d = rem.min()
        t += d * cal.step(len(rem))
        if t > HORIZON_S:
            return HORIZON_S
        rem -= d
        ctx += d
        keep = rem > 0.5
        ctx, rem = ctx[keep], rem[keep]


# ===== Predictors =====


class Snapshot:
    """Instance state at the target's arrival, as seen by an online predictor."""

    def __init__(self, *, view: InstanceView, x, t0: float):
        self.x = x
        self.t0 = t0
        self.running = [r for r in view.running_at(t0) if r is not x]
        self.queue_ahead = [q for q in view.queued_at(t0) if q is not x]
        self.generated = {id(r): view.generated_at(r, t0) for r in self.running}


def _queue_entries(*, snap: Snapshot, decode_of) -> list:
    entries = [(q.input_len, decode_of(q), max(0, q.input_len - q.cached)) for q in snap.queue_ahead]
    x = snap.x
    return entries + [(x.input_len, decode_of(x), max(0, x.input_len - x.cached))]


def predict_oracle(*, snap: Snapshot, cal: Calibration) -> float:
    running = []
    for r in snap.running:
        g = snap.generated[id(r)]
        rem = r.decode_len - g
        if snap.t0 < r.t_decode_start:
            rem += (r.t_decode_start - snap.t0) / cal.step(len(snap.running))
        running.append([r.input_len + g, max(rem, 1)])
    return simulate_wait(running=running, queue=_queue_entries(snap=snap, decode_of=lambda q: q.decode_len), cal=cal)


def predict_mc(*, snap: Snapshot, cal: Calibration, model: DecodeModel, grouped: bool, rng: random.Random) -> list:
    pool = lambda t: model.pool(session_id=t.session_id, turn_id=t.turn_id, grouped=grouped)
    waits = []
    for _ in range(MC_SAMPLES):
        running = []
        for r in snap.running:
            g = snap.generated[id(r)]
            rem = model.sample_remaining(pool(r), g, rng)
            if snap.t0 < r.t_decode_start:
                rem += (r.t_decode_start - snap.t0) / cal.step(len(snap.running))
            running.append([r.input_len + g, rem])
        decode_of = lambda q: model.sample_remaining(pool(q), 0, rng)
        waits.append(simulate_wait(running=running, queue=_queue_entries(snap=snap, decode_of=decode_of), cal=cal))
    return waits


def predict_oracle_v2(*, snap: Snapshot, costs) -> float:
    running, inflight, queue, busy = split_state(snap=snap, decode_of=lambda t, g: t.decode_len - g)
    return simulate_wait_v2(running=running, inflight=inflight, queue=queue, costs=costs, busy_s=busy)


def predict_mc_v2(*, snap: Snapshot, costs, model: DecodeModel, rng: random.Random) -> list:
    pool = lambda t: model.pool(session_id=t.session_id, turn_id=t.turn_id, grouped=False)
    waits = []
    for _ in range(MC_SAMPLES):
        running, inflight, queue, busy = split_state(
            snap=snap, decode_of=lambda t, g: model.sample_remaining(pool(t), g, rng))
        waits.append(simulate_wait_v2(running=running, inflight=inflight, queue=queue, costs=costs, busy_s=busy))
    return waits


def local_step_factor(*, view: InstanceView, snap: Snapshot, cal: Calibration) -> float:
    """Observed recent per-token decode time over the table's, at the current batch size."""
    done = [r for r in view.finished_between(snap.t0 - WINDOW_S, snap.t0) if r.decode_len >= 32]
    if len(done) < 5:
        return 1.0
    observed = statistics.median(r.decode_s / r.decode_len for r in done)
    return observed / cal.step(len(snap.running))


def predict_continuum(*, view: InstanceView, snap: Snapshot) -> float:
    """Continuum-style: recent mean queue delay per unit context, times x's context."""
    recent = view.admitted_between(snap.t0 - WINDOW_S, snap.t0)
    if not recent:
        return 0.0
    per_token = statistics.mean(r.queue_s for r in recent) / statistics.mean(r.input_len for r in recent)
    return per_token * snap.x.input_len


def predict_fluid(*, view: InstanceView, snap: Snapshot, cal: Calibration) -> float:
    """Deficit divided by the recent net release rate, for KV tokens and for slots."""
    done = view.finished_between(snap.t0 - WINDOW_S, snap.t0)
    b = len(snap.running)
    used = sum(r.input_len + snap.generated[id(r)] for r in snap.running)
    need = sum(q.input_len for q in snap.queue_ahead) + snap.x.input_len
    release = sum(r.input_len + r.decode_len for r in done) / WINDOW_S
    net = release - b / cal.step(b)
    deficit = need - (KV_CAPACITY - used)
    t_tok = 0.0 if deficit <= 0 else (deficit / net if net > 0 else HORIZON_S)
    slots_short = len(snap.queue_ahead) + 1 - (MAX_RUNNING - b)
    rate = len(done) / WINDOW_S
    t_cnt = 0.0 if slots_short <= 0 else (slots_short / rate if rate > 0 else HORIZON_S)
    return min(HORIZON_S, max(t_tok, t_cnt))


# ===== Evaluation =====


def _rate_of(run_dir: str) -> float:
    return float(re.search(r"_r([0-9.]+)_", os.path.basename(os.path.normpath(run_dir)) + "_").group(1))


def _select_targets(*, views: dict, t_lo: float, t_hi: dict, rng: random.Random, n_uniform: int, n_queued: int) -> list:
    pool = [(ep, x) for ep, v in views.items() for x in v.turns if t_lo <= x.t_send <= t_hi[ep]]
    queued = [p for p in pool if p[1].t_admit - p[1].t_send > 1.0]
    picked = rng.sample(pool, min(n_uniform, len(pool)))
    picked_q = rng.sample(queued, min(n_queued, len(queued)))
    return [(ep, x, "uniform") for ep, x in picked] + [(ep, x, "queued") for ep, x in picked_q]


def evaluate_run(*, run_dir: str, model: DecodeModel, cal: Calibration, costs, rng: random.Random,
                 n_uniform: int, n_queued: int, pre_collapse: bool) -> list:
    turns = load_turns(os.path.join(run_dir, "loadgen.csv"))
    views = {ep: InstanceView(ts) for ep, ts in turns.items()}
    t_start = min(x.t_send for ts in turns.values() for x in ts)
    info = dict(l.split(": ", 1) for l in open(os.path.join(run_dir, "run_info.txt")).read().splitlines() if ": " in l)
    duration = float(info["duration_s"])
    # Skip warm-up, and the tail where unfinished turns are missing from loadgen.csv;
    # with pre_collapse, also stop 5 min before each instance's collapse onset.
    t_hi = {}
    for ep, v in views.items():
        t_hi[ep] = t_start + duration - 3600
        if pre_collapse:
            onset = collapse_onset(v.turns)
            t_hi[ep] = min(t_hi[ep], onset - 300)
            print(f"  {ep}: collapse onset " + ("never" if onset == float("inf") else f"{(onset - t_start) / 60:.0f} min"))
    targets = _select_targets(
        views=views, t_lo=t_start + 600, t_hi=t_hi,
        rng=rng, n_uniform=n_uniform, n_queued=n_queued,
    )
    rows = []
    for ep, x, sample in targets:
        view = views[ep]
        snap = Snapshot(view=view, x=x, t0=x.t_send)
        mc_g = predict_mc(snap=snap, cal=cal, model=model, grouped=False, rng=rng)
        mc_c = predict_mc(snap=snap, cal=cal, model=model, grouped=True, rng=rng)
        local = cal.scaled(local_step_factor(view=view, snap=snap, cal=cal))
        mc_l = predict_mc(snap=snap, cal=local, model=model, grouped=False, rng=rng)
        mc_v2 = predict_mc_v2(snap=snap, costs=costs, model=model, rng=rng)
        rows.append({
            "run": os.path.basename(os.path.normpath(run_dir)),
            "rate": _rate_of(run_dir),
            "sample": sample,
            "endpoint": ep,
            "t_offset_s": round(x.t_send - t_start, 1),
            "running": len(snap.running),
            "queue_ahead": len(snap.queue_ahead),
            "input_len": x.input_len,
            # Client-side wait until the first forward; the server's queue_time
            # misses the time spent before the scheduler loop receives the request.
            "actual_s": round(x.t_admit - x.t_send, 3),
            "server_queue_s": round(x.queue_s, 3),
            "zero": 0.0,
            "continuum": round(predict_continuum(view=view, snap=snap), 3),
            "fluid": round(predict_fluid(view=view, snap=snap, cal=cal), 3),
            "mc_global_mean": round(statistics.mean(mc_g), 3),
            "mc_global_p50": round(statistics.median(mc_g), 3),
            "mc_global_p90": round(float(np.percentile(mc_g, 90)), 3),
            "mc_grouped_mean": round(statistics.mean(mc_c), 3),
            "mc_grouped_p50": round(statistics.median(mc_c), 3),
            "mc_grouped_p90": round(float(np.percentile(mc_c, 90)), 3),
            "mc_global_local_mean": round(statistics.mean(mc_l), 3),
            "mc_global_local_p90": round(float(np.percentile(mc_l, 90)), 3),
            "oracle": round(predict_oracle(snap=snap, cal=cal), 3),
            "oracle_local": round(predict_oracle(snap=snap, cal=local), 3),
            "mc_v2_mean": round(statistics.mean(mc_v2), 3),
            "mc_v2_p50": round(statistics.median(mc_v2), 3),
            "mc_v2_p90": round(float(np.percentile(mc_v2, 90)), 3),
            "oracle_v2": round(predict_oracle_v2(snap=snap, costs=costs), 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("-o", "--output", required=True, help="per-request predictions CSV")
    parser.add_argument("--n-uniform", type=int, default=300)
    parser.add_argument("--n-queued", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-collapse", action="store_true",
                        help="only evaluate turns sent >= 5 min before the instance's cache-thrashing collapse")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    run_dirs = sorted({d.rstrip("/") for p in args.run_dirs for d in glob.glob(p) if os.path.isdir(d)}, key=_rate_of)
    all_views = [InstanceView(ts) for d in run_dirs for ts in load_turns(os.path.join(d, "loadgen.csv")).values()]
    cal = fit_calibration(all_views, rng)
    costs = fit_step_costs(all_views, rng)
    print(f"step costs: {costs}", flush=True)
    model = DecodeModel(args.corpus)
    rows = []
    for d in run_dirs:
        r = evaluate_run(run_dir=d, model=model, cal=cal, costs=costs, rng=rng, n_uniform=args.n_uniform,
                         n_queued=args.n_queued, pre_collapse=args.pre_collapse)
        print(f"{os.path.basename(d)}: {len(r)} targets", flush=True)
        rows += r
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
