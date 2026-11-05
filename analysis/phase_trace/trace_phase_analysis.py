"""Offline trace analysis: can tool calls on one instance be synchronized?

Questions:
  1. Is a session's turn rhythm persistent (within-session autocorrelation)?
  2. How predictable is the next turn from session history / tool type?
  3. If two sessions start a tool call together, how fast do they drift apart?
  4. How much decode-free time does an instance get with random placement,
     and how much would per-turn wave gating create (and at what wait cost)?

GPU-busy time per turn is modeled from token counts (recorded LLM time
includes API-side queueing): prefill_tokens / PREFILL_TPS + decode_tokens * TPOT.

Usage: python trace_phase_analysis.py [--tpot-ms 15] [--prefill-tps 10000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["hermes", "hermes_v1", "openclaw"]
COLS = [
    "session_id",
    "benchmark",
    "turn_index",
    "prefill_num_tokens",
    "decode_num_tokens",
    "turn_toolcall_latency_ms",
    "tool_names",
]


def load(name, tpot_ms, prefill_tps):
    d = pd.read_csv(ROOT / "data" / f"{name}.csv", usecols=COLS)
    d = d.sort_values(["session_id", "turn_index"]).reset_index(drop=True)
    d["gpu_s"] = d.prefill_num_tokens / prefill_tps + d.decode_num_tokens * tpot_ms / 1e3
    d["tool_s"] = d.turn_toolcall_latency_ms / 1e3
    d["period_s"] = d.gpu_s + d.tool_s
    d["tool_type"] = d.tool_names.map(tool_type)
    return d


def tool_type(s):
    try:
        names = json.loads(s)
    except (TypeError, ValueError):
        return "none"
    return "+".join(sorted(set(names))) if names else "none"


# ---------------------------------------------------------------- 1. autocorr
def lag_autocorr(d, col, lags=(1, 2, 3, 5), demean=True):
    out = {}
    x = np.log(d[col].clip(lower=1e-3))
    if demean:
        x = x - x.groupby(d.session_id).transform("mean")
    for k in lags:
        y = x.groupby(d.session_id).shift(-k)
        m = y.notna() & x.notna()
        out[k] = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 10 else np.nan
    return out


def shuffle_null(d, col, n=20, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = d.copy()
        s[col] = s.groupby("session_id")[col].transform(lambda v: rng.permutation(v.values))
        vals.append(lag_autocorr(s, col, lags=(1,))[1])
    return float(np.mean(vals))


def icc(d, col):
    """Share of log-variance explained by session identity."""
    x = np.log(d[col].clip(lower=1e-3))
    m = x.notna()
    x, g = x[m], d.session_id[m]
    between = x.groupby(g).transform("mean").var()
    return between / x.var()


def eta2(d, col, by):
    x = np.log(d[col].clip(lower=1e-3))
    m = x.notna()
    x, g = x[m], d[by][m]
    return x.groupby(g).transform("mean").var() / x.var()


# ---------------------------------------------------------- 2. predictability
def predict_next(d, col):
    """log2 error of next-turn prediction for several online predictors."""
    rows = []
    for _, s in d.groupby("session_id"):
        v = np.log2(s[col].clip(lower=1e-3).values)
        ok = ~np.isnan(v)
        ew = None
        for i in range(len(v)):
            if not ok[i]:
                continue
            if i > 0 and ok[i - 1]:
                rows.append({"last": v[i - 1], "ewma": ew, "true": v[i]})
            ew = v[i] if ew is None else 0.5 * v[i] + 0.5 * ew
    r = pd.DataFrame(rows).dropna()
    glob = np.median(np.log2(d[col].clip(lower=1e-3).dropna()))
    res = {}
    for name, pred in [("global_median", glob), ("last_turn", r["last"]), ("session_ewma", r["ewma"])]:
        err = (r["true"] - pred).abs()
        res[name] = {"mae_log2": err.mean(), "within_1.5x": (err < np.log2(1.5)).mean()}
    # oracle tool-type mean (tool type is known when the tool call is emitted)
    if col == "tool_s":
        x = np.log2(d.tool_s.clip(lower=1e-3))
        m = x.notna()
        err = (x[m] - x[m].groupby(d.tool_type[m]).transform("mean")).abs()
        res["tool_type_mean(oracle)"] = {"mae_log2": err.mean(), "within_1.5x": (err < np.log2(1.5)).mean()}
    return res


# --------------------------------------------------------- 3. sync persistence
def session_seqs(d):
    seqs = []
    for _, s in d.groupby("session_id"):
        g, t = s.gpu_s.values, s.tool_s.values
        seqs.append((g, t))
    return seqs


def tool_intervals_from(seq, i, K):
    """Tool intervals of turns i..i+K, with tool i starting at t=0."""
    g, t = seq
    out, now = [], 0.0
    for k in range(K + 1):
        j = i + k
        if j >= len(t) or np.isnan(t[j]):
            break
        out.append((now, now + t[j]))
        now += t[j]
        if j + 1 < len(g):
            now += g[j + 1]
    return out


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def sync_persistence(d, K=5, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    seqs = session_seqs(d)
    starts = [(si, i) for si, (g, t) in enumerate(seqs) for i in range(len(t)) if not np.isnan(t[i])]
    tool_len = np.array([seqs[si][1][i] for si, i in starts])
    order = np.argsort(tool_len)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    res = {}
    for mode in ["random", "matched_dur"]:
        ious = [[] for _ in range(K + 1)]
        for _ in range(n_pairs):
            a = rng.integers(len(starts))
            if mode == "random":
                b = rng.integers(len(starts))
            else:
                # partner whose *current* tool duration is the nearest neighbour
                r = rank[a] + rng.choice([-2, -1, 1, 2])
                b = order[np.clip(r, 0, len(order) - 1)]
            if starts[a][0] == starts[b][0]:
                continue
            ia = tool_intervals_from(seqs[starts[a][0]], starts[a][1], K)
            ib = tool_intervals_from(seqs[starts[b][0]], starts[b][1], K)
            for k in range(min(len(ia), len(ib))):
                ious[k].append(iou(ia[k], ib[k]))
        res[mode] = {k: (float(np.mean(v)), len(v)) for k, v in enumerate(ious) if v}
    return res


# --------------------------------------------------------- 4. instance replay
def busy_timeline(seq_iter, horizon, gate_T=None):
    """Busy intervals of one 'slot' that replays sessions back to back.

    With gate_T, a session returning from a tool waits for the next multiple
    of gate_T before its next GPU phase (wave gating). Returns intervals and
    total gating wait.
    """
    busy, now, wait = [], 0.0, 0.0
    for g, t in seq_iter:
        for j in range(len(g)):
            if gate_T and j > 0:
                start = np.ceil(now / gate_T) * gate_T
                wait += start - now
                now = start
            busy.append((now, now + g[j]))
            now += g[j]
            if j < len(t) and not np.isnan(t[j]):
                now += t[j]
            if now > horizon:
                return busy, wait
    return busy, wait


def instance_replay(d, N, horizon=3600.0, gate_T=None, n_inst=20, seed=0):
    rng = np.random.default_rng(seed)
    seqs = session_seqs(d)
    dt = 0.01
    nbin = int(horizon / dt)
    free_frac, win_lens, waits, turns = [], [], 0.0, 0
    for _ in range(n_inst):
        active = np.zeros(nbin, dtype=np.int16)
        for _ in range(N):
            def gen():
                # random start offset: begin mid-session at a random turn
                first = True
                while True:
                    g, t = seqs[rng.integers(len(seqs))]
                    if first:
                        i = rng.integers(len(g))
                        g, t = g[i:], t[i:]
                        first = False
                    yield g, t
            busy, w = busy_timeline(gen(), horizon, gate_T)
            waits += w
            turns += len(busy)
            for a, b in busy:
                ia, ib = int(a / dt), min(int(np.ceil(b / dt)), nbin)
                if ia < nbin:
                    active[ia:ib] += 1
        free = active == 0
        free_frac.append(free.mean())
        # lengths of contiguous decode-free windows
        edges = np.diff(np.concatenate([[0], free.astype(np.int8), [0]]))
        s, e = np.where(edges == 1)[0], np.where(edges == -1)[0]
        win_lens.extend(((e - s) * dt).tolist())
    win = np.array(win_lens)
    total = horizon * n_inst
    return {
        "free_frac": float(np.mean(free_frac)),
        "free_in_win>=1s": float(win[win >= 1].sum() / total),
        "free_in_win>=5s": float(win[win >= 5].sum() / total),
        "mean_wait_per_turn_s": waits / max(turns, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpot-ms", type=float, default=15.0)
    ap.add_argument("--prefill-tps", type=float, default=10000.0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {}
    for name in DATASETS:
        d = load(name, args.tpot_ms, args.prefill_tps)
        r = {}
        sess = d.groupby("session_id").agg(gpu=("gpu_s", "sum"), tool=("tool_s", "sum"))
        r["n_sessions"], r["n_turns"] = int(len(sess)), int(len(d))
        r["tool_frac_overall"] = float(sess.tool.sum() / (sess.tool.sum() + sess.gpu.sum()))
        r["tool_frac_per_session_median"] = float((sess.tool / (sess.tool + sess.gpu)).median())
        r["median_s"] = {c: float(d[c].median()) for c in ["gpu_s", "tool_s", "period_s"]}
        r["autocorr_within"] = {c: lag_autocorr(d, c) for c in ["gpu_s", "tool_s", "period_s"]}
        r["autocorr_raw"] = {c: lag_autocorr(d, c, demean=False) for c in ["gpu_s", "tool_s", "period_s"]}
        r["autocorr_shuffle_null_lag1"] = {c: shuffle_null(d, c) for c in ["gpu_s", "tool_s", "period_s"]}
        r["icc_session"] = {c: float(icc(d, c)) for c in ["gpu_s", "tool_s", "period_s"]}
        r["eta2_tool_type_on_tool_s"] = float(eta2(d, "tool_s", "tool_type"))
        r["same_tool_type_as_prev"] = float((d.tool_type == d.groupby("session_id").tool_type.shift(1)).mean())
        r["predict_next"] = {c: predict_next(d, c) for c in ["gpu_s", "tool_s", "period_s"]}
        r["sync_persistence"] = sync_persistence(d)
        r["instance_replay"] = {}
        for N in [4, 8, 16, 32]:
            r["instance_replay"][f"N={N}"] = {"random": instance_replay(d, N)}
            for T in [1.0, 2.0, 5.0]:
                r["instance_replay"][f"N={N}"][f"gate_T={T}"] = instance_replay(d, N, gate_T=T)
        report[name] = r
        print(f"done {name}", flush=True)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, tuple):
            return [conv(v) for v in o]
        return o

    path = out / f"report_tpot{args.tpot_ms:g}.json"
    path.write_text(json.dumps(conv(report), indent=1))
    print(path)


if __name__ == "__main__":
    main()
