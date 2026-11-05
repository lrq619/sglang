"""TraceLab: do single-round vs multi-round sessions separate into two gap modes?

Terms: session -> round (one user input + the agent loop it triggers) -> turn
(one LLM call + its tools). gap = time from a turn's end to the next turn's
request; tool gap inside a round, user gap at a round boundary.

Filters: drop sessions with any user gap > 1h, or total gap > 2h.

Usage: python round_gap.py [--trace ~/scratch/tracelab/syfi_coding_trace.jsonl.gz]
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
MAX_USER_GAP_S = 3600
MAX_TOTAL_GAP_S = 7200


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_gaps(path):
    by_sid = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            by_sid.setdefault(r["session_id"], []).append(r)
    rows = []
    for sid, turns in by_sid.items():
        turns.sort(key=lambda r: r["round_index"])
        prev_end, prev_tool = None, None
        for r in turns:
            ev = [e for e in r["timing_events"] if e.get("timestamp")]
            users = [ts(e["timestamp"]) for e in ev if e["event_type"] == "user_message"]
            if prev_end is not None:
                if users:
                    rows.append((sid, r["provider"], "user", users[0] - prev_end))
                elif prev_tool is not None:
                    rows.append((sid, r["provider"], "tool", prev_tool))
            lat = [t["tool_wall_latency_ms"] / 1e3 for t in r["tools"] if t.get("tool_wall_latency_ms") is not None]
            prev_tool = max(lat) if lat else None
            ends = [ts(e["timestamp"]) for e in ev] + [ts(t["result_at"]) for t in r["tools"] if t.get("result_at")]
            prev_end = max(ends) if ends else prev_end
    g = pd.DataFrame(rows, columns=["sid", "provider", "kind", "gap"])
    return g[g.gap >= 0], {sid: len(t) for sid, t in by_sid.items()}


def summarize(g, n_turns):
    s = g.groupby("sid").agg(
        provider=("provider", "first"),
        n_gaps=("gap", "size"),
        n_user=("kind", lambda k: (k == "user").sum()),
        total=("gap", "sum"),
        mean=("gap", "mean"),
        median=("gap", "median"),
        max_user=("gap", lambda x: 0.0),
    )
    s["max_user"] = g[g.kind == "user"].groupby("sid").gap.max().reindex(s.index).fillna(0)
    s["user_total"] = g[g.kind == "user"].groupby("sid").gap.sum().reindex(s.index).fillna(0)
    s["n_turns"] = pd.Series(n_turns).reindex(s.index)
    s["rounds"] = np.where(s.n_user > 0, "multi", "single")
    return s


def q(x):
    x = np.asarray(x, float)
    return " / ".join(f"{np.percentile(x, p):.3g}" for p in (10, 50, 90)) + f"  mean {x.mean():.3g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default=str(Path.home() / "scratch/tracelab/syfi_coding_trace.jsonl.gz"))
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)

    g, n_turns = load_gaps(args.trace)
    s = summarize(g, n_turns)
    print(f"sessions with >=1 gap: {len(s)} (of {len(n_turns)}); single {(s.rounds == 'single').sum()}, multi {(s.rounds == 'multi').sum()}")
    drop_user = s.max_user > MAX_USER_GAP_S
    drop_total = s.total > MAX_TOTAL_GAP_S
    for name, m in [("user gap > 1h", drop_user), ("total gap > 2h", drop_total), ("either", drop_user | drop_total)]:
        print(f"  drop {name:15s}: single {(m & (s.rounds == 'single')).sum():5d}  multi {(m & (s.rounds == 'multi')).sum():5d}")
    k = s[~(drop_user | drop_total)]
    k.to_csv(OUT / "sessions_filtered.csv")
    gk = g[g.sid.isin(k.index)].merge(k[["rounds"]], left_on="sid", right_index=True)

    print("\nper-session stats (p10 / p50 / p90  mean):")
    for col in ["n_turns", "n_user", "total", "mean", "median"]:
        for rd in ["single", "multi"]:
            print(f"  {col:8s} {rd:6s} n={(k.rounds == rd).sum():5d}  {q(k[k.rounds == rd][col])}")
    m = k[k.rounds == "multi"]
    print(f"  multi: user-gap share of total gap  {q(m.user_total / m.total.clip(lower=1e-9))}")
    print("\nper-gap stats:")
    for (rd, kind), x in gk.groupby(["rounds", "kind"]).gap:
        print(f"  {rd:6s} {kind:4s} n={len(x):6d}  {q(x)}")
    for rd in ["single", "multi"]:
        for prov in ["claude", "codex"]:
            x = k[(k.rounds == rd) & (k.provider == prov)]["mean"]
            print(f"  session mean gap {rd:6s} {prov:6s} n={len(x):5d}  {q(x)}")

    edges = [0, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 7200]
    print(f"\nper-session mean gap histogram, edges(s) {edges}:")
    for rd in ["single", "multi"]:
        print(f"  {rd:6s}", np.histogram(k[k.rounds == rd]["mean"], edges)[0].tolist())
    print("  all   ", np.histogram(k["mean"], edges)[0].tolist())
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    bins = np.logspace(-2, 4, 49)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in [
        (axes[0], "mean", "Per-session mean gap (s)"),
        (axes[1], "total", "Per-session total gap (s)"),
    ]:
        for rd, c in [("single", "#2a78b5"), ("multi", "#d1495b")]:
            ax.hist(k[k.rounds == rd][col].clip(lower=1e-2), bins=bins, alpha=0.6, color=c, label=f"{rd}-round (n={(k.rounds == rd).sum()})")
        x = k[col].clip(lower=1e-2)
        ax.hist(x, bins=bins, histtype="step", color="k", lw=1.2, label="combined")
        ax.set_xscale("log"), ax.set_title(title), ax.set_ylabel("sessions"), ax.legend()
    ax = axes[2]
    for (rd, kind), c in [(("single", "tool"), "#2a78b5"), (("multi", "tool"), "#f4a259"), (("multi", "user"), "#d1495b")]:
        x = gk[(gk.rounds == rd) & (gk.kind == kind)].gap.clip(lower=1e-2)
        ax.hist(x, bins=bins, alpha=0.5, color=c, density=True, label=f"{rd}-round {kind} gap (n={len(x)})")
    ax.set_xscale("log"), ax.set_title("Per-gap distribution (density)"), ax.legend()
    fig.suptitle("TraceLab after filters (user gap <= 1h, total gap <= 2h)")
    fig.tight_layout()
    fig.savefig(OUT / "round_gap_bimodal.png", dpi=130)
    print(f"\nwrote {OUT / 'round_gap_bimodal.png'}")


if __name__ == "__main__":
    main()
