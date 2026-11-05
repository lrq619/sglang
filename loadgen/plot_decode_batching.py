"""Summarize and plot one or more bench_decode_batching.py runs.

Reads <rundir>/trials.csv of every run, keeps valid trials with rep >= 0, and
normalizes each run by its own B = 1. With several runs, a (N, B) point is taken
from the first run that has it, so later runs only add points. Writes into --out
(default: the first rundir):
  summary.csv (one run) or summary_merged.csv (several runs)
  norm_throughput_vs_bs.png    y(B, N) / y(1, N) vs B, one line per context length N
  tpot_vs_bs.png               mean TPOT vs B, one line per N
Filled markers are batch sizes that divide the attention split budget (64 / B
integral); hollow markers sit between the steps of the split-KV staircase.

Usage:
    python loadgen/plot_decode_batching.py logs/decbatch_<ts>
    python loadgen/plot_decode_batching.py logs/decbatch_<ts> logs/decbatch_ext_n*_<ts>
"""

import argparse
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ordinal blue ramp (dataviz reference palette, steps 300..700): darker = longer context.
RAMP = ["#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
        "#104281", "#0d366b"]
# Fitted from the 1..16 sweep: each request gets floor(64 / B) KV chunks per head group.
SPLIT_BUDGET = 64
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"


def _label(n: int) -> str:
    # 261952 (largest 64-aligned N under the 262144 context limit) prints as 256K.
    return f"{round(n / 1024)}K" if n >= 1024 else str(n)


def summarize(trials):
    valid = [t for t in trials if t["valid"] == "1" and int(t["rep"]) >= 0]
    out = {}
    for n in sorted({int(t["n_ctx"]) for t in valid}):
        rows = {}
        for b in sorted({int(t["bs"]) for t in valid if int(t["n_ctx"]) == n}):
            tp = [float(t["steady_tpot"] if "steady_tpot" in t else t["mean_tpot"]) for t in valid
                  if int(t["n_ctx"]) == n and int(t["bs"]) == b]
            m = statistics.mean(tp)
            rows[b] = {"n": len(tp), "tpot_ms": m * 1e3,
                       "tpot_std_ms": (statistics.stdev(tp) if len(tp) > 1 else 0.0) * 1e3,
                       "throughput": b / m}
        if 1 in rows:
            for b, r in rows.items():
                r["norm"] = r["throughput"] / rows[1]["throughput"]
        out[n] = rows
    return out


def write_summary(summ, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_ctx", "bs", "n_trials", "tpot_ms", "tpot_std_ms",
                    "throughput_tok_s", "norm_throughput"])
        for n, rows in summ.items():
            for b, r in rows.items():
                w.writerow([n, b, r["n"], f"{r['tpot_ms']:.3f}", f"{r['tpot_std_ms']:.3f}",
                            f"{r['throughput']:.2f}",
                            "" if "norm" not in r else f"{r['norm']:.3f}"])


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def merge(summs):
    """First run wins per (N, B); later runs only add points."""
    out = {}
    for summ in summs:
        for n, rows in summ.items():
            dst = out.setdefault(n, {})
            for b, r in rows.items():
                if b not in dst:
                    dst[b] = r
                elif "norm" in r and "norm" in dst[b]:
                    print(f"  overlap N={_label(n)} B={b}: tpot {dst[b]['tpot_ms']:.2f} vs "
                          f"{r['tpot_ms']:.2f} ms, norm {dst[b]['norm']:.2f} vs {r['norm']:.2f}")
    return {n: dict(sorted(rows.items())) for n, rows in sorted(out.items())}


def _end_labels(ax, ends, dx):
    """Direct labels at line ends, pushed apart vertically so they do not overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.04 * (hi - lo)
    placed = []
    for y, x, text in sorted(ends):
        y_txt = max(y, placed[-1] + gap) if placed else y
        placed.append(y_txt)
        ax.annotate(text, (x, y), xytext=(x + dx, y_txt), textcoords="data",
                    va="center", fontsize=8, color=INK)


def plot(summ, key, ylabel, title, path, ideal=False, b_step=1):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    max_b = max((max(r) for r in summ.values() if r), default=16)
    if ideal:
        ax.plot([1, max_b], [1, max_b], color=MUTED, linestyle="--", linewidth=1.2,
                label="ideal (y = B)")
    colors = RAMP[-len(summ):] if len(summ) <= len(RAMP) else RAMP
    ends = []
    for color, (n, rows) in zip(colors, summ.items()):
        bs = [b for b in rows if key in rows[b] and (b == 1 or b % b_step == 0)]
        ys = [rows[b][key] for b in bs]
        if not bs:
            continue
        ax.plot(bs, ys, color=color, linewidth=2, label=f"N = {_label(n)}")
        clean = [SPLIT_BUDGET % b == 0 for b in bs]
        for want, face in ((True, color), (False, "white")):
            xs = [b for b, c in zip(bs, clean) if c == want]
            if xs:
                ax.plot(xs, [y for y, c in zip(ys, clean) if c == want], linestyle="none",
                        marker="o", markersize=4.5, markerfacecolor=face,
                        markeredgecolor=color, markeredgewidth=1.2)
        ends.append((ys[-1], bs[-1], f"{_label(n)}  {ys[-1]:.1f}" + ("x" if ideal else "")))
    if ideal:
        ax.set_ylim(0, max(y for y, _, _ in ends) * 1.25)
    _end_labels(ax, ends, dx=0.02 * max_b)
    _style(ax, "Batch size B", ylabel)
    ax.set_xticks(range(1, max_b + 1) if max_b <= 16 else [1] + list(range(4, max_b + 1, 4)))
    ax.set_xlim(0.5, max_b * 1.16)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("rundirs", nargs="+")
    p.add_argument("--out", default=None, help="output directory (default: first rundir)")
    p.add_argument("--b-step", type=int, default=4,
                   help="plot B = 1 and multiples of this; summary CSVs keep every B")
    args = p.parse_args()
    out = args.out or args.rundirs[0]
    summs = []
    for d in args.rundirs:
        with open(os.path.join(d, "trials.csv")) as f:
            trials = list(csv.DictReader(f))
        n_bad = sum(1 for t in trials if t["valid"] != "1")
        reasons = sorted({t["invalid_reason"] for t in trials if t["valid"] != "1"})
        print(f"{os.path.basename(d)}: {len(trials)} trials, {n_bad} invalid "
              f"{reasons if reasons else ''}")
        summs.append(summarize(trials))
    summ = merge(summs)
    name = "summary.csv" if len(summs) == 1 else "summary_merged.csv"
    write_summary(summ, os.path.join(out, name))
    plot(summ, "norm", "Normalized decode throughput  y(B) / y(1)",
         "Qwen3.8-27B, H200: decode batching gain vs context length N\n"
         "y = B x 128 / decode latency, 128 output tokens, prefill excluded",
         os.path.join(out, "norm_throughput_vs_bs.png"), ideal=True, b_step=args.b_step)
    plot(summ, "tpot_ms", "Mean TPOT (ms)",
         "Qwen3.8-27B, H200: per-token decode latency vs batch size",
         os.path.join(out, "tpot_vs_bs.png"), b_step=args.b_step)
    cols = sorted({b for rows in summ.values() for b in rows if SPLIT_BUDGET % b == 0 or b > 16})
    print("  N    " + " ".join(f"B={b:<4}" for b in cols))
    for n, rows in summ.items():
        print(f"{_label(n):>5}  " + " ".join(
            f"{rows[b]['norm']:<6.2f}" if b in rows and "norm" in rows[b] else "  -   "
            for b in cols))


if __name__ == "__main__":
    main()
