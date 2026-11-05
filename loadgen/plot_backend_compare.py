"""Compare decode batching across attention backends at one context length.

Each --series is "label=rundir[,rundir...]"; the runs of one series are merged as in
plot_decode_batching.py (first run wins per B, each run normalized by its own B = 1).
Writes <out>/backend_compare_n<N>.png with two panels: normalized throughput
y(B) / y(1), and mean TPOT, both vs B (B = 1 and multiples of --b-step).

Usage:
    python loadgen/plot_backend_compare.py --n 32768 --out logs/decbatch_fa3_n32768_<ts> \
        --series flashinfer=logs/decbatch_<ts>,logs/decbatch_ext_n32768_<ts> \
        --series fa3=logs/decbatch_fa3_n32768_<ts>
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_decode_batching import INK, MUTED, _label, _style, merge, summarize

# Categorical slots 1 and 2 of the dataviz reference palette (blue, orange).
COLORS = ["#2a78d6", "#eb6834"]


def _load(rundirs, n_ctx):
    summs = []
    for d in rundirs:
        with open(os.path.join(d, "trials.csv")) as f:
            summs.append(summarize(list(csv.DictReader(f))))
    return merge(summs).get(n_ctx, {})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, required=True, help="context length N to compare")
    p.add_argument("--series", action="append", required=True, help="label=dir[,dir...]")
    p.add_argument("--out", required=True)
    p.add_argument("--b-step", type=int, default=4)
    args = p.parse_args()

    series = []
    for spec in args.series:
        label, dirs = spec.split("=", 1)
        rows = _load(dirs.split(","), args.n)
        bs = [b for b in rows if (b == 1 or b % args.b_step == 0) and "norm" in rows[b]]
        series.append((label, bs, rows))

    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    max_b = max(max(bs) for _, bs, _ in series)
    max_norm = max(r[b]["norm"] for _, bs, r in series for b in bs)
    ax_n.plot([1, max_b], [1, max_b], color=MUTED, linestyle="--", linewidth=1.2,
              label="ideal (y = B)")
    for color, (label, bs, rows) in zip(COLORS, series):
        for ax, key in ((ax_n, "norm"), (ax_t, "tpot_ms")):
            ys = [rows[b][key] for b in bs]
            ax.plot(bs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label)
            ax.annotate(f"{ys[-1]:.1f}" + ("x" if key == "norm" else " ms"), (bs[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
    ticks = [1] + list(range(args.b_step, max_b + 1, args.b_step))
    _style(ax_n, "Batch size B", "Normalized decode throughput  y(B) / y(1)")
    ax_n.set_ylim(0, max_norm * 1.25)
    _style(ax_t, "Batch size B", "Mean TPOT (ms)")
    ax_t.set_ylim(bottom=0)
    for ax in (ax_n, ax_t):
        ax.set_xticks(ticks)
        ax.set_xlim(0, max_b * 1.12)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Qwen3.8-27B, H200, N = {_label(args.n)}: attention backend vs decode "
                 "batching (128 output tokens, prefill excluded)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(args.out, f"backend_compare_n{args.n}.png")
    fig.savefig(path)
    print(path)
    print("   B  " + "  ".join(f"{label:>18}" for label, _, _ in series))
    for b in sorted({b for _, bs, _ in series for b in bs}):
        cells = []
        for _, bs, rows in series:
            r = rows.get(b)
            cells.append(f"{r['norm']:6.2f}x {r['tpot_ms']:7.2f}ms" if r and b in bs else " " * 18)
        print(f"{b:>4}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
