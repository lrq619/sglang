"""Plot queueing-latency prediction error vs Monte Carlo sample count P.

One panel per load regime: the MC line over P, with the two point estimates
(conditional-mean and conditional-median lengths, one simulation each) and the true-length oracle as
reference lines. Reads qp_psweep.py output; needs matplotlib.

Usage:
    python plot_psweep.py work_dirs/queueing_predict_psweep.csv -o work_dirs/psweep.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill, references/palette.md), light surface.
MC_COLOR = "#2a78d6"
POINT_COLOR = "#eb6834"
POINT_MEDIAN_COLOR = "#1baf7a"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

P_VALUES = [1, 2, 4, 8, 16, 32]
PANELS = [
    ("Knee, r = 4.25-4.75", lambda r: 4 < r["rate"] < 5),
    ("Overload, r = 5-6", lambda r: r["rate"] >= 5),
    ("Before collapse, all rates", lambda r: r["pre_collapse"] == 1),
]


def _median_rel_err(rows: list, key: str) -> float:
    a = np.array([r["actual_s"] for r in rows])
    p = np.array([r[key] for r in rows])
    return float(np.median(np.abs(p - a) / a)) * 100


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_path")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv_path)))
    for r in rows:
        for k in r:
            if k not in ("run", "sample", "endpoint"):
                r[k] = float(r[k])
    queued = [r for r in rows if r["sample"] == "queued"]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, pick) in zip(axes, PANELS):
        sel = [r for r in queued if pick(r)]
        mc = [_median_rel_err(sel, f"mc_{p}") for p in P_VALUES]
        point = _median_rel_err(sel, "point")
        point_med = _median_rel_err(sel, "point_median")
        oracle = _median_rel_err(sel, "oracle")
        _style(ax)
        ax.plot(P_VALUES, mc, color=MC_COLOR, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label="Monte Carlo (mean of P sims)")
        ax.axhline(point, color=POINT_COLOR, linewidth=2, linestyle="--", zorder=2,
                   label="Point estimate (conditional-mean lengths)")
        ax.axhline(point_med, color=POINT_MEDIAN_COLOR, linewidth=2, linestyle="-.", zorder=2,
                   label="Point estimate (conditional-median lengths)")
        ax.axhline(oracle, color=INK_MUTED, linewidth=1.5, linestyle=":", zorder=2,
                   label="Oracle (true lengths)")
        ax.annotate(f"MC P=32: {mc[-1]:.1f}%", (P_VALUES[-1], mc[-1]), xytext=(0, -14), textcoords="offset points",
                    fontsize=9, color=INK_SECONDARY, ha="right")
        ax.annotate(f"median: {point_med:.1f}%", (P_VALUES[2], point_med), xytext=(0, 5), textcoords="offset points",
                    fontsize=9, color=INK_SECONDARY, ha="center")
        ax.annotate(f"mean: {point:.1f}%", (P_VALUES[-1], point), xytext=(0, -13), textcoords="offset points",
                    fontsize=9, color=INK_SECONDARY, ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(P_VALUES)
        ax.set_xticklabels([str(p) for p in P_VALUES])
        ax.set_ylim(0, max(point, max(mc)) * 1.15)
        ax.set_title(f"{title}  (n = {len(sel)})", loc="left", fontsize=11, color=INK_PRIMARY)
        ax.set_xlabel("Monte Carlo samples P", color=INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("Median relative error (%)", color=INK_SECONDARY, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Queueing-latency prediction error vs Monte Carlo sample count", x=0.01, y=0.98, ha="left",
                 fontsize=13, color=INK_PRIMARY)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.91), ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
