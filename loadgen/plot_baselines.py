"""Bar chart of queueing-latency prediction error, ours vs baseline ideas.

Median relative error on the queued (wait > 1 s) targets of the knee regime
(4 < r < 5) in qp_psweep.py output. The baselines are reimplementations of each
paper's core idea inside this evaluation, not the original systems:
Continuum = recent per-token mean queue delay, TokenScale = fluid deficit over
recent release rate, Astrolabe = one simulation with conditional-mean lengths.
Needs matplotlib.

Usage:
    python plot_baselines.py work_dirs/queueing_predict_psweep.csv -o work_dirs/baselines.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill, references/palette.md), light surface;
# baselines share one recessive hue so the comparison reads against ours.
OURS_COLOR = "#2a78d6"
BASELINE_COLOR = "#b7b6ad"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

METHODS = [
    ("Continuum", "continuum"),
    ("Astrolabe", "point"),
    ("TokenScale", "fluid"),
    ("Ours", "mc_32"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_path")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv_path))
            if r["sample"] == "queued" and 4 < float(r["rate"]) < 5]
    actual = np.array([float(r["actual_s"]) for r in rows])
    errors = [
        float(np.median(np.abs(np.array([float(r[key]) for r in rows]) - actual) / actual)) * 100
        for _, key in METHODS
    ]

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = [OURS_COLOR if name == "Ours" else BASELINE_COLOR for name, _ in METHODS]
    bars = ax.bar([name for name, _ in METHODS], errors, width=0.6, color=colors, zorder=3)
    for bar, err in zip(bars, errors):
        ax.annotate(f"{err:.1f}%", (bar.get_x() + bar.get_width() / 2, err), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10, color=INK_PRIMARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(errors) * 1.15)
    ax.set_ylabel("Median Error (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Queuing latency prediction", loc="left", fontsize=13, color=INK_PRIMARY, pad=12)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}: " + ", ".join(f"{n}={e:.1f}%" for (n, _), e in zip(METHODS, errors)))


if __name__ == "__main__":
    main()
