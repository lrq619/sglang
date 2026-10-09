"""Plot censored session JCT (mean and p99) against session arrival rate.

Each --series is one group of runs sharing a window length, given as
"label=glob[,glob...]"; groups with different windows are drawn as separate
lines, never joined, because a shorter window caps censored JCT lower. A single
series gets no legend (the title names it). The rate is parsed from the run name
(_r<rate>_). Needs matplotlib (not in the sglang venv; any Python with it works).

Usage:
    python plot_jct.py -o logs/jct \
        --series "3h window=logs/sweep_A_r*,logs/sweep_B_r*"
"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_jct  # noqa: E402

# Reference palette (dataviz skill, references/palette.md): categorical slots
# 1-2, then chart chrome and ink on the light surface.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
SERIES_MARKERS = ["o", "s"]
SERIES_LINESTYLES = ["-", "--"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

METRICS = {
    "mean": ("jct_mean", "Mean censored session JCT"),
    "p99": ("jct_p99", "P99 censored session JCT"),
}


def _rate(run_dir: str) -> float:
    m = re.search(r"_r([0-9.]+)_", os.path.basename(os.path.normpath(run_dir)) + "_")
    return float(m.group(1))


def _load_series(spec: str) -> tuple:
    label, _, patterns = spec.partition("=")
    dirs = sorted(
        {d for p in patterns.split(",") for d in glob.glob(p) if os.path.isdir(d)},
        key=_rate,
    )
    if not dirs:
        sys.exit(f"no run dirs match {patterns}")
    rows = [analyze_jct.summarize(run_dir=d, corpus_override=None) for d in dirs]
    window_s = float(rows[0]["duration_s"])
    return label, [_rate(d) for d in dirs], rows, window_s


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metric(*, series: list, metric: str, out_path: str, show_window_caps: bool):
    key, title = METRICS[metric]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for i, (label, rates, rows, window_s) in enumerate(series):
        values = [r[key] for r in rows]
        color = SERIES_COLORS[i]
        ax.plot(
            rates, values, color=color, linewidth=2, linestyle=SERIES_LINESTYLES[i],
            marker=SERIES_MARKERS[i], markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label, zorder=3,
        )
        if len(series) > 1:
            # Direct label at the line end, in ink rather than the series color.
            ax.annotate(
                label, (rates[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK_SECONDARY,
            )
        if show_window_caps and window_s <= max(max(r[key] for r in s[2]) for s in series) * 1.3:
            ax.axhline(window_s, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
            ax.annotate(
                f"{label} cap {window_s:.0f}s", (rates[0], window_s),
                xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK_MUTED,
            )

    if len(series) == 1:
        title = f"{title} ({series[0][0]})"
    ax.set_title(title, loc="left", fontsize=14, color=INK_PRIMARY, pad=14)
    ax.set_xlabel("Session arrival rate (sessions/min)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Seconds", color=INK_SECONDARY, fontsize=11)
    ax.set_ylim(bottom=0)
    all_rates = sorted({r for s in series for r in s[1]})
    ax.set_xticks(all_rates)
    ax.set_xticklabels([f"{r:g}" for r in all_rates])
    ax.set_xlim(all_rates[0] - 0.3, all_rates[-1] + (1.6 if len(series) > 1 else 0.3))
    if len(series) > 1:
        ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--series", action="append", required=True, help='"label=glob[,glob...]"')
    parser.add_argument("-o", "--output-prefix", required=True, help="writes <prefix>_mean.png, <prefix>_p99.png")
    parser.add_argument("--font", default="DejaVu Sans")
    args = parser.parse_args()

    plt.rcParams["font.family"] = args.font
    plt.rcParams["axes.unicode_minus"] = False
    series = [_load_series(s) for s in args.series]
    plot_metric(series=series, metric="mean", out_path=f"{args.output_prefix}_mean.png", show_window_caps=False)
    plot_metric(series=series, metric="p99", out_path=f"{args.output_prefix}_p99.png", show_window_caps=True)


if __name__ == "__main__":
    main()
