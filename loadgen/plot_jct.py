"""Plot censored session JCT (mean and p99) against session arrival rate.

Each --series is one group of runs sharing a window length, given as
"label=glob[,glob...]"; groups with different windows are drawn as separate
lines, never joined, because a shorter window caps censored JCT lower. A single
series gets no legend (the title names it). The rate is parsed from the run name
(_r<rate>_). Needs matplotlib (not in the sglang venv; any Python with it works).

Usage:
    python plot_jct.py -o logs/jct \
        --series "3h window=logs/sweep_A_r*,logs/sweep_B_r*"
    python plot_jct.py -o logs/jct --colors "#2a78d6,#898781" \
        --series "evict-only=logs/sweep_A_r*" --series "oracle=logs/pincost_r*"
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
SERIES_MARKERS = ["o", "s", "D"]
SERIES_LINESTYLES = ["-", "--", ":"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Arbitrary; about one text line at the default figure size.
LABEL_GAP_FRAC = 0.05

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


def _label_line_ends(ax, *, series: list, key: str):
    """Direct label at each line end, in ink rather than the series color;
    labels closer than LABEL_GAP_FRAC of the y range are pushed apart."""
    ends = sorted(
        ((s[2][-1][key], s[1][-1], s[0]) for s in series), key=lambda e: e[0]
    )
    top = max(max(r[key] for r in s[2]) for s in series)
    gap = top * LABEL_GAP_FRAC
    placed = []
    for y, x, label in ends:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(
            label, (x, y), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=10, color=INK_SECONDARY,
        )


def plot_metric(
    *, series: list, metric: str, out_path: str, show_window_caps: bool, colors: list
):
    key, title = METRICS[metric]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for i, (label, rates, rows, window_s) in enumerate(series):
        values = [r[key] for r in rows]
        color = colors[i]
        ax.plot(
            rates, values, color=color, linewidth=2, linestyle=SERIES_LINESTYLES[i],
            marker=SERIES_MARKERS[i], markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label, zorder=3,
        )

    if len(series) > 1:
        _label_line_ends(ax, series=series, key=key)

    if show_window_caps:
        top = max(max(r[key] for r in s[2]) for s in series)
        # One cap line per distinct window length, however many series share it.
        for window_s in sorted({s[3] for s in series}):
            if window_s > top * 1.3:
                continue
            ax.axhline(window_s, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
            ax.annotate(
                f"{window_s / 3600:g}h window cap {window_s:.0f}s",
                (min(min(s[1]) for s in series), window_s),
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
    ax.set_xlim(all_rates[0] - 0.3, all_rates[-1] + (0.9 if len(series) > 1 else 0.3))
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
    parser.add_argument(
        "--colors",
        default=",".join(SERIES_COLORS),
        help="comma-separated line colors, one per --series in order",
    )
    args = parser.parse_args()

    plt.rcParams["font.family"] = args.font
    plt.rcParams["axes.unicode_minus"] = False
    series = [_load_series(s) for s in args.series]
    colors = args.colors.split(",")
    if len(colors) < len(series):
        sys.exit(f"--colors has {len(colors)} colors for {len(series)} series")
    for metric, caps in (("mean", False), ("p99", True)):
        plot_metric(
            series=series, metric=metric, out_path=f"{args.output_prefix}_{metric}.png",
            show_window_caps=caps, colors=colors,
        )


if __name__ == "__main__":
    main()
