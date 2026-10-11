"""Plot each instance's KV release rate over time, as the fluid predictor sees it.

Release rate at time t = sum over turns finishing in (t - window, t] of the
tokens they free (input_len + decode_len), divided by the window. One panel per
run, one line per endpoint; also prints the coefficient of variation after
warm-up. Needs matplotlib (any Python with it).

Usage:
    python plot_release_rate.py logs/sweep_*_r* -o work_dirs/release_rate --windows 300 30
"""

import argparse
import glob
import math
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_state import load_turns  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-2, light-surface chrome.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
STEP_S = 10.0
WARMUP_S = 600.0


def _rate_of(run_dir: str) -> float:
    return float(re.search(r"_r([0-9.]+)_", os.path.basename(os.path.normpath(run_dir)) + "_").group(1))


def release_series(*, turns: list, t_start: float, t_stop: float, window_s: float) -> tuple:
    ends = np.array(sorted(t.t_end for t in turns))
    freed = np.array([t.input_len + t.decode_len for t in sorted(turns, key=lambda t: t.t_end)], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(freed)])
    grid = np.arange(t_start + window_s, t_stop, STEP_S)
    hi = np.searchsorted(ends, grid, side="right")
    lo = np.searchsorted(ends, grid - window_s, side="right")
    return (grid - t_start) / 60.0, (cum[hi] - cum[lo]) / window_s


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_window(*, runs: list, window_s: float, out_path: str) -> list:
    ncols = 3
    nrows = math.ceil(len(runs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    ymax = 0.0
    stats = []
    series = []
    for run_dir, turns_by_ep in runs:
        t_start = min(t.t_send for ts in turns_by_ep.values() for t in ts)
        t_stop = max(t.t_end for ts in turns_by_ep.values() for t in ts)
        per_ep = []
        for i, ep in enumerate(sorted(turns_by_ep)):
            x, y = release_series(turns=turns_by_ep[ep], t_start=t_start, t_stop=t_stop, window_s=window_s)
            per_ep.append((ep, x, y))
            steady = y[x * 60 >= WARMUP_S]
            if len(steady) > 1 and steady.mean() > 0:
                stats.append((_rate_of(run_dir), i, steady.mean(), statistics.pstdev(steady) / steady.mean()))
            ymax = max(ymax, y.max())
        series.append((run_dir, per_ep))
    for ax, (run_dir, per_ep) in zip(axes.flat, series):
        _style(ax)
        for i, (ep, x, y) in enumerate(per_ep):
            ax.plot(x, y / 1000, color=SERIES_COLORS[i], linewidth=1.0 if window_s < 120 else 1.6,
                    label=f"instance {i}")
        cvs = [s[3] for s in stats if s[0] == _rate_of(run_dir)]
        ax.set_title(f"r = {_rate_of(run_dir):g} sessions/min   (CV after 10 min: "
                     + ", ".join(f"{c:.2f}" for c in cvs) + ")",
                     loc="left", fontsize=9.5, color=INK_PRIMARY)
        ax.set_ylim(0, ymax / 1000 * 1.05)
    for ax in axes.flat[len(series):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Minutes since run start", color=INK_SECONDARY, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("K tokens freed / s", color=INK_SECONDARY, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=2)
    fig.suptitle(f"KV release rate per instance ({window_s:g} s sliding window)", x=0.01, ha="left",
                 fontsize=13, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("-o", "--output-prefix", required=True)
    parser.add_argument("--windows", nargs="+", type=float, default=[300.0])
    args = parser.parse_args()

    dirs = sorted({d.rstrip("/") for p in args.run_dirs for d in glob.glob(p) if os.path.isdir(d)}, key=_rate_of)
    runs = [(d, load_turns(os.path.join(d, "loadgen.csv"))) for d in dirs]
    for w in args.windows:
        stats = plot_window(runs=runs, window_s=w, out_path=f"{args.output_prefix}_{int(w)}s.png")
        print(f"window {w:g}s: rate, instance, mean K tok/s, CV")
        for r, i, m, cv in stats:
            print(f"  r={r:<5g} inst{i}  mean={m / 1000:6.1f}  CV={cv:.2f}")


if __name__ == "__main__":
    main()
