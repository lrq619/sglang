"""Plot the cumulative tool-call share of session time per instance over a run.

At time t, per endpoint: (tool-call wait elapsed by t) / (session lifetime elapsed
by t), summed over the sessions routed there. A session lives from its arrival to
its last turn's end (or the window end if cut); its tool-call wait is the gap
after each finished turn, turn_toolcall_latency. Everything else is LLM time
(queueing, prefill, decode). One subplot per run. Needs matplotlib.

Usage:
    python plot_toolcall_share.py logs/A logs/B --titles "pin-only" "evict-only" -o share.png
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_S = 30
# Arbitrary; smooths the 1 s metrics poll enough to ignore single-request blips.
QUEUE_WINDOW_S = 120
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _intervals(run_dir: str) -> tuple:
    """endpoint -> (alive intervals, tool intervals) in seconds since window start."""
    sessions = list(csv.DictReader(open(os.path.join(run_dir, "sessions.csv"))))
    w0 = float(sessions[0]["window_start"])
    w1 = float(sessions[0]["window_end"])
    turns = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run_dir, "loadgen.csv"))):
        if r["status"] == "ok":
            turns[r["replica_id"]].append(r)
    alive, tool = defaultdict(list), defaultdict(list)
    for s in sessions:
        if not s["t_arrival"] or float(s["t_arrival"]) >= w1:
            continue
        ep = s["endpoint"]
        end = float(s["t_end"]) if s["status"] == "done" else w1
        alive[ep].append((float(s["t_arrival"]) - w0, min(end, w1) - w0))
        for r in turns[s["replica_id"]]:
            if r["turn_toolcall_latency"]:
                t_end = float(r["t_end"])
                tool_end = min(t_end + float(r["turn_toolcall_latency"]), w1)
                tool[ep].append((t_end - w0, tool_end - w0))
    return alive, tool, w1 - w0


def _cumulative(intervals: list, grid: list) -> list:
    """Total length of intervals clipped to [0, t], for each t in grid."""
    return [sum(max(0.0, min(b, t) - a) for a, b in intervals if a < t) for t in grid]


def _queue_onset(run_dir: str) -> dict:
    """endpoint -> seconds since window start after which the queue never drains.

    Rough: the last time the rolling mean of num_queue_reqs over QUEUE_WINDOW_S is
    below 1; None if it is still below 1 in the last QUEUE_WINDOW_S of the run.
    """
    w0 = float(next(csv.DictReader(open(os.path.join(run_dir, "sessions.csv"))))["window_start"])
    by_ep = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))):
        if r["num_queue_reqs"]:
            by_ep[r["endpoint"]].append((float(r["t"]) - w0, float(r["num_queue_reqs"])))
    onset = {}
    for ep, rows in by_ep.items():
        rows.sort()
        last_drained, window, total, lo = 0.0, [], 0.0, 0
        for t, q in rows:
            window.append((t, q))
            total += q
            while window[lo][0] < t - QUEUE_WINDOW_S:
                total -= window[lo][1]
                lo += 1
            if t >= QUEUE_WINDOW_S and total / (len(window) - lo) < 1:
                last_drained = t
        onset[ep] = None if last_drained >= rows[-1][0] - QUEUE_WINDOW_S else last_drained
    return onset


def _share_series(run_dir: str) -> tuple:
    alive, tool, span = _intervals(run_dir)
    grid = [STEP_S * i for i in range(1, int(span // STEP_S) + 1)]
    series = {}
    for ep in sorted(alive):
        a = _cumulative(alive[ep], grid)
        t = _cumulative(tool[ep], grid)
        series[ep] = [100.0 * x / y if y else 0.0 for x, y in zip(t, a)]
    return grid, series


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--titles", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    titles = args.titles or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]

    plt.rcParams["font.family"] = "DejaVu Sans"
    ncols = min(len(args.run_dirs), 2)
    nrows = -(-len(args.run_dirs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.6 * nrows),
                             dpi=160, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, run_dir, title in zip(axes.flat, args.run_dirs, titles):
        grid, series = _share_series(run_dir)
        onset = _queue_onset(run_dir)
        ax.set_facecolor(SURFACE)
        for i, (ep, ys) in enumerate(series.items()):
            color = COLORS[i % len(COLORS)]
            name = f"instance {i} (:{ep.rsplit(':', 1)[1]})"
            ax.plot([t / 3600 for t in grid], ys, color=color, linewidth=2, label=name)
            if onset[ep] is None:
                print(f"{title} {name}: final {ys[-1]:.1f}%, no sustained queue")
                continue
            x = onset[ep] / 3600
            ax.axvline(x, color=color, linewidth=1, linestyle=":", alpha=0.7, zorder=1)
            ax.plot([x], [0], marker="^", markersize=10, color=color, markeredgecolor=SURFACE,
                    clip_on=False, zorder=4)
            ax.annotate(f"queuing\n{onset[ep] / 60:.0f} min", (x, 0), xytext=(4, 6 + 26 * i),
                        textcoords="offset points", fontsize=9, color=color)
            print(f"{title} {name}: final {ys[-1]:.1f}%, queuing from {onset[ep] / 60:.1f} min")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, grid[-1] / 3600)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time since load start (h)", color=INK2, fontsize=11)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
        ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK2)
    for row in axes:
        row[0].set_ylabel("Tool-call share of session time (%)", color=INK2, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
