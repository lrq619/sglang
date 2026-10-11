"""Predict every turn of one run with the v2 Monte Carlo predictor and plot
ground-truth vs predicted wait (client-side, send -> first forward) over time.

Step costs and the decode-length model are calibrated as in qp_predict.py
(on all --calib-runs). Needs matplotlib and numpy.

Usage:
    python plot_queue_pred.py logs/sweep_X_r2_* --calib-runs "logs/sweep_*_r*" \
        --corpus data/hermes_clean.jsonl -o work_dirs/queue_pred_r2
"""

import argparse
import csv
import glob
import os
import random
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qp_predict import DecodeModel, Snapshot, _rate_of, predict_mc_v2  # noqa: E402
from qp_sim2 import fit_step_costs  # noqa: E402
from qp_state import InstanceView, collapse_onset, load_turns  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-2, light-surface chrome.
TRUTH_COLOR = "#2a78d6"
PRED_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def predict_run(*, run_dir: str, costs, model: DecodeModel, rng: random.Random) -> tuple:
    """(rows, collapse onset in minutes per instance, None if it never collapses)."""
    turns = load_turns(os.path.join(run_dir, "loadgen.csv"))
    t_start = min(x.t_send for ts in turns.values() for x in ts)
    rows = []
    onsets = []
    for i, ep in enumerate(sorted(turns)):
        view = InstanceView(turns[ep])
        onset = collapse_onset(view.turns)
        onsets.append(None if onset == float("inf") else (onset - t_start) / 60.0)
        for x in view.turns:
            waits = predict_mc_v2(snap=Snapshot(view=view, x=x, t0=x.t_send), costs=costs, model=model, rng=rng)
            rows.append({
                "instance": i,
                "t_min": (x.t_send - t_start) / 60.0,
                "actual_s": x.t_admit - x.t_send,
                "pred_mean_s": statistics.mean(waits),
                "pred_p90_s": float(np.percentile(waits, 90)),
            })
    return rows, onsets


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def plot(*, rows: list, onsets: list, rate: float, out_path: str):
    n_inst = 1 + max(r["instance"] for r in rows)
    fig, axes = plt.subplots(n_inst, 1, figsize=(13, 3.4 * n_inst), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ymax = max(max(r["actual_s"], r["pred_mean_s"]) for r in rows)
    for i, ax in enumerate(np.atleast_1d(axes)):
        _style(ax)
        rs = [r for r in rows if r["instance"] == i]
        t = np.array([r["t_min"] for r in rs])
        ax.scatter(t, [r["actual_s"] for r in rs], s=14, color=TRUTH_COLOR, marker="o",
                   edgecolors=SURFACE, linewidths=0.6, label="ground truth", zorder=3)
        ax.scatter(t, [r["pred_mean_s"] for r in rs], s=16, color=PRED_COLOR, marker="x",
                   linewidths=1.0, label="predicted (v2 Monte Carlo mean)", zorder=4)
        title = f"instance {i}"
        if onsets[i] is not None:
            ax.axvline(onsets[i], color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
            ax.annotate("cache-thrashing collapse", (onsets[i], ymax * 1.02), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_MUTED, va="top")
            title += f"  (collapse at {onsets[i]:.0f} min)"
        ax.set_title(title, loc="left", fontsize=10, color=INK_SECONDARY)
        ax.set_ylabel("Wait until first forward (s)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylim(-0.03 * ymax, ymax * 1.08)
    np.atleast_1d(axes)[-1].set_xlabel("Minutes since run start", color=INK_SECONDARY, fontsize=10)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=2)
    fig.suptitle(f"Queueing latency: ground truth vs predicted (r = {rate:g} sessions/min)",
                 x=0.01, ha="left", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--calib-runs", required=True, help="glob of runs to calibrate step costs on")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("-o", "--output-prefix", required=True, help="writes <prefix>.png and <prefix>.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    calib = [d for d in glob.glob(args.calib_runs) if os.path.isdir(d)]
    costs = fit_step_costs([InstanceView(ts) for d in calib
                            for ts in load_turns(os.path.join(d, "loadgen.csv")).values()], rng)
    model = DecodeModel(args.corpus)
    run_dir = glob.glob(args.run_dir)[0]
    rows, onsets = predict_run(run_dir=run_dir, costs=costs, model=model, rng=rng)
    with open(f"{args.output_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = np.array([r["actual_s"] for r in rows])
    p = np.array([r["pred_mean_s"] for r in rows])
    print(f"{len(rows)} turns | actual >1s: {(a > 1).sum()} | predicted >1s: {(p > 1).sum()} | "
          f"both >1s: {((a > 1) & (p > 1)).sum()} | MAE {np.abs(p - a).mean():.3f}s")
    plot(rows=rows, onsets=onsets, rate=_rate_of(run_dir), out_path=f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
