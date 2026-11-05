"""Bar chart of mean censored JCT by queuing policy, pin-only vs evict-only.

r=8, HiCache ratio 2, first hour of load (a session still open at the hour counts
until then). Queuing policies:
  FCFS       - plain first come, first served
  Queue jump - later turns of a session are tried first and passed over when they
               do not fit (--prefix-pin-admit-first; for evict-only with
               --prefix-pin-admit-first-scope continuation)
  Continuum  - program-level FCFS: every turn ordered by its session's arrival,
               strictly (--enable-priority-scheduling, loadgen --session-priority)
Needs matplotlib.

Usage:
    python plot_queue_policy_bars.py -o logs/r8_jct_by_queue_policy.png
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = os.path.expanduser("~/scratch/logs/")
WINDOW_S = 3600
POLICIES = ["FCFS", "Queue jump", "Continuum"]
# (label, color, run glob per policy; None where it was not run)
SERIES = [
    ("pin-only", "#2a78d6", ["pinfcfs2_r8_h200_*", "pinonly2_r8_h200_*", None]),
    ("evict-only", "#eb6834", ["cmp2_evict_r8_h200_*", "evictcont2_r8_h200_*", "evictprog2_r8_h200_*"]),
]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def mean_jct(run_dir: str) -> float:
    sessions = list(csv.DictReader(open(os.path.join(run_dir, "sessions.csv"))))
    cut = float(sessions[0]["window_start"]) + WINDOW_S
    jct = []
    for s in sessions:
        if not s["t_arrival"] or float(s["t_arrival"]) > cut:
            continue
        done = s["status"] == "done" and s["t_end"] and float(s["t_end"]) <= cut
        jct.append((float(s["t_end"]) if done else cut) - float(s["t_arrival"]))
    return sum(jct) / len(jct)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.36
    for i, (label, color, globs) in enumerate(SERIES):
        offset = (i - (len(SERIES) - 1) / 2) * width
        for j, g in enumerate(globs):
            if g is None:
                continue
            y = mean_jct(sorted(glob.glob(LOGS + g))[-1])
            print(f"{label:10s} {POLICIES[j]:10s} {y:.0f} s")
            ax.bar(j + offset, y, width=width - 0.04, color=color, zorder=3,
                   label=label if j == 0 else None)
            ax.annotate(f"{y:.0f}", (j + offset, y), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax.tick_params(axis="x", colors=INK2, labelsize=11, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels(POLICIES)
    ax.set_ylim(0, 1100)
    ax.set_xlabel("Queuing policy", color=INK2, fontsize=11)
    ax.set_ylabel("Mean JCT (s)", color=INK2, fontsize=11)
    ax.set_title("r=8, HiCache ratio 2: mean JCT by queuing policy", loc="left",
                 fontsize=14, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
