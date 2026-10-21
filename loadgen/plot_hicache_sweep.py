"""Plot a policy metric at r=8 against the effective HiCache (CPU) size.

Effective size = host KV + mamba pools minus the device mirror that write-through
keeps there (71.9 GB per server for Qwen3.8-27B on H200); ratio <= 1 leaves none.
Metrics are taken over the first hour of load: finished sessions, or the mean
censored JCT (a session still open at the hour counts until then). Needs matplotlib.

Usage:
    python plot_hicache_sweep.py --metric finished -o logs/r8_finished_vs_cpu_cache.png
    python plot_hicache_sweep.py --metric mean_jct -o logs/r8_meanjct_vs_cpu_cache.png
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
# Host KV + mamba pools per server from the launch logs; ratio 1 only mirrors the
# device (half of the ratio-2 pools).
MIRROR_GB = (62.72 + 81.13) / 2
# (hicache ratio, host pools GB, run prefix of the first evict/oracle sweep)
POINTS = [(0.5, 15.68 + 20.32, "hostcap0.5"), (1.25, 39.20 + 50.65, "hostcap1.25"),
          (1.5, 47.04 + 60.81, "hostcap1.5"), (2.0, 62.72 + 81.13, "cmp2")]
# Run globs per point, in POINTS order; None where the policy was not run.
SERIES = [
    ("pin-only", ["pinonly0.5_r8_h200_*", None, None, "pinonly2_r8_h200_*"], "#2a78d6", "o"),
    ("evict-only", [f"{pre}_evict_r8_h200_*" for _, _, pre in POINTS], "#eb6834", "s"),
    ("oracle", [f"lifeoracle{r:g}_r8_h200_*" for r, _, _ in POINTS], "#1baf7a", "D"),
]
METRICS = {
    # key: (y label, title, y max, label gap in y units)
    "finished": ("Finished sessions", "r=8: finished sessions vs CPU cache size", 340, 14),
    "mean_jct": ("Mean JCT (s)", "r=8: mean JCT vs CPU cache size", 1200, 50),
}
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def metric(run_dir: str, key: str) -> float:
    sessions = list(csv.DictReader(open(os.path.join(run_dir, "sessions.csv"))))
    cut = float(sessions[0]["window_start"]) + WINDOW_S
    finished, jct = 0, []
    for s in sessions:
        if not s["t_arrival"] or float(s["t_arrival"]) > cut:
            continue
        done = s["status"] == "done" and s["t_end"] and float(s["t_end"]) <= cut
        finished += bool(done)
        jct.append((float(s["t_end"]) if done else cut) - float(s["t_arrival"]))
    return finished if key == "finished" else sum(jct) / len(jct)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    ylabel, title, ymax, gap = METRICS[args.metric]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [max(0.0, total - MIRROR_GB) for _, total, _ in POINTS]
    ends = []
    for label, globs, color, marker in SERIES:
        pts = [(x, metric(glob.glob(LOGS + g)[0], args.metric)) for x, g in zip(xs, globs)
               if g is not None]
        print(label, [(round(x, 1), round(y, 1)) for x, y in pts])
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, linewidth=2,
                marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label, zorder=3)
        ends.append((pts[-1][1], label))
    # Direct labels at the line ends, pushed apart when the ends nearly coincide.
    placed = []
    for y, label in sorted(ends):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(label, (xs[-1], y), xytext=(xs[-1] + 3, y_text), textcoords="data",
                    va="center", fontsize=10, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.0f}\n(ratio {'<=1' if r < 1 else f'{r:g}'})"
                        for x, (r, _, _) in zip(xs, POINTS)])
    ax.set_xlim(-4, xs[-1] + 14)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Effective HiCache (GB)", color=INK2, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=11)
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=12)
    ax.legend(frameon=False, loc="lower right" if args.metric == "finished" else "lower left",
              fontsize=10, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
