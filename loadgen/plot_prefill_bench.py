"""Plot prefill throughput vs context length, one curve per bench_prefill.py run.

Each run dir holds bench.csv and run_info.txt (whose "label:" names the curve); the
curve is the median over reps at each length. Needs matplotlib.

Usage:
    python plot_prefill_bench.py logs/prefillbench_*_graph logs/prefillbench_*_eager -o out.png
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#8a5cd1"]
MARKERS = ["o", "s", "D", "^"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _label(run_dir: str) -> str:
    for line in open(os.path.join(run_dir, "run_info.txt")):
        if line.startswith("label:"):
            return line.split(":", 1)[1].strip()
    return os.path.basename(run_dir)


def _curve(run_dir: str) -> list:
    by_len = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run_dir, "bench.csv"))):
        by_len[int(r["length"])].append(float(r["tok_per_s"]))
    return [(n, statistics.median(v)) for n, v in sorted(by_len.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, d in enumerate(args.run_dirs):
        pts = _curve(d)
        label = _label(d)
        print(label, [(n, round(t)) for n, t in pts])
        ax.plot([n for n, _ in pts], [t for _, t in pts], color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)], linewidth=2, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)", color=INK2, fontsize=11)
    ax.set_ylabel("Prefill throughput (tok/s)", color=INK2, fontsize=11)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Single-request prefill throughput, Qwen3.8-27B on 1x H200", loc="left",
                 fontsize=13, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
