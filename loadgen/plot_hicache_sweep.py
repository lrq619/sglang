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
# Host KV + mamba pools per server from the launch logs; ratio 1 only mirrors the
# device (half of the ratio-2 pools).
MIRROR_GB = (62.72 + 81.13) / 2
# pin-only does not depend on the host size, so a series whose label starts with
# FLAT draws its one value across every point.
FLAT = "pin-only"
_HOSTCAP = [(0.5, 15.68 + 20.32, "hostcap0.5"), (1.25, 39.20 + 50.65, "hostcap1.25"),
            (1.5, 47.04 + 60.81, "hostcap1.5"), (2.0, 62.72 + 81.13, "cmp2")]
_TOOL2X = [(3.0, 94.09 + 121.62, "tool2x3h_hc3"), (7.0, 219.54 + 283.56, "tool2x3h_hc7")]
_R5 = [(1.0, 31.36 + 40.64, "hc1"), (3.0, 94.09 + 121.62, "hc3")]
# Ratio 0 marks the pin-only runs, which have no HiCache.
_TL = [(0.0, 0.0, None), (1.25, 39.20 + 50.65, "hc1.25"), (1.5, 47.04 + 60.81, "hc1.5"),
       (2.0, 62.72 + 81.13, "hc2"), (3.0, 94.09 + 121.62, "hc3"), (4.0, 125.45 + 162.10, "hc4")]
SWEEPS = {
    # points: (hicache ratio, host pools GB, run prefix of the evict run)
    # series: (label, run glob per point or None where not run, color, marker)
    # limits: metric key -> (y max, label gap in y units)
    "r8": {
        "window_s": 3600,
        "title": "r=8",
        "points": _HOSTCAP,
        "series": [
            # plain FCFS (no queue jump), run only at ratio 2
            ("pin-only", [None, None, None, "pinfcfs2_r8_h200_*"], "#2a78d6", "o"),
            ("evict-only", [f"{pre}_evict_r8_h200_*" for _, _, pre in _HOSTCAP], "#eb6834", "s"),
            ("oracle", [f"lifeoracle{r:g}_r8_h200_*" for r, _, _ in _HOSTCAP], "#1baf7a", "D"),
        ],
        "limits": {"finished": (340, 14), "mean_jct": (1200, 50)},
    },
    "tool2x3h": {
        "window_s": 10800,
        "title": "r=8, tool-call x2, 3 h",
        "points": _TOOL2X,
        "series": [
            # admit-first (pinned next turns jump the queue), run only at ratio 3
            ("pin-only", ["tool2x3h_hc3_pinjump_r8_h200_*", None], "#2a78d6", "o"),
            ("evict-only", [f"{pre}_evict_r8_h200_*" for _, _, pre in _TOOL2X], "#eb6834", "s"),
        ],
        "limits": {"finished": (1100, 40), "mean_jct": (2800, 110)},
    },
}
# r=5, 3 h, original and 2x tool-call latencies; pin-only runs have no HiCache.
for _scale in (1, 2):
    SWEEPS[f"tool{_scale}x3h_r5"] = {
        "window_s": 10800,
        "title": f"r=5, tool-call x{_scale}, 3 h",
        "points": _R5,
        "series": [
            ("pin-only (FCFS)", [f"tool{_scale}x3h_pinfcfs_r5_h200_*", None], "#8a5cd1", "o"),
            ("pin-only (admit-first)", [f"tool{_scale}x3h_pinjump_r5_h200_*", None], "#2a78d6", "o"),
            ("evict-only", [f"tool{_scale}x3h_{pre}_evict_r5_h200_*" for _, _, pre in _R5],
             "#eb6834", "s"),
        ],
        "limits": {"finished": (1000, 40), "mean_jct": (2800, 110)},
    }
# trace_lab single / multi round at twice their per-GPU edge rate, 2 GPUs per run, 3 h.
# limits: arrivals in 3 h are 700 (single) and 284 (multi).
for _trace, _r, _limits in (
    ("single", "4", {"finished": (720, 28), "mean_jct": (2400, 90)}),
    ("multi", "1.6", {"finished": (300, 12), "mean_jct": (4000, 150)}),
):
    SWEEPS[f"tl{_trace}_r{_r}"] = {
        "window_s": 10800,
        "title": f"trace_lab {_trace}-round, r={_r}, 3 h",
        "points": _TL,
        "series": [
            ("pin-only (admit-first)", [f"tl{_trace}_pinjump_r{_r}_h200_*"] + [None] * (len(_TL) - 1),
             "#2a78d6", "o"),
            ("evict-only", [None] + [f"tl{_trace}_{pre}_evict_r{_r}_h200_*" for _, _, pre in _TL[1:]],
             "#eb6834", "s"),
        ],
        "limits": _limits,
        # Ticks at 0/18/36 GB are too close for "(ratio x)" captions; put ratios on a top axis.
        "ratio_axis": True,
        "gap_arrow": True,
    }
METRICS = {
    # key: (y label, title suffix)
    "finished": ("Finished sessions", "finished sessions vs CPU cache size"),
    "mean_jct": ("Mean JCT (s)", "mean JCT vs CPU cache size"),
}
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def metric(*, run_dir: str, key: str, window_s: int) -> float:
    sessions = list(csv.DictReader(open(os.path.join(run_dir, "sessions.csv"))))
    cut = float(sessions[0]["window_start"]) + window_s
    finished, jct = 0, []
    for s in sessions:
        if not s["t_arrival"] or float(s["t_arrival"]) > cut:
            continue
        done = s["status"] == "done" and s["t_end"] and float(s["t_end"]) <= cut
        finished += bool(done)
        jct.append((float(s["t_end"]) if done else cut) - float(s["t_arrival"]))
    return finished if key == "finished" else sum(jct) / len(jct)


def _draw_gap_arrow(ax, *, final: dict, x: float, unit: float, metric_key: str) -> None:
    """Two-headed arrow from the pin-only line to evict-only's last point, labelled
    with evict-only's relative gain over pin-only (lower JCT, more finished)."""
    pin = next(v for k, v in final.items() if k.startswith(FLAT))
    evict = final["evict-only"]
    if metric_key == "mean_jct":
        text = f"{(pin - evict) / pin:.0%}\nlower JCT"
    else:
        text = f"{(evict - pin) / pin:+.0%}\nfinished"
    x_arrow = x - 6 * unit
    ax.annotate("", xy=(x_arrow, evict), xytext=(x_arrow, pin),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.3,
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text(x_arrow - 2 * unit, (pin + evict) / 2, text, ha="right", va="center",
            fontsize=10.5, color=INK, fontweight="bold")


def _plot_sweep(ax, *, sweep: dict, metric_key: str, title: str) -> None:
    points = sweep["points"]
    ylabel, _ = METRICS[metric_key]
    ymax, gap = sweep["limits"][metric_key]
    ax.set_facecolor(SURFACE)
    xs = [max(0.0, total - MIRROR_GB) for _, total, _ in points]
    ends = []
    final = {}
    for label, globs, color, marker in sweep["series"]:
        pts = [(x, metric(run_dir=glob.glob(LOGS + g)[0], key=metric_key, window_s=sweep["window_s"]))
               for x, g in zip(xs, globs) if g is not None]
        print(label, [(round(x, 1), round(y, 1)) for x, y in pts])
        if label.startswith(FLAT):
            ax.plot(xs, [pts[0][1]] * len(xs), color=color, linewidth=2, label=label, zorder=3)
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, linewidth=2,
                marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=None if label.startswith(FLAT) else label, zorder=3)
        ends.append((pts[-1][1], label))
        final[label] = pts[-1][1]
    if sweep.get("gap_arrow"):
        _draw_gap_arrow(ax, final=final, x=xs[-1], unit=xs[-1] / 72, metric_key=metric_key)
    # Direct labels at the line ends, pushed apart when the ends nearly coincide.
    # Offsets were tuned on a 72 GB axis; scale them with the axis.
    unit = xs[-1] / 72
    placed = []
    for y, label in sorted(ends):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(label, (xs[-1], y), xytext=(xs[-1] + 3 * unit, y_text), textcoords="data",
                    va="center", fontsize=10, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    if sweep.get("ratio_axis"):
        ax.set_xticklabels([f"{x:.0f}" for x in xs])
    else:
        ax.set_xticklabels([f"{x:.0f}\n" + ("(no HiCache)" if r == 0 else f"(ratio {'<=1' if r < 1 else f'{r:g}'})")
                            for x, (r, _, _) in zip(xs, points)])
    ax.set_xlim(-4 * unit, xs[-1] + 14 * unit)
    if sweep.get("ratio_axis"):
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top.set_xticks(xs)
        top.set_xticklabels(["none" if r == 0 else f"{r:g}" for r, _, _ in points])
        top.set_xlabel("HiCache ratio", color=INK2, fontsize=10)
        top.tick_params(colors=MUTED, labelsize=9.5)
        for side in ("top", "right", "left", "bottom"):
            top.spines[side].set_visible(side == "top")
        top.spines["top"].set_color(BASE)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Effective HiCache (GB)", color=INK2, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=11)
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=12)
    ax.legend(frameon=False, loc="lower right" if metric_key == "finished" else "lower left",
              fontsize=10, labelcolor=INK2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--sweep", choices=SWEEPS, nargs="+", default=["r8"],
                        help="several sweeps are drawn side by side")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    _, title_suffix = METRICS[args.metric]

    plt.rcParams["font.family"] = "DejaVu Sans"
    n = len(args.sweep)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4.8), dpi=160, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, key in zip(axes[0], args.sweep):
        sweep = SWEEPS[key]
        title = f"{sweep['title']}: {title_suffix}" if n == 1 else sweep["title"]
        _plot_sweep(ax, sweep=sweep, metric_key=args.metric, title=title)
    if n > 1:
        fig.suptitle(title_suffix[0].upper() + title_suffix[1:], fontsize=15, color=INK)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
