"""Break an SGLang server's startup time into phases from its log and plot them.

Each run dir is one launch (see startup_profile.pbs): run_info.txt holds the unix
time the launch command was issued (launch_t) and servers/server0_*.log the server
log. Phases end at these log lines (second resolution); a phase whose marker is
missing (e.g. no HiCache) is zero:
    imports          launch -> first log line
    init             -> "Init parallel begin" (tokenizer, scheduler spawn)
    load_weight      -> "Load weight end"
    memory_pools     -> "Memory pool end" (device KV + mamba)
    kernel_init      -> "Capture target prefill CUDA graph begin"
    cuda_graph_prefill / cuda_graph_decode
    radix_cache_init -> "Init Unified Radix Cache"
    hicache_host_alloc -> "Tree cache initialized" (host KV + mamba pools)
    http_and_warmup  -> "The server is fired up and ready to roll!"
Needs matplotlib.

Usage:
    python plot_startup.py logs/startup_* -o logs/startup_breakdown.png
"""

import argparse
import datetime as dt
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (phase, regex of the log line that ends it); None ends at the first log line.
PHASES = [
    ("imports", None),
    ("init", r"Init parallel begin"),
    ("load_weight", r"Load weight end"),
    ("memory_pools", r"Memory pool end"),
    ("kernel_init", r"Capture target prefill CUDA graph begin"),
    ("cuda_graph_prefill", r"Capture target prefill CUDA graph end"),
    ("cuda_graph_decode", r"Capture target decode CUDA graph end"),
    ("radix_cache_init", r"Init Unified Radix Cache"),
    ("hicache_host_alloc", r"Tree cache initialized"),
    ("http_and_warmup", r"The server is fired up and ready to roll"),
]
COLORS = ["#c3c2b7", "#898781", "#2a78d6", "#8a5cd1", "#52514e", "#eb6834", "#f2a65a",
          "#1baf7a", "#d64a7a", "#e1c84a"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")


def _read_info(run_dir: str) -> dict:
    info = {}
    for line in open(os.path.join(run_dir, "run_info.txt")):
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def breakdown(run_dir: str) -> dict:
    """phase -> seconds for one launch."""
    launch_t = float(_read_info(run_dir)["launch_t"])
    log = glob.glob(os.path.join(run_dir, "servers", "server0_*.log"))[0]
    stamped = []
    for line in open(log, errors="replace"):
        m = _TS.match(line)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            stamped.append((t, line))
    out, prev = {}, launch_t
    for phase, pattern in PHASES:
        if pattern is None:
            end = stamped[0][0]
        else:
            hits = [t for t, line in stamped if re.search(pattern, line)]
            end = hits[0] if hits else prev
        out[phase] = max(0.0, end - prev)
        prev = max(prev, end)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    runs = [(_read_info(d).get("label", os.path.basename(d)), breakdown(d)) for d in args.run_dirs]
    for label, b in runs:
        print(f"{label}: total {sum(b.values()):.0f}s  " + "  ".join(f"{k}={v:.0f}" for k, v in b.items()))

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 1.6 + 1.1 * len(runs)), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(len(runs)))[::-1]
    for (phase, _), color in zip(PHASES, COLORS):
        left = [sum(v for k, v in b.items() if [p for p, _ in PHASES].index(k) < [p for p, _ in PHASES].index(phase))
                for _, b in runs]
        widths = [b[phase] for _, b in runs]
        ax.barh(ys, widths, left=left, color=color, height=0.55, label=phase.replace("_", " "),
                edgecolor=SURFACE, linewidth=0.8)
        for y, l, w in zip(ys, left, widths):
            if w >= 12:
                ax.text(l + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if color not in ("#c3c2b7", "#e1c84a", "#f2a65a") else INK)
    for y, (_, b) in zip(ys, runs):
        total = sum(b.values())
        ax.text(total + 4, y, f"{total:.0f} s", va="center", fontsize=10, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in runs], fontsize=10, color=INK2)
    ax.set_xlabel("Seconds since launch", color=INK2, fontsize=11)
    ax.set_xlim(0, max(sum(b.values()) for _, b in runs) * 1.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("SGLang startup breakdown, Qwen3.8-27B on 1x H200", loc="left", fontsize=14,
                 color=INK, pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5,
              fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
