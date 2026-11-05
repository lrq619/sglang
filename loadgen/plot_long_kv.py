"""Summarize and plot a bench_long_kv.py run.

Reads <rundir>/trials.csv, keeps valid trials, and writes:
  summary.csv      per point: mean/std over trials of the per-trial short-request
                   mean TPOT / E2E / TTFT, and slowdowns vs A and vs C(L_min)
  tpot_vs_L.png    short-request mean TPOT vs stressor context length
  e2e_vs_L.png     short-request mean E2E latency vs stressor context length
A mix sweep (several stressor counts at fixed concurrency) writes summary.csv,
tpot_vs_mix.png and e2e_vs_mix.png instead, keyed by the number of stressors.

Usage:
    python loadgen/plot_long_kv.py logs/longkv_<ts>
"""

import argparse
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _ms(v):
    return float(v) * 1e3


def _stats(vals):
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def summarize(trials):
    """{point: {metric: (mean, std), 'n': ..., 'n_invalid': ...}}, point 0 means A."""
    points = sorted({int(t["stress_len"]) for t in trials})
    out = {}
    for L in points:
        rows = [t for t in trials if int(t["stress_len"]) == L]
        valid = [t for t in rows if t["valid"] == "1"]
        out[L] = {
            "n": len(valid), "n_invalid": len(rows) - len(valid),
            "invalid_reasons": sorted({t["invalid_reason"] for t in rows if t["valid"] != "1"}),
        }
        for m in ("mean_tpot", "mean_e2e", "mean_ttft"):
            out[L][m] = _stats([_ms(t[m]) for t in valid if t[m]])
    return out


def write_summary(summ, path):
    a_tpot, a_e2e = summ.get(0, {}).get("mean_tpot", (None,))[0], summ.get(0, {}).get("mean_e2e", (None,))[0]
    c_pts = [L for L in summ if L > 0 and summ[L]["n"]]
    lmin = min(c_pts) if c_pts else None
    ref_tpot = summ[lmin]["mean_tpot"][0] if lmin else None
    ref_e2e = summ[lmin]["mean_e2e"][0] if lmin else None
    fields = ["point", "stress_len", "n_valid", "n_invalid",
              "tpot_ms_mean", "tpot_ms_std", "e2e_ms_mean", "e2e_ms_std",
              "ttft_ms_mean", "ttft_ms_std",
              "tpot_minus_A_ms", "tpot_slowdown_vs_A", "tpot_minus_Lmin_ms", "tpot_slowdown_vs_Lmin",
              "e2e_slowdown_vs_A", "e2e_slowdown_vs_Lmin", "invalid_reasons"]
    rows = []
    for L, s in sorted(summ.items()):
        tpot, e2e, ttft = s["mean_tpot"], s["mean_e2e"], s["mean_ttft"]
        div = lambda x, y: x / y if x is not None and y else None
        sub = lambda x, y: x - y if x is not None and y is not None else None
        rows.append({
            "point": "A" if L == 0 else f"C({L})", "stress_len": L,
            "n_valid": s["n"], "n_invalid": s["n_invalid"],
            "tpot_ms_mean": tpot[0], "tpot_ms_std": tpot[1],
            "e2e_ms_mean": e2e[0], "e2e_ms_std": e2e[1],
            "ttft_ms_mean": ttft[0], "ttft_ms_std": ttft[1],
            "tpot_minus_A_ms": sub(tpot[0], a_tpot), "tpot_slowdown_vs_A": div(tpot[0], a_tpot),
            "tpot_minus_Lmin_ms": None if L == 0 else sub(tpot[0], ref_tpot),
            "tpot_slowdown_vs_Lmin": None if L == 0 else div(tpot[0], ref_tpot),
            "e2e_slowdown_vs_A": div(e2e[0], a_e2e),
            "e2e_slowdown_vs_Lmin": None if L == 0 else div(e2e[0], ref_e2e),
            "invalid_reasons": " | ".join(s["invalid_reasons"]),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def _fmt_len(L):
    return f"{L // 1024}K" if L >= 1024 else str(L)


def plot(summ, metric, ylabel, path, title):
    c_pts = [L for L in sorted(summ) if L > 0 and summ[L]["n"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if c_pts:
        ys = [summ[L][metric][0] for L in c_pts]
        es = [summ[L][metric][1] for L in c_pts]
        ax.errorbar(c_pts, ys, yerr=es, marker="o", capsize=4, color="#2a6fdb",
                    label="C(L): 8 short + 8 stressors")
    if 0 in summ and summ[0]["n"]:
        m, s = summ[0][metric]
        ax.axhline(m, color="#555555", ls="--", label="A: 8 short alone")
        ax.axhspan(m - s, m + s, color="#555555", alpha=0.12, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(c_pts)
    ax.set_xticklabels([_fmt_len(L) for L in c_pts])
    ax.set_xlabel("Stressor context length L (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===== Mix sweep: fixed concurrency, varying number of stressors =====


def summarize_mix(trials):
    """{n_stress: {...}} over valid trials, same layout as summarize()."""
    out = {}
    for k in sorted({int(t["n_stress"]) for t in trials}):
        rows = [t for t in trials if int(t["n_stress"]) == k]
        valid = [t for t in rows if t["valid"] == "1"]
        out[k] = {
            "n_short": int(rows[0]["n_short"]), "stress_len": int(rows[0]["stress_len"]),
            "n": len(valid), "n_invalid": len(rows) - len(valid),
            "invalid_reasons": sorted({t["invalid_reason"] for t in rows if t["valid"] != "1"}),
        }
        for m in ("mean_tpot", "mean_e2e", "mean_ttft"):
            out[k][m] = _stats([_ms(t[m]) for t in valid if t[m]])
    return out


def write_summary_mix(summ, path):
    ref = summ.get(0)
    rows = []
    for k, s in sorted(summ.items()):
        row = {"mix": f"{s['n_short']}:{k}", "n_short": s["n_short"], "n_stress": k,
               "stress_len": s["stress_len"], "n_valid": s["n"], "n_invalid": s["n_invalid"]}
        for m, name in (("mean_tpot", "tpot"), ("mean_e2e", "e2e"), ("mean_ttft", "ttft")):
            mean, std = s[m]
            row[f"{name}_ms_mean"], row[f"{name}_ms_std"] = mean, std
            r0 = ref[m][0] if ref and ref["n"] else None
            row[f"{name}_slowdown_vs_0"] = mean / r0 if mean is not None and r0 else None
        row["invalid_reasons"] = " | ".join(s["invalid_reasons"])
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_mix(summ, metric, ylabel, path, title):
    ks = [k for k in sorted(summ) if summ[k]["n"]]
    L = max(s["stress_len"] for s in summ.values())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(ks, [summ[k][metric][0] for k in ks], yerr=[summ[k][metric][1] for k in ks],
                marker="o", capsize=4, color="#2a6fdb",
                label=f"shorts (512 ctx) with k stressors ({_fmt_len(L)} ctx)")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{summ[k]['n_short']}:{k}" for k in ks])
    ax.set_xlabel("short : stressor (total concurrency fixed)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main_mix(trials, rundir):
    summ = summarize_mix(trials)
    rows = write_summary_mix(summ, os.path.join(rundir, "summary.csv"))
    plot_mix(summ, "mean_tpot", "Short-request mean TPOT (ms)",
             os.path.join(rundir, "tpot_vs_mix.png"), "Short-request TPOT vs stressor count")
    plot_mix(summ, "mean_e2e", "Short-request mean E2E latency (ms)",
             os.path.join(rundir, "e2e_vs_mix.png"), "Short-request E2E vs stressor count")
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    print(f"{'mix':>6} {'n':>3} {'inv':>3} {'TPOT ms':>14} {'x16:0':>6} {'E2E ms':>16} {'x16:0':>6} "
          f"{'TTFT ms':>8}")
    for r in rows:
        print(f"{r['mix']:>6} {r['n_valid']:>3} {r['n_invalid']:>3} "
              f"{f(r['tpot_ms_mean'])}+-{f(r['tpot_ms_std'])} {f(r['tpot_slowdown_vs_0'], 3):>6} "
              f"{f(r['e2e_ms_mean'], 1)}+-{f(r['e2e_ms_std'], 1)} {f(r['e2e_slowdown_vs_0'], 3):>6} "
              f"{f(r['ttft_ms_mean'], 1):>8}  {r['invalid_reasons']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("rundir")
    args = p.parse_args()
    trials = [t for t in _load(os.path.join(args.rundir, "trials.csv")) if int(t["rep"]) >= 0]
    if len({t.get("n_stress") for t in trials if t.get("n_stress") not in (None, "0")}) > 1:
        return main_mix(trials, args.rundir)
    summ = summarize(trials)
    rows = write_summary(summ, os.path.join(args.rundir, "summary.csv"))
    plot(summ, "mean_tpot", "Short-request mean TPOT (ms)",
         os.path.join(args.rundir, "tpot_vs_L.png"), "Short-request TPOT vs stressor context")
    plot(summ, "mean_e2e", "Short-request mean E2E latency (ms)",
         os.path.join(args.rundir, "e2e_vs_L.png"), "Short-request E2E vs stressor context")
    f = lambda v, d=2: "-" if v in (None, "") else f"{v:.{d}f}"
    print(f"{'point':>10} {'n':>3} {'inv':>3} {'TPOT ms':>14} {'xA':>6} {'xLmin':>6} "
          f"{'E2E ms':>16} {'xA':>6} {'xLmin':>6}")
    for r in rows:
        print(f"{r['point']:>10} {r['n_valid']:>3} {r['n_invalid']:>3} "
              f"{f(r['tpot_ms_mean'])}+-{f(r['tpot_ms_std'])} {f(r['tpot_slowdown_vs_A'], 3):>6} "
              f"{f(r['tpot_slowdown_vs_Lmin'], 3):>6} "
              f"{f(r['e2e_ms_mean'], 1)}+-{f(r['e2e_ms_std'], 1)} {f(r['e2e_slowdown_vs_A'], 3):>6} "
              f"{f(r['e2e_slowdown_vs_Lmin'], 3):>6}  {r['invalid_reasons']}")


if __name__ == "__main__":
    main()
