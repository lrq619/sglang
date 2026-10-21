"""TokenCake baseline on the Hermes dataset (data/hermes.csv).

Online, causal per-type EWMA (alpha=0.5) over a single wall-clock event stream
spanning all sessions. Reports both granularities:
  - per-call : label = single tool latency, predict at that tool's start
  - per-turn : label = turn_toolcall_latency_ms, predict at turn start (llm_end);
               turn prediction = per-tool EWMA predictions combined the way
               Hermes executes the batch (parallel -> max, serial -> sum + 1s*(n-1)).
"""
import argparse, bisect, csv, collections, json, random, sys

csv.field_size_limit(sys.maxsize)

LETTERS = "ABCDEFGHIJKLM"
EDGES_MS = [100 * 2 ** k for k in range(12)]  # 100ms .. 204.8s
SHELL_TOOLS = {"terminal", "bash", "execute_bash", "exec"}
# Hermes runs a batch concurrently only if every tool is parallel-safe and the
# path-scoped ones touch distinct paths (verified against recorded turn latencies).
PAR_SAFE = {"read_file", "search_files", "patch", "write_file"}
PATH_SCOPED = {"read_file", "patch", "write_file"}
SERIAL_GAP_MS = 1000.0
ALPHA = 0.5
COLD_MS = 1000.0
TAIL = {"H": 7}  # long tail = >=6.4s (~P88 of Hermes turns)


def bucket_of(ms):
    return bisect.bisect_right(EDGES_MS, ms)


def ctype(name, args):
    if name in SHELL_TOOLS:
        cmd = (args.get("command") or "").strip()
        return cmd.split()[0] if cmd else name
    return name


def is_parallel(calls):
    if len(calls) < 2 or any(c["name"] not in PAR_SAFE for c in calls):
        return False
    paths = [c["args"].get("path") for c in calls if c["name"] in PATH_SCOPED]
    return len(paths) == len(set(paths))


def load(path):
    rows = list(csv.DictReader(open(path)))
    turns = []
    for r in rows:
        names = json.loads(r["tool_names"] or "[]")
        if not names or r["turn_toolcall_latency_source"] != "recorded":
            continue
        lat = json.loads(r["per_tool_latency_ms"])
        rel = json.loads(r["per_tool_latency_reliable"])
        tcs = json.loads(r["response"])["tool_calls"]
        assert len(tcs) == len(names) == len(lat)
        calls = []
        for tc, n, l, ok in zip(tcs, names, lat, rel):
            try:
                a = json.loads(tc["function"]["arguments"])
            except Exception:
                a = {}
            calls.append({"name": n, "args": a if isinstance(a, dict) else {},
                          "lat": float(l), "reliable": bool(ok)})
        for c in calls:
            c["type"] = ctype(c["name"], c["args"])
        t0 = int(r["llm_end_ns"]) / 1e6  # ms
        par = is_parallel(calls)
        off = 0.0
        for i, c in enumerate(calls):
            if par:
                c["start"] = t0
            else:
                off += SERIAL_GAP_MS if i else 0.0
                c["start"] = t0 + off
                off += c["lat"]
            c["end"] = c["start"] + c["lat"]
        turns.append({"session": r["session_id"], "bench": r["benchmark"],
                      "turn": int(r["turn_index"]), "t0": t0, "par": par,
                      "lat": float(r["turn_toolcall_latency_ms"]), "calls": calls})
    return turns


def split_sessions(turns, seed=0):
    by = collections.defaultdict(set)
    for t in turns:
        by[t["bench"]].add(t["session"])
    dev = set()
    rng = random.Random(seed)
    for b in sorted(by):
        ss = sorted(by[b])
        rng.shuffle(ss)
        dev.update(ss[: len(ss) // 2])
    return dev


class EWMA:
    def __init__(self):
        self.per = {}
        self.sys = None

    def update(self, typ, ms):
        self.per[typ] = ms if typ not in self.per else ALPHA * ms + (1 - ALPHA) * self.per[typ]
        self.sys = ms if self.sys is None else ALPHA * ms + (1 - ALPHA) * self.sys

    def predict(self, typ):
        if typ in self.per:
            return self.per[typ]
        return self.sys if self.sys is not None else COLD_MS


def run(turns):
    """Single causal event stream. Ends (kind 0) before starts (kind 1) at equal time."""
    ev = []
    for ti, t in enumerate(turns):
        ev.append((t["t0"], 1, "turn", ti, None))
        for ci, c in enumerate(t["calls"]):
            ev.append((c["start"], 1, "call", ti, ci))
            if c["reliable"]:
                ev.append((c["end"], 0, "end", ti, ci))
    ev.sort(key=lambda e: (e[0], e[1]))
    m = EWMA()
    for _, _, kind, ti, ci in ev:
        t = turns[ti]
        if kind == "end":
            c = t["calls"][ci]
            m.update(c["type"], c["lat"])
        elif kind == "call":
            c = t["calls"][ci]
            c["pred_ms"] = m.predict(c["type"])
        else:  # turn: predict every tool of the batch at turn start, then compose
            ps = [m.predict(c["type"]) for c in t["calls"]]
            t["pred_ms"] = max(ps) if t["par"] else sum(ps) + SERIAL_GAP_MS * (len(ps) - 1)


def metrics(pairs):
    """pairs: list of (true_bucket, pred_bucket)."""
    n = len(pairs)
    out = {"n": n,
           "exact": sum(a == b for a, b in pairs) / n,
           "within1": sum(abs(a - b) <= 1 for a, b in pairs) / n}
    for name, k in TAIL.items():
        tp = sum(a >= k and b >= k for a, b in pairs)
        pos = sum(a >= k for a, _ in pairs)
        pp = sum(b >= k for _, b in pairs)
        out[f">={name}"] = {"pos": pos, "recall": tp / pos if pos else float("nan"),
                            "precision": tp / pp if pp else float("nan")}
    return out


def fmt(name, m):
    s = f"{name:<16} n={m['n']:<5} exact={m['exact']*100:5.1f}%  ±1={m['within1']*100:5.1f}%"
    for k in TAIL:
        t = m[f">={k}"]
        s += f"  >={k}(pos={t['pos']}) R={t['recall']*100:5.1f}% P={t['precision']*100:5.1f}%"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/hermes.csv")
    ap.add_argument("--out", default="analysis/latency_pred/results/tokencake_hermes.jsonl")
    args = ap.parse_args()
    turns = load(args.csv)
    run(turns)
    dev = split_sessions(turns)

    with open(args.out, "w") as f:
        for t in turns:
            f.write(json.dumps({
                "session": t["session"], "bench": t["bench"], "turn": t["turn"],
                "split": "dev" if t["session"] in dev else "test", "parallel": t["par"],
                "lat_ms": t["lat"], "pred_ms": t["pred_ms"],
                "true": LETTERS[bucket_of(t["lat"])], "pred": LETTERS[bucket_of(t["pred_ms"])],
                "calls": [{"type": c["type"], "lat_ms": c["lat"], "pred_ms": c["pred_ms"],
                           "true": LETTERS[bucket_of(c["lat"])],
                           "pred": LETTERS[bucket_of(c["pred_ms"])]} for c in t["calls"]],
            }) + "\n")

    for level in ("per-turn", "per-call"):
        print(f"== {level} ==")
        groups = collections.defaultdict(list)
        for t in turns:
            sp = "dev" if t["session"] in dev else "test"
            items = [t] if level == "per-turn" else [c for c in t["calls"] if c["reliable"]]
            for x in items:
                p = (bucket_of(x["lat"]), bucket_of(x["pred_ms"]))
                for g in ("all", sp, "bench:" + t["bench"]):
                    groups[g].append(p)
        for g in ["all", "dev", "test"] + sorted(k for k in groups if k.startswith("bench:")):
            print(fmt(g, metrics(groups[g])))


if __name__ == "__main__":
    main()
