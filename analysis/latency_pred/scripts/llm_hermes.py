"""LLM suffix query (route1, per-turn) for tool-phase latency on Hermes.

Per turn: prefix = the agent's exact token context through this turn's reply
(prefill/decode token ids from data/hermes.csv, so it hits the radix cache),
suffix = one user message with the question plus online, causal evidence (same
information TokenCake has). One constrained letter is decoded; the A-M top
logprobs give a 13-bucket distribution.

  build : write prompts (evidence snapshot per turn) to a jsonl, no GPU needed
  query : send them to an SGLang server (/generate), write predictions
  eval  : compare against TokenCake on the identical turn set
"""
import argparse, bisect, collections, csv, difflib, json, math, os, random, re, sys, time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import tokencake_hermes as tc  # noqa: E402

csv.field_size_limit(sys.maxsize)
L = tc.LETTERS
TAIL_K = 7  # >=H (6.4 s)
BUCKET_TABLE = ("Buckets: A <0.1s | B 0.1-0.2s | C 0.2-0.4s | D 0.4-0.8s | E 0.8-1.6s | "
                "F 1.6-3.2s | G 3.2-6.4s | H 6.4-12.8s | I 12.8-25.6s | J 25.6-51s | "
                "K 51-102s | L 102-205s | M >=205s")
MODEL = os.environ.get("LP_MODEL", "/data/projects/12004493/models/Qwen3.8-27B")


# ---------------------------------------------------------------- text helpers
_HDR = re.compile(r"\s(?:-H|--header|-A|--user-agent)\s+(?:'[^']*'|\"[^\"]*\"|\S+)")
_REDIR = re.compile(r"\s*(?:2>/dev/null|2>&1|>/dev/null)")


def norm_cmd(cmd):
    cmd = _REDIR.sub("", _HDR.sub("", cmd))
    return " ".join(cmd.split())[:110]


def call_text(c):
    n, a = c["name"], c["args"]
    if n in tc.SHELL_TOOLS:
        return norm_cmd(a.get("command") or "")
    if n == "execute_code":
        return "execute_code: " + " ".join((a.get("code") or a.get("command") or "").split())[:100]
    if n == "read_file":
        return f"read_file {a.get('path', '')}"
    if n == "write_file":
        return f"write_file {a.get('path', '')} ({len(a.get('content') or '')} chars)"
    if n == "patch":
        return f"patch {a.get('path', '')}"
    if n == "search_files":
        return f"search_files {a.get('pattern', '')!r} in {a.get('path', '')}"[:110]
    if n == "process":
        return f"process {a.get('action', '')}"
    return n


def turn_text(t):
    return " ; ".join(call_text(c) for c in t["calls"])[:220]


def combo(t):
    cnt = collections.Counter(c["type"] for c in t["calls"])
    return " + ".join(f"{k}" if v == 1 else f"{k} x{v}" for k, v in sorted(cnt.items()))


def raw_args(c, limit=600):
    s = json.dumps(c["args"], ensure_ascii=False) if c["args"] else c.get("raw", "")
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def timeout_s(t):
    """Upper bound from explicit timeouts, only when every call carries one."""
    ts = []
    for c in t["calls"]:
        v = c["args"].get("timeout")
        try:
            ts.append(float(v))
        except (TypeError, ValueError):
            return None
    if not ts:
        return None
    return max(ts) if t["par"] else sum(ts) + (len(ts) - 1)


def fmt_s(ms):
    return f"{ms / 1000:.3f} s"


def dist_line(lats):
    c = collections.Counter(tc.bucket_of(x) for x in lats)
    n = len(lats)
    return " · ".join(f"{L[b]} {round(100 * v / n)}%" for b, v in sorted(c.items()))


def hist_line(text, ms):
    return f"    {text} -> {L[tc.bucket_of(ms)]} ({fmt_s(ms)})"


# ---------------------------------------------------------------- evidence
class History:
    def __init__(self):
        self.ewma = tc.EWMA()
        self.calls = collections.defaultdict(list)  # type -> [(text, ms)]
        self.turns = []  # completed: dict(session, text, combo, ms)
        self.by_combo = collections.defaultdict(list)
        self.by_session = collections.defaultdict(list)

    def end_call(self, c):
        self.ewma.update(c["type"], c["lat"])
        self.calls[c["type"]].append((call_text(c), c["lat"]))

    def end_turn(self, t):
        rec = {"session": t["session"], "text": turn_text(t), "combo": combo(t), "ms": t["lat"]}
        self.turns.append(rec)
        self.by_combo[rec["combo"]].append(rec)
        self.by_session[t["session"]].append(rec)

    def similar(self, text, k=6, window=3000):
        seen, scored = set(), []
        for r in self.turns[-window:]:
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            sm = difflib.SequenceMatcher(None, text, r["text"], autojunk=False)
            if sm.real_quick_ratio() < 0.3 or sm.quick_ratio() < 0.3:
                continue
            scored.append((sm.ratio(), r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]


def kind_block(h, t, stable):
    """Per-tool-kind history (TokenCake's information) + turn-level history of the same mix."""
    out = []
    types = list(dict.fromkeys(c["type"] for c in t["calls"]))
    for ty in types:
        hs = h.calls.get(ty, [])
        if not hs:
            out.append(f"Earlier calls of kind `{ty}`: none yet.")
            continue
        e = h.ewma.per[ty]
        lats = [x for _, x in hs]
        line = f"Earlier calls of kind `{ty}` in this environment: n={len(hs)}"
        if stable:
            b, v = collections.Counter(tc.bucket_of(x) for x in lats).most_common(1)[0]
            line += f" ({round(100 * v / len(lats))}% fall in {L[b]})"
        out.append(line)
        out.append(f"  bucket distribution: {dist_line(lats)}")
        out.append(f"  recent average (EWMA): {fmt_s(e)} -> {L[tc.bucket_of(e)]}")
        out.append("  most recent:")
        out += [hist_line(tx, ms) for tx, ms in reversed(hs[-4:])]
    if len(t["calls"]) > 1:
        ps = [h.ewma.predict(c["type"]) for c in t["calls"]]
        est = max(ps) if t["par"] else sum(ps) + tc.SERIAL_GAP_MS * (len(ps) - 1)
        out.append(f"Combining the per-kind averages for this turn gives {fmt_s(est)} -> {L[tc.bucket_of(est)]}.")
    key = combo(t)
    ts = h.by_combo.get(key, [])
    if ts and len(t["calls"]) > 1:
        out.append(f"Earlier turns with the same tool mix ({key}): n={len(ts)}")
        out.append(f"  bucket distribution: {dist_line([r['ms'] for r in ts])}")
        out += [hist_line(r["text"], r["ms"]) for r in reversed(ts[-4:])]
    return out


def build_suffix(h, t, stable):
    n = len(t["calls"])
    if n == 1:
        head = ["How long will this tool call take to run?",
                f"Tool call: {t['calls'][0]['name']} {raw_args(t['calls'][0])}"]
    else:
        mode = ("run in parallel; the turn lasts as long as the slowest call" if t["par"] else
                "run one after another with about 1 s between consecutive calls")
        head = [f"How long will the tool calls of this turn take to run in total? There are {n} calls, {mode}."]
        head += [f"  {i + 1}. {c['name']} {raw_args(c, 300)}" for i, c in enumerate(t["calls"])]
    body = ["", BUCKET_TABLE, ""]
    kinds = kind_block(h, t, stable)
    sess = h.by_session.get(t["session"], [])
    if stable:
        body += kinds
        same = [r for r in sess if r["combo"] == combo(t)]
        if same:
            body.append("This session, earlier turns of the same kind, most recent first:")
            body += [hist_line(r["text"], r["ms"]) for r in reversed(same[-4:])]
    else:
        body += kinds
        if sess:
            body.append("This session, most recent turns first:")
            body += [hist_line(r["text"], r["ms"]) for r in reversed(sess[-4:])]
        sim = h.similar(turn_text(t))
        if sim:
            body.append("Most similar earlier turns:")
            body += [hist_line(r["text"], r["ms"]) for r in sim]
    to = timeout_s(t)
    if to is not None:
        body.append(f"This turn's timeout: {to:g} s (so at most {L[tc.bucket_of(to * 1000)]}).")
    body.append("")
    if stable:
        body.append("Follow the history of the same kind unless this call itself gives a clear "
                    "reason to differ (different target, scale, flags or timeout).")
    body.append("Do not call any tool. Answer with the letter of the bucket only.")
    return "\n".join(head + body)


# ---------------------------------------------------------------- data
def load_all(csv_path, clean_path):
    turns = tc.load(csv_path)
    rows = {(r["session_id"], int(r["turn_index"])): r for r in csv.DictReader(open(csv_path))}
    clean = {json.loads(l)["session_id"] for l in open(clean_path)}
    return turns, rows, clean


def contexts(rows, clean):
    """Token context the model saw through each turn's reply, for clean sessions."""
    by = collections.defaultdict(list)
    for (s, k), r in rows.items():
        if s in clean:
            by[s].append(r)
    out = {}
    for s, rs in by.items():
        rs.sort(key=lambda r: int(r["turn_index"]))
        ctx, prev_dec = [], 0
        for r in rs:
            pre = json.loads(r["prefill_token_ids"])
            dec = json.loads(r["decode_token_ids"])
            if r["prompt_reset"] == "True":
                ctx = list(pre)
            else:
                if r["prev_response_in_history"] != "True" and prev_dec:
                    ctx = ctx[:-prev_dec]
                ctx = ctx + pre
            ctx = ctx + dec
            prev_dec = len(dec)
            out[(s, int(r["turn_index"]))] = list(ctx)
    return out


def stable_types(turns, dev, min_n=10, frac=0.95):
    lat = collections.defaultdict(list)
    for t in turns:
        if t["session"] in dev:
            for c in t["calls"]:
                lat[c["type"]].append(tc.bucket_of(c["lat"]))
    res = {}
    for ty, bs in lat.items():
        b, v = collections.Counter(bs).most_common(1)[0]
        if len(bs) >= min_n:
            res[ty] = (len(bs), v / len(bs), L[b])
    return res


def cmd_build(a):
    turns, rows, clean = load_all(a.csv, a.clean)
    dev = tc.split_sessions(turns)
    st = stable_types(turns, dev)
    stable = {ty for ty, (_, f, _) in st.items() if f >= 0.95}
    print("dev type concentration (n, top-bucket share, bucket):", file=sys.stderr)
    for ty, v in sorted(st.items(), key=lambda kv: -kv[1][0]):
        print(f"  {ty:<14} n={v[0]:<4} {v[1] * 100:5.1f}% {v[2]}{'  STABLE' if ty in stable else ''}", file=sys.stderr)
    tc.run(turns)  # TokenCake predictions on the same stream
    ev = []
    for ti, t in enumerate(turns):
        ev.append((t["t0"], 1, "turn", ti, None))
        ev.append((t["t0"] + t["lat"], 0, "tend", ti, None))
        for ci, c in enumerate(t["calls"]):
            if c["reliable"]:
                ev.append((c["end"], 0, "cend", ti, ci))
    ev.sort(key=lambda e: (e[0], e[1], e[2] != "cend"))
    h = History()
    out = open(a.out, "w")
    n = 0
    for _, _, kind, ti, ci in ev:
        t = turns[ti]
        if kind == "cend":
            h.end_call(t["calls"][ci])
        elif kind == "tend":
            h.end_turn(t)
        elif t["session"] in clean:
            is_stable = all(c["type"] in stable for c in t["calls"])
            out.write(json.dumps({
                "session": t["session"], "bench": t["bench"], "turn": t["turn"],
                "split": "dev" if t["session"] in dev else "test",
                "route": "hist5" if is_stable else "hist3", "lat_ms": t["lat"],
                "tokencake_ms": t["pred_ms"], "suffix": build_suffix(h, t, is_stable),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} prompts -> {a.out}", file=sys.stderr)


# ---------------------------------------------------------------- query
SUFFIX_PRE = "\n<|im_start|>user\n"
SUFFIX_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_query(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    letter_id = {tok.convert_tokens_to_ids(c): c for c in L}
    turns, rows, clean = load_all(a.csv, a.clean)
    ctxs = contexts(rows, clean)
    prompts = [json.loads(l) for l in open(a.prompts)]
    if a.split != "all":
        prompts = [p for p in prompts if p["split"] == a.split]
    prompts.sort(key=lambda p: (p["session"], p["turn"]))
    done = set()
    if os.path.exists(a.out):
        done = {(r["session"], r["turn"]) for r in map(json.loads, open(a.out))}
    out = open(a.out, "a")
    url = a.url.rstrip("/") + "/generate"
    for i, p in enumerate(prompts):
        if (p["session"], p["turn"]) in done:
            continue
        base = ctxs[(p["session"], p["turn"])]
        if a.warm:  # the agent's own request already cached the context
            post(url, {"input_ids": base, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
        ids = base + tok.encode(SUFFIX_PRE + p["suffix"] + SUFFIX_POST, add_special_tokens=False)
        t0 = time.time()
        r = post(url, {"input_ids": ids,
                       "sampling_params": {"max_new_tokens": 1, "temperature": 0, "regex": "[A-M]"},
                       "return_logprob": True, "top_logprobs_num": 20, "logprob_start_len": -1})
        dt = time.time() - t0
        mi = r["meta_info"]
        top = mi["output_top_logprobs"][0]
        probs = {letter_id[tid]: math.exp(lp) for lp, tid, *_ in top if tid in letter_id}
        z = sum(probs.values()) or 1.0
        dist = [probs.get(c, 0.0) / z for c in L]
        out.write(json.dumps({**{k: p[k] for k in ("session", "bench", "turn", "split", "route", "lat_ms", "tokencake_ms")},
                              "pred": r["text"].strip()[:1], "dist": dist, "ttft_s": dt,
                              "prompt_tokens": mi.get("prompt_tokens"), "cached_tokens": mi.get("cached_tokens")}) + "\n")
        out.flush()
        if i % 25 == 0:
            print(f"[{i}/{len(prompts)}] {p['session']}:{p['turn']} pred={r['text']!r} true={L[tc.bucket_of(p['lat_ms'])]} "
                  f"ttft={dt * 1000:.0f}ms cached={mi.get('cached_tokens')}/{mi.get('prompt_tokens')}", file=sys.stderr)


# ---------------------------------------------------------------- eval
def tail_stats(y, flag):
    tp = sum(a and b for a, b in zip(y, flag))
    pos, pp = sum(y), sum(flag)
    return (tp / pos if pos else float("nan")), (tp / pp if pp else float("nan"))


def summarize(rs, theta):
    yt = [tc.bucket_of(r["lat_ms"]) for r in rs]
    llm = [L.index(r["pred"]) for r in rs]
    tkc = [tc.bucket_of(r["tokencake_ms"]) for r in rs]
    ptail = [sum(r["dist"][TAIL_K:]) for r in rs]
    y = [b >= TAIL_K for b in yt]
    n = len(rs)
    res = {"n": n, "pos": sum(y)}
    res["tkc_exact"] = sum(a == b for a, b in zip(yt, tkc)) / n
    res["llm_exact"] = sum(a == b for a, b in zip(yt, llm)) / n
    res["tkc_w1"] = sum(abs(a - b) <= 1 for a, b in zip(yt, tkc)) / n
    res["llm_w1"] = sum(abs(a - b) <= 1 for a, b in zip(yt, llm)) / n
    res["tkc_R"], res["tkc_P"] = tail_stats(y, [b >= TAIL_K for b in tkc])
    res["llm_argmax_R"], res["llm_argmax_P"] = tail_stats(y, [b >= TAIL_K for b in llm])
    res["llm_R"], res["llm_P"] = tail_stats(y, [p >= theta for p in ptail])
    return res


def bootstrap(rs, theta, B=2000, seed=0):
    rng = random.Random(seed)
    keys = ["exact", "R", "P"]
    diffs = {k: [] for k in keys}
    for _ in range(B):
        s = [rs[rng.randrange(len(rs))] for _ in rs]
        m = summarize(s, theta)
        diffs["exact"].append(m["llm_exact"] - m["tkc_exact"])
        for k in ("R", "P"):
            d = m[f"llm_{k}"] - m[f"tkc_{k}"]
            if not math.isnan(d):
                diffs[k].append(d)
    ci = {}
    for k, v in diffs.items():
        v.sort()
        ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float("nan"),) * 2
    return ci


def pick_theta(rs):
    """Smallest theta whose tail precision >= TokenCake's precision on these turns."""
    tkc_P = summarize(rs, 1.0)["tkc_P"]
    for th in [x / 100 for x in range(1, 100)]:
        m = summarize(rs, th)
        if not math.isnan(m["llm_P"]) and m["llm_P"] >= tkc_P:
            return th
    return 0.99


def cmd_eval(a):
    rs = [json.loads(l) for l in open(a.preds)]
    dev = [r for r in rs if r["split"] == "dev"]
    th_dev = pick_theta(dev) if dev else 0.25
    print(f"theta picked on dev (precision >= TokenCake): {th_dev:.2f}")
    groups = {"all": rs, "dev": dev, "test": [r for r in rs if r["split"] == "test"]}
    for b in sorted({r["bench"] for r in rs}):
        groups["bench:" + b] = [r for r in rs if r["bench"] == b]
    for r in ("hist3", "hist5"):
        groups["route:" + r] = [x for x in rs if x["route"] == r]
    for theta_name, th in (("theta=0.25", 0.25), (f"theta_dev={th_dev:.2f}", th_dev)):
        print(f"\n== tail >=H (6.4 s), LLM flags tail when P(>=H) >= {th:.2f} [{theta_name}] ==")
        print(f"{'group':<26}{'n':>5}{'pos':>5} | {'exact TKC':>9} {'LLM':>6} | {'±1 TKC':>7} {'LLM':>6} |"
              f" {'tail R TKC':>10} {'LLM':>6} | {'tail P TKC':>10} {'LLM':>6} | argmax R/P")
        for g, x in groups.items():
            if not x:
                continue
            m = summarize(x, th)
            print(f"{g:<26}{m['n']:>5}{m['pos']:>5} | {m['tkc_exact'] * 100:8.1f}% {m['llm_exact'] * 100:5.1f}% |"
                  f" {m['tkc_w1'] * 100:6.1f}% {m['llm_w1'] * 100:5.1f}% | {m['tkc_R'] * 100:9.1f}% {m['llm_R'] * 100:5.1f}% |"
                  f" {m['tkc_P'] * 100:9.1f}% {m['llm_P'] * 100:5.1f}% | {m['llm_argmax_R'] * 100:.1f}/{m['llm_argmax_P'] * 100:.1f}")
        for g in ("all", "dev", "test"):
            if groups[g]:
                ci = bootstrap(groups[g], th)
                print(f"  bootstrap LLM-TKC 95% CI [{g}]: exact [{ci['exact'][0] * 100:+.1f}, {ci['exact'][1] * 100:+.1f}]"
                      f"  tail R [{ci['R'][0] * 100:+.1f}, {ci['R'][1] * 100:+.1f}]  tail P [{ci['P'][0] * 100:+.1f}, {ci['P'][1] * 100:+.1f}]")
    tt = sorted(r["ttft_s"] for r in rs)
    hit = sorted((r["cached_tokens"] or 0) / max(1, r["prompt_tokens"] or 1) for r in rs)
    print(f"\nsuffix query latency median/P90: {tt[len(tt) // 2] * 1000:.0f} / {tt[int(0.9 * len(tt))] * 1000:.0f} ms;"
          f" cache hit ratio median {hit[len(hit) // 2] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "query", "eval"])
    ap.add_argument("--csv", default="data/hermes.csv")
    ap.add_argument("--clean", default="data/hermes_clean.jsonl")
    ap.add_argument("--prompts", default="analysis/latency_pred/results/llm_hermes_prompts.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--url", default=os.environ.get("LP_URL", "http://127.0.0.1:30000"))
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        a.out = a.out or a.prompts
        cmd_build(a)
    elif a.cmd == "query":
        a.out = a.out or "analysis/latency_pred/results/llm_hermes_preds.jsonl"
        cmd_query(a)
    else:
        cmd_eval(a)


if __name__ == "__main__":
    main()
