"""Build hermes_v1.csv (the hermes.csv replay format, .claude/REPLAY.md) from the
qwen38_fp8_corpus_r0 archive of Hermes runs on terminal-bench.

Source of truth is each cell's logs/wire.jsonl: the exact request every agent LLM
call sent (messages, tools, chat template kwargs) with its start and end time and,
in the streamed tail, the usage. sessions.jsonl is not used for prompts: Hermes
rewrites old tool results in the history it sends, so its stored transcript is
not what the model saw. Calls without agent_tracker (the compaction summary and
other auxiliary requests) and non-200 calls are skipped.

Per task (agent_tracker.session_id), calls in start order give one row each:
  * the prompt is the request rendered with the model's chat template
    (enable_thinking / preserve_thinking as sent), cut right after the
    generation prompt's <think>;
  * the response is the assistant message the next request appended to this
    one's messages; decode = render(messages + response) past that cut, ending
    with <|im_end|>; the last call takes its response from sessions.jsonl;
  * the next row extends the previous context (prompt_reset False) when its
    prompt starts with prompt + response (prev_response_in_history True) or
    with the prompt alone (False); otherwise the history was rewritten (tool
    result pruning, compaction) and the row carries its full prompt
    (prompt_reset True);
  * turn_toolcall_latency_ms = next call start - this call end.

Every split sits right after a special token (<think>, <|im_end|>), so segments
tokenize independently; --tokenize-url checks this per row by also tokenizing
each full prompt and comparing it with the concatenated segments.

Usage (tokenization on a running SGLang server of the same model):
    python build_hermes_v1.py data/qwen38_fp8_corpus_r0.zip -o data/hermes_v1.csv \\
        --tokenize-url http://127.0.0.1:30000
    python build_hermes_v1.py data/qwen38_fp8_corpus_r0.zip --dry-run   # no tokenizer
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Optional

DEFAULT_MODEL = "/data/projects/12004493/models/Qwen3.8-27B"
THINK = "<think>"
IM_END = "<|im_end|>"
CSV_COLUMNS = [
    "session_id", "benchmark", "task", "turn_index", "prompt_reset",
    "prev_response_in_history", "prompt_delta", "response", "prefill_token_ids",
    "decode_token_ids", "prefill_num_tokens", "decode_num_tokens",
    "turn_toolcall_latency_ms", "turn_toolcall_latency_source", "per_tool_latency_ms",
    "tool_names", "per_tool_latency_reliable", "llm_start_ns", "llm_end_ns",
    "recorded_context_tokens", "recorded_decode_tokens", "prompt_available",
    "response_source",
]
USAGE_RE = re.compile(r'"usage":\s*(\{[^{}]*\})')


# ===== Rendering =====


def load_template(model: str):
    """The model's tokenizer, used only to render its chat template (the same
    jinja environment vLLM and SGLang render with)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def to_template_message(m: dict) -> dict:
    """OpenAI wire message -> what the chat template reads: reasoning as
    reasoning_content, tool-call arguments as a mapping (as vLLM renders them)."""
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m["role"] == "assistant":
        if m.get("reasoning") is not None:
            out["reasoning_content"] = m["reasoning"]
        elif m.get("reasoning_content") is not None:
            out["reasoning_content"] = m["reasoning_content"]
        calls = m.get("tool_calls")
        if isinstance(calls, str):
            calls = json.loads(calls)
        if calls:
            fixed = []
            for c in calls:
                fn = dict(c.get("function", c))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"arguments": args}
                fixed.append({**c, "function": fn})
            out["tool_calls"] = fixed
    if m["role"] == "tool" and m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    return out


def render(tpl, *, messages, tools, kwargs, add_generation_prompt) -> str:
    return tpl.apply_chat_template(
        [to_template_message(m) for m in messages],
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **kwargs,
    )


def cut_prompt(text: str) -> str:
    """Prompt through the generation prompt's <think>; its trailing newline
    starts the decode, as in hermes.csv."""
    i = text.rfind(THINK)
    assert i >= 0 and text[i + len(THINK):].strip() == "", text[-80:]
    return text[: i + len(THINK)]


def cut_full(text: str) -> str:
    """Context after the response: through its <|im_end|>."""
    i = text.rfind(IM_END)
    assert i >= 0, text[-80:]
    return text[: i + len(IM_END)]


# ===== Inputs =====


def read_calls(zf: zipfile.ZipFile) -> dict:
    """task run id -> its successful agent calls, in start order."""
    calls = defaultdict(list)
    for name in sorted(n for n in zf.namelist() if n.endswith("logs/wire.jsonl")):
        with zf.open(name) as fh:
            for line in fh:
                w = json.loads(line)
                req = w["request"]
                at = req.get("agent_tracker")
                if not at or w.get("status") != 200:
                    continue
                usage = None
                found = USAGE_RE.findall(w.get("response_tail") or "")
                if found:
                    try:
                        usage = json.loads(found[-1])
                    except json.JSONDecodeError:
                        usage = None
                calls[at["session_id"]].append(
                    {
                        "t": w["t"],
                        "t_end": w["t_end"],
                        "messages": req["messages"],
                        "tools": req.get("tools"),
                        "kwargs": req.get("chat_template_kwargs") or {},
                        "usage": usage,
                    }
                )
    for v in calls.values():
        v.sort(key=lambda c: c["t"])
    return calls


def read_final_responses(zf: zipfile.ZipFile) -> dict:
    """task run id -> last assistant message of its last session (sessions.jsonl)."""
    out = {}
    for name in zf.namelist():
        if not name.endswith("/sessions.jsonl") or "/aries/" in name or "/runs/" not in name:
            continue
        task = name.split("/")[-2]
        with zf.open(name) as fh:
            rows = sorted((json.loads(l) for l in fh), key=lambda r: r["started_at"])
        if not rows:
            continue
        assistants = [m for m in rows[-1]["messages"] if m["role"] == "assistant"]
        if assistants:
            out[task] = assistants[-1]
    return out


# ===== Rows =====


def build_session(tpl, run_id: str, calls: list, final: Optional[dict], stats: Counter):
    """Rows as text segments; tokenized later."""
    task_run = run_id.split("-hm_nocompact_rotate_rr-")[-1]
    task = re.sub(r"-\d{3}$", "", task_run)
    sid = hashlib.sha1(run_id.encode()).hexdigest()[:16]
    rows = []
    prev = None  # (prompt_cut, full_cut, messages)
    for k, c in enumerate(calls):
        prompt = cut_prompt(
            render(tpl, messages=c["messages"], tools=c["tools"], kwargs=c["kwargs"],
                   add_generation_prompt=True)
        )
        nxt = calls[k + 1] if k + 1 < len(calls) else None
        n = len(c["messages"])
        response, source = None, "none"
        if nxt is not None and len(nxt["messages"]) > n and nxt["messages"][:n] == c["messages"] \
                and nxt["messages"][n]["role"] == "assistant":
            response, source = nxt["messages"][n], "wire"
        elif nxt is None and final is not None:
            response, source = final, "sessions.jsonl"
        if response is not None:
            full = cut_full(
                render(tpl, messages=c["messages"] + [response], tools=c["tools"],
                       kwargs=c["kwargs"], add_generation_prompt=False)
            )
            if not full.startswith(prompt):
                stats["response_not_prefix"] += 1
                response, full = None, None
        else:
            full = None
        stats[f"response_{source if response is not None else 'none'}"] += 1

        if prev is None:
            reset, in_hist, delta = True, False, prompt
        elif prev[1] is not None and prompt.startswith(prev[1]):
            reset, in_hist, delta = False, True, prompt[len(prev[1]):]
        elif prompt.startswith(prev[0]):
            reset, in_hist, delta = False, False, prompt[len(prev[0]):]
        else:
            reset, in_hist, delta = True, False, prompt
        stats["reset" if reset else ("extend_after_response" if in_hist else "extend_without_response")] += 1
        new_msgs = c["messages"] if reset else c["messages"][len(prev[2]) + (1 if in_hist else 0):]

        usage = c["usage"] or {}
        latency = (nxt["t"] - c["t_end"]) * 1000 if nxt is not None else None
        tool_names = []
        if response is not None:
            tc = response.get("tool_calls")
            if isinstance(tc, str):
                tc = json.loads(tc)
            tool_names = [(x.get("function") or x).get("name") for x in (tc or [])]
        rows.append(
            {
                "session_id": sid,
                "benchmark": "terminal_bench",
                "task": task,
                "turn_index": k + 1,
                "prompt_reset": reset,
                "prev_response_in_history": in_hist,
                "prompt_delta": json.dumps(new_msgs, ensure_ascii=False),
                "response": json.dumps(
                    {k2: response.get(k2) for k2 in ("content", "reasoning", "tool_calls")}
                    if response is not None else None, ensure_ascii=False),
                "_prefill_text": delta,
                "_decode_text": full[len(prompt):] if full is not None else None,
                "_prompt_text": prompt,
                "turn_toolcall_latency_ms": None if latency is None else round(latency, 3),
                "turn_toolcall_latency_source": "recorded" if latency is not None else "",
                "per_tool_latency_ms": "[]",
                "tool_names": json.dumps(tool_names),
                "per_tool_latency_reliable": "[]",
                "llm_start_ns": int(c["t"] * 1e9),
                "llm_end_ns": int(c["t_end"] * 1e9),
                "recorded_context_tokens": usage.get("prompt_tokens"),
                "recorded_decode_tokens": usage.get("completion_tokens"),
                "prompt_available": True,
                "response_source": source if response is not None else "none",
            }
        )
        prev = (prompt, full, c["messages"])
    return rows


# ===== Tokenization =====


class Tokenizer:
    def __init__(self, url: str):
        import requests

        self._session = requests.Session()
        self._url = url.rstrip("/") + "/tokenize"

    def __call__(self, texts: list) -> list:
        out = []
        for i in range(0, len(texts), 16):
            r = self._session.post(
                self._url, json={"prompt": texts[i:i + 16], "add_special_tokens": False},
                timeout=600,
            )
            r.raise_for_status()
            tokens = r.json()["tokens"]
            out += tokens if isinstance(tokens[0], list) else [tokens]
        return out


def tokenize_session(tok, rows: list, stats: Counter) -> None:
    prefill = tok([r["_prefill_text"] for r in rows])
    decode_idx = [i for i, r in enumerate(rows) if r["_decode_text"] is not None]
    decode = dict(zip(decode_idx, tok([rows[i]["_decode_text"] for i in decode_idx])))
    full = tok([r["_prompt_text"] for r in rows])
    ctx_before = ctx_after = None
    for i, r in enumerate(rows):
        if r["prompt_reset"]:
            ctx = prefill[i]
        elif r["prev_response_in_history"]:
            ctx = ctx_after + prefill[i]
        else:
            ctx = ctx_before + prefill[i]
        if ctx != full[i]:
            stats["token_mismatch"] += 1
        d = decode.get(i, [])
        usage_decode = r["recorded_decode_tokens"]
        r["prefill_token_ids"] = json.dumps(prefill[i])
        r["decode_token_ids"] = json.dumps(d)
        r["prefill_num_tokens"] = len(prefill[i])
        r["decode_num_tokens"] = usage_decode if usage_decode is not None else len(d)
        if usage_decode is not None and d:
            stats["decode_diff_abs_sum"] += abs(usage_decode - len(d))
        ctx_before, ctx_after = ctx, ctx + d


# ===== Main =====


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("archive")
    p.add_argument("-o", "--output", default="data/hermes_v1.csv")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenize-url")
    p.add_argument("--dry-run", action="store_true", help="render and split only")
    args = p.parse_args()
    if not args.dry_run and not args.tokenize_url:
        p.error("--tokenize-url is required unless --dry-run")

    tpl = load_template(args.model)
    zf = zipfile.ZipFile(args.archive)
    calls = read_calls(zf)
    finals = read_final_responses(zf)
    stats = Counter()
    tok = None if args.dry_run else Tokenizer(args.tokenize_url)
    writer = None
    if not args.dry_run:
        csv.field_size_limit(sys.maxsize)
        f = open(args.output, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
    chars = Counter()
    for n, (run_id, cs) in enumerate(sorted(calls.items())):
        rows = build_session(tpl, run_id, cs, finals.get(run_id.split("-hm_nocompact_rotate_rr-")[-1]), stats)
        stats["sessions"] += 1
        stats["rows"] += len(rows)
        chars["prefill"] += sum(len(r["_prefill_text"]) for r in rows)
        chars["prompt"] += sum(len(r["_prompt_text"]) for r in rows)
        if tok is not None:
            tokenize_session(tok, rows, stats)
            for r in rows:
                writer.writerow(r)
        print(f"[{n + 1}/{len(calls)}] {run_id[-50:]}: {len(rows)} rows, "
              f"{sum(r['prompt_reset'] for r in rows)} resets", flush=True)
    print(json.dumps(dict(stats), indent=1))
    print(f"prefill chars {chars['prefill']:,} vs full-prompt chars {chars['prompt']:,} "
          f"({chars['prefill'] / max(1, chars['prompt']):.1%})")


if __name__ == "__main__":
    main()
