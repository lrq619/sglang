"""Fit StepCosts by microbenchmarking an idle SGLang server (offline profiling).

Prefill: isolated requests over an (extend, prefix) grid with max_new_tokens=1;
y = prefill_finished_time - forward_entry_time from meta_info (needs
--enable-metrics), features summed over the request's chunks.
Decode: B concurrent streamed requests with ignore_eos over a (batch, context)
grid; y = per-token interval inside the window where all B decode together.

Usage:
    python -m controller.queuing_predictor.profile_costs --url http://127.0.0.1:30000 \
        --output controller/queuing_predictor/costs/qwen3.8-27b_h100.json
    # Refit from the printed points of an earlier run, no server needed:
    python -m controller.queuing_predictor.profile_costs --from-log profile_costs.log \
        --chunk 32768 --output ...
"""

from __future__ import annotations

import argparse
import asyncio
import os
import json
import re
import statistics
import subprocess
import time
from typing import List, Tuple

import aiohttp
import numpy as np

from controller.queuing_predictor.costs import StepCosts

# Plain-text token ids, clear of the special-token range of Qwen tokenizers.
TOKEN_LO, TOKEN_HI = 1000, 100000

PREFILL_EXTENDS = [256, 1024, 4096, 8192, 16384, 32768]
PREFILL_PREFIXES = [0, 8192, 32768, 65536, 131072]
DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 48, 64]
DECODE_CONTEXTS = [1024, 8192, 32768]


def _tokens(rng: np.random.Generator, n: int) -> List[int]:
    return rng.integers(TOKEN_LO, TOKEN_HI, size=n).tolist()


async def _generate(http, url: str, ids: List[int], max_new: int) -> dict:
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new, "ignore_eos": True},
    }
    async with http.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        return (await resp.json())["meta_info"]


async def _stream(http, url: str, ids: List[int], max_new: int) -> List[Tuple[float, int]]:
    """(monotonic receive time, completion tokens so far) per streamed chunk."""
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new, "ignore_eos": True},
        "stream": True,
    }
    out = []
    async with http.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            meta = json.loads(line[5:])["meta_info"]
            out.append((time.monotonic(), meta["completion_tokens"]))
    return out


async def _flush(http, url: str) -> None:
    async with http.post(f"{url}/flush_cache") as resp:
        await resp.read()


def _chunk_features(*, extend: int, prefix: int, chunk: int) -> Tuple[float, float, float]:
    steps, lin, quad = 0, 0.0, 0.0
    done, left = prefix, extend
    while left > 0:
        e = min(left, chunk)
        steps += 1
        lin += e
        quad += e * (done + e / 2) / 1e6
        done += e
        left -= e
    return steps, lin, quad


async def profile_prefill(http, args, cfg, rng) -> List[Tuple[Tuple[float, float, float], float]]:
    rows = []
    limit = min(args.max_context, cfg["max_total_num_tokens"] // 2)
    chunk = cfg["chunked_prefill_size"] or (1 << 40)
    for prefix in PREFILL_PREFIXES:
        for extend in PREFILL_EXTENDS:
            if prefix + extend > limit:
                continue
            for _ in range(args.repeats):
                head = _tokens(rng, prefix)
                if prefix:
                    await _generate(http, args.url, head, 1)
                meta = await _generate(http, args.url, head + _tokens(rng, extend), 1)
                y = meta["prefill_finished_time"] - meta["forward_entry_time"]
                cached = meta.get("cached_tokens", 0)
                e = prefix + extend - cached
                rows.append((_chunk_features(extend=e, prefix=cached, chunk=chunk), y))
                print(f"prefill E={e:6d} P={cached:6d} {y * 1e3:8.1f} ms", flush=True)
    return rows


async def profile_decode(http, args, cfg, rng) -> List[Tuple[Tuple[float, float, float], float]]:
    rows = []
    n = args.decode_tokens
    budget = int(cfg["max_total_num_tokens"] * 0.7)
    for ctx in DECODE_CONTEXTS:
        for b in DECODE_BATCHES:
            if b > cfg["max_running_requests"] or b * (ctx + n) > budget:
                continue
            traces = await asyncio.gather(
                *(_stream(http, args.url, _tokens(rng, ctx), n) for _ in range(b))
            )
            lo = max(t[0][0] for t in traces)
            hi = min(t[-1][0] for t in traces)
            per_req = []
            for t in traces:
                inside = [k for ts, k in t if lo <= ts <= hi]
                if len(inside) >= 2 and inside[-1] > inside[0]:
                    per_req.append((hi - lo) / (inside[-1] - inside[0]))
            if not per_req:
                continue
            y = statistics.median(per_req)
            x = (1.0, float(b), b * (ctx + n / 2) / 1e6)
            rows.append((x, y))
            print(f"decode B={b:3d} ctx={ctx:6d} {y * 1e3:7.2f} ms/step", flush=True)
            await _flush(http, args.url)
    return rows


def _fit(rows, *, relative: bool) -> Tuple[np.ndarray, float]:
    x = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    # Relative-error weighting so the many small prefills are not swamped.
    w = 1.0 / y if relative else np.ones_like(y)
    coef, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    rel_err = np.abs(x @ coef - y) / y
    return coef, float(np.median(rel_err))


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def main_async(args) -> None:
    rng = np.random.default_rng(args.seed)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        async with http.get(f"{args.url}/scheduler_snapshot") as resp:
            snap = (await resp.json())[0]
        if snap["running"] or snap["num_waiting"]:
            raise SystemExit("server is not idle; profile on a dedicated instance")
        cfg = snap["config"]
        print(f"config: {cfg}", flush=True)
        await _flush(http, args.url)
        await _generate(http, args.url, _tokens(rng, 64), 4)  # warm up
        pre = await profile_prefill(http, args, cfg, rng)
        await _flush(http, args.url)
        dec = await profile_decode(http, args, cfg, rng)
    with open(args.output + ".points.json", "w") as f:
        json.dump({"prefill": pre, "decode": dec, "config": cfg}, f)
    _fit_and_save(pre=pre, dec=dec, output=args.output,
                  meta={"url": args.url, "model_tag": args.model_tag, "config": cfg})


def _fit_and_save(*, pre, dec, output: str, meta: dict) -> None:
    p, p_err = _fit(pre, relative=True)
    d, d_err = _fit(dec, relative=False)
    costs = StepCosts(
        d0=float(d[0]), d1=float(d[1]), d2=float(d[2]),
        p0=float(p[0]), p1=float(p[1]), p2=float(p[2]),
        meta={
            **meta,
            "gpu": _gpu_name(),
            "profiled_unix": time.time(),
            "prefill_median_rel_err": p_err,
            "decode_median_rel_err": d_err,
            "num_prefill_points": len(pre),
            "num_decode_points": len(dec),
        },
    )
    costs.save(output)
    print(f"wrote {output}: d={d.tolist()} p={p.tolist()} "
          f"median rel err prefill={p_err:.3f} decode={d_err:.3f}")


def refit_from_log(args) -> None:
    pre, dec = [], []
    with open(args.from_log) as f:
        for line in f:
            m = re.match(r"prefill E=\s*(\d+) P=\s*(\d+)\s+([\d.]+) ms", line)
            if m:
                e, prefix, ms = int(m[1]), int(m[2]), float(m[3])
                x = _chunk_features(extend=e, prefix=prefix, chunk=args.chunk)
                pre.append((x, ms / 1e3))
                continue
            m = re.match(r"decode B=\s*(\d+) ctx=\s*(\d+)\s+([\d.]+) ms/step", line)
            if m:
                b, ctx, ms = int(m[1]), int(m[2]), float(m[3])
                n = args.decode_tokens
                dec.append(((1.0, float(b), b * (ctx + n / 2) / 1e6), ms / 1e3))
    _fit_and_save(pre=pre, dec=dec, output=args.output,
                  meta={"refit_from": os.path.abspath(args.from_log),
                        "model_tag": args.model_tag, "chunk": args.chunk})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url")
    p.add_argument("--from-log", help="refit from a previous run's printed points")
    p.add_argument("--chunk", type=int, default=32768, help="chunked_prefill_size, --from-log only")
    p.add_argument("--output", required=True)
    p.add_argument("--model-tag", default="")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--max-context", type=int, default=262144)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.from_log:
        refit_from_log(args)
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
