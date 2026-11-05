"""Measure single-request prefill throughput of a running SGLang server vs context length.

One request at a time (batch 1), max_new_tokens=1, fresh random input_ids per request
so nothing hits the prefix cache. Prefill time is server-side,
prefill_finished_time - forward_entry_time from meta_info (needs --enable-metrics),
so HTTP and queueing are excluded. Writes one CSV row per request.

Usage:
    python bench_prefill.py --url http://127.0.0.1:30000 -o bench.csv
"""

import argparse
import csv
import json
import random
import statistics
import urllib.request

DEFAULT_LENGTHS = [2**k for k in range(4, 18)]  # 16 .. 131072
# Arbitrary id range inside the Qwen3.8 vocab that avoids special tokens.
TOKEN_LO, TOKEN_HI = 1000, 150000


def _generate(url: str, input_ids: list) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["meta_info"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    ids = lambda n: [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]

    for _ in range(args.warmup):
        _generate(args.url, ids(1024))
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "rep", "prompt_tokens", "cached_tokens", "prefill_s", "tok_per_s"])
        for n in args.lengths:
            tps = []
            for rep in range(args.reps):
                meta = _generate(args.url, ids(n))
                prefill_s = meta["prefill_finished_time"] - meta["forward_entry_time"]
                cached = meta.get("cached_tokens", 0)
                uncached = meta["prompt_tokens"] - cached
                tps.append(uncached / prefill_s)
                w.writerow([n, rep, meta["prompt_tokens"], cached, f"{prefill_s:.6f}", f"{tps[-1]:.1f}"])
                f.flush()
            print(f"len={n:7d} median prefill throughput {statistics.median(tps):10.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
