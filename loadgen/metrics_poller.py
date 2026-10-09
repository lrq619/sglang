"""Sample SGLang scheduler gauges from each endpoint's /metrics at a fixed rate.

One CSV row per endpoint per tick, flushed as written. t is a unix timestamp
(the time the scrape was sent), on the same clock as loadgen's t_send / t_end.
Runs until SIGINT / SIGTERM. The servers must run with --enable-metrics.

The gauges are refreshed by the scheduler on every prefill batch, every
--decode-log-interval decode steps (default 40) and when idle, so consecutive
samples can repeat a value.

Usage:
    python metrics_poller.py \
        --endpoints http://10.0.0.1:30000,http://10.0.0.2:30000 \
        --output metrics.csv --interval 1
"""

import argparse
import asyncio
import csv
import signal
import time

import aiohttp

GAUGES = {
    "num_queue_reqs": "sglang:num_queue_reqs",
    "num_running_reqs": "sglang:num_running_reqs",
    "token_usage": "sglang:token_usage",
    "full_token_usage": "sglang:full_token_usage",
    "mamba_usage": "sglang:mamba_usage",
    # Free slots per pool; the first time either hits ~0 marks the onset of
    # memory pressure (eviction or blocking), identical across cache policies.
    "kv_available_tokens": "sglang:kv_available_tokens",
    "mamba_available_tokens": "sglang:mamba_available_tokens",
}
CSV_COLUMNS = ["t", "endpoint", *GAUGES, "error"]


def _parse_gauges(text: str) -> dict:
    """Pick GAUGES out of Prometheus text; a gauge with several label sets is summed."""
    by_metric = {metric: name for name, metric in GAUGES.items()}
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(" ", 1)[0]
        name = by_metric.get(metric)
        if name is not None:
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return values


async def _scrape(*, http: aiohttp.ClientSession, endpoint: str) -> dict:
    t = time.time()
    row = {"t": f"{t:.3f}", "endpoint": endpoint, "error": ""}
    try:
        async with http.get(f"{endpoint}/metrics") as resp:
            resp.raise_for_status()
            row.update(_parse_gauges(await resp.text()))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


async def run(args):
    endpoints = [e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # A scrape slower than one interval is cut, so ticks never pile up.
    timeout = aiohttp.ClientTimeout(total=args.interval * 0.9)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        async with aiohttp.ClientSession(timeout=timeout) as http:
            next_tick = time.monotonic()
            while not stop.is_set():
                rows = await asyncio.gather(
                    *(_scrape(http=http, endpoint=ep) for ep in endpoints)
                )
                writer.writerows(rows)
                f.flush()
                # Absolute schedule: sampling does not drift by the scrape time.
                next_tick += args.interval
                delay = next_tick - time.monotonic()
                if delay < 0:
                    next_tick = time.monotonic()
                    continue
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoints", required=True, help="comma-separated SGLang base URLs"
    )
    parser.add_argument("--output", required=True, help="time-series CSV")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
