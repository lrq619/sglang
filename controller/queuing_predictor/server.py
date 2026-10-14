"""HTTP API of the queueing-latency predictor sidecar.

Usage:
    python -m controller.queuing_predictor.server \
        --instances http://127.0.0.1:30000,http://127.0.0.1:30001 \
        --costs controller/queuing_predictor/costs/qwen3.8-27b_h100.json --port 31000

POST /predict
    {"instances": [...],          # optional; default every --instances entry
     "input_len": 20311,          # or "input_ids": [...]
     "prefix_len": 19800,         # optional cached prefix of the target; default
                                  # /match_prefix on input_ids, else the instance's
                                  # recent cache-hit ratio * input_len
     "t_send_unix": 1791966978.5, # optional: predict from the last poll before it
     "max_new_tokens": 4096,
     "samples": 8}
    -> {"ready": true, "pool_size": ..., "predictions": [{"instance", "dp_rank",
        "queue_mean_s", "queue_p50_s", "queue_p90_s", "prefix_tokens", "prefill_s",
        "total_mean_s", "snapshot_age_s", ...}]}
    Until the pool holds --min-pool finished turns: {"ready": false, ...}, no predictions.
GET /health    pool size, per-instance poll status, unsupported configs
GET /pool      pool quantiles

--dump-snapshots PATH appends every polled snapshot as a gzip JSON line, and
--log-predicts PATH every /predict input with the snapshot it used, so
replay.py can re-predict offline (e.g. with true decode lengths).

--shadow-log PATH appends one CSV row per request the first time a poll sees it
in a waiting queue: the prediction for that request with only the entries ahead
of it, so predicted queue time = wait_age_s + pred_mean_s can be joined by rid
against the measured queue time without touching the client.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import logging
import time
from typing import Dict, List, Optional, Set

import aiohttp
import msgspec
import numpy as np
from aiohttp import web

from controller.queuing_predictor.client import InstanceClient
from controller.queuing_predictor.costs import StepCosts
from controller.queuing_predictor.pool import DecodePool
from controller.queuing_predictor.predictor import predict
from controller.queuing_predictor.simulator import UnsupportedConfig, check_supported

logger = logging.getLogger(__name__)

SHADOW_COLUMNS = [
    "t_unix", "instance", "dp_rank", "rid", "position", "wait_age_s",
    "snapshot_age_s", "pred_mean_s", "pred_p50_s", "pred_p90_s", "prefill_s",
    "input_len", "prefix_tokens", "admit_prefix_tokens", "prefix_known", "num_running", "step_mode",
    "kv_free", "pool_size", "predict_ms",
]


class PredictorService:
    def __init__(self, *, args: argparse.Namespace, costs: StepCosts):
        self.args = args
        self.costs = costs
        self.pool = DecodePool(cap=args.pool_cap, min_size=args.min_pool)
        self.rng = np.random.default_rng(args.seed)
        self.clients: Dict[str, InstanceClient] = {}
        self._http: aiohttp.ClientSession = None
        self._poller: asyncio.Task = None
        self._shadow_seen: Set[str] = set()
        self._shadow_writer: Optional[csv.DictWriter] = None
        self._shadow_file = None
        self._dump_file = None
        self._predict_log = None

    async def start(self, app: web.Application) -> None:
        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.args.http_timeout)
        )
        for url in self.args.instances.split(","):
            self._client(url)
        if self.args.shadow_log:
            self._shadow_file = open(self.args.shadow_log, "a", newline="")
            self._shadow_writer = csv.DictWriter(self._shadow_file, SHADOW_COLUMNS)
            if self._shadow_file.tell() == 0:
                self._shadow_writer.writeheader()
        if self.args.dump_snapshots:
            self._dump_file = gzip.open(self.args.dump_snapshots, "at")
        if self.args.log_predicts:
            self._predict_log = open(self.args.log_predicts, "a")
        self._poller = asyncio.create_task(self._poll_loop())

    async def stop(self, app: web.Application) -> None:
        self._poller.cancel()
        await self._http.close()
        for f in (self._shadow_file, self._dump_file, self._predict_log):
            if f is not None:
                f.close()

    def _client(self, url: str) -> InstanceClient:
        url = url.rstrip("/")
        if url not in self.clients:
            self.clients[url] = InstanceClient(
                url=url,
                http=self._http,
                on_finished=lambda rec: self.pool.add(rec["output_len"]),
                max_waiting=self.args.max_waiting,
            )
        return self.clients[url]

    async def _poll_loop(self) -> None:
        while True:
            results = await asyncio.gather(
                *(c.poll() for c in self.clients.values()), return_exceptions=True
            )
            for url, r in zip(self.clients, results):
                if isinstance(r, Exception):
                    logger.warning("poll %s failed: %r", url, r)
                    continue
                if self._dump_file is not None:
                    self._dump_file.write(
                        json.dumps({"url": url, "received_unix": time.time(), "snaps": r[1]})
                        + "\n"
                    )
                if self._shadow_writer is not None and self.pool.ready:
                    try:
                        self._shadow_predict(url=url, got=r)
                    except Exception:
                        logger.exception("shadow prediction on %s failed", url)
            if self._dump_file is not None:
                self._dump_file.flush()
            await asyncio.sleep(self.args.poll_interval)

    async def handle_predict(self, request: web.Request) -> web.Response:
        t0 = time.perf_counter()
        body = await request.json()
        if "input_len" in body:
            input_len = int(body["input_len"])
        else:
            input_len = len(body["input_ids"])
        urls = body.get("instances") or list(self.clients)
        if not self.pool.ready:
            return web.json_response(
                {
                    "ready": False,
                    "pool_size": len(self.pool),
                    "min_pool": self.pool.min_size,
                    "predictions": [],
                }
            )
        t_send = body.get("t_send_unix")
        results = await asyncio.gather(
            *(self._predict_url(url=u, body=body, input_len=input_len, t_send=t_send)
              for u in urls),
            return_exceptions=True,
        )
        predictions = []
        for url, got in zip(urls, results):
            if isinstance(got, Exception):
                predictions.append({"instance": url, "error": repr(got)})
            else:
                predictions += got
        return web.json_response(
            {
                "ready": True,
                "pool_size": len(self.pool),
                "predictions": predictions,
                "latency_ms": (time.perf_counter() - t0) * 1e3,
            }
        )

    async def _predict_url(
        self, *, url: str, body: dict, input_len: int, t_send: Optional[float]
    ) -> List[dict]:
        """With t_send_unix, predict from the last poll before the send, so a request
        sent concurrently with this call never queues behind itself."""
        client = self._client(url)
        if t_send is None:
            received, snaps = await client.snapshot()
            ages = {s["dp_rank"]: time.monotonic() - received for s in snaps}
        else:
            got = client.snapshot_before(t_send)
            if got is None:
                raise LookupError(f"no snapshot polled before t_send={t_send}")
            snaps = got[1]
            ages = {s["dp_rank"]: t_send - s["snapshot_unix"] for s in snaps}
        matched: Dict[int, dict] = {}
        if body.get("prefix_len") is None and "input_ids" in body:
            matched = await client.match_prefix(body["input_ids"])
        out = []
        for snap in snaps:
            rank = snap["dp_rank"]
            ratio = client.prefix_ratio(rank)
            prefix_len = body.get("prefix_len")
            prefix_path = None
            if prefix_len is None and rank in matched:
                prefix_len = matched[rank]["matched_tokens"]
                prefix_path = matched[rank].get("prefix_path")
            prefix_source = "match" if prefix_len is not None else "ratio"
            if prefix_len is None:
                prefix_len = int(input_len * ratio)
            try:
                pred = predict(
                    snap=snap,
                    snapshot_age_s=max(0.0, ages[rank]),
                    pool=self.pool,
                    costs=self.costs,
                    input_len=input_len,
                    prefix_len=int(prefix_len),
                    max_new_tokens=int(
                        body.get("max_new_tokens", self.args.default_max_new_tokens)
                    ),
                    prefix_ratio=ratio,
                    samples=int(body.get("samples", self.args.samples)),
                    rng=self.rng,
                    cap_by_max_new=self.args.cap_by_max_new,
                    prefix_path=prefix_path,
                )
            except UnsupportedConfig as e:
                out.append({"instance": url, "dp_rank": rank, "error": str(e)})
                continue
            row = {
                "instance": url,
                "dp_rank": rank,
                **msgspec.to_builtins(pred),
                "prefix_source": prefix_source,
                "snapshot_age_s": ages[rank],
                "pool_size": len(self.pool),
            }
            out.append(row)
            if self._predict_log is not None:
                record = {
                    "rid": body.get("rid"),
                    "t_send_unix": t_send,
                    "snapshot_unix": snap["snapshot_unix"],
                    "input_len": input_len,
                    "max_new_tokens": body.get("max_new_tokens"),
                    "prefix_len": int(prefix_len),
                    "prefix_path": prefix_path,
                    "prefix_ratio": ratio,
                    **row,
                }
                self._predict_log.write(json.dumps(record) + "\n")
                self._predict_log.flush()
        return out

    def _shadow_predict(self, *, url: str, got) -> None:
        received, snaps = got
        client = self.clients[url]
        for snap in snaps:
            try:
                check_supported(snap["config"])
            except UnsupportedConfig:
                continue
            ratio = client.prefix_ratio(snap["dp_rank"])
            for i, w in enumerate(snap["waiting"]):
                if w["rid"] in self._shadow_seen:
                    continue
                self._shadow_seen.add(w["rid"])
                if w["is_retracted"]:
                    continue
                self._shadow_writer.writerow(
                    self._shadow_row(
                        url=url, snap=snap, index=i, ratio=ratio, received=received
                    )
                )
        self._shadow_file.flush()

    def _shadow_row(
        self, *, url: str, snap: dict, index: int, ratio: float, received: float
    ) -> dict:
        w = snap["waiting"][index]
        prefix_known = w["prefix_len"] is not None
        prefix = w["prefix_len"] if prefix_known else int(w["input_len"] * ratio)
        ahead = dict(snap, waiting=snap["waiting"][:index], num_waiting=index)
        t0 = time.perf_counter()
        pred = predict(
            snap=ahead,
            snapshot_age_s=0.0,
            pool=self.pool,
            costs=self.costs,
            input_len=w["input_len"],
            prefix_len=prefix,
            max_new_tokens=w["max_new_tokens"],
            prefix_ratio=ratio,
            samples=self.args.samples,
            rng=self.rng,
            cap_by_max_new=self.args.cap_by_max_new,
            prefix_path=w.get("prefix_path"),
        )
        return {
            "t_unix": time.time(),
            "instance": url,
            "dp_rank": snap["dp_rank"],
            "rid": w["rid"],
            "position": index,
            "wait_age_s": w["wait_queue_entry_age_s"],
            "snapshot_age_s": time.monotonic() - received,
            "pred_mean_s": pred.queue_mean_s,
            "pred_p50_s": pred.queue_p50_s,
            "pred_p90_s": pred.queue_p90_s,
            "prefill_s": pred.prefill_s,
            "input_len": w["input_len"],
            "prefix_tokens": pred.prefix_tokens,
            "admit_prefix_tokens": pred.admit_prefix_tokens,
            "prefix_known": int(prefix_known),
            "num_running": pred.num_running,
            "step_mode": snap["step"]["mode"] if snap["step"] else "",
            "kv_free": snap["memory"]["kv_available"] + snap["memory"]["kv_evictable"],
            "pool_size": len(self.pool),
            "predict_ms": (time.perf_counter() - t0) * 1e3,
        }

    async def handle_health(self, request: web.Request) -> web.Response:
        instances = {}
        for url, c in self.clients.items():
            unsupported = None
            for snap in c.last_snapshots:
                try:
                    check_supported(snap["config"])
                except UnsupportedConfig as e:
                    unsupported = str(e)
            instances[url] = {
                "last_snapshot_unix": c.last_snapshot_unix,
                "last_error": c.last_error,
                "finished_seen": c.num_finished_seen,
                "finished_lost": c.num_lost,
                "unsupported_config": unsupported,
            }
        return web.json_response(
            {
                "ready": self.pool.ready,
                "pool_size": len(self.pool),
                "pool_total_added": self.pool.total_added,
                "min_pool": self.pool.min_size,
                "instances": instances,
            }
        )

    async def handle_pool(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"pool_size": len(self.pool), "quantiles": self.pool.quantiles()}
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--instances", required=True, help="comma-separated SGLang URLs")
    p.add_argument("--costs", required=True, help="StepCosts JSON (profile_costs.py)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=31000)
    p.add_argument("--min-pool", type=int, default=500)
    p.add_argument("--pool-cap", type=int, default=20000)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--max-waiting", type=int, default=4096)
    p.add_argument("--default-max-new-tokens", type=int, default=4096)
    p.add_argument("--http-timeout", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shadow-log", help="CSV of first-sight predictions per queued rid")
    p.add_argument("--dump-snapshots", help="gzip JSON lines of every polled snapshot")
    p.add_argument("--log-predicts", help="JSON lines of every /predict input and result")
    p.add_argument(
        "--cap-by-max-new",
        action="store_true",
        help="clip drawn decode lengths at max_new_tokens (an oracle under loadgen)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    service = PredictorService(args=args, costs=StepCosts.load(args.costs))
    app = web.Application()
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    app.router.add_post("/predict", service.handle_predict)
    app.router.add_get("/health", service.handle_health)
    app.router.add_get("/pool", service.handle_pool)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
