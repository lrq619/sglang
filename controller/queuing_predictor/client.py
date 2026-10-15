"""Async client for one SGLang instance's GET /scheduler_snapshot."""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# Arbitrary; the snapshot is rebuilt between scheduler steps, ~tens of ms apart.
SNAPSHOT_TTL_S = 0.05
# Arbitrary; long enough to smooth a few sessions, short enough to track drift.
PREFIX_WINDOW = 500


class InstanceClient:
    """Snapshot cache, finished-ring cursor and cache-hit estimate of one instance."""

    def __init__(
        self,
        *,
        url: str,
        http: aiohttp.ClientSession,
        on_finished: Callable[[dict], None],
        max_waiting: int,
    ):
        self.url = url.rstrip("/")
        self._http = http
        self._on_finished = on_finished
        self._max_waiting = max_waiting
        # Per DP rank: last seq handed to on_finished.
        self._cursor: Dict[int, int] = {}
        self._cache: Optional[Tuple[float, List[dict]]] = None
        # Recent polls, so a prediction can use the last state before its send time.
        self._history: Deque[Tuple[float, List[dict]]] = collections.deque(maxlen=8)
        self._lock = asyncio.Lock()
        # (input_len, cached_tokens) of recent finishes, for the prefix estimate.
        self._recent: Dict[int, Deque[Tuple[int, int]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=PREFIX_WINDOW)
        )
        self.last_snapshot_unix: Optional[float] = None
        self.num_finished_seen = 0
        self.num_lost = 0
        self.last_error: Optional[str] = None

    async def snapshot(self) -> Tuple[float, List[dict]]:
        """(local monotonic receive time, one snapshot per DP rank), TTL-cached."""
        async with self._lock:
            if (
                self._cache is not None
                and time.monotonic() - self._cache[0] < SNAPSHOT_TTL_S
            ):
                return self._cache
            return await self._fetch()

    async def poll(self) -> Tuple[float, List[dict]]:
        """Refresh the snapshot and feed new finished requests to the pool."""
        async with self._lock:
            return await self._fetch()

    def snapshot_before(self, t_unix: float) -> Optional[Tuple[float, List[dict]]]:
        """Newest polled snapshot taken no later than t_unix (server clock)."""
        for got in reversed(self._history):
            if max(s["snapshot_unix"] for s in got[1]) <= t_unix:
                return got
        return None

    def last_running_entry(self, rid: str) -> Optional[dict]:
        """The newest polled running entry of rid, if any recent poll saw it."""
        for _, snaps in reversed(self._history):
            for snap in snaps:
                for r in snap["running"]:
                    if r["rid"] == rid:
                        return r
        return None

    async def unpin_prefix(self, rid: str) -> str:
        async with self._http.post(
            f"{self.url}/unpin_prefix", json={"target_rid": rid}
        ) as resp:
            return (await resp.json())["status"]

    async def match_prefix(self, input_ids: List[int]) -> Dict[int, dict]:
        """Read-only {matched_tokens, prefix_path} of input_ids per DP rank."""
        async with self._http.post(
            f"{self.url}/match_prefix", json={"input_ids": input_ids}
        ) as resp:
            resp.raise_for_status()
            return {r["dp_rank"]: r for r in await resp.json()}

    @property
    def last_snapshots(self) -> List[dict]:
        return self._cache[1] if self._cache is not None else []

    @property
    def last_received(self) -> float:
        """Local monotonic time the last snapshot arrived."""
        return self._cache[0] if self._cache is not None else float("-inf")

    def prefix_ratio(self, dp_rank: int) -> float:
        recent = self._recent.get(dp_rank)
        if not recent:
            return 0.0
        total = sum(i for i, _ in recent)
        return sum(c for _, c in recent) / total if total else 0.0

    async def _fetch(self) -> Tuple[float, List[dict]]:
        # A cursor of -1 asks for the whole ring on first contact; afterwards the
        # smallest per-rank cursor so no rank misses entries.
        since = min(self._cursor.values()) if self._cursor else -1
        params = {"finished_since": str(since), "max_waiting": str(self._max_waiting)}
        try:
            async with self._http.get(
                f"{self.url}/scheduler_snapshot", params=params
            ) as resp:
                resp.raise_for_status()
                snaps = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.last_error = repr(e)
            raise
        received = time.monotonic()
        self.last_error = None
        self.last_snapshot_unix = time.time()
        for snap in snaps:
            self._ingest_finished(snap)
        self._cache = (received, snaps)
        self._history.append(self._cache)
        return self._cache

    def _ingest_finished(self, snap: dict) -> None:
        rank = snap["dp_rank"]
        cursor = self._cursor.get(rank, -1)
        if snap["finished_seq_head"] < cursor:
            logger.warning("%s dp%d: finished ring reset; server restarted?", self.url, rank)
            cursor = -1
        if snap["finished_seq_tail"] > cursor + 1 and cursor >= 0:
            self.num_lost += snap["finished_seq_tail"] - cursor - 1
        for rec in snap["finished"]:
            if rec["seq"] <= cursor:
                continue
            cursor = rec["seq"]
            self.num_finished_seen += 1
            if rec["finish_reason"] == "abort":
                continue
            self._recent[rank].append((rec["input_len"], rec["cached_tokens"]))
            self._on_finished(rec)
        self._cursor[rank] = max(cursor, snap["finished_seq_head"])
