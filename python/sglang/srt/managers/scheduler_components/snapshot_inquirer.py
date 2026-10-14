"""Builds the GET /scheduler_snapshot payload and keeps the ring of finished
requests that an external queueing-latency predictor learns decode lengths from.

Runs between scheduler loop iterations (from process_input_requests), never
during a forward; under the overlap loop the last launched batch may still be
on the GPU, which StepSnapshot.result_processed reports.
"""

from __future__ import annotations

import collections
import time
from array import array
from typing import TYPE_CHECKING, Any, Callable, Deque, List, Optional

import msgspec

from sglang.srt.managers.io_struct import (
    GetSchedulerSnapshotReq,
    GetSchedulerSnapshotReqOutput,
    MatchPrefixReqInput,
    MatchPrefixReqOutput,
)
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS
from sglang.srt.managers.scheduler_snapshot import (
    ChunkedReqSnapshot,
    FinishedReqRecord,
    MemorySnapshot,
    NewTokenRatioSnapshot,
    RunningReqSnapshot,
    SchedulerSnapshot,
    SnapshotConfig,
    StepSnapshot,
    WaitingReqSnapshot,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
        NewTokenRatioTracker,
    )
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )

# Arbitrary; at ~10 finishes/s this holds ~2 h, far longer than a 1 s poller lags.
FINISHED_RING_CAPACITY = 65536


class FinishedEntry(msgspec.Struct, kw_only=True, frozen=True):
    seq: int
    rid: str
    input_len: int
    output_len: int
    cached_tokens: int
    finish_reason: str
    finish_monotonic: float


class FinishedReqRing:
    """Bounded ring of finished requests with a monotonically increasing seq."""

    def __init__(self, *, capacity: int = FINISHED_RING_CAPACITY):
        self._records: Deque[FinishedEntry] = collections.deque(maxlen=capacity)
        self._next_seq = 0

    @property
    def seq_head(self) -> int:
        return self._next_seq - 1

    @property
    def seq_tail(self) -> int:
        return self._records[0].seq if self._records else self._next_seq

    def append(
        self,
        *,
        rid: str,
        input_len: int,
        output_len: int,
        cached_tokens: int,
        finish_reason: str,
        finish_monotonic: float,
    ) -> None:
        self._records.append(
            FinishedEntry(
                seq=self._next_seq,
                rid=rid,
                input_len=input_len,
                output_len=output_len,
                cached_tokens=cached_tokens,
                finish_reason=finish_reason,
                finish_monotonic=finish_monotonic,
            )
        )
        self._next_seq += 1

    def since(self, seq: int) -> List[FinishedEntry]:
        """Records with seq > `seq`, oldest first."""
        if not self._records or seq >= self.seq_head:
            return []
        start = max(0, seq + 1 - self.seq_tail)
        return [self._records[i] for i in range(start, len(self._records))]


class SchedulerSnapshotInquirer:
    def __init__(
        self,
        *,
        dp_rank: Optional[int],
        max_total_num_tokens: int,
        max_running_requests: int,
        chunked_prefill_size: Optional[int],
        page_size: int,
        schedule_policy: str,
        enable_mixed_chunk: bool,
        enable_overlap: bool,
        pool_stats_observer: SchedulerPoolStatsObserver,
        new_token_ratio_tracker: NewTokenRatioTracker,
        get_running_batch: Callable[[], ScheduleBatch],
        get_waiting_queue: Callable[[], List[Req]],
        get_chunked_req: Callable[[], Optional[Req]],
        get_last_batch: Callable[[], Optional[ScheduleBatch]],
        get_forward_ct: Callable[[], int],
        get_eviction_ranks: Callable[[], Optional[dict]],
        get_shared_locked_tokens: Callable[[Any], Optional[int]],
        peek_prefix_path_for_req: Callable[[Req, Optional[dict]], Optional[list]],
        peek_prefix_path_for_tokens: Callable[[array, Optional[dict]], Optional[list]],
        clip_max_new_tokens: int = CLIP_MAX_NEW_TOKENS,
    ):
        # None without DP.
        self.dp_rank = dp_rank if dp_rank is not None else 0
        self.max_total_num_tokens = max_total_num_tokens
        self.max_running_requests = max_running_requests
        self.chunked_prefill_size = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size > 0
            else None
        )
        self.page_size = page_size
        self.schedule_policy = schedule_policy
        self.enable_mixed_chunk = enable_mixed_chunk
        self.enable_overlap = enable_overlap
        self.clip_max_new_tokens = clip_max_new_tokens
        self.pool_stats_observer = pool_stats_observer
        self.new_token_ratio_tracker = new_token_ratio_tracker
        self.get_running_batch = get_running_batch
        self.get_waiting_queue = get_waiting_queue
        self.get_chunked_req = get_chunked_req
        self.get_last_batch = get_last_batch
        self.get_forward_ct = get_forward_ct
        self.get_eviction_ranks = get_eviction_ranks
        self.get_shared_locked_tokens = get_shared_locked_tokens
        self.peek_prefix_path_for_req = peek_prefix_path_for_req
        self.peek_prefix_path_for_tokens = peek_prefix_path_for_tokens
        self.finished_ring = FinishedReqRing()

    def handle(self, req: GetSchedulerSnapshotReq) -> GetSchedulerSnapshotReqOutput:
        return GetSchedulerSnapshotReqOutput(
            snapshot=self.get_snapshot(
                finished_since=req.finished_since, max_waiting=req.max_waiting
            )
        )

    def handle_match_prefix(self, req: MatchPrefixReqInput) -> MatchPrefixReqOutput:
        path = self.peek_prefix_path_for_tokens(
            array("q", req.input_ids), self.get_eviction_ranks()
        )
        return MatchPrefixReqOutput(
            dp_rank=self.dp_rank,
            matched_tokens=_path_len(path),
            prefix_path=path,
        )

    def record_finished(self, req: Req) -> None:
        """Called once per request, when its final output is streamed."""
        output_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        self.finished_ring.append(
            rid=req.rid,
            input_len=len(req.origin_input_ids),
            output_len=output_len,
            cached_tokens=req.cached_tokens,
            finish_reason=req.finished_reason.to_json()["type"],
            finish_monotonic=time.monotonic(),
        )

    def get_snapshot(
        self, *, finished_since: Optional[int], max_waiting: int
    ) -> SchedulerSnapshot:
        now_mono = time.monotonic()
        now_perf = time.perf_counter()
        waiting_queue = self.get_waiting_queue()
        head = waiting_queue[: max(0, max_waiting)]
        # One tree walk per snapshot, skipped when nothing waits.
        ranks = self.get_eviction_ranks() if head else None
        waiting = [
            _waiting(
                req,
                prefix_path=self.peek_prefix_path_for_req(req, ranks),
                now_perf=now_perf,
            )
            for req in head
        ]
        chunked_req = self.get_chunked_req()
        last_batch = self.get_last_batch()
        return SchedulerSnapshot(
            dp_rank=self.dp_rank,
            snapshot_unix=time.time(),
            snapshot_monotonic=now_mono,
            forward_ct=self.get_forward_ct(),
            config=self._config(),
            new_token_ratio=self._new_token_ratio(),
            memory=self._memory(),
            step=self._step(batch=last_batch, now_mono=now_mono),
            running=self._running(
                last_batch=last_batch, chunked_req=chunked_req, now_perf=now_perf
            ),
            chunked=_chunked(chunked_req),
            num_waiting=len(waiting_queue),
            waiting=waiting,
            num_retracted=sum(1 for req in waiting_queue if req.is_retracted),
            finished_seq_head=self.finished_ring.seq_head,
            finished_seq_tail=self.finished_ring.seq_tail,
            finished=self._finished(finished_since=finished_since, now_mono=now_mono),
            build_ms=(time.perf_counter() - now_perf) * 1e3,
        )

    def _config(self) -> SnapshotConfig:
        return SnapshotConfig(
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            chunked_prefill_size=self.chunked_prefill_size,
            page_size=self.page_size,
            schedule_policy=self.schedule_policy,
            enable_mixed_chunk=self.enable_mixed_chunk,
            enable_overlap=self.enable_overlap,
            clip_max_new_tokens=self.clip_max_new_tokens,
        )

    def _new_token_ratio(self) -> NewTokenRatioSnapshot:
        tracker = self.new_token_ratio_tracker
        return NewTokenRatioSnapshot(
            current=tracker.current,
            init=tracker.init,
            min=tracker.min,
            decay=tracker.decay,
        )

    def _memory(self) -> MemorySnapshot:
        stats = self.pool_stats_observer.get_pool_stats()
        return MemorySnapshot(
            kv_available=int(stats.full_available_size),
            kv_evictable=int(stats.full_evictable_size),
            kv_used=int(stats.full_num_used),
            mamba_available=stats.mamba_available_size,
            mamba_evictable=stats.mamba_evictable_size,
            mamba_used=stats.mamba_num_used,
        )

    def _step(
        self, *, batch: Optional[ScheduleBatch], now_mono: float
    ) -> Optional[StepSnapshot]:
        if batch is None or batch.forward_mode is None:
            return None
        is_extend = batch.forward_mode.is_extend()
        return StepSnapshot(
            forward_iter=batch.forward_iter,
            mode=batch.forward_mode.name.lower(),
            launch_age_s=(
                now_mono - batch.launch_ts if batch.launch_ts is not None else None
            ),
            result_processed=not self.enable_overlap,
            num_reqs=len(batch.reqs),
            extend_tokens=list(batch.extend_lens) if is_extend else [],
            prefix_tokens=list(batch.prefix_lens) if is_extend else [],
        )

    def _running(
        self,
        *,
        last_batch: Optional[ScheduleBatch],
        chunked_req: Optional[Req],
        now_perf: float,
    ) -> List[RunningReqSnapshot]:
        # An extend batch's requests join the running batch only at the next
        # get_next_batch_to_run, so they are listed here as prefilling.
        out = [
            self._running_req(req, prefilling=False, now_perf=now_perf)
            for req in self.get_running_batch().reqs
            if not req.finished()
        ]
        if last_batch is not None and last_batch.forward_mode.is_extend():
            seen = {id(req) for req in self.get_running_batch().reqs}
            out += [
                self._running_req(req, prefilling=True, now_perf=now_perf)
                for req in last_batch.reqs
                if id(req) not in seen and req is not chunked_req and not req.finished()
            ]
        return out

    def _running_req(
        self, req: Req, *, prefilling: bool, now_perf: float
    ) -> RunningReqSnapshot:
        return RunningReqSnapshot(
            rid=req.rid,
            input_len=len(req.origin_input_ids),
            output_len=len(req.output_ids),
            max_new_tokens=req.sampling_params.max_new_tokens,
            cached_tokens=req.cached_tokens,
            prefilling=prefilling,
            forward_entry_age_s=_age(req.time_stats.forward_entry_time, now_perf),
            shared_prefix_tokens=(
                self.get_shared_locked_tokens(req.last_node)
                if req.last_node is not None
                else 0
            ),
        )

    def _finished(
        self, *, finished_since: Optional[int], now_mono: float
    ) -> List[FinishedReqRecord]:
        if finished_since is None:
            return []
        return [
            FinishedReqRecord(
                seq=r.seq,
                rid=r.rid,
                input_len=r.input_len,
                output_len=r.output_len,
                cached_tokens=r.cached_tokens,
                finish_reason=r.finish_reason,
                finish_age_s=now_mono - r.finish_monotonic,
            )
            for r in self.finished_ring.since(finished_since)
        ]


def _age(ts: float, now: float) -> Optional[float]:
    return now - ts if ts > 0.0 else None


def _chunked(req: Optional[Req]) -> Optional[ChunkedReqSnapshot]:
    if req is None or req.extend_range is None:
        return None
    # extend_range.end already counts the chunk in flight.
    return ChunkedReqSnapshot(
        rid=req.rid,
        input_len=len(req.origin_input_ids),
        prefilled_len=req.extend_range.end,
        max_new_tokens=req.sampling_params.max_new_tokens,
        cached_tokens=req.cached_tokens,
    )


def _path_len(path: Optional[list]) -> Optional[int]:
    if path is None:
        return None
    return path[-1][0] if path else 0


def _waiting(
    req: Req, *, prefix_path: Optional[list], now_perf: float
) -> WaitingReqSnapshot:
    return WaitingReqSnapshot(
        rid=req.rid,
        input_len=len(req.origin_input_ids),
        output_len=len(req.output_ids),
        max_new_tokens=req.sampling_params.max_new_tokens,
        prefix_len=_path_len(prefix_path),
        prefix_path=prefix_path,
        wait_queue_entry_age_s=_age(req.time_stats.wait_queue_entry_time, now_perf),
        is_retracted=req.is_retracted,
    )
