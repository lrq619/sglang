"""Payload of GET /scheduler_snapshot: one scheduler's queue state for an
external queueing-latency predictor.

All lengths are tokens. Every time is an age in seconds before
`snapshot_monotonic`, converted at build time from the source field's own
clock, so a reader on another machine never mixes clocks.
"""

from __future__ import annotations

from typing import List, Optional

import msgspec


class SnapshotConfig(msgspec.Struct, kw_only=True, frozen=True):
    max_total_num_tokens: int
    # Effective value, after the mamba-slot cap.
    max_running_requests: int
    # None when chunked prefill is disabled.
    chunked_prefill_size: Optional[int]
    page_size: int
    schedule_policy: str
    enable_mixed_chunk: bool
    enable_overlap: bool
    # SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION, the cap PrefillAdder applies to
    # max_new_tokens when reserving decode headroom.
    clip_max_new_tokens: int


class NewTokenRatioSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    current: float
    init: float
    min: float
    decay: float


class MemorySnapshot(msgspec.Struct, kw_only=True, frozen=True):
    kv_available: int
    kv_evictable: int
    kv_used: int
    # None unless the model has a mamba state pool.
    mamba_available: Optional[int] = None
    mamba_evictable: Optional[int] = None
    mamba_used: Optional[int] = None


class StepSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    """The last batch launched to the GPU."""

    forward_iter: int
    # "extend" | "decode" | "mixed" | "idle" | another ForwardMode name.
    mode: str
    launch_age_s: Optional[float]
    # False under the overlap loop: the batch may still occupy the GPU.
    result_processed: bool
    num_reqs: int
    # Per request, extend/mixed batches only.
    extend_tokens: List[int] = []
    prefix_tokens: List[int] = []


class RunningReqSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    """A request holding KV that decodes once the step in flight ends."""

    rid: str
    input_len: int
    output_len: int
    max_new_tokens: int
    cached_tokens: int
    # True if its (last) prefill chunk is the step in flight, so it is not in
    # the running batch yet.
    prefilling: bool
    forward_entry_age_s: Optional[float]
    # Prefix tokens another request also locks; they stay allocated when this one
    # finishes. None if the cache cannot tell.
    shared_prefix_tokens: Optional[int] = None


class ChunkedReqSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    """The request mid chunked prefill, with chunks left after the step in flight."""

    rid: str
    input_len: int
    # Prompt tokens written to KV once the step in flight ends.
    prefilled_len: int
    max_new_tokens: int
    cached_tokens: int


class WaitingReqSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    rid: str
    input_len: int
    output_len: int
    max_new_tokens: int
    # Device prefix a match would reuse at snapshot time (read-only peek, so it
    # can shrink by admission if evicted meanwhile); None if the cache cannot peek.
    prefix_len: Optional[int]
    wait_queue_entry_age_s: Optional[float]
    is_retracted: bool
    # Match boundaries behind prefix_len, shallowest first, as [prefix length,
    # device tokens evicted before this boundary is lost]; the rank is None when it
    # cannot be evicted. Lets a reader shrink the prefix as eviction advances.
    prefix_path: Optional[List[List[Optional[int]]]] = None


class FinishedReqRecord(msgspec.Struct, kw_only=True, frozen=True):
    seq: int
    rid: str
    input_len: int
    output_len: int
    cached_tokens: int
    # OpenAI-style type: "stop" | "length" | "abort" | ...
    finish_reason: str
    finish_age_s: float


class SchedulerSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    dp_rank: int
    snapshot_unix: float
    snapshot_monotonic: float
    forward_ct: int
    config: SnapshotConfig
    new_token_ratio: NewTokenRatioSnapshot
    memory: MemorySnapshot
    step: Optional[StepSnapshot]
    running: List[RunningReqSnapshot]
    chunked: Optional[ChunkedReqSnapshot]
    num_waiting: int
    # Head of the waiting queue in scheduling order, capped at max_waiting.
    waiting: List[WaitingReqSnapshot]
    num_retracted: int
    # Largest seq ever assigned; -1 before the first finish.
    finished_seq_head: int
    # Smallest seq still in the ring; a client whose cursor is below it lost entries.
    finished_seq_tail: int
    # Entries with seq > finished_since, oldest first; empty when it is None.
    finished: List[FinishedReqRecord] = []
    # Scheduler-side time to build this snapshot.
    build_ms: float = 0.0
