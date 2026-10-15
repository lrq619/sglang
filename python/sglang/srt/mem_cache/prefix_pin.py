"""Externally controlled prefix pins (``--enable-prefix-pin``).

A pin keeps one finished request's cached prefix -- the KV path from the root to
the node its tokens landed on, plus that node's mamba state -- locked in the
radix tree until it is unpinned, so an agent session survives a tool call
without being evicted. The server only supplies the mechanism; which requests to
pin, and for how long, is the caller's decision.

Pins are taken before the request finishes: ``/pin_prefix`` on a running rid, or
``pin_on_finish`` on the generate request. A finished request's node may already
be evicted, or keep its KV but lose its mamba state, so a pin that arrives after
the finish is refused rather than locking a node of unknown residency.

No pin can block progress forever:
  * a budget caps pinned KV tokens and pinned mamba states;
  * when the scheduler finds the queue head unadmittable with nothing running,
    or decode short of memory, or too few free mamba states for an admission,
    it force-releases pins in ``release_order`` (see ``release_until``);
  * ``release_pin_of`` on the next turn hands the prefix over: once that request
    is admitted and holds its own lock, the pin is dropped.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Optional

import msgspec

from sglang.srt.arg_groups.choices import PREFIX_PIN_RELEASE_ORDER_CHOICES
from sglang.srt.mem_cache.base_prefix_cache import CachedPrefix
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import (
        BasePrefixCache,
        DecLockRefParams,
    )

logger = logging.getLogger(__name__)

# Mamba states one admission may take: the extra_buffer reservation of 5 per
# request (kv_cache_configurator, overlap on) plus a match-time copy-on-write.
MAMBA_STATES_PER_ADMISSION = 6

# Arbitrary bounds on bookkeeping for rids the registry may never hear about again.
_MAX_PENDING = 4096
_MAX_REMEMBERED_RIDS = 16384


class PinStatus:
    PINNED = "pinned"
    PENDING = "pending"
    ALREADY_PINNED = "already_pinned"
    ALREADY_FINISHED = "already_finished"
    NOTHING_CACHED = "nothing_cached"
    OVER_BUDGET = "over_budget"
    UNPINNED = "unpinned"
    NOT_PINNED = "not_pinned"
    FORCE_RELEASED = "force_released"


# Why a pin request was turned down; each is a metrics label.
PIN_REFUSE_REASONS = (
    PinStatus.OVER_BUDGET,
    PinStatus.NOTHING_CACHED,
    PinStatus.ALREADY_FINISHED,
)


class ForceReleaseReason:
    """Why the scheduler dropped a pin to make progress; each is a metrics label."""

    # Queue head unadmittable while nothing runs to free memory.
    ADMISSION = "admission"
    # Decode short of KV memory; pins go before running requests are retracted.
    DECODE = "decode"
    # Too few free mamba states for an admission's slots and copy-on-write.
    MAMBA_HEADROOM = "mamba_headroom"


FORCE_RELEASE_REASONS = (
    ForceReleaseReason.ADMISSION,
    ForceReleaseReason.DECODE,
    ForceReleaseReason.MAMBA_HEADROOM,
)


class PinEntry(msgspec.Struct, frozen=True):
    rid: str
    node_id: int
    dec_params: DecLockRefParams
    num_tokens: int
    pinned_at: float
    # Absolute monotonic time the caller expects the next turn; None if unknown.
    expected_return_at: Optional[float]


class PinResult(msgspec.Struct, frozen=True):
    status: str
    num_tokens: int = 0


class PrefixPinRegistry:
    """Pin bookkeeping and tree locks for one scheduler's radix cache."""

    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        kv_capacity: int,
        mamba_capacity: Optional[int],
        mamba_free_states: Optional[Callable[[], int]],
        max_kv_fraction: float,
        max_mamba_fraction: float,
        release_order: str,
        clock: Callable[[], float] = time.monotonic,
    ):
        assert release_order in PREFIX_PIN_RELEASE_ORDER_CHOICES, release_order
        self._tree_cache = tree_cache
        self._max_kv_tokens = max_kv_fraction * kv_capacity
        self._max_mamba_states = (
            max_mamba_fraction * mamba_capacity if mamba_capacity is not None else None
        )
        self._mamba_capacity = mamba_capacity
        self._mamba_free_states = mamba_free_states
        self._release_order = release_order
        self._clock = clock

        self._pins: dict[str, PinEntry] = {}
        # rid -> expected_return_s, for pins requested before the request finished.
        self._pending: OrderedDict[str, Optional[float]] = OrderedDict()
        self._finished: OrderedDict[str, None] = OrderedDict()
        self._force_released: OrderedDict[str, None] = OrderedDict()
        self._pinned_kv_tokens = 0
        # Cumulative counts by reason, exported as Prometheus counters.
        self.refused = dict.fromkeys(PIN_REFUSE_REASONS, 0)
        self.force_released = dict.fromkeys(FORCE_RELEASE_REASONS, 0)

    # ===== Queries =====

    def has_pins(self) -> bool:
        return bool(self._pins)

    @property
    def pinned_kv_tokens(self) -> int:
        return self._pinned_kv_tokens

    @property
    def pinned_mamba_states(self) -> int:
        # One pin locks exactly its anchor node's mamba state.
        return len(self._pins) if self._max_mamba_states is not None else 0

    @property
    def pinned_mamba_usage(self) -> float:
        if not self._mamba_capacity:
            return 0.0
        return self.pinned_mamba_states / self._mamba_capacity

    @property
    def num_force_released(self) -> int:
        return sum(self.force_released.values())

    def snapshot(self) -> list[dict]:
        now = self._clock()
        return [
            {
                "rid": e.rid,
                "num_tokens": e.num_tokens,
                "pinned_for_s": round(now - e.pinned_at, 3),
                "expected_return_in_s": (
                    None
                    if e.expected_return_at is None
                    else round(e.expected_return_at - now, 3)
                ),
            }
            for e in self._pins.values()
        ]

    # ===== Control plane =====

    def request_pin(self, *, rid: str, expected_return_s: Optional[float]) -> PinResult:
        if rid in self._pins:
            return PinResult(
                status=PinStatus.ALREADY_PINNED, num_tokens=self._pins[rid].num_tokens
            )
        if rid in self._finished:
            self.refused[PinStatus.ALREADY_FINISHED] += 1
            return PinResult(status=PinStatus.ALREADY_FINISHED)
        # Not seen finishing: running, queued, or not arrived yet.
        _remember(self._pending, rid, expected_return_s, limit=_MAX_PENDING)
        return PinResult(status=PinStatus.PENDING)

    def unpin(self, rid: str) -> str:
        if rid in self._pins:
            self._release(rid)
            return PinStatus.UNPINNED
        if self._pending.pop(rid, _MISSING) is not _MISSING:
            return PinStatus.UNPINNED
        if rid in self._force_released:
            return PinStatus.FORCE_RELEASED
        return PinStatus.NOT_PINNED

    def clear(self) -> None:
        """Drop every pin and intent; call before the tree is reset."""
        for rid in list(self._pins):
            self._release(rid)
        self._pending.clear()

    # ===== Request lifecycle =====

    def on_request_released(self, req: Req, *, cached: Optional[CachedPrefix]) -> None:
        """A request's KV row is being released; ``cached`` is where its KV
        landed in the tree (None when nothing was inserted)."""
        if not req.finished():
            # Retraction: the request will be re-run and released again.
            return
        _remember(self._finished, req.rid, None, limit=_MAX_REMEMBERED_RIDS)
        wanted = req.pin_on_finish or req.rid in self._pending
        expected_return_s = self._pending.pop(req.rid, req.pin_expected_return_s)
        if not wanted:
            return
        if cached is None or cached.num_tokens == 0:
            cached = self._deepest_reusable_prefix(req)
        result = self._pin(
            rid=req.rid, cached=cached, expected_return_s=expected_return_s
        )
        if result.status != PinStatus.PINNED:
            self.refused[result.status] += 1
            logger.debug("prefix pin for rid=%s not taken: %s", req.rid, result.status)

    def handover(self, reqs) -> None:
        """Admitted requests now hold their own prefix locks; drop the pins
        they asked to take over (``release_pin_of``)."""
        for req in reqs:
            if req.release_pin_of is not None:
                self.unpin(req.release_pin_of)
                req.release_pin_of = None

    # ===== Safety valve =====

    def release_until(self, satisfied: Callable[[], bool], *, reason: str) -> int:
        """Force-release pins in release order until ``satisfied()``; returns
        how many were released."""
        released = 0
        while self._pins and not satisfied():
            self.release_one(reason=reason)
            released += 1
        return released

    def release_one(self, *, reason: str) -> bool:
        """``reason`` is one of FORCE_RELEASE_REASONS."""
        if not self._pins:
            return False
        victim = min(self._pins.values(), key=self._victim_key)
        logger.info(
            "Force-releasing prefix pin rid=%s (%d tokens): %s",
            victim.rid,
            victim.num_tokens,
            reason,
        )
        self._release(victim.rid)
        _remember(self._force_released, victim.rid, None, limit=_MAX_REMEMBERED_RIDS)
        self.force_released[reason] += 1
        return True

    def ensure_mamba_headroom(self, states: int) -> int:
        """Keep at least ``states`` mamba states free or evictable, so a
        match-time copy-on-write and a new request's slots can always be had."""
        if self._mamba_free_states is None:
            return 0
        return self.release_until(
            lambda: self._mamba_free_states() >= states,
            reason=ForceReleaseReason.MAMBA_HEADROOM,
        )

    # ===== Internals =====

    def _pin(
        self,
        *,
        rid: str,
        cached: Optional[CachedPrefix],
        expected_return_s: Optional[float],
    ) -> PinResult:
        if cached is None or cached.num_tokens == 0:
            return PinResult(status=PinStatus.NOTHING_CACHED)
        # Overlapping pins share path tokens; counting each in full keeps the
        # budget conservative.
        if self._pinned_kv_tokens + cached.num_tokens > self._max_kv_tokens:
            return PinResult(status=PinStatus.OVER_BUDGET)
        if (
            self._max_mamba_states is not None
            and len(self._pins) + 1 > self._max_mamba_states
        ):
            return PinResult(status=PinStatus.OVER_BUDGET)
        receipt = self._tree_cache.inc_lock_ref(cached.node_id)
        now = self._clock()
        self._pins[rid] = PinEntry(
            rid=rid,
            node_id=cached.node_id,
            dec_params=receipt.to_dec_params(),
            num_tokens=cached.num_tokens,
            pinned_at=now,
            expected_return_at=None
            if expected_return_s is None
            else now + expected_return_s,
        )
        self._pinned_kv_tokens += cached.num_tokens
        return PinResult(status=PinStatus.PINNED, num_tokens=cached.num_tokens)

    def _deepest_reusable_prefix(self, req: Req) -> Optional[CachedPrefix]:
        """The deepest node of req's context a match would reuse now; read-only.
        With mamba extra_buffer a finish inserts only up to the last mamba
        checkpoint, so a turn that added fewer tokens than the track interval
        inserts nothing, while the path its next turn reuses is still cached."""
        path = self._tree_cache.peek_prefix_path(
            RadixKey(
                token_ids=req.origin_input_ids + req.output_ids,
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            )
        )
        if not path:
            return None
        num_tokens, node_id = path[-1]
        return CachedPrefix(node_id=node_id, num_tokens=num_tokens)

    def _release(self, rid: str) -> None:
        entry = self._pins.pop(rid)
        self._tree_cache.dec_lock_ref(entry.node_id, entry.dec_params)
        self._pinned_kv_tokens -= entry.num_tokens

    def _victim_key(self, e: PinEntry):
        """Smallest key is released first."""
        if self._release_order == "latest_return":
            return -(math.inf if e.expected_return_at is None else e.expected_return_at)
        if self._release_order == "largest":
            return -e.num_tokens
        return e.pinned_at


_MISSING = object()


def _remember(store: OrderedDict, key: str, value, *, limit: int) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)
