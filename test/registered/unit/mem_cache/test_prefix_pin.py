"""Unit tests for srt/mem_cache/prefix_pin.py (PrefixPinRegistry).

The registry's correctness rests on its tree locks: every pin takes exactly one
lock and every release gives back that lock with its own receipt, whatever path
(unpin, handover, safety valve, clear) does the releasing. The fake tree below
counts locks per node and rejects a release without a matching acquire.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.base_prefix_cache import CachedPrefix, IncLockRefResult
from sglang.srt.mem_cache.prefix_pin import (
    ForceReleaseReason,
    PinStatus,
    PrefixPinRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CountingTree:
    """Tree-cache boundary: lock counts per node, receipts checked on release."""

    def __init__(self):
        self.locks: dict[int, int] = {}

    def inc_lock_ref(self, node_id):
        self.locks[node_id] = self.locks.get(node_id, 0) + 1
        return IncLockRefResult(delta=1, node_id=node_id)

    def dec_lock_ref(self, node_id, params):
        assert params.node_id == node_id, "receipt released on another node"
        assert self.locks.get(node_id, 0) > 0, "release without a held lock"
        self.locks[node_id] -= 1

    def held(self) -> int:
        return sum(self.locks.values())


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _req(
    rid, *, finished=True, pin_on_finish=False, expected=None, release_pin_of=None
):
    return SimpleNamespace(
        rid=rid,
        finished=lambda: finished,
        pin_on_finish=pin_on_finish,
        pin_expected_return_s=expected,
        release_pin_of=release_pin_of,
    )


def _cached(node_id, num_tokens=100):
    return CachedPrefix(node_id=node_id, num_tokens=num_tokens)


class TestPrefixPinRegistry(CustomTestCase):
    def _registry(
        self,
        *,
        kv_capacity=10_000,
        mamba_capacity=None,
        order="oldest",
        mamba_free=None,
        kv_fraction=0.5,
        mamba_fraction=0.5,
    ):
        self.tree = _CountingTree()
        self.clock = _Clock()
        return PrefixPinRegistry(
            tree_cache=self.tree,
            kv_capacity=kv_capacity,
            mamba_capacity=mamba_capacity,
            mamba_free_states=mamba_free,
            max_kv_fraction=kv_fraction,
            max_mamba_fraction=mamba_fraction,
            release_order=order,
            clock=self.clock,
        )

    def _pin(self, reg, rid, node_id, *, num_tokens=100, expected=None):
        reg.on_request_released(
            _req(rid, pin_on_finish=True, expected=expected),
            cached=_cached(node_id, num_tokens),
        )

    def test_pending_pin_survives_retraction_and_locks_once_on_finish(self):
        reg = self._registry()
        self.assertEqual(
            reg.request_pin(rid="a", expected_return_s=None).status, PinStatus.PENDING
        )
        # A retracted request is released without finishing; it will run again.
        reg.on_request_released(_req("a", finished=False), cached=None)
        self.assertEqual(self.tree.held(), 0)
        reg.on_request_released(_req("a"), cached=_cached(7))
        self.assertEqual(self.tree.locks, {7: 1})
        self.assertEqual(reg.unpin("a"), PinStatus.UNPINNED)
        self.assertEqual(self.tree.held(), 0)
        # Idempotent: a second unpin must not release again.
        self.assertEqual(reg.unpin("a"), PinStatus.NOT_PINNED)
        self.assertEqual(self.tree.held(), 0)

    def test_unrequested_and_uncached_requests_take_no_lock(self):
        reg = self._registry()
        reg.on_request_released(_req("plain"), cached=_cached(1))
        reg.on_request_released(_req("aborted", pin_on_finish=True), cached=None)
        reg.on_request_released(
            _req("empty", pin_on_finish=True), cached=_cached(2, num_tokens=0)
        )
        self.assertEqual(self.tree.held(), 0)
        self.assertFalse(reg.has_pins())

    def test_pin_after_finish_is_refused(self):
        reg = self._registry()
        reg.on_request_released(_req("done"), cached=_cached(3))
        result = reg.request_pin(rid="done", expected_return_s=None)
        self.assertEqual(result.status, PinStatus.ALREADY_FINISHED)
        self.assertEqual(self.tree.held(), 0)
        self.assertEqual(reg.refused[PinStatus.ALREADY_FINISHED], 1)

    def test_budget_refusals_take_no_lock(self):
        # KV: 0.5 * 1000 = 500 tokens.
        reg = self._registry(kv_capacity=1000)
        self._pin(reg, "a", 1, num_tokens=400)
        self._pin(reg, "b", 2, num_tokens=200)
        self.assertEqual(self.tree.locks, {1: 1})
        self.assertEqual(reg.refused[PinStatus.OVER_BUDGET], 1)
        # Mamba: 0.5 * 4 = 2 states, one per pin.
        reg = self._registry(mamba_capacity=4)
        for i, rid in enumerate("abc"):
            self._pin(reg, rid, i)
        self.assertEqual(len(reg.snapshot()), 2)
        self.assertEqual(self.tree.held(), 2)
        self.assertEqual(reg.refused[PinStatus.OVER_BUDGET], 1)
        self.assertEqual(reg.pinned_mamba_usage, 0.5)

    def test_release_orders(self):
        for order, expected_first in [
            # Unknown return time counts as never coming back: released first.
            ("latest_return", "unknown"),
            ("oldest", "early"),
            ("largest", "big"),
        ]:
            with self.subTest(order=order):
                reg = self._registry(order=order)
                self._pin(reg, "early", 1, num_tokens=10, expected=50.0)
                self.clock.now = 1.0
                self._pin(reg, "big", 2, num_tokens=900, expected=5.0)
                self.clock.now = 2.0
                self._pin(reg, "unknown", 3, num_tokens=20, expected=None)
                reg.release_one(reason=ForceReleaseReason.ADMISSION)
                remaining = {p["rid"] for p in reg.snapshot()}
                self.assertNotIn(expected_first, remaining)
                self.assertEqual(len(remaining), 2)
                self.assertEqual(reg.unpin(expected_first), PinStatus.FORCE_RELEASED)
                self.assertEqual(reg.force_released[ForceReleaseReason.ADMISSION], 1)

    def test_release_until_stops_once_satisfied(self):
        reg = self._registry()
        for i in range(4):
            self._pin(reg, f"r{i}", i)
        decode = ForceReleaseReason.DECODE
        self.assertEqual(reg.release_until(lambda: True, reason=decode), 0)
        self.assertEqual(
            reg.release_until(lambda: self.tree.held() <= 1, reason=decode), 3
        )
        self.assertEqual(self.tree.held(), 1)
        # Never satisfiable: drains every pin and stops instead of looping.
        self.assertEqual(reg.release_until(lambda: False, reason=decode), 1)
        self.assertEqual(self.tree.held(), 0)
        self.assertEqual(reg.force_released[decode], 4)
        self.assertEqual(reg.num_force_released, 4)

    def test_mamba_headroom_releases_only_what_is_needed(self):
        # Each released pin frees two mamba states (its own plus a collateral one).
        reg = self._registry(
            mamba_capacity=100, mamba_free=lambda: 2 * (3 - self.tree.held())
        )
        for i in range(3):
            self._pin(reg, f"r{i}", i)
        self.assertEqual(reg.ensure_mamba_headroom(3), 2)
        self.assertEqual(reg.force_released[ForceReleaseReason.MAMBA_HEADROOM], 2)
        self.assertEqual(self.tree.held(), 1)
        self.assertEqual(reg.ensure_mamba_headroom(3), 0)

    def test_handover_releases_once_and_clears_the_request_field(self):
        reg = self._registry()
        self._pin(reg, "turn1", 1)
        turn2 = _req("turn2", release_pin_of="turn1")
        reg.handover([turn2])
        self.assertEqual(self.tree.held(), 0)
        self.assertIsNone(turn2.release_pin_of)
        # Chunked prefill re-submits the same request next step; turn1 is pinned
        # again by then and must not be released by the stale field.
        self._pin(reg, "turn1", 1)
        reg.handover([turn2])
        self.assertEqual(self.tree.locks, {1: 1})

    def test_clear_releases_every_lock_and_intent(self):
        reg = self._registry()
        self._pin(reg, "a", 1)
        self._pin(reg, "b", 1)  # two pins on one node hold two locks
        reg.request_pin(rid="c", expected_return_s=None)
        self.assertEqual(self.tree.locks, {1: 2})
        reg.clear()
        self.assertEqual(self.tree.held(), 0)
        reg.on_request_released(_req("c"), cached=_cached(9))
        self.assertEqual(self.tree.held(), 0)


if __name__ == "__main__":
    unittest.main()
