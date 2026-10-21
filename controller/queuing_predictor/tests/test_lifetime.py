"""Lifetime predictor and the resource-cost pin decision (lifetime.py)."""

import unittest

from controller.queuing_predictor.costs import StepCosts
from controller.queuing_predictor.lifetime import (
    UNBOUNDED_S,
    CounterRates,
    decide_pin,
    parse_counters,
    predict_lifetime,
)

COSTS = StepCosts(d0=0.02, d1=0.0005, d2=0.0, p0=0.01, p1=7.5e-5, p2=0.0)
TOTAL = 1_000_000


def _snap(*, free=0, evictable=100_000, waiting=10, waited_s=30.0, host=None, running=20):
    mem = {"kv_available": free, "kv_evictable": evictable, "kv_used": TOTAL - free - evictable,
           "mamba_available": None, "mamba_evictable": None, "mamba_used": None}
    if host is not None:
        mem.update(host_kv_total=host[0], host_kv_available=host[1])
    return {
        "config": {"max_total_num_tokens": TOTAL, "chunked_prefill_size": 32768,
                   "admit_pinned_first": True},
        "memory": mem,
        "running": [{"rid": f"r{i}"} for i in range(running)],
        "num_waiting": waiting,
        "waiting": [{"wait_queue_entry_age_s": waited_s, "admits_first": False}] * waiting,
    }


def _rates(*, evicted=0.0, backup=0.0):
    r = CounterRates()
    r.add(t=0.0, counters={"evicted_tokens": 0.0, "backup_kv_tokens": 0.0})
    r.add(t=10.0, counters={"evicted_tokens": 10 * evicted, "backup_kv_tokens": 10 * backup})
    return r


def _decide(snap, rates, toolcall_s=1.0):
    return decide_pin(snap=snap, rates=rates, costs=COSTS, rid="x", path_tokens=20_000,
                      shared_tokens=0, reuse_tokens=20_000, toolcall_s=toolcall_s)


class TestLifetime(unittest.TestCase):
    def test_no_eviction_pressure_is_unbounded_and_not_pinned(self):
        life = predict_lifetime(snap=_snap(), rates=_rates())
        self.assertEqual(life.device_s, UNBOUNDED_S)
        d = _decide(_snap(), _rates())
        self.assertEqual((d.nopin_outcome, d.decision), ("device", "nopin"))

    def test_device_only_under_pressure_pins(self):
        # 100k evictable older tokens at 10k/s: gone after 10 s, before the
        # 1 s tool call plus 30 s in the queue is over.
        rates = _rates(evicted=10_000)
        self.assertAlmostEqual(predict_lifetime(snap=_snap(), rates=rates).device_s, 10.0)
        d = _decide(_snap(), rates)
        self.assertEqual((d.nopin_outcome, d.decision), ("lost", "pin"))

    def test_ample_host_reloads_instead_of_pinning(self):
        # Host holds 1M over the 1M device mirror: 1M host-only + 100k at 10k/s.
        snap = _snap(host=(2_000_000, 0))
        rates = _rates(evicted=10_000, backup=10_000)
        life = predict_lifetime(snap=snap, rates=rates)
        self.assertAlmostEqual(life.host_s, 110.0)
        # Holding 20k of 1M for 10 s delays 10 waiting requests 2 s; a ~21 ms
        # reload stalls 30 requests 0.63 s.
        d = _decide(snap, rates, toolcall_s=10.0)
        self.assertEqual((d.nopin_outcome, d.decision), ("host", "nopin"))
        # For a 1 s tool call the pin (0.2 s) is cheaper than the reload.
        self.assertEqual(_decide(snap, rates).decision, "pin")

    def test_host_no_larger_than_device_mirror_adds_little(self):
        snap = _snap(host=(500_000, 0))
        life = predict_lifetime(snap=snap, rates=_rates(evicted=10_000, backup=10_000))
        self.assertAlmostEqual(life.host_s, 10.0)

    def test_long_tool_call_is_released_even_when_it_will_be_lost(self):
        # Holding 20k of 1M for 1000 s delays 10 waiting requests 200 s;
        # re-prefilling 20k (1.5 s) stalls 30 requests 45 s.
        d = _decide(_snap(), _rates(evicted=10_000), toolcall_s=1000.0)
        self.assertEqual((d.nopin_outcome, d.decision), ("lost", "nopin"))


class TestParseCounters(unittest.TestCase):
    def test_sums_label_sets_and_splits_pools(self):
        text = "\n".join([
            'sglang:evicted_tokens_total{tp_rank="0"} 5.0',
            'sglang:hicache_backup_tokens_total{pool="kv",tp_rank="0"} 7.0',
            'sglang:hicache_backup_tokens_total{pool="mamba",tp_rank="0"} 2.0',
            "# HELP sglang:evicted_tokens_total x",
        ])
        self.assertEqual(parse_counters(text), {"evicted_tokens": 5.0, "backup_kv_tokens": 7.0,
                                                "backup_mamba_states": 2.0})


if __name__ == "__main__":
    unittest.main()
