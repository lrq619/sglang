"""Run from the repo root: python -m unittest controller.queuing_predictor.tests.test_simulator"""

import os
import random
import sys
import unittest

import numpy as np

from controller.queuing_predictor.costs import StepCosts
from controller.queuing_predictor.pool import PAST_MAX_TAIL, DecodePool
from controller.queuing_predictor.predictor import predict
from controller.queuing_predictor.simulator import (
    HORIZON_S,
    UnsupportedConfig,
    build_scenario,
    simulate,
    simulate_wait,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "loadgen")
)
import qp_sim2  # noqa: E402

COSTS = StepCosts(d0=0.02, d1=0.0005, d2=0.0, p0=0.01, p1=2e-5, p2=1e-4)


def _snap(*, running=(), waiting=(), chunked=None, free=10**7, max_running=64,
          chunk=8192, step=None, ratio=0.1):
    return {
        "dp_rank": 0,
        "config": {
            "max_total_num_tokens": free,
            "max_running_requests": max_running,
            "chunked_prefill_size": chunk,
            "page_size": 1,
            "schedule_policy": "fcfs",
            "enable_mixed_chunk": False,
            "enable_overlap": True,
            "clip_max_new_tokens": 4096,
            "waiting_prefix_matched": False,
        },
        "new_token_ratio": {"current": ratio, "init": 0.7, "min": ratio, "decay": 0.0},
        "memory": {"kv_available": free, "kv_evictable": 0, "kv_used": 0},
        "step": step,
        "running": [
            {"rid": f"r{i}", "input_len": c, "output_len": g, "max_new_tokens": 4096,
             "cached_tokens": 0, "prefilling": False, "forward_entry_age_s": 1.0}
            for i, (c, g) in enumerate(running)
        ],
        "chunked": chunked,
        "num_waiting": len(waiting),
        "waiting": [
            {"rid": f"w{i}", "input_len": c, "output_len": 0, "max_new_tokens": 4096,
             "prefix_len": p, "wait_queue_entry_age_s": 1.0, "is_retracted": False}
            for i, (c, p) in enumerate(waiting)
        ],
        "num_retracted": 0,
        "finished_seq_head": -1,
        "finished_seq_tail": 0,
        "finished": [],
    }


def _scenario(snap, *, target=(1000, 0), max_new=4096):
    return build_scenario(
        snap=snap, target_input_len=target[0], target_prefix_len=target[1],
        target_max_new=max_new, prefix_ratio=0.0, costs=COSTS,
    )


class TestPool(unittest.TestCase):
    def test_conditional_draws_exceed_generated(self):
        pool = DecodePool(cap=5, min_size=3)
        for x in [10, 20, 30, 40, 50, 60]:
            pool.add(x)
        # cap keeps the 5 newest; 10 is gone.
        self.assertEqual(pool.sorted_lengths().tolist(), [20, 30, 40, 50, 60])
        rng = np.random.default_rng(0)
        g = np.array([0, 25, 59, 60, 1000] * 200)
        rem = pool.sample_remaining(g, rng)
        self.assertTrue((rem[g < 60] >= 1).all())
        self.assertTrue(((g + rem)[g == 25] >= 30).all())
        self.assertTrue(((g + rem)[g == 59] == 60).all())
        self.assertTrue((rem[g >= 60] == PAST_MAX_TAIL).all())
        # Every eligible length gets drawn, including the largest.
        self.assertEqual(set((g + rem)[g == 0].tolist()), {20, 30, 40, 50, 60})


class TestSimulator(unittest.TestCase):
    def test_slot_bound_wait_is_time_to_first_finish(self):
        """With one slot taken, the target waits exactly for the decode to end."""
        sc = _scenario(_snap(running=[(500, 10)], max_running=1))
        wait = simulate_wait(sc=sc, running_remaining=np.array([7]), chunked_decode=0,
                             queue_decode=np.array([0]), costs=COSTS)
        self.assertAlmostEqual(wait, 7 * COSTS.decode_step(batch=1, ctx_tokens=0))

    def test_memory_bound_target_waits_for_release(self):
        """Admission needs extend + min(max_new, clip) free beyond the ratio reserve."""
        # free 3000; running reserves 0.1 * 4096 = 409.6; target needs 1000 + 4096.
        sc = _scenario(_snap(running=[(2000, 0)], free=3000), target=(1000, 0))
        wait = simulate_wait(sc=sc, running_remaining=np.array([5]), chunked_decode=0,
                             queue_decode=np.array([0]), costs=COSTS)
        # 5000 free after the finish returns 2005 tokens, still short: never admitted.
        self.assertEqual(wait, HORIZON_S)
        # 2000 + 1000 > 3000 - 409.6 now, fits once the 2005 tokens return.
        sc = _scenario(_snap(running=[(2000, 0)], free=3000), target=(2000, 0),
                       max_new=1000)
        wait = simulate_wait(sc=sc, running_remaining=np.array([5]), chunked_decode=0,
                             queue_decode=np.array([0]), costs=COSTS)
        self.assertAlmostEqual(wait, 5 * COSTS.decode_step(batch=1, ctx_tokens=0))

    def test_busy_step_delays_admission(self):
        step = {"forward_iter": 1, "mode": "extend", "launch_age_s": 0.01,
                "result_processed": False, "num_reqs": 1,
                "extend_tokens": [8192], "prefix_tokens": [0]}
        sc = _scenario(_snap(step=step))
        expected = COSTS.prefill_step([(8192, 0)]) - 0.01
        self.assertAlmostEqual(sc.busy_s, expected)
        wait = simulate_wait(sc=sc, running_remaining=np.zeros(0), chunked_decode=0,
                             queue_decode=np.array([0]), costs=COSTS)
        self.assertAlmostEqual(wait, expected)

    def test_matches_offline_simulator_when_memory_is_ample(self):
        """Slot- and chunk-bound dynamics must agree with loadgen/qp_sim2, the
        simulator the offline accuracy numbers were measured with."""
        py_rng = random.Random(1)
        orig = (qp_sim2.KV_CAPACITY, qp_sim2.MAX_RUNNING, qp_sim2.CHUNK_TOKENS)
        qp_sim2.KV_CAPACITY, qp_sim2.MAX_RUNNING, qp_sim2.CHUNK_TOKENS = 10**9, 8, 8192
        try:
            for trial in range(30):
                running = [(py_rng.randint(100, 20000), py_rng.randint(0, 50))
                           for _ in range(py_rng.randint(0, 8))]
                waiting = [(py_rng.randint(100, 30000), 0)
                           for _ in range(py_rng.randint(0, 6))]
                run_rem = [py_rng.randint(1, 400) for _ in running]
                q_dec = [py_rng.randint(2, 400) for _ in waiting]
                target = (py_rng.randint(100, 20000), 0)
                sc = _scenario(_snap(running=running, waiting=waiting, max_running=8,
                                     chunk=8192), target=target)
                online = simulate_wait(
                    sc=sc, running_remaining=np.array(run_rem), chunked_decode=0,
                    # Online counts the first token as produced by the prefill.
                    queue_decode=np.array([d + 1 for d in q_dec] + [0]), costs=COSTS,
                )
                offline = qp_sim2.simulate_wait_v2(
                    running=[[c + g, r] for (c, g), r in zip(running, run_rem)],
                    inflight=[],
                    queue=[qp_sim2.Pending(ctx=c, extend_left=c, decode=d)
                           for (c, _), d in zip(waiting, q_dec)]
                    + [qp_sim2.Pending(ctx=target[0], extend_left=target[0], decode=1)],
                    costs=qp_sim2.StepCosts(*[getattr(COSTS, k) for k in
                                              ("d0", "d1", "d2", "p0", "p1", "p2")]),
                )
                self.assertAlmostEqual(online, offline, places=9, msg=f"trial {trial}")
        finally:
            qp_sim2.KV_CAPACITY, qp_sim2.MAX_RUNNING, qp_sim2.CHUNK_TOKENS = orig

    def _evicting_snap(self, *, max_running=1, kv_evictable=100000):
        # A request decoding 100 more tokens into 10 free KV tokens: about 90
        # tokens of LRU eviction by the time it finishes.
        snap = _snap(running=[(500, 0)], max_running=max_running)
        snap["memory"] = {"kv_available": 10, "kv_evictable": kv_evictable, "kv_used": 0}
        return snap

    def _admit_prefix(self, snap, path):
        sc = build_scenario(snap=snap, target_input_len=2000, target_prefix_len=1500,
                            target_max_new=100, prefix_ratio=0.0, costs=COSTS,
                            target_prefix_path=path)
        return simulate(sc=sc, running_remaining=np.array([100]), chunked_decode=0,
                        queue_decode=np.array([0]), costs=COSTS)[1]

    def test_prefix_lost_once_eviction_passes_its_rank(self):
        """A waiting prefix is gone once allocation beyond free KV exceeds its rank
        by its first admission attempt; deeper boundaries go first."""
        snap = self._evicting_snap()
        self.assertEqual(self._admit_prefix(snap, [[600, None], [1000, 500], [1500, 40]]), 1000)
        self.assertEqual(self._admit_prefix(snap, [[600, None], [1000, 80]]), 600)
        self.assertEqual(self._admit_prefix(snap, None), 1500)

    def test_only_memory_blocked_head_is_protected(self):
        """The scheduler re-matches (refreshing LRU) a head it attempts but cannot
        fit in memory; a head blocked by the request-slot cap is never attempted."""
        path = [[1500, 0]]
        # A free slot, but 500 extend + 100 decode > 910 - 409.6 reserve.
        memory_blocked = self._evicting_snap(max_running=2, kv_evictable=900)
        self.assertEqual(self._admit_prefix(memory_blocked, path), 1500)
        slot_blocked = self._evicting_snap(max_running=1)
        self.assertEqual(self._admit_prefix(slot_blocked, path), 0)

    def test_finish_keeps_prefix_another_request_locks(self):
        """A finishing request frees its context minus the prefix others still
        lock; counting it all admits a memory-blocked head one finish too early."""
        snap = _snap(running=[(2000, 0), (3000, 0)], max_running=4)
        snap["running"][0]["shared_prefix_tokens"] = 500
        snap["memory"] = {"kv_available": 10, "kv_evictable": 1000, "kv_used": 0}
        sc = build_scenario(snap=snap, target_input_len=2300, target_prefix_len=0,
                            target_max_new=100, prefix_ratio=0.0, costs=COSTS)
        wait = simulate_wait(sc=sc, running_remaining=np.array([10, 50]),
                             chunked_decode=0, queue_decode=np.array([0]), costs=COSTS)
        # After A: 990 + 2010 - 500 = 2500 free, 409.6 reserved for B < 2400 needed.
        step = lambda batch: COSTS.decode_step(batch=batch, ctx_tokens=0)
        self.assertAlmostEqual(wait, 10 * step(2) + 40 * step(1))

    def test_unsupported_config_refused(self):
        snap = _snap()
        snap["config"]["enable_mixed_chunk"] = True
        with self.assertRaises(UnsupportedConfig):
            _scenario(snap)


class TestPredict(unittest.TestCase):
    def test_monte_carlo_end_to_end(self):
        pool = DecodePool(cap=1000, min_size=1)
        for x in range(50, 550, 10):
            pool.add(x)
        snap = _snap(running=[(5000, 20)] * 8, waiting=[(4000, 3000)] * 5, max_running=8)
        pred = predict(snap=snap, snapshot_age_s=0.0, pool=pool, costs=COSTS,
                       input_len=2000, prefix_len=0, max_new_tokens=4096,
                       prefix_ratio=0.0, samples=8, rng=np.random.default_rng(0))
        self.assertGreater(pred.queue_mean_s, 0)
        self.assertLessEqual(pred.queue_p50_s, pred.queue_p90_s)
        self.assertAlmostEqual(pred.total_mean_s, pred.queue_mean_s + pred.prefill_s)


if __name__ == "__main__":
    unittest.main()
