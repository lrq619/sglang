import time
import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.managers.io_struct import (
    GetSchedulerSnapshotReq,
    GetSchedulerSnapshotReqOutput,
    _msgpack_decoder,
)
from sglang.srt.managers.scheduler_components.snapshot_inquirer import (
    FinishedReqRing,
    SchedulerSnapshotInquirer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.msgpack_utils import enc_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _ring_append(ring, rid):
    ring.append(
        rid=rid,
        input_len=10,
        output_len=3,
        cached_tokens=0,
        finish_reason="stop",
        finish_monotonic=time.monotonic(),
    )


class TestFinishedReqRing(CustomTestCase):
    def test_cursor_semantics_across_wraparound(self):
        ring = FinishedReqRing(capacity=4)
        self.assertEqual((ring.seq_head, ring.seq_tail), (-1, 0))
        self.assertEqual(ring.since(-1), [])
        _ring_append(ring, "first")
        self.assertEqual([r.seq for r in ring.since(-1)], [0])
        ring = FinishedReqRing(capacity=4)

        for i in range(6):
            _ring_append(ring, f"r{i}")
        # seq 0 and 1 fell off; a reader at cursor 0 sees tail 2 > 0 + 1 (a gap).
        self.assertEqual((ring.seq_head, ring.seq_tail), (5, 2))
        self.assertEqual([r.seq for r in ring.since(-1)], [2, 3, 4, 5])
        self.assertEqual([r.seq for r in ring.since(0)], [2, 3, 4, 5])
        self.assertEqual([r.seq for r in ring.since(3)], [4, 5])
        self.assertEqual(ring.since(5), [])
        self.assertEqual(ring.since(99), [])


def _req(rid, *, input_len, output_len=0, max_new=100, finished=False, **kw):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[0] * input_len,
        output_ids=[0] * output_len,
        sampling_params=SimpleNamespace(max_new_tokens=max_new),
        cached_tokens=kw.get("cached_tokens", 0),
        is_retracted=False,
        extend_range=kw.get("extend_range"),
        time_stats=SimpleNamespace(
            forward_entry_time=0.0, wait_queue_entry_time=time.perf_counter()
        ),
        finished=lambda: finished,
        last_node=None,
    )


def _make_inquirer(
    *, running, waiting, chunked, last_batch, peek=lambda req, ranks: None, ranks=None
):
    pool_stats = SimpleNamespace(
        full_available_size=1000,
        full_evictable_size=200,
        full_num_used=800,
        mamba_available_size=None,
        mamba_evictable_size=None,
        mamba_num_used=None,
    )
    return SchedulerSnapshotInquirer(
        dp_rank=None,
        max_total_num_tokens=2000,
        max_running_requests=8,
        chunked_prefill_size=64,
        page_size=1,
        schedule_policy="fcfs",
        enable_mixed_chunk=False,
        enable_overlap=True,
        pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: pool_stats),
        new_token_ratio_tracker=SimpleNamespace(
            current=0.3, init=0.7, min=0.1, decay=0.01
        ),
        get_running_batch=lambda: SimpleNamespace(reqs=running),
        get_waiting_queue=lambda: waiting,
        get_chunked_req=lambda: chunked,
        get_last_batch=lambda: last_batch,
        get_forward_ct=lambda: 7,
        get_eviction_ranks=ranks or (lambda: {}),
        get_shared_locked_tokens=lambda node: 0,
        peek_prefix_path_for_req=peek,
        peek_prefix_path_for_tokens=lambda token_ids, ranks: [[len(token_ids) // 2, 7]],
    )


class TestSchedulerSnapshotInquirer(CustomTestCase):
    def test_extend_step_requests_are_running_but_chunked_is_not(self):
        """Requests whose last chunk is in flight are not in running_batch yet; a
        predictor that misses them under-counts the batch it will wait behind."""
        decoding = _req("dec", input_len=50, output_len=5)
        done = _req("done", input_len=10, output_len=2, finished=True)
        new = _req("new", input_len=30, output_len=0)
        chunked = _req("chunk", input_len=200, extend_range=SimpleNamespace(end=64))
        step = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            forward_iter=7,
            launch_ts=time.monotonic(),
            reqs=[new, chunked],
            extend_lens=[30, 64],
            prefix_lens=[0, 0],
        )
        inquirer = _make_inquirer(
            running=[decoding, done], waiting=[], chunked=chunked, last_batch=step
        )
        snap = inquirer.get_snapshot(finished_since=-1, max_waiting=10)
        self.assertEqual(
            [(r.rid, r.prefilling) for r in snap.running],
            [("dec", False), ("new", True)],
        )
        self.assertEqual((snap.chunked.rid, snap.chunked.prefilled_len), ("chunk", 64))
        self.assertEqual(snap.step.mode, "extend")
        self.assertEqual(snap.step.extend_tokens, [30, 64])

    def test_waiting_cap_bounds_tree_walks(self):
        """Peeking and ranking walk the radix tree inside the scheduler loop: one
        rank walk per snapshot, none when nothing waits, and no peeks past
        max_waiting, or a long queue stalls every poll."""
        waiting = [_req(f"w{i}", input_len=100) for i in range(5)]
        peeked, rank_walks = [], []
        inquirer = _make_inquirer(
            running=[], waiting=waiting, chunked=None, last_batch=None,
            peek=lambda req, ranks: peeked.append(req.rid) or [[16, None], [40, 9]],
            ranks=lambda: rank_walks.append(1) or {},
        )
        snap = inquirer.get_snapshot(finished_since=-1, max_waiting=2)
        self.assertEqual(snap.num_waiting, 5)
        self.assertEqual([w.prefix_len for w in snap.waiting], [40, 40])
        self.assertEqual(snap.waiting[0].prefix_path, [[16, None], [40, 9]])
        self.assertEqual(peeked, ["w0", "w1"])
        self.assertEqual(len(rank_walks), 1)
        self.assertIsNone(snap.step)

        waiting.clear()
        inquirer.get_snapshot(finished_since=-1, max_waiting=2)
        self.assertEqual(len(rank_walks), 1)

    def test_reply_round_trips_through_ipc_codec(self):
        inquirer = _make_inquirer(
            running=[_req("dec", input_len=50, output_len=5)],
            waiting=[_req("w0", input_len=100)],
            chunked=None,
            last_batch=None,
        )
        inquirer.record_finished(
            SimpleNamespace(
                rid="f0",
                origin_input_ids=[0] * 9,
                output_ids=[0] * 4,
                finished_len=3,
                cached_tokens=2,
                finished_reason=SimpleNamespace(to_json=lambda: {"type": "length"}),
            )
        )
        snap = inquirer.handle(GetSchedulerSnapshotReq()).snapshot
        self.assertEqual(snap.finished, [])
        encoded = msgspec.msgpack.encode(
            inquirer.handle(GetSchedulerSnapshotReq(finished_since=-1)),
            enc_hook=enc_hook,
        )
        decoded = _msgpack_decoder.decode(encoded)
        self.assertIsInstance(decoded, GetSchedulerSnapshotReqOutput)
        self.assertEqual(decoded.snapshot.running[0].rid, "dec")
        self.assertEqual(
            [(f.rid, f.output_len) for f in decoded.snapshot.finished], [("f0", 3)]
        )


if __name__ == "__main__":
    unittest.main()
