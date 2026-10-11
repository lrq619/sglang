"""Reconstruct per-instance scheduler state from a loadgen run, for offline
queueing-latency prediction (see work_dirs/queuing_latency.md).

Each finished turn in loadgen.csv gives its send time, end time and the
server-measured queue / prefill / decode spans, so on every endpoint a request is
    queued   in [t_send,  t_admit)          t_admit = t_end - decode - prefill
    running  in [t_admit, t_end)            (prefill then decode)
Turns cut at the window end are not in loadgen.csv, so states near the end of
a run miss requests that never finished; callers evaluate only early enough.
"""

import bisect
import csv
from collections import defaultdict
from typing import NamedTuple


class Turn(NamedTuple):
    endpoint: str
    replica_id: int
    session_id: str
    turn_id: int
    t_send: float
    t_admit: float
    t_decode_start: float
    t_end: float
    input_len: int
    cached: int
    decode_len: int
    queue_s: float
    prefill_s: float
    decode_s: float


def load_turns(path: str) -> dict:
    """endpoint -> turns sorted by t_send."""
    by_ep = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            continue
        t_end = float(r["t_end"])
        decode_s = float(r["turn_decode_latency"])
        prefill_s = float(r["turn_prefill_latency"])
        t_decode_start = t_end - decode_s
        by_ep[r["endpoint"]].append(
            Turn(
                endpoint=r["endpoint"],
                replica_id=int(r["replica_id"]),
                session_id=r["session_id"],
                turn_id=int(r["turn_id"]),
                t_send=float(r["t_send"]),
                t_admit=t_decode_start - prefill_s,
                t_decode_start=t_decode_start,
                t_end=t_end,
                input_len=int(r["input_len"]),
                cached=int(r["cached_tokens"] or 0),
                decode_len=int(r["completion_tokens"]),
                queue_s=float(r["turn_queuing_latency"]),
                prefill_s=prefill_s,
                decode_s=decode_s,
            )
        )
    for turns in by_ep.values():
        turns.sort(key=lambda t: t.t_send)
    return dict(by_ep)


class InstanceView:
    """Fast point-in-time queries over one endpoint's turns."""

    def __init__(self, turns: list):
        self.turns = turns
        self._by_admit = sorted(turns, key=lambda t: t.t_admit)
        self._admit_times = [t.t_admit for t in self._by_admit]
        self._send_times = [t.t_send for t in turns]
        # Turns can stay in flight for long; bound the backward scan by the longest.
        self._max_span = max((t.t_end - t.t_send) for t in turns)

    def running_at(self, t: float) -> list:
        lo = bisect.bisect_left(self._admit_times, t - self._max_span)
        hi = bisect.bisect_right(self._admit_times, t)
        return [x for x in self._by_admit[lo:hi] if x.t_end > t]

    def queued_at(self, t: float) -> list:
        """Requests sent before t and not yet admitted, in FCFS (send) order."""
        lo = bisect.bisect_left(self._send_times, t - self._max_span)
        hi = bisect.bisect_left(self._send_times, t)
        return [x for x in self.turns[lo:hi] if x.t_admit > t]

    def generated_at(self, turn: Turn, t: float) -> int:
        """Decode tokens produced by t, interpolated over the decode span."""
        if t <= turn.t_decode_start:
            return 0
        frac = min(1.0, (t - turn.t_decode_start) / max(turn.decode_s, 1e-6))
        return int(frac * turn.decode_len)

    def finished_between(self, t0: float, t1: float) -> list:
        lo = bisect.bisect_left(self._admit_times, t0 - self._max_span)
        hi = bisect.bisect_right(self._admit_times, t1)
        return [x for x in self._by_admit[lo:hi] if t0 <= x.t_end < t1]

    def admitted_between(self, t0: float, t1: float) -> list:
        lo = bisect.bisect_left(self._admit_times, t0)
        hi = bisect.bisect_left(self._admit_times, t1)
        return self._by_admit[lo:hi]


# Rolling hit ratio sits near 0.9 before collapse and near 0.25 after it.
COLLAPSE_HIT_RATIO = 0.5
COLLAPSE_WINDOW_S = 300.0


def collapse_onset(turns: list) -> float:
    """First time the rolling prefix-hit ratio drops below COLLAPSE_HIT_RATIO and
    never recovers (cache-thrashing collapse); inf if the instance never collapses."""
    done = sorted(turns, key=lambda t: t.t_end)
    ends = [t.t_end for t in done]
    onset = float("inf")
    t_end = ends[-1]
    t = ends[0] + COLLAPSE_WINDOW_S
    while t <= t_end:
        lo = bisect.bisect_right(ends, t - COLLAPSE_WINDOW_S)
        hi = bisect.bisect_right(ends, t)
        win = done[lo:hi]
        if win:
            hit = sum(x.cached for x in win) / max(1, sum(x.input_len for x in win))
            if hit < COLLAPSE_HIT_RATIO:
                onset = min(onset, t)
            else:
                onset = float("inf")
        t += 10.0
    return onset
