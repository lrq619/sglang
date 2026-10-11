"""Step-level admission simulator with prefill compute (layer 1 of the queueing
predictor, see work_dirs/queuing_latency.md).

Mirrors SGLang's default scheduling (enable_mixed_chunk=False, fcfs):
  * whenever a request can be admitted, the next step is a prefill batch of up
    to CHUNK_TOKENS tokens: in-flight chunked prefills first, then queue heads;
    a request larger than the remaining budget is chunked;
  * running requests do not decode during a prefill step;
  * otherwise every running request decodes one token per step.
A request's queueing ends when its first chunk is scheduled.

Step costs come from loadgen data: decode steps from turns whose decode span
saw no other prefill, prefill steps from turns prefilled alone.
"""

import bisect
from typing import NamedTuple

import numpy as np

KV_CAPACITY = 957105
MAX_RUNNING = 52
NEW_TOKEN_RATIO = 0.7 * 0.14
CLIP_MAX_NEW = 4096
CHUNK_TOKENS = 32768
HORIZON_S = 1800.0


class StepCosts(NamedTuple):
    """decode step = d0 + d1 * batch + d2 * context_Mtok;
    prefill step = p0 + sum_i(p1 * E_i + p2 * E_i * (P_i + E_i / 2) / 1e6)."""

    d0: float
    d1: float
    d2: float
    p0: float
    p1: float
    p2: float

    def decode_step(self, batch: int, ctx_tokens: float) -> float:
        return self.d0 + self.d1 * batch + self.d2 * ctx_tokens / 1e6

    def prefill_step(self, chunks: list) -> float:
        """chunks: (extend tokens this step, prefix tokens before them)."""
        return self.p0 + sum(self.p1 * e + self.p2 * e * (p + e / 2) / 1e6 for e, p in chunks)


def fit_step_costs(views: list, rng) -> StepCosts:
    dec_x, dec_y, pre_x, pre_y = [], [], [], []
    for v in views:
        prefill_iv = sorted((t.t_admit, t.t_decode_start) for t in v.turns)
        starts = [a for a, _ in prefill_iv]
        for x in rng.sample(v.turns, min(1500, len(v.turns))):
            if x.decode_len >= 32 and x.decode_s > 0.2:
                lo = bisect.bisect_left(starts, x.t_decode_start - 120)
                hi = bisect.bisect_right(starts, x.t_end)
                overlap = sum(max(0.0, min(b, x.t_end) - max(a, x.t_decode_start)) for a, b in prefill_iv[lo:hi])
                # Arbitrary cut: a decode span this clean is dominated by decode steps.
                if overlap < 0.05 * x.decode_s:
                    run = v.running_at((x.t_decode_start + x.t_end) / 2)
                    dec_x.append((len(run), sum(r.input_len for r in run) / 1e6))
                    dec_y.append(x.decode_s / x.decode_len)
            if x.prefill_s > 0 and len(v.admitted_between(x.t_admit - 0.5, x.t_decode_start)) == 1:
                e = max(1, x.input_len - x.cached)
                pre_x.append((e, x.cached))
                pre_y.append(x.prefill_s)
    dx, dy = np.array(dec_x), np.array(dec_y)
    d, *_ = np.linalg.lstsq(np.c_[np.ones(len(dx)), dx[:, 0], dx[:, 1]], dy, rcond=None)
    e, p = np.array(pre_x, dtype=float).T
    py = np.array(pre_y)
    a = np.c_[np.ones(len(e)), e, e * (p + e / 2) / 1e6]
    # Relative-error weighting so the many small prefills are not swamped.
    w = 1.0 / py
    c, *_ = np.linalg.lstsq(a * w[:, None], py * w, rcond=None)
    return StepCosts(d0=d[0], d1=d[1], d2=d[2], p0=c[0], p1=c[1], p2=c[2])


class Pending(NamedTuple):
    """A request not yet decoding: ctx when prefilled, tokens left to prefill, decode length."""

    ctx: int
    extend_left: int
    decode: int


def simulate_wait_v2(*, running: list, inflight: list, queue: list, costs: StepCosts, busy_s: float = 0.0) -> float:
    """Seconds until the last queue entry's first chunk is scheduled.

    running: [ctx, remaining_decode]; inflight: admitted Pending still prefilling;
    queue: Pending in FCFS order ending with the target; busy_s: time left in the
    step the GPU is executing at t0 (the scheduler admits nothing before it ends).
    """
    ctx = [float(r[0]) for r in running]
    rem = [float(r[1]) for r in running]
    inflight = [list(p) for p in inflight]
    reserved = _reserved(ctx=ctx, rem=rem, inflight=inflight)
    qi = 0
    t = busy_s
    while t <= HORIZON_S:
        budget = CHUNK_TOKENS
        chunks = []
        for p in inflight:
            if budget <= 0:
                break
            take = min(p[1], budget)
            chunks.append((take, p[0] - p[1]))
            p[1] -= take
            budget -= take
        while budget > 0 and qi < len(queue) and len(rem) + len(inflight) < MAX_RUNNING:
            q = queue[qi]
            if q.ctx + min(q.decode, CLIP_MAX_NEW) >= KV_CAPACITY - reserved:
                break
            if qi == len(queue) - 1:
                return t
            take = min(q.extend_left, budget)
            chunks.append((take, q.ctx - q.extend_left))
            inflight.append([q.ctx, q.extend_left - take, q.decode])
            reserved += q.ctx + NEW_TOKEN_RATIO * min(q.decode, CLIP_MAX_NEW)
            budget -= take
            qi += 1
        if chunks:
            t += costs.prefill_step(chunks)
            for p in [p for p in inflight if p[1] <= 0]:
                ctx.append(float(p[0]))
                rem.append(float(p[2]))
            inflight = [p for p in inflight if p[1] > 0]
            continue
        if not rem:
            return HORIZON_S
        # Nothing admissible: decode until the next request finishes.
        d = min(rem)
        t += d * costs.decode_step(len(rem), sum(ctx))
        alive = [i for i, r in enumerate(rem) if r - d > 0.5]
        ctx = [ctx[i] + d for i in alive]
        rem = [rem[i] - d for i in alive]
        reserved = _reserved(ctx=ctx, rem=rem, inflight=inflight)
    return HORIZON_S


def _reserved(*, ctx: list, rem: list, inflight: list) -> float:
    """PrefillAdder's offset: held context plus decode headroom of admitted requests."""
    held = sum(ctx) + NEW_TOKEN_RATIO * sum(min(r, CLIP_MAX_NEW) for r in rem)
    return held + sum(p[0] + NEW_TOKEN_RATIO * min(p[2], CLIP_MAX_NEW) for p in inflight)


def split_state(*, snap, decode_of) -> tuple:
    """(running, inflight, queue, busy_s) for simulate_wait_v2 from a Snapshot.

    decode_of(turn, generated) -> remaining decode tokens. A request still in
    prefill at t0 is mid-way through a chunk: progress is interpolated over its
    prefill span, the step in flight ends at the next CHUNK_TOKENS boundary, and
    only the tokens after that boundary remain to schedule.
    """
    running, inflight = [], []
    busy_s = 0.0
    for r in snap.running:
        g = snap.generated[id(r)]
        if snap.t0 < r.t_decode_start:
            extend = max(1, r.input_len - r.cached)
            left_frac = min(1.0, (r.t_decode_start - snap.t0) / max(r.prefill_s, 1e-6))
            done = extend * (1.0 - left_frac)
            boundary = min(extend, (int(done // CHUNK_TOKENS) + 1) * CHUNK_TOKENS)
            busy_s = max(busy_s, (boundary - done) / extend * r.prefill_s)
            if boundary < extend:
                inflight.append(Pending(ctx=r.input_len, extend_left=int(extend - boundary),
                                        decode=decode_of(r, 0)))
            else:
                running.append([r.input_len, max(1, decode_of(r, 0))])
        else:
            running.append([r.input_len + g, max(1, decode_of(r, g))])
    queue = [Pending(ctx=q.input_len, extend_left=max(1, q.input_len - q.cached), decode=decode_of(q, 0))
             for q in snap.queue_ahead]
    x = snap.x
    queue.append(Pending(ctx=x.input_len, extend_left=max(1, x.input_len - x.cached), decode=decode_of(x, 0)))
    return running, inflight, queue, busy_s
