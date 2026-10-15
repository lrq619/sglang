"""Pin-or-not decision at the end of a turn, by simulating both worlds.

At decision time a turn has just finished and its KV path (path_tokens long,
shared_tokens of it also locked by other requests) is pinned. Its next turn R
arrives toolcall_s later and can reuse reuse_tokens of that path.

  pin:    the unique part stays locked; R is admitted with the whole reusable
          prefix, and the pin's leftover tail returns to the evictable pool then.
  no pin: the unique part is evictable at once, newest in LRU order, so it goes
          after every other evictable token; R keeps what survives the eviction
          by the time the scheduler first attempts it, and prefills the rest.

Both worlds start from the same snapshot, draw the same decode lengths and run
until every request present (running, chunked, waiting, plus R) has finished.
The cost of a world is the mean completion time over that fixed set, so a pin
pays for the queue it holds back and no pin pays for R's re-prefill, which also
stalls every decode running at the time. New arrivals other than R are not
simulated; they are assumed to cost both worlds about the same.

The event loop extends simulator.simulate (same admission, chunking, eviction
and memory rules), which stops at the target's admission instead.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import msgspec
import numpy as np

from controller.queuing_predictor.costs import StepCosts
from controller.queuing_predictor.pool import DecodePool
from controller.queuing_predictor.simulator import Pending, Scenario, build_scenario

# Simulated seconds past R's arrival before unfinished requests are cut.
HORIZON_AFTER_ARRIVAL_S = 1800.0
# Arbitrary; enough boundaries that partial eviction of the unique part is
# resolved to ~1/16 of its length.
UNIQUE_BOUNDARIES = 16


class Arrival(msgspec.Struct, kw_only=True, frozen=True):
    t: float
    req: Pending
    decode: int
    # Tokens returned to the evictable pool when it is admitted (pin handover).
    release_on_admit: float = 0.0


class WorldResult(msgspec.Struct, kw_only=True, frozen=True):
    mean_finish_s: float
    num_requests: int
    num_truncated: int
    # Queue time summed over the waiting requests and R.
    sum_wait_s: float
    # Seconds from R's arrival to its first chunk, and its cached prefix then.
    r_wait_s: float
    r_prefix: int
    # Admission time per queue entry in order, R last if it was admitted.
    admit_s: Tuple[float, ...]


class PinDecision(msgspec.Struct, kw_only=True, frozen=True):
    decision: str
    # mean completion time, pin world minus no-pin world; < 0 favours the pin.
    delta_mean_finish_s: float
    pin_mean_finish_s: float
    nopin_mean_finish_s: float
    pin_sum_wait_s: float
    nopin_sum_wait_s: float
    pin_r_wait_s: float
    nopin_r_wait_s: float
    nopin_r_prefix: float
    nopin_reprefill_tokens: float
    nopin_reprefill_s: float
    num_requests: int
    num_truncated: int
    num_running: int
    num_waiting: int
    unique_tokens: int
    evictable_other: int
    samples: int


def decide_pin(
    *,
    snap: dict,
    snapshot_age_s: float,
    pool: DecodePool,
    costs: StepCosts,
    rid: str,
    path_tokens: int,
    shared_tokens: int,
    reuse_tokens: int,
    next_input_len: int,
    next_decode_tokens: int,
    toolcall_s: float,
    samples: int,
    rng: np.random.Generator,
) -> PinDecision:
    # The finished turn may still be listed as running in a snapshot taken
    # just before it finished; its KV is held either way, as the pin holds it.
    snap = dict(snap, running=[r for r in snap["running"] if r["rid"] != rid])
    sc = build_scenario(
        snap=snap,
        target_input_len=1,
        target_prefix_len=0,
        target_max_new=1,
        prefix_ratio=0.0,
        costs=costs,
    )
    shared = min(shared_tokens, path_tokens)
    reuse = min(reuse_tokens, path_tokens, max(0, next_input_len - 1))
    unique = path_tokens - shared
    evictable_other = int(snap["memory"]["kv_evictable"])
    t_arrive = snapshot_age_s + toolcall_s
    pin_arrival = Arrival(
        t=t_arrive,
        req=_next_turn(ctx=next_input_len, max_new=next_decode_tokens,
                       path=((reuse, None),)),
        decode=next_decode_tokens,
        release_on_admit=float(max(0, path_tokens - max(reuse, shared))),
    )
    nopin_arrival = Arrival(
        t=t_arrive,
        req=_next_turn(
            ctx=next_input_len,
            max_new=next_decode_tokens,
            path=_nopin_path(shared=shared, reuse=reuse, path_tokens=path_tokens,
                             evictable_other=evictable_other),
        ),
        decode=next_decode_tokens,
    )
    queue_generated = np.array([q.generated for q in sc.queue], dtype=np.int64)
    horizon = t_arrive + HORIZON_AFTER_ARRIVAL_S
    pin_runs, nopin_runs = [], []
    for _ in range(samples):
        # Common random numbers: both worlds see the same decode lengths.
        draws = dict(
            running_remaining=pool.sample_remaining(sc.running_generated, rng),
            chunked_decode=int(pool.sample_remaining(np.zeros(1), rng)[0]),
            queue_decode=pool.sample_remaining(queue_generated, rng),
        )
        pin_runs.append(
            run_world(sc=sc, arrival=pin_arrival, extra_free=0.0, horizon_s=horizon,
                      costs=costs, **draws)
        )
        nopin_runs.append(
            run_world(sc=sc, arrival=nopin_arrival, extra_free=float(unique),
                      horizon_s=horizon, costs=costs, **draws)
        )
    pin_mean = _mean(pin_runs, "mean_finish_s")
    nopin_mean = _mean(nopin_runs, "mean_finish_s")
    nopin_prefix = _mean(nopin_runs, "r_prefix")
    reprefill = max(0.0, reuse - nopin_prefix)
    return PinDecision(
        decision="pin" if pin_mean < nopin_mean else "nopin",
        delta_mean_finish_s=pin_mean - nopin_mean,
        pin_mean_finish_s=pin_mean,
        nopin_mean_finish_s=nopin_mean,
        pin_sum_wait_s=_mean(pin_runs, "sum_wait_s"),
        nopin_sum_wait_s=_mean(nopin_runs, "sum_wait_s"),
        pin_r_wait_s=_mean(pin_runs, "r_wait_s"),
        nopin_r_wait_s=_mean(nopin_runs, "r_wait_s"),
        nopin_r_prefix=nopin_prefix,
        nopin_reprefill_tokens=reprefill,
        nopin_reprefill_s=_reprefill_s(costs=costs, reuse=reuse, kept=nopin_prefix,
                                       chunk=sc.chunk),
        num_requests=pin_runs[0].num_requests,
        num_truncated=max(r.num_truncated for r in pin_runs + nopin_runs),
        num_running=len(sc.running_ctx),
        num_waiting=len(sc.queue),
        unique_tokens=unique,
        evictable_other=evictable_other,
        samples=samples,
    )


def _next_turn(*, ctx: int, max_new: int, path: tuple) -> Pending:
    return Pending(
        ctx=ctx,
        prefix=max((n for n, _ in path), default=0),
        generated=0,
        max_new=max_new,
        prefix_path=path,
    )


def _nopin_path(
    *, shared: int, reuse: int, path_tokens: int, evictable_other: int
) -> tuple:
    """Boundaries of R's reusable prefix once unpinned: the shared part stays
    locked; a unique boundary at length L goes after every other evictable token
    and the path's deeper tokens, path_tokens - L of them."""
    path = [(min(shared, reuse), None)] if shared > 0 else [(0, None)]
    if reuse > shared:
        for L in np.linspace(shared, reuse, UNIQUE_BOUNDARIES + 1)[1:]:
            L = int(round(L))
            path.append((L, evictable_other + path_tokens - L))
    return tuple(path)


def _reprefill_s(*, costs: StepCosts, reuse: int, kept: float, chunk: int) -> float:
    lost = int(round(reuse - kept))
    if lost <= 0:
        return 0.0
    return costs.isolated_prefill(extend=lost, prefix=int(kept), chunk=chunk) - costs.p0


def _mean(runs: List[WorldResult], field: str) -> float:
    return float(np.mean([getattr(r, field) for r in runs]))


def run_world(
    *,
    sc: Scenario,
    arrival: Optional[Arrival],
    extra_free: float,
    horizon_s: float,
    running_remaining: np.ndarray,
    chunked_decode: int,
    queue_decode: np.ndarray,
    costs: StepCosts,
) -> WorldResult:
    """Simulate sc (target ignored) plus one timed arrival until every request
    finishes or horizon_s; extra_free adds evictable tokens at the start."""
    ctx = sc.running_ctx.copy()
    rem = running_remaining.astype(float)
    max_left = sc.running_max_new - sc.running_generated
    shared = sc.running_shared.copy()
    # [final ctx, extend left, tokens done, decode left, max_new left, shared]
    inflight = []
    if sc.chunked is not None:
        inflight.append(
            [sc.chunked.ctx, sc.chunked.ctx - sc.chunked_done, sc.chunked_done,
             chunked_decode, sc.chunked.max_new, 0]
        )
    queue: List[Pending] = list(sc.queue)
    decodes: List[float] = [float(d) for d in queue_decode]
    arrived_at: List[float] = [0.0] * len(queue)
    pending = arrival
    num_requests = len(rem) + len(inflight) + len(queue) + (arrival is not None)
    r_index = len(queue) if arrival is not None else -1
    r_wait, r_prefix = math.nan, 0

    free = sc.free_tokens + extra_free
    ratio = sc.ratio
    clip = sc.clip
    qi = 0
    t = sc.busy_s
    allocated = 0.0
    head_qi, head_prefix = -1, 0
    finish_sum, done_n = 0.0, 0
    admit: List[float] = []
    wait_sum = 0.0
    while done_n < num_requests and t <= horizon_s:
        if pending is not None and t >= pending.t:
            queue.append(pending.req)
            decodes.append(float(pending.decode))
            arrived_at.append(pending.t)
            pending = None
        offset = ratio * float(np.minimum(np.maximum(max_left, 0), clip).sum())
        budget = sc.chunk
        chunks = []
        pass_reserve = 0.0
        for p in inflight:
            if budget <= 0:
                break
            take = min(p[1], budget)
            chunks.append((take, p[2]))
            p[1] -= take
            p[2] += take
            free -= take
            allocated += take
            budget -= take
            if p[1] <= 0:
                pass_reserve += min(p[4], clip)
        slots = len(rem) + len(inflight)
        while budget > 0 and qi < len(queue) and slots < sc.max_running:
            q = queue[qi]
            if head_qi != qi:
                head_qi = qi
                head_prefix = q.surviving_prefix(max(0.0, allocated - sc.kv_available))
            prefix = head_prefix
            extend = max(1, q.ctx - prefix)
            max_new_left = max(0, q.max_new - q.generated)
            if extend + min(max_new_left, clip) > free - offset - pass_reserve:
                break
            admit.append(t)
            wait_sum += t - arrived_at[qi]
            if qi == r_index:
                r_wait, r_prefix = t - arrived_at[qi], prefix
                free += arrival.release_on_admit
            take = min(extend, budget)
            chunks.append((take, prefix))
            locked = q.locked_prefix(prefix)
            free -= prefix - locked + take
            allocated += take
            if take >= extend:
                pass_reserve += min(max_new_left, clip)
            inflight.append(
                [q.ctx, extend - take, prefix + take, decodes[qi], max_new_left, locked]
            )
            budget -= take
            slots += 1
            qi += 1
        if chunks:
            t += costs.prefill_step(chunks)
            ctx, rem, max_left, shared, free, n = _join_prefilled(
                inflight=inflight, ctx=ctx, rem=rem, max_left=max_left, shared=shared,
                free=free,
            )
            finish_sum += n * t
            done_n += n
            inflight = [p for p in inflight if p[1] > 0]
            continue
        if len(rem) == 0:
            if pending is not None:
                t = max(t, pending.t)
                continue
            # Idle and the head never fits: nothing left can make progress.
            break
        d = float(rem.min())
        n = len(rem)
        if pending is not None:
            step = costs.decode_step(batch=n, ctx_tokens=float(ctx.sum()))
            d = min(d, max(1.0, math.ceil((pending.t - t) / step)))
        t += d * costs.decode_step(batch=n, ctx_tokens=float(ctx.sum()) + n * d / 2)
        free -= d * n
        allocated += d * n
        ratio = max(sc.ratio_min, ratio - sc.ratio_decay * d)
        ctx = ctx + d
        rem = rem - d
        max_left = max_left - d
        done = rem <= 0.5
        if done.any():
            k = int(done.sum())
            finish_sum += k * t
            done_n += k
            free += float((ctx[done] - shared[done]).sum())
            ctx, rem, max_left = ctx[~done], rem[~done], max_left[~done]
            shared = shared[~done]
    truncated = num_requests - done_n
    finish_sum += truncated * max(t, horizon_s)
    if arrival is not None and math.isnan(r_wait):
        r_wait = max(t, horizon_s) - arrival.t
        r_prefix = queue[r_index].surviving_prefix(
            max(0.0, allocated - sc.kv_available)
        ) if r_index < len(queue) else arrival.req.prefix
    return WorldResult(
        mean_finish_s=finish_sum / max(1, num_requests),
        num_requests=num_requests,
        num_truncated=truncated,
        sum_wait_s=wait_sum,
        r_wait_s=r_wait,
        r_prefix=int(r_prefix),
        admit_s=tuple(admit),
    )


def _join_prefilled(*, inflight: list, ctx, rem, max_left, shared, free: float):
    """simulator._join_prefilled, also counting requests done at their first token."""
    new = [p for p in inflight if p[1] <= 0]
    if not new:
        return ctx, rem, max_left, shared, free, 0
    new_ctx = np.array([p[0] + 1 for p in new], dtype=float)
    new_rem = np.array([max(0.0, p[3] - 1) for p in new], dtype=float)
    new_max_left = np.array([p[4] - 1 for p in new], dtype=float)
    new_shared = np.array([p[5] for p in new], dtype=float)
    done = new_rem <= 0.5
    free += float((new_ctx[done] - new_shared[done]).sum())
    return (
        np.concatenate([ctx, new_ctx[~done]]),
        np.concatenate([rem, new_rem[~done]]),
        np.concatenate([max_left, new_max_left[~done]]),
        np.concatenate([shared, new_shared[~done]]),
        free,
        int(done.sum()),
    )
