"""Step-level admission simulator over a live /scheduler_snapshot, ported from
loadgen/qp_sim2.simulate_wait_v2 and split_state.

Mirrors SGLang's default scheduling (fcfs, enable_mixed_chunk=False):
  * whenever a request can be admitted, the next step is a prefill batch of up
    to `chunk` tokens: the in-flight chunked request first, then queue heads;
    a request larger than the remaining budget is chunked;
  * running requests do not decode during a prefill step;
  * otherwise every running request decodes one token per step.
A request's queueing ends when its first chunk is scheduled.

Prefix eviction: a waiting request does not lock its cached prefix, so it is
lost once LRU eviction reaches it. Each prefix boundary carries the device
tokens that must be evicted before it goes (SchedulerSnapshot prefix_path);
eviction since the snapshot is the allocation beyond the snapshot's free KV.
Once the scheduler attempts the head (a request slot is free) it re-matches it
on every pass, refreshing its LRU position, so a request keeps the prefix it
still had at its first admission attempt; a head blocked only by the slot cap is
never matched and keeps losing its prefix.

Shared prefixes: locking a prefix another request already holds frees nothing,
and a finishing request releases its context minus what another holder still
locks (RunningReqSnapshot.shared_prefix_tokens; for a newly admitted request,
the part of its prefix_path that cannot be evicted).

Memory follows PrefillAdder: a queue head is admitted only if
extend + min(max_new_tokens, clip) fits in available + evictable KV minus
new_token_ratio * min(max_new_tokens - generated, clip) per running request
and the full reservation of requests admitted in the same pass. Sampled decode
lengths only decide when requests finish; a finished request's context returns
to the evictable cache.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import msgspec
import numpy as np

from controller.queuing_predictor.costs import StepCosts

HORIZON_S = 1800.0


class UnsupportedConfig(ValueError):
    pass


class Pending(msgspec.Struct, kw_only=True, frozen=True):
    """A queued request: prompt plus already generated tokens (retracted)."""

    ctx: int
    prefix: int
    generated: int
    max_new: int
    # (prefix length, eviction rank or None if never evicted), shallowest first;
    # empty when unknown, and then `prefix` is kept regardless of eviction.
    prefix_path: Tuple[Tuple[int, Optional[int]], ...] = ()

    @property
    def extend(self) -> int:
        return max(1, self.ctx - self.prefix)

    def locked_prefix(self, prefix: int) -> int:
        """Part of a `prefix`-long match another request already locks."""
        locked = [n for n, rank in self.prefix_path if rank is None and n <= prefix]
        return max(locked, default=0)

    def surviving_prefix(self, evicted: float) -> int:
        """Cached prefix left after `evicted` device tokens went by LRU."""
        if not self.prefix_path:
            return self.prefix
        alive = [n for n, rank in self.prefix_path if rank is None or rank >= evicted]
        return min(max(alive, default=0), max(0, self.ctx - 1))


class Scenario(msgspec.Struct, kw_only=True, frozen=True):
    """Deterministic part of one instance's state; decode lengths are drawn per sample."""

    running_ctx: np.ndarray
    running_generated: np.ndarray
    running_max_new: np.ndarray
    # Tokens of each running request's prefix another request also locks.
    running_shared: np.ndarray
    # The request mid chunked prefill: final ctx, tokens left, tokens done.
    chunked: Optional[Pending]
    chunked_done: int
    queue: List[Pending]
    target: Pending
    free_tokens: float
    # Unused device KV at the snapshot; allocation beyond it evicts.
    kv_available: float
    busy_s: float
    chunk: int
    max_running: int
    clip: int
    ratio: float
    ratio_min: float
    ratio_decay: float


def check_supported(config: dict) -> None:
    if config["schedule_policy"] != "fcfs":
        raise UnsupportedConfig(f"schedule_policy={config['schedule_policy']}")
    if config["enable_mixed_chunk"]:
        raise UnsupportedConfig("enable_mixed_chunk")


def build_scenario(
    *,
    snap: dict,
    target_input_len: int,
    target_prefix_len: int,
    target_max_new: int,
    target_prefix_path: Optional[list] = None,
    prefix_ratio: float,
    costs: StepCosts,
) -> Scenario:
    """`prefix_ratio` estimates the cached share of a waiting request's prompt
    when the scheduler has not matched its prefix yet."""
    config = snap["config"]
    check_supported(config)
    running = snap["running"]
    ctx = np.array([r["input_len"] + r["output_len"] for r in running], dtype=float)
    chunk = config["chunked_prefill_size"] or (1 << 40)
    ratio = snap["new_token_ratio"]
    memory = snap["memory"]
    chunked = snap["chunked"]
    return Scenario(
        running_ctx=ctx,
        running_generated=np.array([r["output_len"] for r in running], dtype=np.int64),
        running_max_new=np.array([r["max_new_tokens"] for r in running], dtype=float),
        running_shared=np.array(
            [r.get("shared_prefix_tokens") or 0 for r in running], dtype=float
        ),
        chunked=(
            None
            if chunked is None
            else Pending(
                ctx=chunked["input_len"],
                prefix=chunked["prefilled_len"],
                generated=0,
                max_new=chunked["max_new_tokens"],
            )
        ),
        chunked_done=0 if chunked is None else chunked["prefilled_len"],
        queue=_queue(snap=snap, prefix_ratio=prefix_ratio),
        target=Pending(
            ctx=target_input_len,
            prefix=min(target_prefix_len, max(0, target_input_len - 1)),
            generated=0,
            max_new=target_max_new,
            prefix_path=_path(target_prefix_path),
        ),
        free_tokens=float(memory["kv_available"] + memory["kv_evictable"]),
        kv_available=float(memory["kv_available"]),
        busy_s=_busy_s(snap=snap, running_ctx=float(ctx.sum()), costs=costs),
        chunk=int(chunk),
        max_running=config["max_running_requests"],
        clip=config["clip_max_new_tokens"],
        ratio=ratio["current"],
        ratio_min=ratio["min"],
        ratio_decay=ratio["decay"],
    )


def _queue(*, snap: dict, prefix_ratio: float) -> List[Pending]:
    out = []
    for w in snap["waiting"]:
        ctx = w["input_len"] + w["output_len"]
        prefix = w["prefix_len"]
        if prefix is None:
            prefix = int(w["input_len"] * prefix_ratio)
        out.append(
            Pending(
                ctx=ctx,
                prefix=min(prefix, max(0, ctx - 1)),
                generated=w["output_len"],
                max_new=w["max_new_tokens"],
                prefix_path=_path(w.get("prefix_path")),
            )
        )
    # The reply carries only the head of a long queue; stand in for the rest
    # by cycling through the head.
    missing = snap["num_waiting"] - len(out)
    if missing > 0 and out:
        out += [out[i % len(out)] for i in range(missing)]
    return out


def _path(path: Optional[list]) -> Tuple[Tuple[int, Optional[int]], ...]:
    return tuple((int(n), rank) for n, rank in path) if path else ()


def _busy_s(*, snap: dict, running_ctx: float, costs: StepCosts) -> float:
    """Time left in the step the GPU executes at snapshot time."""
    step = snap["step"]
    if step is None or step["result_processed"] or step["launch_age_s"] is None:
        return 0.0
    if step["mode"] == "extend":
        cost = costs.prefill_step(zip(step["extend_tokens"], step["prefix_tokens"]))
    elif step["mode"] == "decode":
        cost = costs.decode_step(batch=step["num_reqs"], ctx_tokens=running_ctx)
    else:
        return 0.0
    return max(0.0, cost - step["launch_age_s"])


def simulate_wait(**kwargs) -> float:
    return simulate(**kwargs)[0]


def simulate(
    *,
    sc: Scenario,
    running_remaining: np.ndarray,
    chunked_decode: int,
    queue_decode: np.ndarray,
    costs: StepCosts,
) -> Tuple[float, int]:
    """(Seconds from the snapshot until the target's first chunk is scheduled,
    the target's cached prefix then).

    running_remaining: decode tokens left per running request; chunked_decode:
    total decode length of the chunked request; queue_decode: decode tokens left
    per queue entry (the target's is never used).
    """
    ctx = sc.running_ctx.copy()
    rem = running_remaining.astype(float)
    # Decode budget left before max_new_tokens, for the reservation.
    max_left = sc.running_max_new - sc.running_generated
    shared = sc.running_shared.copy()
    # [final ctx, extend left, tokens done, decode left, max_new left, shared]
    inflight = []
    if sc.chunked is not None:
        inflight.append(
            [
                sc.chunked.ctx,
                sc.chunked.ctx - sc.chunked_done,
                sc.chunked_done,
                chunked_decode,
                sc.chunked.max_new,
                0,
            ]
        )
    queue = sc.queue + [sc.target]
    free = sc.free_tokens
    ratio = sc.ratio
    clip = sc.clip
    qi = 0
    t = sc.busy_s
    allocated = 0.0
    # Prefix of the current head, frozen at its first admission attempt.
    head_qi, head_prefix = -1, 0
    while t <= HORIZON_S:
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
            if qi == len(queue) - 1:
                return t, prefix
            take = min(extend, budget)
            chunks.append((take, prefix))
            # Locking the cached prefix takes it out of the evictable pool,
            # except the part another request already locks.
            locked = q.locked_prefix(prefix)
            free -= prefix - locked + take
            allocated += take
            if take >= extend:
                pass_reserve += min(max_new_left, clip)
            inflight.append(
                [q.ctx, extend - take, prefix + take, queue_decode[qi], max_new_left, locked]
            )
            budget -= take
            slots += 1
            qi += 1
        if chunks:
            t += costs.prefill_step(chunks)
            ctx, rem, max_left, shared, free = _join_prefilled(
                inflight=inflight, ctx=ctx, rem=rem, max_left=max_left, shared=shared,
                free=free,
            )
            inflight = [p for p in inflight if p[1] > 0]
            continue
        if len(rem) == 0:
            # Idle and the target still does not fit: it never gets admitted.
            return HORIZON_S, _target_prefix(queue, head_qi, head_prefix, allocated, sc)
        d = float(rem.min())
        n = len(rem)
        t += d * costs.decode_step(batch=n, ctx_tokens=float(ctx.sum()) + n * d / 2)
        free -= d * n
        allocated += d * n
        ratio = max(sc.ratio_min, ratio - sc.ratio_decay * d)
        ctx = ctx + d
        rem = rem - d
        max_left = max_left - d
        done = rem <= 0.5
        if done.any():
            free += float((ctx[done] - shared[done]).sum())
            ctx, rem, max_left = ctx[~done], rem[~done], max_left[~done]
            shared = shared[~done]
    return HORIZON_S, _target_prefix(queue, head_qi, head_prefix, allocated, sc)


def _target_prefix(queue, head_qi, head_prefix, allocated, sc) -> int:
    if head_qi == len(queue) - 1:
        return head_prefix
    return queue[-1].surviving_prefix(max(0.0, allocated - sc.kv_available))


def _join_prefilled(*, inflight: list, ctx, rem, max_left, shared, free: float):
    """Requests whose last chunk just ran start decoding with their first token out."""
    new = [p for p in inflight if p[1] <= 0]
    if not new:
        return ctx, rem, max_left, shared, free
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
    )
