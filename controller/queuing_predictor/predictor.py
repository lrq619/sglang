"""Monte Carlo over decode lengths drawn from the pool, ported from
loadgen/qp_online.predict_mc_online."""

from __future__ import annotations

from typing import Optional

import msgspec
import numpy as np

from controller.queuing_predictor.costs import StepCosts
from controller.queuing_predictor.pool import DecodePool
from controller.queuing_predictor.simulator import build_scenario, simulate


class Prediction(msgspec.Struct, kw_only=True, frozen=True):
    queue_mean_s: float
    queue_p50_s: float
    queue_p90_s: float
    # Target's cached prefix now, and its mean at admission after eviction.
    prefix_tokens: int
    admit_prefix_tokens: float
    prefill_s: float
    total_mean_s: float
    num_running: int
    num_waiting: int


def predict(
    *,
    snap: dict,
    snapshot_age_s: float,
    pool: DecodePool,
    costs: StepCosts,
    input_len: int,
    prefix_len: int,
    max_new_tokens: int,
    prefix_ratio: float,
    samples: int,
    rng: np.random.Generator,
    cap_by_max_new: bool = False,
    prefix_path: Optional[list] = None,
) -> Prediction:
    """`cap_by_max_new` clips each drawn length at the request's max_new_tokens;
    off by default because loadgen sets max_new_tokens to the true length."""
    sc = build_scenario(
        snap=snap,
        target_input_len=input_len,
        target_prefix_len=prefix_len,
        target_max_new=max_new_tokens,
        prefix_ratio=prefix_ratio,
        costs=costs,
        target_prefix_path=prefix_path,
    )
    queue_generated = np.array([q.generated for q in sc.queue], dtype=np.int64)
    queue_max_left = np.array(
        [max(1, q.max_new - q.generated) for q in sc.queue], dtype=np.int64
    )
    running_max_left = np.maximum(
        1, sc.running_max_new.astype(np.int64) - sc.running_generated
    )
    waits = np.empty(samples)
    admit_prefix = np.empty(samples)
    for i in range(samples):
        running_rem = pool.sample_remaining(sc.running_generated, rng)
        queue_rem = pool.sample_remaining(queue_generated, rng)
        chunked_decode = int(pool.sample_remaining(np.zeros(1), rng)[0])
        if cap_by_max_new:
            running_rem = np.minimum(running_rem, running_max_left)
            queue_rem = np.minimum(queue_rem, queue_max_left)
            if sc.chunked is not None:
                chunked_decode = min(chunked_decode, sc.chunked.max_new)
        # Queue entries map 1:1; the target's slot is never read.
        waits[i], admit_prefix[i] = simulate(
            sc=sc,
            running_remaining=running_rem,
            chunked_decode=chunked_decode,
            queue_decode=np.append(queue_rem, 0),
            costs=costs,
        )
    waits = np.maximum(0.0, waits - snapshot_age_s)
    prefill_s = float(
        np.mean(
            [
                costs.isolated_prefill(
                    extend=max(1, sc.target.ctx - int(p)), prefix=int(p), chunk=sc.chunk
                )
                for p in admit_prefix
            ]
        )
    )
    mean = float(waits.mean())
    return Prediction(
        queue_mean_s=mean,
        queue_p50_s=float(np.quantile(waits, 0.5)),
        queue_p90_s=float(np.quantile(waits, 0.9)),
        prefix_tokens=sc.target.prefix,
        admit_prefix_tokens=float(admit_prefix.mean()),
        prefill_s=prefill_s,
        total_mean_s=mean + prefill_s,
        num_running=len(sc.running_ctx),
        num_waiting=len(sc.queue),
    )
