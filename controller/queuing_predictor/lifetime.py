"""KV lifetime predictor and the pin-or-not decision built on it.

A turn that just finished leaves its path newest in both LRU orders. Unpinned, it
stays on device until eviction has gone through everything older there, and on
host (HiCache, write-through) until host writes have gone through everything
older there; that is the LRU characteristic time of each tier (Che's
approximation): older content divided by the rate that consumes it.

  L_dev  = (free device + evictable device) / device eviction rate
  L_host = (free host + host-only + evictable device) / host write rate, at least
           L_dev; per host pool (KV tokens, mamba states), the shorter one wins

The device evictable content is older than the turn and will be host-only before
it, so it counts on host too. Rates come from the server's counters.

The decision compares the two resource wastes, not who queues first:

  pin    holds N unique tokens idle for the hold H: (N * H / M) * Q waiting
         requests delayed
  no pin re-prefills the reusable prefix if it outlives neither tier by the time
         the next turn is admitted, or reloads it from host; either stalls the
         B running and Q waiting requests
"""

from __future__ import annotations

import collections
from typing import Deque, Optional, Tuple

import msgspec

from controller.queuing_predictor.costs import StepCosts

# Window the counter rates are taken over; arbitrary, long enough to smooth
# bursts of a few prefill steps.
RATE_WINDOW_S = 60.0
# Below this a tier is not under eviction pressure and the lifetime is unbounded.
MIN_RATE = 1e-3
UNBOUNDED_S = 1e9
# Measured on 2xH200, Qwen3.8-27B fp8 KV: marginal prefill time per host-loaded
# token (0.75 ms per 1k), plus one mamba state (~156 MB) over PCIe.
RELOAD_S_PER_TOKEN = 0.75e-6
RELOAD_S_PER_MAMBA_STATE = 0.006


class CounterRates:
    """Per-second rates of the server counters the predictor needs, over the
    last RATE_WINDOW_S of /metrics scrapes."""

    def __init__(self):
        self._samples: Deque[Tuple[float, dict]] = collections.deque()

    def add(self, *, t: float, counters: dict) -> None:
        self._samples.append((t, counters))
        while self._samples and t - self._samples[0][0] > RATE_WINDOW_S:
            self._samples.popleft()

    def rate(self, key: str) -> Optional[float]:
        if len(self._samples) < 2:
            return None
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        if key not in c0 or key not in c1 or t1 <= t0:
            return None
        return max(0.0, (c1[key] - c0[key]) / (t1 - t0))


class Lifetime(msgspec.Struct, kw_only=True, frozen=True):
    device_s: float
    host_s: float
    # Inputs, for the log.
    device_evict_rate: Optional[float]
    host_kv_write_rate: Optional[float]
    host_mamba_write_rate: Optional[float]


def predict_lifetime(*, snap: dict, rates: CounterRates) -> Lifetime:
    mem = snap["memory"]
    total = snap["config"]["max_total_num_tokens"]
    free_dev, evictable_dev = mem["kv_available"], mem["kv_evictable"]
    evict = rates.rate("evicted_tokens")
    device_s = _divide(free_dev + evictable_dev, evict)
    w_kv = rates.rate("backup_kv_tokens")
    w_mamba = rates.rate("backup_mamba_states")
    if mem.get("host_kv_total") is None:
        return Lifetime(device_s=device_s, host_s=device_s, device_evict_rate=evict,
                        host_kv_write_rate=w_kv, host_mamba_write_rate=w_mamba)
    host_free = mem["host_kv_available"]
    # Write-through: host also mirrors what is still on device.
    host_only = max(0, mem["host_kv_total"] - host_free - (total - free_dev))
    host_s = _divide(host_free + host_only + evictable_dev, w_kv)
    if mem.get("host_mamba_total") is not None and mem.get("mamba_used") is not None:
        m_free = mem["host_mamba_available"]
        m_device = mem["mamba_used"] + (mem["mamba_evictable"] or 0)
        m_host_only = max(0, mem["host_mamba_total"] - m_free - m_device)
        host_s = min(host_s, _divide(m_free + m_host_only + (mem["mamba_evictable"] or 0), w_mamba))
    return Lifetime(device_s=device_s, host_s=max(device_s, host_s), device_evict_rate=evict,
                    host_kv_write_rate=w_kv, host_mamba_write_rate=w_mamba)


def _divide(amount: float, rate: Optional[float]) -> float:
    if rate is None or rate < MIN_RATE:
        return UNBOUNDED_S
    return amount / rate


class PinDecision(msgspec.Struct, kw_only=True, frozen=True):
    decision: str
    # Predicted outcome if not pinned: "device" | "host" | "lost".
    nopin_outcome: str
    cost_pin: float
    cost_nopin: float
    hold_pin_s: float
    hold_nopin_s: float
    next_turn_queue_s: float
    lifetime_device_s: float
    lifetime_host_s: float
    prefill_s: float
    reload_s: float
    unique_tokens: int
    reuse_tokens: int
    num_running: int
    num_waiting: int
    device_evict_rate: Optional[float]
    host_kv_write_rate: Optional[float]
    host_mamba_write_rate: Optional[float]


def decide_pin(
    *,
    snap: dict,
    rates: CounterRates,
    costs: StepCosts,
    rid: str,
    path_tokens: int,
    shared_tokens: int,
    reuse_tokens: int,
    toolcall_s: float,
) -> PinDecision:
    config = snap["config"]
    total = config["max_total_num_tokens"]
    chunk = config["chunked_prefill_size"] or (1 << 40)
    running = [r for r in snap["running"] if r["rid"] != rid]
    queued = snap["waiting"]
    q, b = snap["num_waiting"], len(running)
    unique = max(0, path_tokens - min(shared_tokens, path_tokens))
    reuse = min(reuse_tokens, path_tokens)
    life = predict_lifetime(snap=snap, rates=rates)
    # A pinned next turn jumps the queue; an unpinned one waits like a new
    # arrival, roughly as long as the oldest of those already waiting.
    jumps = config.get("admit_pinned_first", False)
    fcfs_ages = [w["wait_queue_entry_age_s"] or 0.0 for w in queued
                 if not (jumps and w.get("admits_first"))]
    queue_s = max(fcfs_ages, default=0.0)
    hold_pin = toolcall_s if jumps else toolcall_s + queue_s
    hold_nopin = toolcall_s + queue_s
    if hold_nopin <= life.device_s:
        outcome = "device"
    elif hold_nopin <= life.host_s:
        outcome = "host"
    else:
        outcome = "lost"
    prefill_s = costs.isolated_prefill(extend=max(1, reuse), prefix=0, chunk=chunk)
    reload_s = RELOAD_S_PER_TOKEN * reuse + RELOAD_S_PER_MAMBA_STATE
    cost_pin = unique * hold_pin / total * q
    stall = {"device": 0.0, "host": reload_s, "lost": prefill_s}[outcome]
    cost_nopin = stall * (b + q)
    return PinDecision(
        decision="pin" if cost_pin < cost_nopin else "nopin",
        nopin_outcome=outcome,
        cost_pin=cost_pin,
        cost_nopin=cost_nopin,
        hold_pin_s=hold_pin,
        hold_nopin_s=hold_nopin,
        next_turn_queue_s=queue_s,
        lifetime_device_s=min(life.device_s, UNBOUNDED_S),
        lifetime_host_s=min(life.host_s, UNBOUNDED_S),
        prefill_s=prefill_s,
        reload_s=reload_s,
        unique_tokens=unique,
        reuse_tokens=reuse,
        num_running=b,
        num_waiting=q,
        device_evict_rate=life.device_evict_rate,
        host_kv_write_rate=life.host_kv_write_rate,
        host_mamba_write_rate=life.host_mamba_write_rate,
    )


def parse_counters(text: str) -> dict:
    """The counters predict_lifetime needs, summed over label sets, from
    Prometheus text."""
    out = {"evicted_tokens": 0.0, "backup_kv_tokens": 0.0, "backup_mamba_states": 0.0}
    seen = set()
    for line in text.splitlines():
        if line.startswith("sglang:evicted_tokens_total{"):
            key = "evicted_tokens"
        elif line.startswith("sglang:hicache_backup_tokens_total{"):
            key = "backup_kv_tokens" if 'pool="kv"' in line else (
                "backup_mamba_states" if 'pool="mamba"' in line else None)
        else:
            continue
        if key is None:
            continue
        try:
            out[key] += float(line.rsplit(" ", 1)[1])
            seen.add(key)
        except ValueError:
            continue
    return {k: v for k, v in out.items() if k in seen}
