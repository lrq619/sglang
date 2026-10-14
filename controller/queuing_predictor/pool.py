"""Global decode-length pool (the empirical CDF), ported from
loadgen/qp_online.OnlinePool and qp_predict.DecodeModel.sample_remaining.

Holds the output lengths of the most recent `cap` finished turns across every
instance; prediction is refused until it has `min_size` entries.
"""

from __future__ import annotations

import collections
from typing import Deque, Dict

import numpy as np

# Arbitrary; a request already past every observed length gets this short tail.
PAST_MAX_TAIL = 64


class DecodePool:
    def __init__(self, *, cap: int, min_size: int):
        self.cap = cap
        self.min_size = min_size
        self._fifo: Deque[int] = collections.deque()
        self._sorted: np.ndarray = np.empty(0, dtype=np.int64)
        self._dirty = False
        self.total_added = 0

    def __len__(self) -> int:
        return len(self._fifo)

    @property
    def ready(self) -> bool:
        return len(self._fifo) >= self.min_size

    def add(self, output_len: int) -> None:
        self._fifo.append(int(output_len))
        if len(self._fifo) > self.cap:
            self._fifo.popleft()
        self._dirty = True
        self.total_added += 1

    def sorted_lengths(self) -> np.ndarray:
        if self._dirty:
            self._sorted = np.sort(np.fromiter(self._fifo, dtype=np.int64))
            self._dirty = False
        return self._sorted

    def sample_remaining(
        self, generated: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Remaining decode tokens D - g with D ~ pool | D > g, one draw per entry."""
        pool = self.sorted_lengths()
        n = len(pool)
        generated = np.asarray(generated, dtype=np.int64)
        lo = np.searchsorted(pool, generated, side="right")
        past = lo >= n
        idx = lo + (rng.random(len(lo)) * np.maximum(n - lo, 1)).astype(np.int64)
        idx = np.minimum(idx, n - 1)
        out = pool[idx] - generated
        out[past] = PAST_MAX_TAIL
        return out

    def quantiles(self) -> Dict[str, float]:
        pool = self.sorted_lengths()
        if len(pool) == 0:
            return {}
        qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        out = {f"p{int(q * 100)}": float(np.quantile(pool, q)) for q in qs}
        out["mean"] = float(pool.mean())
        out["max"] = float(pool[-1])
        return out
