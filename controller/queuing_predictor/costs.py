"""Step cost model, ported from loadgen/qp_sim2.StepCosts.

decode step  = d0 + d1 * batch + d2 * context_Mtok
prefill step = p0 + sum_i(p1 * E_i + p2 * E_i * (P_i + E_i / 2) / 1e6)
where E_i is the chunk's extend tokens and P_i the tokens already in KV before it.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import msgspec


class StepCosts(msgspec.Struct, kw_only=True, frozen=True):
    d0: float
    d1: float
    d2: float
    p0: float
    p1: float
    p2: float
    # Free-form provenance: model, GPU, server flags, fit residuals.
    meta: dict = {}

    def decode_step(self, *, batch: int, ctx_tokens: float) -> float:
        return self.d0 + self.d1 * batch + self.d2 * ctx_tokens / 1e6

    def prefill_step(self, chunks: Iterable[Tuple[float, float]]) -> float:
        """chunks: (extend tokens this step, prefix tokens before them)."""
        return self.p0 + sum(
            self.p1 * e + self.p2 * e * (p + e / 2) / 1e6 for e, p in chunks
        )

    def isolated_prefill(self, *, extend: int, prefix: int, chunk: int) -> float:
        """Prefill time of one request alone, split into chunk-sized steps."""
        total, done = 0.0, prefix
        left = max(1, extend)
        while left > 0:
            take = min(left, chunk)
            total += self.prefill_step([(take, done)])
            done += take
            left -= take
        return total

    @classmethod
    def load(cls, path: str) -> StepCosts:
        with open(path, "rb") as f:
            return msgspec.json.decode(f.read(), type=cls)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(msgspec.json.format(msgspec.json.encode(self), indent=2))
