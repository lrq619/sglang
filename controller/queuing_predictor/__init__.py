"""Online queueing-latency predictor sidecar for SGLang (see
.claude/queuing_predictor.md).

Reads GET /scheduler_snapshot from each instance, keeps a global pool of
finished decode lengths, and answers POST /predict with a Monte Carlo over a
step-level admission simulator. Standalone: aiohttp + numpy + msgspec, no
sglang import.
"""
