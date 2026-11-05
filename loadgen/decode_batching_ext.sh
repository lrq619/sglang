#!/usr/bin/env bash
#PBS -N decbatch_ext
#PBS -P 12004493
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
# Extension of the decode batching sweep to larger B and longer contexts: runs
# decode_batching.sh once per config, each with its own server, so the mamba state
# pool can be sized per N. Every config writes its own flat LOG_ROOT/<RUN_PREFIX>_n<N>_<ts>/.
#
# CONFIGS: space-separated "N:mamba_slots:B1/B2/..." entries, run in order (qsub -v
# splits on commas, so the batch sizes are slash-separated).
#   qsub -V -v CONFIGS="261952:28:1/4/8 32768:152:1/16/20/24" loadgen/decode_batching_ext.sh
#
# Memory setup, chosen so that up to ~2.1M KV tokens fit on one H200: mem-fraction 0.90
# with the prefill CUDA graph off (0.88 OOMs in its capture), and 3 mamba slots per
# request (extra_buffer_lazy + SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK) instead of 5.
# --mamba-max-states-per-path 1 keeps only the tail state of each warmed prompt;
# otherwise per-chunk interior states (N / 32K per prompt) overflow the small pool and
# LRU evicts other prompts' tail states, so their prefix hit drops to 0.
# None of these change the decode forward path.
set -uo pipefail

REPO=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
: "${CONFIGS:?set CONFIGS}"
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
export MAMBA_RADIX_CACHE_STRATEGY=${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=${SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK:-1}

for cfg in $CONFIGS; do
    IFS=: read -r n slots bss <<< "$cfg"
    echo "[$(date +%H:%M:%S)] config N=$n slots=$slots B=$bss"
    RUN_NAME="${RUN_PREFIX:-decbatch_ext}_n$n" \
        EXTRA_SERVER_ARGS="--max-mamba-cache-size $slots --mamba-max-states-per-path 1 --disable-prefill-cuda-graph ${EXTRA_SERVER_ARGS:-}" \
        BENCH_ARGS="--lens $n --bs-list ${bss//\// } --reps ${REPS:-3}" \
        bash "$REPO/loadgen/decode_batching.sh"
    echo "[$(date +%H:%M:%S)] config N=$n exit=$?"
done
