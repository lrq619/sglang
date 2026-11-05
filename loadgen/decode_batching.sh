#!/usr/bin/env bash
#PBS -N decbatch
#PBS -P 12004493
#PBS -l select=1:ngpus=2
#PBS -l walltime=01:00:00
#PBS -j oe
# Decode batching gain vs context length: one server per GPU, bench_decode_batching.py
# splits the context lengths across them, then plot and stop the servers.
# Output: LOG_ROOT/<RUN_NAME>_<timestamp>/ (flat, one run).
#
# Usage:
#   qsub -V loadgen/decode_batching.sh
#   qsub -V -l select=1:ngpus=1 -l walltime=00:30:00 \
#       -v RUN_NAME=decbatch_smoke,BENCH_ARGS="--lens 1024 98304 --reps 1" loadgen/decode_batching.sh
#
# Env knobs: RUN_NAME (default decbatch), BENCH_ARGS, PLOT_PYTHON, and every knob of
# launch_servers.sh. Divergence from the cookbook H200 cell, memory split only:
# the mamba state pool is pinned to 112 slots (0.88 mem-fraction OOMs in prefill graph capture)
# so that 16 requests x 96K context fit in the KV pool (default pool: 957K tokens).
set -uo pipefail

REPO=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
LOG_ROOT=${LOG_ROOT:-$HOME/scratch/logs}
MODEL=${MODEL:-/data/projects/12004493/models/Qwen3.8-27B}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:---max-mamba-cache-size 112}
export PYTHON MODEL MEM_FRACTION_STATIC EXTRA_SERVER_ARGS

if [ -z "${CUDA_HOME:-}" ]; then
    set +u
    source /etc/profile > /dev/null 2>&1
    module load cuda/13.0
    set -u
fi

LOGDIR="$LOG_ROOT/${RUN_NAME:-decbatch}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR/servers"
exec > >(tee -a "$LOGDIR/run.log") 2>&1
log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    trap - EXIT INT TERM
    local pgid
    for pgid in $(cat "$LOGDIR/servers/pids" 2> /dev/null); do kill -TERM -- "-$pgid" 2> /dev/null; done
    sleep 10
    for pgid in $(cat "$LOGDIR/servers/pids" 2> /dev/null); do kill -KILL -- "-$pgid" 2> /dev/null; done
    log "cleanup done"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

{
    echo "host: $(hostname)"
    echo "start: $(date -Is)"
    echo "git: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "model: $MODEL"
    echo "mem_fraction_static: $MEM_FRACTION_STATIC"
    echo "extra_server_args: $EXTRA_SERVER_ARGS"
    echo "attention_backend: ${ATTENTION_BACKEND:-flashinfer}"
    echo "mamba_radix_cache_strategy: ${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"
    echo "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK: ${SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK:-}"
    echo "bench_args: ${BENCH_ARGS:-}"
    echo "PBS_JOBID: ${PBS_JOBID:-}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    "$PYTHON" -c "import sglang; print('sglang', sglang.__version__)"
} > "$LOGDIR/run_info.txt" 2>&1
git diff > "$LOGDIR/git_diff.patch"

if [ -d "$MODEL" ]; then
    log "preloading $MODEL into page cache"
    find "$MODEL" -maxdepth 1 -type f -name '*.safetensors' -print0 |
        xargs -0 -P 8 -I{} dd if={} of=/dev/null bs=16M status=none
fi
log "launching servers"
EP=$(HOST_IP=127.0.0.1 LOG_DIR="$LOGDIR/servers" bash "$REPO/loadgen/launch_servers.sh")
[ -n "$EP" ] || { log "server launch failed"; exit 1; }
log "endpoints: $EP"
grep -hE "max_total_num_tokens|max_running_requests is capped|KV Cache is allocated|Mamba Cache is allocated" \
    "$LOGDIR"/servers/server*.log | tee -a "$LOGDIR/run_info.txt"

log "bench start"
# shellcheck disable=SC2086
"$PYTHON" -u "$REPO/loadgen/bench_decode_batching.py" --urls "$EP" --outdir "$LOGDIR" ${BENCH_ARGS:-}
log "bench exit=$?"
log "server log: $(cat "$LOGDIR"/servers/server*.log | grep -c 'Retract requests') retraction events" |
    tee -a "$LOGDIR/run_info.txt"
"${PLOT_PYTHON:-$HOME/scratch/mplenv/bin/python}" "$REPO/loadgen/plot_decode_batching.py" "$LOGDIR"
log "done: $LOGDIR"
