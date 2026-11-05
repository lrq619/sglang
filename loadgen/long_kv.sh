#!/usr/bin/env bash
#PBS -N longkv
#PBS -P 12004493
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
# Long-KV decode interference experiment on one GPU (see loadgen/README_long_kv.md):
# launch one server with launch_servers.sh defaults, run bench_long_kv.py, plot,
# then stop the server. Output: LOG_ROOT/<RUN_NAME>_<timestamp>/ (flat, one run).
#
# Usage:
#   bash loadgen/long_kv.sh                                   # on a GPU node
#   qsub -v BENCH_ARGS="--reps 1 --stress-lens 128 8192" loadgen/long_kv.sh
#
# Env knobs:
#   RUN_NAME    logdir prefix (default longkv)
#   PLOT_PYTHON interpreter with matplotlib (default ~/scratch/mplenv/bin/python)
#   BENCH_ARGS  extra bench_long_kv.py flags (word-split)
#   Plus every knob of launch_servers.sh (MODEL, MEM_FRACTION_STATIC, ...).
set -uo pipefail

REPO=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
LOG_ROOT=${LOG_ROOT:-$HOME/scratch/logs}
MODEL=${MODEL:-/data/projects/12004493/models/Qwen3.8-27B}
export PYTHON MODEL

if [ -z "${CUDA_HOME:-}" ]; then
    set +u
    source /etc/profile > /dev/null 2>&1
    module load cuda/13.0
    set -u
fi

LOGDIR="$LOG_ROOT/${RUN_NAME:-longkv}_$(date +%Y%m%d_%H%M%S)"
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
    echo "bench_args: ${BENCH_ARGS:-}"
    echo "extra_server_args: ${EXTRA_SERVER_ARGS:-}"
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
log "launching server"
EP=$(HOST_IP=127.0.0.1 LOG_DIR="$LOGDIR/servers" bash "$REPO/loadgen/launch_servers.sh")
[ -n "$EP" ] || { log "server launch failed"; exit 1; }
EP=${EP%%,*}
log "endpoint: $EP"
grep -hE "max_total_num_tokens|max_running_requests is capped|KV Cache is allocated|Mamba Cache is allocated" \
    "$LOGDIR"/servers/server*.log | tee -a "$LOGDIR/run_info.txt"

log "bench start"
# shellcheck disable=SC2086
"$PYTHON" -u "$REPO/loadgen/bench_long_kv.py" --url "$EP" --outdir "$LOGDIR" ${BENCH_ARGS:-}
log "bench exit=$?"
log "server log: $(grep -c 'Retract requests' "$LOGDIR"/servers/server*.log) retraction events" |
    tee -a "$LOGDIR/run_info.txt"
"${PLOT_PYTHON:-$HOME/scratch/mplenv/bin/python}" "$REPO/loadgen/plot_long_kv.py" "$LOGDIR"
log "done: $LOGDIR"
