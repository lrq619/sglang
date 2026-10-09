#!/usr/bin/env bash
# Run dry_run.sh once per arrival rate, each with freshly launched servers.
#
# Usage (on a GPU node):
#   bash loadgen/sweep.sh
#   RATES="6 12" DURATION=600 bash loadgen/sweep.sh
#
# Env knobs:
#   CORPUS    prepare_corpus.py output (default <repo>/data/hermes_clean.jsonl)
#   RATES     arrival rates in sessions/min (default "3 6 12 24")
#   TRACE     arrivals path template, %r = rate (default <repo>/data/arrivals_r%r_d30.jsonl)
#   DURATION  loadgen run length per rate in seconds (default 1800)
#   Plus every knob of dry_run.sh / launch_servers.sh.
#   SWEEP_TAG name prefix shared by this sweep's runs (default sweep_<timestamp>)
# Runs are flat siblings in LOG_ROOT (no per-sweep folder, see
# .claude/rules/experiment-logs-layout.md): LOG_ROOT/<SWEEP_TAG>_r<rate>_<timestamp>/,
# with the sweep's own log at LOG_ROOT/<SWEEP_TAG>.log.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CORPUS=${CORPUS:-$REPO/data/hermes_clean.jsonl}
export DURATION=${DURATION:-1800}
RATES=${RATES:-"3 6 12 24"}
TRACE=${TRACE:-$REPO/data/arrivals_r%r_d30.jsonl}
export LOG_ROOT=${LOG_ROOT:-$HOME/scratch/logs}
SWEEP_TAG=${SWEEP_TAG:-sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_ROOT/$SWEEP_TAG.log"; }

# Only this sweep's GPUs (CUDA_VISIBLE_DEVICES, when set) must be idle;
# other sweeps may share the node.
wait_gpus_idle() {
    local busy
    for _ in $(seq 60); do
        busy=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
            awk -F', ' -v vis="${CUDA_VISIBLE_DEVICES:-}" '
                BEGIN { n = split(vis, a, ","); for (i = 1; i <= n; i++) allow[a[i]] = 1 }
                $3 >= 1024 && (n == 0 || ($1 in allow) || ($2 in allow))' | wc -l)
        [ "$busy" -eq 0 ] && return 0
        sleep 2
    done
    return 1
}

log "sweep $SWEEP_TAG: rates=[$RATES] duration=${DURATION}s gpus=${CUDA_VISIBLE_DEVICES:-all} port=${BASE_PORT:-30000} corpus=$CORPUS -> $LOG_ROOT"
for r in $RATES; do
    trace=${TRACE//%r/$r}
    if [ ! -f "$trace" ]; then
        log "r=$r: missing $trace, skipped"
        continue
    fi
    if ! wait_gpus_idle; then
        log "r=$r: GPUs still busy after 120s, aborting sweep"
        exit 1
    fi
    log "r=$r: start ($trace)"
    ARRIVALS=$trace RUN_NAME=${SWEEP_TAG}_r$r bash "$REPO/loadgen/dry_run.sh" > /dev/null 2>&1
    rc=$?
    log "r=$r: $([ $rc -eq 0 ] && echo OK || echo "FAILED (rc=$rc)") $(ls -d "$LOG_ROOT/${SWEEP_TAG}_r${r}"_* | tail -1)"
done
log "sweep done"
