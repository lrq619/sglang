#!/usr/bin/env bash
# Run several sweep.sh instances side by side on one node, one per GPU group.
#
# GPUs are split into groups of GPUS_PER_GROUP; rates are dealt round-robin to
# the groups, so group g runs rates g, g+G, g+2G, ... one after another, each
# with freshly launched servers. Every run is a flat sibling under LOG_ROOT
# (.claude/rules/experiment-logs-layout.md), all sharing one SWEEP_TAG prefix.
#
# Usage (on a GPU node):
#   RATES="8 8.5 9 9.5 10 10.5 11 11.5" TRACE=data/arrivals_r%r_d60.jsonl \
#       DURATION=3600 bash loadgen/sweep_multi.sh
#
# Env knobs:
#   RATES           arrival rates in sessions/min (required)
#   GPUS_PER_GROUP  GPUs (= servers) per run (default 2)
#   PORT_STRIDE     port offset between groups (default 10; group g uses 30000 + g*stride)
#   Plus every knob of sweep.sh / dry_run.sh / launch_servers.sh.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${RATES:?set RATES, e.g. RATES=\"8 9 10\"}"
GPUS_PER_GROUP=${GPUS_PER_GROUP:-2}
PORT_STRIDE=${PORT_STRIDE:-10}
export LOG_ROOT=${LOG_ROOT:-$HOME/scratch/logs}
export SWEEP_TAG=${SWEEP_TAG:-sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_ROOT/$SWEEP_TAG.log"; }

mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
num_groups=$((${#ALL_GPUS[@]} / GPUS_PER_GROUP))
if [ "$num_groups" -eq 0 ]; then
    log "need at least $GPUS_PER_GROUP GPUs, found ${#ALL_GPUS[@]}"
    exit 1
fi
read -ra rates <<< "$RATES"

log "sweep_multi $SWEEP_TAG: ${#rates[@]} runs on $num_groups groups of $GPUS_PER_GROUP GPUs"
pids=()
for g in $(seq 0 $((num_groups - 1))); do
    group_gpus=$(IFS=,; echo "${ALL_GPUS[*]:$((g * GPUS_PER_GROUP)):$GPUS_PER_GROUP}")
    group_rates=()
    for i in $(seq "$g" "$num_groups" $((${#rates[@]} - 1))); do
        group_rates+=("${rates[$i]}")
    done
    [ ${#group_rates[@]} -eq 0 ] && continue
    log "group $g: gpus=$group_gpus port=$((30000 + g * PORT_STRIDE)) rates=[${group_rates[*]}]"
    CUDA_VISIBLE_DEVICES=$group_gpus BASE_PORT=$((30000 + g * PORT_STRIDE)) \
        RATES="${group_rates[*]}" bash "$REPO/loadgen/sweep.sh" \
        > "$LOG_ROOT/${SWEEP_TAG}_group$g.out" 2>&1 &
    pids+=($!)
    # Staggered so the groups do not all hit the page-cache preload at once.
    sleep 20
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
log "sweep_multi done (rc=$rc)"
exit $rc
