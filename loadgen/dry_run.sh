#!/usr/bin/env bash
# End-to-end loadgen dry run on the current node:
#   0. create a timestamped logdir under ~/scratch/logs
#   1. preload the checkpoint into page cache, then launch one SGLang server
#      per available GPU and wait until all are ready
#   2. run loadgen for DURATION seconds, sampling server gauges into metrics.csv
#   3. tear down loadgen, the poller and every server; kill leftover GPU processes
#   4. verify that the loadgen and metrics output is complete in the logdir
#
# Usage (on a GPU node):
#   bash loadgen/dry_run.sh
#   CORPUS=hermes_clean.jsonl DURATION=600 RATE_PER_MIN=10 bash loadgen/dry_run.sh
#   CORPUS=data/hermes_clean.jsonl ARRIVALS=data/arrivals_r3_d3.jsonl bash loadgen/dry_run.sh
#
# Env knobs:
#   CORPUS        prepare_corpus.py output; when unset, a synthetic corpus is generated
#   DURATION      loadgen run length in seconds (default 180)
#   ARRIVALS      make_arrivals.py output; when unset, generated from RATE_PER_MIN
#   RATE_PER_MIN  session arrival rate (default 6)
#   LOG_ROOT      parent of the logdir (default ~/scratch/logs)
#   RUN_NAME      logdir prefix, LOG_ROOT/<RUN_NAME>_<timestamp> (default dryrun)
#   METRICS_INTERVAL  /metrics sampling period in seconds (default 1)
#   PRELOAD_JOBS  parallel readers for the page-cache preload (default 8; 0 skips it)
#   LOADGEN_EXTRA_ARGS  extra loadgen.py flags (word-split), e.g. "--pin"
#   PREDICTOR_COSTS  StepCosts JSON for the queueing predictor sidecar; profiled on the
#                    first (idle) server when the file is missing, then the sidecar runs
#                    alongside loadgen and logs first-sight predictions to predictions.csv
#   PREDICTOR_PORT   sidecar port (default 32000; 31000+ are the servers' NCCL ports)
#   Plus every knob of launch_servers.sh (MODEL, CONTEXT_LENGTH, BASE_PORT, ...).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
DURATION=${DURATION:-180}
RATE_PER_MIN=${RATE_PER_MIN:-6}
LOG_ROOT=${LOG_ROOT:-$HOME/scratch/logs}
METRICS_INTERVAL=${METRICS_INTERVAL:-1}
PRELOAD_JOBS=${PRELOAD_JOBS:-8}
MODEL=${MODEL:-/data/projects/12004493/models/Qwen3.8-27B}
export PYTHON MODEL

# Non-interactive ssh sessions lack CUDA_HOME, which DeepGEMM asserts on at import.
if [ -z "${CUDA_HOME:-}" ]; then
    set +u
    source /etc/profile > /dev/null 2>&1
    module load cuda/13.0
    set -u
fi

# ===== 0. logdir =====
LOGDIR="$LOG_ROOT/${RUN_NAME:-dryrun}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR/servers"
exec > >(tee -a "$LOGDIR/dry_run.log") 2>&1
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "logdir: $LOGDIR"
{
    echo "host: $(hostname)"
    echo "start: $(date -Is)"
    echo "git: $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
    echo "duration_s: $DURATION"
    echo "rate_per_min: $RATE_PER_MIN"
    echo "model: $MODEL"
    echo "mamba_full_memory_ratio: ${MAMBA_FULL_MEMORY_RATIO:-1.3}"
    echo "corpus: ${CORPUS:-synthetic}"
    echo "arrivals: ${ARRIVALS:-generated}"
    echo "extra_server_args: ${EXTRA_SERVER_ARGS:-}"
    echo "loadgen_extra_args: ${LOADGEN_EXTRA_ARGS:-}"
    echo "mem_fraction_static: ${MEM_FRACTION_STATIC:-0.85}"
    echo "predictor_costs: ${PREDICTOR_COSTS:-}"
    echo "PBS_JOBID: ${PBS_JOBID:-}"
} > "$LOGDIR/run_info.txt"
nvidia-smi > "$LOGDIR/nvidia-smi.before.txt" 2>&1

LOADGEN_PID=
POLLER_PID=
PREDICTOR_PID=
stop_poller() {
    if [ -n "$POLLER_PID" ] && kill -0 "$POLLER_PID" 2> /dev/null; then
        kill -TERM "$POLLER_PID" 2> /dev/null
        wait "$POLLER_PID" 2> /dev/null
    fi
    POLLER_PID=
}
cleanup() {
    trap - EXIT INT TERM
    log "cleanup: stopping loadgen, poller and servers"
    if [ -n "$LOADGEN_PID" ] && kill -0 "$LOADGEN_PID" 2> /dev/null; then
        kill -INT "$LOADGEN_PID" 2> /dev/null
        sleep 5
        kill -KILL "$LOADGEN_PID" 2> /dev/null
    fi
    stop_poller
    if [ -n "$PREDICTOR_PID" ] && kill -0 "$PREDICTOR_PID" 2> /dev/null; then
        kill -TERM "$PREDICTOR_PID" 2> /dev/null
    fi
    # Scoped to this run's process groups and GPUs: other runs may share the node.
    local pgid
    for pgid in $(cat "$LOGDIR/servers/pids" 2> /dev/null); do
        kill -TERM -- "-$pgid" 2> /dev/null
    done
    for _ in $(seq 30); do
        _own_pgids_alive || break
        sleep 1
    done
    for pgid in $(cat "$LOGDIR/servers/pids" 2> /dev/null); do
        kill -KILL -- "-$pgid" 2> /dev/null
    done
    # Last resort: anything of ours still holding one of this run's GPUs.
    local pid
    for pid in $(_own_gpu_pids); do
        if [ "$(ps -o user= -p "$pid" 2> /dev/null)" = "$USER" ]; then
            log "cleanup: kill -9 leftover GPU process $pid ($(ps -o comm= -p "$pid"))"
            kill -KILL "$pid" 2> /dev/null
        fi
    done
    sleep 3
    nvidia-smi > "$LOGDIR/nvidia-smi.after.txt" 2>&1
    log "cleanup: processes left on this run's GPUs: $(_own_gpu_pids | wc -w)"
}
_own_pgids_alive() {
    local pgid
    for pgid in $(cat "$LOGDIR/servers/pids" 2> /dev/null); do
        pgrep -g "$pgid" > /dev/null && return 0
    done
    return 1
}
_own_gpu_pids() {
    [ -s "$LOGDIR/servers/gpus" ] || return 0
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2> /dev/null |
        tr -d ' ' | awk -F, 'NR == FNR { own[$1] = 1; next } ($2 in own) { print $1 }' \
            "$LOGDIR/servers/gpus" -
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ===== 1. servers =====
# safetensors are mmap-loaded, and page faults on Lustre are slow (~70 MB/s per
# server measured); parallel sequential reads first fill the page cache at GB/s.
if [ "$PRELOAD_JOBS" -gt 0 ] && [ -d "$MODEL" ]; then
    log "preloading $MODEL into page cache ($PRELOAD_JOBS readers)"
    t0=$SECONDS
    find "$MODEL" -maxdepth 1 -type f -name '*.safetensors' -print0 |
        xargs -0 -P "$PRELOAD_JOBS" -I{} dd if={} of=/dev/null bs=16M status=none
    log "preload done: $(du -sh "$MODEL" | cut -f1) in $((SECONDS - t0))s"
fi
log "launching servers"
ENDPOINTS=$(HOST_IP=127.0.0.1 LOG_DIR="$LOGDIR/servers" bash "$REPO/loadgen/launch_servers.sh")
if [ -z "$ENDPOINTS" ]; then
    log "server launch failed; see $LOGDIR/servers/"
    exit 1
fi
log "endpoints: $ENDPOINTS"
for f in "$LOGDIR"/servers/server*_port*.log; do
    log "$(basename "$f"): $(grep -o 'load_weight=[0-9.]*' "$f" | tail -1)," \
        "$(grep -o 'cuda_graph={prefill=[0-9.]*, decode=[0-9.]*' "$f" | tail -1)"
done

# ===== 1b. queueing predictor sidecar =====
if [ -n "${PREDICTOR_COSTS:-}" ]; then
    if [ ! -s "$PREDICTOR_COSTS" ]; then
        log "profiling step costs on ${ENDPOINTS%%,*} -> $PREDICTOR_COSTS"
        (cd "$REPO" && "$PYTHON" -u -m controller.queuing_predictor.profile_costs \
            --url "${ENDPOINTS%%,*}" --output "$PREDICTOR_COSTS" --model-tag "$MODEL") \
            > "$LOGDIR/profile_costs.log" 2>&1 || { log "profiling failed"; exit 1; }
        tail -n 1 "$LOGDIR/profile_costs.log"
    fi
    cp "$PREDICTOR_COSTS" "$LOGDIR/step_costs.json"
    (cd "$REPO" && exec "$PYTHON" -u -m controller.queuing_predictor.server \
        --instances "$ENDPOINTS" --costs "$PREDICTOR_COSTS" \
        --port "${PREDICTOR_PORT:-32000}" --poll-interval 0.25 \
        --shadow-log "$LOGDIR/predictions.csv" \
        --dump-snapshots "$LOGDIR/snapshots.jsonl.gz" \
        --log-predicts "$LOGDIR/predict_log.jsonl" \
        --pin-log "$LOGDIR/pin_decisions.csv") > "$LOGDIR/predictor.log" 2>&1 &
    PREDICTOR_PID=$!
    until curl -sf "http://127.0.0.1:${PREDICTOR_PORT:-32000}/health" > /dev/null; do
        kill -0 "$PREDICTOR_PID" 2> /dev/null || { log "predictor sidecar died"; exit 1; }
        sleep 1
    done
    LOADGEN_EXTRA_ARGS="${LOADGEN_EXTRA_ARGS:-} --predictor http://127.0.0.1:${PREDICTOR_PORT:-32000}"
    log "predictor sidecar on :${PREDICTOR_PORT:-32000} -> predictions.csv, turn_predicted_queuing_latency"
fi

# ===== 2. loadgen =====
if [ -z "${CORPUS:-}" ]; then
    CORPUS="$LOGDIR/corpus_synthetic.jsonl"
    "$PYTHON" "$REPO/loadgen/make_synthetic_corpus.py" -o "$CORPUS"
fi
if [ -n "${ARRIVALS:-}" ]; then
    cp "$ARRIVALS" "$LOGDIR/arrivals.jsonl"
else
    "$PYTHON" "$REPO/loadgen/make_arrivals.py" "$CORPUS" -o "$LOGDIR/arrivals.jsonl" \
        --rate-per-min "$RATE_PER_MIN" --duration-min "$(awk -v d="$DURATION" 'BEGIN { print d / 60 }')"
fi

"$PYTHON" -u "$REPO/loadgen/metrics_poller.py" \
    --endpoints "$ENDPOINTS" \
    --output "$LOGDIR/metrics.csv" \
    --interval "$METRICS_INTERVAL" \
    > "$LOGDIR/metrics_poller.log" 2>&1 &
POLLER_PID=$!
log "metrics poller: every ${METRICS_INTERVAL}s -> metrics.csv"

log "loadgen: ${DURATION}s"
"$PYTHON" -u "$REPO/loadgen/loadgen.py" \
    --corpus "$CORPUS" \
    --arrivals "$LOGDIR/arrivals.jsonl" \
    --endpoints "$ENDPOINTS" \
    --output "$LOGDIR/loadgen.csv" \
    --sessions-output "$LOGDIR/sessions.csv" \
    --duration "$DURATION" \
    --progress-interval 30 \
    ${LOADGEN_EXTRA_ARGS:-} \
    > "$LOGDIR/loadgen.log" 2>&1 &
LOADGEN_PID=$!
# Grace period past --duration for loadgen to cancel in-flight requests and exit.
for _ in $(seq $((DURATION + 60))); do
    kill -0 "$LOADGEN_PID" 2> /dev/null || break
    sleep 1
done
wait "$LOADGEN_PID" 2> /dev/null
LOADGEN_RC=$?
log "loadgen exited with $LOADGEN_RC"

# ===== 3. teardown =====
# A few idle samples after the cut show the servers draining.
sleep 5
stop_poller
for i in $(seq 0 $(($(wc -l < "$LOGDIR/servers/pids") - 1))); do
    curl -s "http://127.0.0.1:$((${BASE_PORT:-30000} + i))/metrics" \
        > "$LOGDIR/servers/server${i}_metrics.prom" 2> /dev/null
done
cleanup

# ===== 4. verify =====
sync
read -r rows ok < <("$PYTHON" -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(len(rows), sum(r["status"] == "ok" for r in rows))
' "$LOGDIR/loadgen.csv" 2> /dev/null || echo "0 0")
log "loadgen.csv: $rows rows ($ok ok)"
read -r m_rows m_ok < <("$PYTHON" -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(len(rows), sum(not r["error"] and r["token_usage"] != "" for r in rows))
' "$LOGDIR/metrics.csv" 2> /dev/null || echo "0 0")
log "metrics.csv: $m_rows samples ($m_ok with gauges)"
read -r s_rows s_arrived < <("$PYTHON" -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(len(rows), sum(r["status"] != "not_arrived" for r in rows))
' "$LOGDIR/sessions.csv" 2> /dev/null || echo "0 0")
log "sessions.csv: $s_rows sessions ($s_arrived arrived)"
"$PYTHON" "$REPO/loadgen/analyze_jct.py" "$LOGDIR" || log "analyze_jct failed"
tail -n 3 "$LOGDIR/loadgen.log"
echo "end: $(date -Is)" >> "$LOGDIR/run_info.txt"
ls -la "$LOGDIR"
if [ "$LOADGEN_RC" -ne 0 ] || [ "$ok" -eq 0 ] || [ "$m_ok" -eq 0 ] || [ "$s_arrived" -eq 0 ] \
    || ! grep -q "^finished:" "$LOGDIR/loadgen.log"; then
    log "DRY RUN FAILED"
    exit 1
fi
log "DRY RUN OK: $LOGDIR"
