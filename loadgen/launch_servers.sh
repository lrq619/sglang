#!/usr/bin/env bash
# Launch one single-GPU SGLang server per available GPU and print the endpoints
# as a comma-separated list (the format loadgen.py --endpoints expects).
#
# Recipe: docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx, H200 single-GPU cell.
# Divergences from the cookbook cell:
#   --context-length 262144                     pinned to the native maximum
#   --enable-metrics                            loadgen reads queue/prefill/decode times
#   --mamba-radix-cache-strategy extra_buffer   prefix hits at ctx + output (REPLAY.md 7.1)
#   --mamba-full-memory-ratio 1.3               cookbook formula 23460/L at L ~18K, the mean
#                                               running-request length measured on hermes r12;
#                                               the default 0.9 capped concurrency at 44
# CUDA graph capture and the radix (prefix) cache are both on by default; this
# script deliberately passes neither --disable-cuda-graph nor --disable-radix-cache.
#
# Usage:
#   bash loadgen/launch_servers.sh
#   MODEL=Qwen/Qwen3.8-27B-FP8 BASE_PORT=31000 bash loadgen/launch_servers.sh
#
# Env knobs:
#   MODEL            checkpoint path or HF id (default: project copy of Qwen3.8-27B, BF16)
#   PYTHON           interpreter with sglang  (default: <repo>/.venv/bin/python)
#   CONTEXT_LENGTH   max tokens per request   (default 262144, the native max)
#   MAMBA_FULL_MEMORY_RATIO  mamba state pool : KV pool memory (default 1.3)
#   BASE_PORT        port of the first server (default 30000; server i uses +i)
#   HOST_IP          address put in the printed endpoints (default: first `hostname -I`)
#   MAX_USED_MIB     a GPU counts as available if its used memory is below this
#   LOG_DIR          per-server logs, pids and endpoints.txt (default logs/servers)
#   READY_TIMEOUT    seconds to wait for /health on each server (default 1800)
#
# Each server runs in its own process group; LOG_DIR/pids lists the group ids,
# so `kill -- -<pgid>` stops a server together with its scheduler subprocesses.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-/data/projects/12004493/models/Qwen3.8-27B}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO:-1.3}
BASE_PORT=${BASE_PORT:-30000}
MAX_USED_MIB=${MAX_USED_MIB:-1024}
LOG_DIR=${LOG_DIR:-logs/servers}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
HOST_IP=${HOST_IP:-$(hostname -I | awk '{print $1}')}

mkdir -p "$LOG_DIR"

# Idle GPUs by UUID, restricted to CUDA_VISIBLE_DEVICES (indices or UUIDs) when set.
mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v max="$MAX_USED_MIB" -v vis="${CUDA_VISIBLE_DEVICES:-}" '
        BEGIN { n = split(vis, a, ","); for (i = 1; i <= n; i++) allow[a[i]] = 1 }
        $3 < max && (n == 0 || ($1 in allow) || ($2 in allow)) { print $2 }'
)
if [ ${#GPUS[@]} -eq 0 ]; then
    echo "No available GPUs (used memory < ${MAX_USED_MIB} MiB)" >&2
    exit 1
fi
echo "Launching ${#GPUS[@]} server(s) on GPU(s): ${GPUS[*]}" >&2
printf '%s\n' "${GPUS[@]}" > "$LOG_DIR/gpus"

PIDS=()
ENDPOINTS=()
LOGS=()
: > "$LOG_DIR/pids"
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    port=$((BASE_PORT + i))
    log="$LOG_DIR/server${i}_port${port}.log"
    echo "gpu=$gpu port=$port" > "$log"
    CUDA_VISIBLE_DEVICES=$gpu setsid "$PYTHON" -m sglang.launch_server \
        --model-path "$MODEL" \
        --trust-remote-code \
        --context-length "$CONTEXT_LENGTH" \
        --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static 0.85 \
        --attention-backend flashinfer \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        --mamba-radix-cache-strategy extra_buffer \
        --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-metrics \
        --host 0.0.0.0 \
        --port "$port" \
        --nccl-port $((port + 1000)) \
        >> "$log" 2>&1 < /dev/null &
    PIDS+=($!)
    echo $! >> "$LOG_DIR/pids"
    ENDPOINTS+=("http://${HOST_IP}:${port}")
    LOGS+=("$log")
done

# Wait until every server answers /health (or one of them dies).
deadline=$((SECONDS + READY_TIMEOUT))
for i in "${!ENDPOINTS[@]}"; do
    ep=${ENDPOINTS[$i]}
    until curl -sf "$ep/health" > /dev/null; do
        if ! kill -0 "${PIDS[$i]}" 2> /dev/null; then
            echo "Server $ep exited; see ${LOGS[$i]}" >&2
            exit 1
        fi
        if [ $SECONDS -ge $deadline ]; then
            echo "Timed out waiting for $ep" >&2
            exit 1
        fi
        sleep 10
    done
    echo "ready: $ep" >&2
done

# stdout: just the endpoint list, so callers can do EP=$(bash launch_servers.sh)
(IFS=,; echo "${ENDPOINTS[*]}") | tee "$LOG_DIR/endpoints.txt"
