#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/load_config.sh"
load_lifemate_config "$SCRIPT_DIR"

pid_file="${LIFEMATE_VLLM_PID_FILE:-/tmp/vllm.pid}"
log_file=/tmp/vllm.log
health_url="http://$VLLM_HOST:$VLLM_PORT/health"

server_ready() {
    curl --fail --silent --noproxy '*' --max-time 1 "$health_url" >/dev/null
}

if server_ready; then
    echo "-> vLLM đã chạy tại $health_url"
    exit 0
fi

base_model_name="$MODEL_NAME"
lora_args=()
if [[ -n "$ZAC_LLM_LORA_ADAPTER" ]]; then
    base_model_name="${MODEL_NAME}-base"
    lora_args=(--enable-lora --lora-modules "$MODEL_NAME=$ZAC_LLM_LORA_ADAPTER")
fi

SECONDS=0
nohup vllm serve "$ZAC_LLM_PATH" \
    --host "$VLLM_HOST" --port "$VLLM_PORT" \
    --served-model-name "$base_model_name" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    "${lora_args[@]}" </dev/null >>"$log_file" 2>&1 &
server_pid=$!

stop_server() {
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    rm -f -- "$pid_file"
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$server_pid" > "$pid_file"

while kill -0 "$server_pid" 2>/dev/null; do
    if server_ready; then
        trap - EXIT INT TERM
        echo "-> vLLM sẵn sàng sau ${SECONDS}s ($health_url)"
        exit 0
    fi
    if (( SECONDS >= VLLM_MAX_WAIT )); then
        echo "[LỖI] vLLM chưa sẵn sàng sau ${VLLM_MAX_WAIT}s. Xem $log_file." >&2
        exit 1
    fi
    sleep 1
done

echo "[LỖI] vLLM đã dừng khi khởi động. Xem $log_file." >&2
exit 1
