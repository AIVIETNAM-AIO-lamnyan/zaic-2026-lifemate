#!/bin/bash
set -e

# ==============================================================================
# SCRIPT INFERENCE CHÍNH THỨC CHO ZALO AI CHALLENGE 2026 - TRACK LIFEMATE
# Tuân thủ định dạng trang 6-7 và 10 trong Docker_Guide_ZAC2026_LifeMate.pdf
# ==============================================================================

echo "=========================================================="
echo "=== [LifeMate] BẮT ĐẦU QUÁ TRÌNH INFERENCE (PREDICT.SH) ==="
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "-> Thư mục thực thi code: $SCRIPT_DIR"

source "$SCRIPT_DIR/scripts/load_config.sh"
load_lifemate_config "$SCRIPT_DIR"

# 2. Chuẩn bị file kết quả và dừng nếu file đã tồn tại.
if [[ ! "$SUBMISSION_FILE_NAME" =~ ^submission_[A-Za-z0-9._-]+\.jsonl$ && "$SUBMISSION_FILE_NAME" != "submission.jsonl" ]]; then
    echo "[LỖI] Tên file phải bắt đầu bằng submission_ và kết thúc bằng .jsonl."
    exit 1
fi

RESULT_DIR="$ZAC_RESULT_DIR"
mkdir -p "$RESULT_DIR"
SUBMISSION_PATH="$RESULT_DIR/$SUBMISSION_FILE_NAME"
if [[ -e "$SUBMISSION_PATH" ]]; then
    echo "[LỖI] File kết quả đã tồn tại: $SUBMISSION_PATH"
    exit 1
fi

# 3. Cấu hình biến môi trường
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ZAC_RESULT_DIR="$RESULT_DIR"
export SUBMISSION_FILE_NAME

# Hàm dọn dẹp tiến trình nền khi script kết thúc
vllm_pid_file="$(mktemp)"
cleanup() {
    if [[ -s "$vllm_pid_file" ]]; then
        echo "-> Đang giải phóng tiến trình nền vLLM..."
        kill -TERM "$(cat "$vllm_pid_file")" 2>/dev/null || true
    fi
    rm -f -- "$vllm_pid_file"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 4. Khởi động vLLM (Qwen3.5-4B FP8 native safetensors). Script in thời gian load model.
LIFEMATE_VLLM_PID_FILE="$vllm_pid_file" bash "$SCRIPT_DIR/start_vllm.sh"

# 5. Chạy pipeline dự đoán chính thức
echo "-> Đang thực thi dự đoán (predict.py)..."
python "$SCRIPT_DIR/predict.py"

echo "-> Đã ghi kết quả vào $SUBMISSION_PATH"
