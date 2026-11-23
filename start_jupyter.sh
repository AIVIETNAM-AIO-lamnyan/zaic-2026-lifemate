#!/bin/bash
set -e

# ==============================================================================
# SCRIPT KHỞI ĐỘNG JUPYTER LAB CHO BTC ĐÁNH GIÁ THỜI GIAN (START_JUPYTER.SH)
# Tuân thủ định dạng trang 12 mục 7 trong Docker_Guide_ZAC2026_LifeMate.pdf
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/scripts/load_config.sh"
load_lifemate_config "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-zac2026}"

# Khởi động vLLM nếu chưa chạy (predict_notebook.ipynb gọi cùng endpoint).
bash "$SCRIPT_DIR/start_vllm.sh"

cd "$SCRIPT_DIR"
echo "-> Khởi động Jupyter Lab tại cổng $JUPYTER_PORT..."
# Khởi động Jupyter Lab theo đúng cú pháp trang 11 trong Docker_Guide_ZAC2026_LifeMate.pdf
jupyter lab --port "$JUPYTER_PORT" --ip "$JUPYTER_HOST" --NotebookApp.password="$JUPYTER_TOKEN" --NotebookApp.token="$JUPYTER_TOKEN" --allow-root --no-browser
