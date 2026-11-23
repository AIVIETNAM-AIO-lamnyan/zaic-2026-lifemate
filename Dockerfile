# ==============================================================================
# DOCKERFILE CHÍNH THỨC NỘP BÀI ZALO AI CHALLENGE 2026 - LIFEMATE
# Base Image: CUDA 12.9 (≤ 13.2 theo guide BTC; driver 595.58.03).
# Pin tag thay vì :latest để không vô tình vượt trần CUDA của server chấm.
# vLLM 0.32.0 phục vụ Qwen3.5-4B FP8 native qua OpenAI API.
# ==============================================================================

FROM vllm/vllm-openai:v0.32.0-cu129

WORKDIR /code

# 1. Cài đặt các thư viện hệ thống cần thiết (âm thanh, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Cài đặt các gói phụ thuộc Python cho inference
COPY requirements-infer.txt /code/requirements-infer.txt
RUN pip install --no-cache-dir -r /code/requirements-infer.txt

# 3. Khắc phục torchcodec mismatch: cài wheel khớp với CUDA 12.9 của base image
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cu129 \
    'torchcodec==0.16.0+cu129'

# 4. Sao chép toàn bộ mã nguồn và trọng số mô hình offline vào /code
COPY . /code

# Cấu hình môi trường offline
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=False

# Phân quyền thực thi cho các script
RUN chmod +x /code/predict.sh /code/start_vllm.sh /code/start_jupyter.sh /code/predict.py

# Tạo thư mục kết quả mặc định
RUN mkdir -p /result

# Nhận lệnh khởi động từ hệ thống chấm
ENTRYPOINT []
CMD ["/bin/bash", "/code/predict.sh"]
