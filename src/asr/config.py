"""Cấu hình của module ASR: danh sách model và đường dẫn dữ liệu.

Model mặc định là **Gipformer 1.5 (68M, Zipformer RNN-T) đã fine-tune** trên
tập train (``gipformer_finetune``, cấu hình ft06: nhiễu trắng + SpecAugment),
giải mã beam search 4 + hotword lấy từ lịch sử hội thoại, chạy ONNX trên
CPU, rồi hậu xử lý số đọc tắt (``postprocess.py``). Trên 200 hội thoại
validation (không dùng để train): WER 6,75 % (6,86 % trước hậu xử lý;
Gipformer gốc 14,1 %, PhoWhisper-large ~25 %), nghe đúng 86 % tên riêng.
Các model còn lại được giữ trong ``MODELS`` để có thể đổi model chỉ bằng
một tham số, ví dụ
``model="phowhisper_large"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..lifemate_config import load_config


_LIFEMATE_CONFIG = load_config()

# Âm thanh của cuộc thi: WAV 16 kHz, mono, PCM 16-bit.
SAMPLE_RATE = 16000

# Thư mục gốc của repo: src/asr/config.py -> lùi 3 cấp.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thư mục chứa các split (train/, public_test/, ...). Biến môi trường
# ZAC_DATA_ROOT cho phép đổi vị trí khi dữ liệu được mount ở nơi khác.
DATA_ROOT = Path(_LIFEMATE_CONFIG["paths"]["data_root"])


# Model fine-tune được lưu ở đây (thư mục checkpoints/ đã nằm trong
# .gitignore: trọng số không được đưa lên git).
# Ghi đè bằng biến môi trường ZAC_FINETUNED_DIR nếu model nằm ở chỗ khác.
FINETUNED_DIR = Path(os.environ.get(
    "ZAC_FINETUNED_DIR",
    PROJECT_ROOT / "checkpoints" / "asr" / "phowhisper-large-zac"))


# Model Gipformer đã fine-tune, luôn resolve từ image (không tìm trong checkpoints/).
# Thư mục này chứa các file ONNX và tokenizer cần thiết. Cấu hình qua
# paths.gipformer_model trong config.toml hoặc ghi đè bằng ZAC_GIPFORMER_MODEL.
GIPFORMER_MODEL = _LIFEMATE_CONFIG["paths"]["gipformer_model"]


@dataclass(frozen=True)
class ModelSpec:
    """Mô tả một model ASR.

    Attributes:
        key: Tên ngắn gọn dùng để chọn model (``phowhisper_large``).
        model_id: Id trên Hugging Face Hub, hoặc đường dẫn thư mục chứa
            trọng số khi chạy offline.
        backend: Họ model, quyết định wrapper trong ``backends.BACKENDS``.
        language: Gợi ý ngôn ngữ theo quy ước của từng backend
            (``vietnamese`` cho Whisper, ``Vietnamese`` cho Qwen3-ASR,
            ``vi-VN`` cho Nemotron). ``None``: để model tự nhận diện.
        batch_size: Số đoạn âm thanh tối đa được giải mã cùng lúc.
        options: Tham số riêng của từng backend.
    """

    key: str
    model_id: str
    backend: str
    language: str | None = None
    batch_size: int = 8
    options: dict[str, Any] = field(default_factory=dict)


# Danh sách model đã benchmark.
_SPECS = (
    # Gipformer fine-tune: beam search 4 + hotword (điểm 1,0, chọn trên dev),
    # 4 luồng CPU.
    # Cấu hình chọn trên validation (gipformer_finetune/DECISIONS.md mục 14).
    ModelSpec("gipformer_zac", GIPFORMER_MODEL, "sherpa_onnx", batch_size=32,
              options={"decoding_method": "modified_beam_search",
                       "max_active_paths": 4, "hotwords_score": 1.0,
                       "num_threads": 4}),
    # PhoWhisper chỉ được fine-tune trên tiếng Việt: không cần gợi ý ngôn ngữ.
    ModelSpec("phowhisper_large", "vinai/PhoWhisper-large", "hf_whisper"),
    # PhoWhisper-large đã fine-tune LoRA trên tập train (src/asr/finetune).
    # Nhãn khi học có prefix <|vi|>, nên khi nhận dạng cũng chỉ định tiếng
    # Việt cho khớp.
    ModelSpec("phowhisper_large_zac", str(FINETUNED_DIR), "hf_whisper",
              language="vietnamese"),
    ModelSpec("qwen3_asr_1_7b", "Qwen/Qwen3-ASR-1.7B", "qwen_asr",
              language="Vietnamese", options={"max_new_tokens": 256}),
    ModelSpec("whisper_large_v3", "openai/whisper-large-v3", "hf_whisper",
              language="vietnamese"),
    ModelSpec("phowhisper_medium", "vinai/PhoWhisper-medium", "hf_whisper"),
    ModelSpec("whisper_large_v3_turbo", "openai/whisper-large-v3-turbo",
              "hf_whisper", language="vietnamese"),
    ModelSpec("qwen3_asr_0_6b", "Qwen/Qwen3-ASR-0.6B", "qwen_asr",
              language="Vietnamese", options={"max_new_tokens": 256}),
    # Nemotron (RNN-T) chỉ hỗ trợ batch 1 trong transformers.
    ModelSpec("nemotron_asr_0_6b", "nvidia/nemotron-3.5-asr-streaming-0.6b",
              "hf_rnnt", language="vi-VN", batch_size=1),
)

#: Tra cứu model theo ``key``.
MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}

#: Model được dùng khi người gọi không chỉ định.
DEFAULT_MODEL = "gipformer_zac"
