"""Nạp cấu hình LifeMate từ config.toml và các biến môi trường.

Tham khảo cấu hình
==================

Giá trị mặc định nằm trong config.toml. Đường dẫn tương đối tới dữ liệu, kết
quả, model và file eval IDs được tính từ thư mục gốc repository. Các biến môi
trường dưới đây ghi đè giá trị TOML tương ứng.

[paths]
  data_root (mặc định /data, ZAC_DATA_ROOT)
    Thư mục chứa các split train, public_test và private_test.
  result_dir (mặc định /result, ZAC_RESULT_DIR)
    Nơi ghi file dự đoán JSONL và nơi evaluator đọc kết quả mặc định.
  gipformer_model (mặc định saved_models/asr/gipformer1.5-zac-ft06,
                   ZAC_GIPFORMER_MODEL)
    Thư mục chứa model ASR Gipformer.
  llm_model (mặc định saved_models/llm/Qwen3.5-4B, ZAC_LLM_PATH)
    Thư mục model Qwen cục bộ được vLLM phục vụ.
  llm_lora_adapter (mặc định rỗng, ZAC_LLM_LORA_ADAPTER)
    Thư mục LoRA tùy chọn áp dụng trên model Qwen. Để trống để dùng model gốc.

[vllm]
  model_name (mặc định qwen3.5:4b, MODEL_NAME)
    Tên model vLLM công bố và pipeline gửi trong các request.
  host (mặc định 127.0.0.1, VLLM_HOST) và port (mặc định 8000, VLLM_PORT)
    Địa chỉ bind và cổng của server vLLM tương thích OpenAI.
  max_model_len (mặc định 16384, MAX_MODEL_LEN)
    Độ dài context tối đa khi khởi động vLLM.
  gpu_memory_utilization (mặc định 0.85, GPU_MEMORY_UTILIZATION)
    Tỷ lệ bộ nhớ GPU dành cho vLLM.
  max_wait_seconds (mặc định 600, VLLM_MAX_WAIT)
    Thời gian tối đa launcher chờ health endpoint của vLLM.

[prediction]
  split (mặc định public_test)
    Split dùng để dự đoán hoặc đánh giá. Giá trị hỗ trợ là train, public_test
    và private_test. Không có biến môi trường ghi đè.
  eval_ids (mặc định rỗng, ZAC_EVAL_IDS)
    File text tùy chọn, mỗi dòng chứa một ID train. Giới hạn lần chạy train
    vào các ID này và buộc pipeline dùng ASR thay transcript có sẵn. Để trống
    khi chạy split khác hoặc chạy toàn bộ train.
  submission_file_name (mặc định submission.jsonl, SUBMISSION_FILE_NAME)
    Tên file JSONL dự đoán trong result_dir. predict.sh chấp nhận submission.jsonl
    hoặc tên bắt đầu bằng submission_ và kết thúc bằng .jsonl.
  agent_thinking_mode (mặc định false, AGENT_THINKING_MODE)
    Bật chế độ thinking của Qwen khi giá trị là true.
  vllm_url (mặc định rỗng, VLLM_URL)
    Endpoint Chat Completions. Nếu để trống, URL được tạo từ vllm.host và
    vllm.port.

[jupyter]
  host (mặc định 0.0.0.0, JUPYTER_HOST) và port (mặc định 9777, JUPYTER_PORT)
    Địa chỉ bind và cổng của JupyterLab.
  output_file_name (mặc định jupyter_submission.jsonl,
                    JUPYTER_OUTPUT_FILE_NAME)
    Tên file dự đoán notebook trong result_dir. Notebook ghi đè file này khi chạy.

[finetune]
  output_dir và resume_from_checkpoint: thư mục adapter và checkpoint tùy chọn.
  sample_limit = 0 dùng toàn bộ train; các khóa còn lại điều khiển LoRA, loss,
  batch, loss_chunk_size, lịch học, validation và early stopping. Thinking dùng cùng
  prediction.agent_thinking_mode để khớp định dạng inference.

start_jupyter.sh đọc thêm JUPYTER_TOKEN. Mặc định token là zac2026. Có thể đặt
biến này để dùng token Jupyter khác.
"""

from __future__ import annotations

import argparse
import base64
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"

ENVIRONMENT_KEYS = {
    "ZAC_DATA_ROOT": "paths.data_root",
    "ZAC_RESULT_DIR": "paths.result_dir",
    "ZAC_GIPFORMER_MODEL": "paths.gipformer_model",
    "ZAC_LLM_PATH": "paths.llm_model",
    "ZAC_LLM_LORA_ADAPTER": "paths.llm_lora_adapter",
    "MODEL_NAME": "vllm.model_name",
    "VLLM_HOST": "vllm.host",
    "VLLM_PORT": "vllm.port",
    "MAX_MODEL_LEN": "vllm.max_model_len",
    "GPU_MEMORY_UTILIZATION": "vllm.gpu_memory_utilization",
    "VLLM_MAX_WAIT": "vllm.max_wait_seconds",
    "ZAC_EVAL_IDS": "prediction.eval_ids",
    "SUBMISSION_FILE_NAME": "prediction.submission_file_name",
    "AGENT_THINKING_MODE": "prediction.agent_thinking_mode",
    "VLLM_URL": "prediction.vllm_url",
    "JUPYTER_HOST": "jupyter.host",
    "JUPYTER_PORT": "jupyter.port",
    "JUPYTER_OUTPUT_FILE_NAME": "jupyter.output_file_name",
}


def _set_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    section, key = dotted_path.split(".", maxsplit=1)
    config[section][key] = value


def _parse_environment_value(raw_value: str, current_value: Any) -> Any:
    if isinstance(current_value, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    return raw_value


def _resolve_path(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def load_config(
    config_path: str | Path = CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load TOML settings, apply environment overrides, and resolve paths."""
    path = Path(config_path).expanduser().resolve()
    root = path.parent
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)

    config.setdefault("paths", {}).setdefault("llm_lora_adapter", "")
    finetune = config.setdefault("finetune", {})
    finetune.setdefault(
        "output_dir",
        config["paths"]["llm_lora_adapter"] or "saved_models/qwen35_4b_zaloai_lora",
    )
    finetune.setdefault("resume_from_checkpoint", "")

    environment = os.environ if environ is None else environ
    for environment_key, dotted_path in ENVIRONMENT_KEYS.items():
        if environment_key not in environment:
            continue
        section, key = dotted_path.split(".", maxsplit=1)
        current_value = config[section][key]
        _set_path(
            config,
            dotted_path,
            _parse_environment_value(environment[environment_key], current_value),
        )

    for key in ("data_root", "result_dir", "gipformer_model", "llm_model"):
        config["paths"][key] = _resolve_path(config["paths"][key], root)

    if config["paths"]["llm_lora_adapter"]:
        config["paths"]["llm_lora_adapter"] = _resolve_path(
            config["paths"]["llm_lora_adapter"], root
        )

    config["finetune"]["output_dir"] = _resolve_path(
        config["finetune"]["output_dir"], root
    )
    if config["finetune"]["resume_from_checkpoint"]:
        config["finetune"]["resume_from_checkpoint"] = _resolve_path(
            config["finetune"]["resume_from_checkpoint"], root
        )

    if config["prediction"]["eval_ids"]:
        config["prediction"]["eval_ids"] = _resolve_path(
            config["prediction"]["eval_ids"], root
        )

    if not config["prediction"]["vllm_url"]:
        host = config["vllm"]["host"]
        port = config["vllm"]["port"]
        config["prediction"]["vllm_url"] = (
            f"http://{host}:{port}/v1/chat/completions"
        )

    return config


def _shell_values(config: dict[str, Any]) -> dict[str, str]:
    values = {
        "ZAC_DATA_ROOT": config["paths"]["data_root"],
        "ZAC_RESULT_DIR": config["paths"]["result_dir"],
        "ZAC_GIPFORMER_MODEL": config["paths"]["gipformer_model"],
        "ZAC_LLM_PATH": config["paths"]["llm_model"],
        "ZAC_LLM_LORA_ADAPTER": config["paths"]["llm_lora_adapter"],
        "MODEL_NAME": config["vllm"]["model_name"],
        "VLLM_HOST": config["vllm"]["host"],
        "VLLM_PORT": config["vllm"]["port"],
        "MAX_MODEL_LEN": config["vllm"]["max_model_len"],
        "GPU_MEMORY_UTILIZATION": config["vllm"]["gpu_memory_utilization"],
        "VLLM_MAX_WAIT": config["vllm"]["max_wait_seconds"],
        "ZAC_EVAL_IDS": config["prediction"]["eval_ids"],
        "SUBMISSION_FILE_NAME": config["prediction"]["submission_file_name"],
        "AGENT_THINKING_MODE": str(config["prediction"]["agent_thinking_mode"]).lower(),
        "VLLM_URL": config["prediction"]["vllm_url"],
        "JUPYTER_HOST": config["jupyter"]["host"],
        "JUPYTER_PORT": config["jupyter"]["port"],
        "JUPYTER_OUTPUT_FILE_NAME": config["jupyter"]["output_file_name"],
    }
    return {key: str(value) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nạp cấu hình LifeMate cho các launcher."
    )
    parser.add_argument("--shell", action="store_true", help="Output base64 key-value pairs")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    if not args.shell:
        parser.error("use --shell when invoking the config loader")

    for key, value in _shell_values(load_config(args.config)).items():
        encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
        print(f"{key}={encoded}")


if __name__ == "__main__":
    main()
