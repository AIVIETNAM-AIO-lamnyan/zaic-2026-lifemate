"""Điều phối ASR, vLLM, ToolEnv và ghi kết quả cho mỗi lần chạy."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..lifemate_config import load_config
from .agent import run_agent
from .dataset import Dataset, open_dataset, read_eval_ids


CONFIG = load_config()


def query_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Gửi một lượt chat đến endpoint tương thích OpenAI của vLLM."""
    payload: dict[str, Any] = {
        "model": CONFIG["vllm"]["model_name"],
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": 512,
        "chat_template_kwargs": {
            "enable_thinking": CONFIG["prediction"]["agent_thinking_mode"],
        },
    }
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"

    request = urllib.request.Request(
        CONFIG["prediction"]["vllm_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def seed_prediction() -> None:
    """Cố định seed của pipeline dự đoán."""
    random.seed(42)
    import numpy as np
    import torch
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_prediction_asr() -> None:
    """Nạp ASR mặc định vào bộ nhớ của tiến trình."""
    from src.asr import load_asr_model

    print("[predict] Đang nạp mô hình ASR...")
    asr_start = time.perf_counter()
    load_asr_model(device="auto")
    print(f"[predict] Thời gian nạp ASR: {time.perf_counter() - asr_start:.2f}s")


def select_prediction_samples(
    dataset: Dataset, eval_ids: set[str] | None,
) -> list[dict[str, Any]]:
    """Kiểm tra danh sách ID và chọn các mẫu cần dự đoán."""
    if eval_ids is not None and dataset.split != "train":
        raise ValueError("prediction.eval_ids chỉ dùng được với prediction.split = 'train'.")
    available_ids = {sample["id"] for sample in dataset.samples}
    if eval_ids is not None and not eval_ids <= available_ids:
        missing = sorted(eval_ids - available_ids)
        raise ValueError(f"Không tìm thấy ID đánh giá trong split '{dataset.split}': {missing}")
    return [
        sample for sample in dataset.samples
        if eval_ids is None or sample["id"] in eval_ids
    ]


def predict_samples(
    dataset: Dataset,
    samples: list[dict[str, Any]],
    output_path: str | Path,
    *,
    eval_ids: set[str] | None,
    overwrite: bool = False,
) -> int:
    """Dự đoán các mẫu đã chọn và ghi kết quả JSONL."""
    from src.asr import get_user_request_from_audio

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    print(f"[predict] Bắt đầu split '{dataset.split}' với {len(samples)} mẫu.")

    count = 0
    run_start = time.perf_counter()
    with output.open(mode, encoding="utf-8") as stream:
        for sample in samples:
            # Tính thời gian từ lúc tạo session như yêu cầu định dạng chấm.
            sample_start = time.perf_counter()
            env = dataset.tool_env(dataset.split, sample["id"], root=dataset.root)

            def transcribe(sample_id: str) -> tuple[str, list[str]]:
                return get_user_request_from_audio(
                    sample_id,
                    split=dataset.split,
                    data_root=dataset.root,
                )
            run_agent(
                sample,
                env,
                catalog_tools=dataset.catalog_tools,
                eval_ids=eval_ids,
                transcribe=transcribe,
                query_model=query_llm,
            )
            tool_calls = env.get_tool_calls()
            state_hash = env.get_state_hash()
            # Thời gian chấm gồm cả lúc tạo ToolEnv và chạy ASR, LLM, tool.
            elapsed_ms = int((time.perf_counter() - sample_start) * 1000)
            # Ghi ngay sau mỗi mẫu để file giữ được phần đã chạy nếu tiến trình dừng.
            stream.write(json.dumps({
                "id": sample["id"],
                "tool_calls": tool_calls,
                "state_hash": state_hash,
                "time": elapsed_ms,
            }, ensure_ascii=False) + "\n")
            stream.flush()
            count += 1
            if count == 1 or count % 10 == 0:
                print(f"[predict] Đã xử lý {count}/{len(samples)} mẫu ({elapsed_ms} ms mẫu gần nhất).")

    elapsed = time.perf_counter() - run_start
    average = elapsed / max(count, 1)
    print(f"[predict] Hoàn tất {count} mẫu trong {elapsed:.2f}s (trung bình {average:.2f}s/mẫu) -> {output}")
    return count


def run_prediction(
    *,
    data_root: str | Path,
    output_path: str | Path,
    split: str,
    eval_ids_path: str | Path | None = None,
    overwrite: bool = False,
) -> int:
    """Chạy một split và trả về số mẫu đã ghi vào submission."""
    seed_prediction()
    eval_ids = read_eval_ids(eval_ids_path)
    dataset = open_dataset(data_root, split)
    samples = select_prediction_samples(dataset, eval_ids)
    load_prediction_asr()
    return predict_samples(
        dataset, samples, output_path, eval_ids=eval_ids, overwrite=overwrite,
    )
