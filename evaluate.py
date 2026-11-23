#!/usr/bin/env python
"""Chấm file dự đoán theo split và ID trong config.toml."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from src.lifemate_config import load_config
from src.zalo_ai.dataset import open_dataset

CONFIG = load_config()
DATA_ROOT = Path(CONFIG["paths"]["data_root"])
TIME_LIMIT_MS = 30_000


def read_jsonl(path: Path) -> list[dict]:
    """Đọc các bản ghi JSON Lines."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_ids(path: Path) -> set[str]:
    """Đọc danh sách ID dùng để chấm."""
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def fuzzy_score(left: str, right: str) -> float:
    """Tính token F1 sau khi chuẩn hóa chữ thường và Unicode NFC."""
    left_tokens = Counter(unicodedata.normalize("NFC", left).lower().split())
    right_tokens = Counter(unicodedata.normalize("NFC", right).lower().split())
    common = sum((left_tokens & right_tokens).values())
    if not common:
        return 0.0
    precision = common / sum(left_tokens.values())
    recall = common / sum(right_tokens.values())
    return 2 * precision * recall / (precision + recall)


def calls_match(predicted: dict, gold: dict, fuzzy_fields: dict[str, set[str]]) -> bool:
    """So khớp tên tool, đối số chính xác và các trường văn bản gần đúng."""
    if predicted.get("name") != gold.get("name"):
        return False

    name = predicted["name"]
    predicted_args = predicted.get("arguments", {})
    gold_args = gold.get("arguments", {})
    fuzzy = fuzzy_fields.get(name, set())
    predicted_exact = {key: value for key, value in predicted_args.items() if key not in fuzzy}
    gold_exact = {key: value for key, value in gold_args.items() if key not in fuzzy}
    if predicted_exact != gold_exact:
        return False

    for key in fuzzy:
        if key in predicted_args and key in gold_args:
            if fuzzy_score(str(predicted_args[key]), str(gold_args[key])) <= 0.6:
                return False
        elif key in predicted_args or key in gold_args:
            return False
    return True


def match_calls(predicted: list[dict], gold: list[dict],
                fuzzy_fields: dict[str, set[str]]) -> int:
    """Tìm số cặp tool call khớp tối đa, mỗi call chỉ được ghép một lần."""
    # Ghép tối đa để thứ tự khác nhau không làm mất các cặp call vẫn khớp.
    matched_gold = {}

    def augment(predicted_index: int, seen: set[int]) -> bool:
        for gold_index, gold_call in enumerate(gold):
            if gold_index in seen or not calls_match(
                    predicted[predicted_index], gold_call, fuzzy_fields):
                continue
            seen.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = predicted_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(predicted)))


def replay_calls(tool_env, calls: list[dict]) -> str:
    """Phát lại tool call theo thứ tự và lấy hash từ môi trường cuộc thi."""
    for call in calls:
        tool_env.call(call["name"], call.get("arguments", {}))
    return tool_env.get_state_hash()


def evaluate_train(predictions_path: Path, dataset, eval_ids_path: str | None) -> None:
    all_training = {row["id"]: row for row in dataset.samples}
    ids = (
        read_ids(Path(eval_ids_path))
        if eval_ids_path
        else set(all_training)
    )
    if not ids:
        raise SystemExit("Không có mẫu train trong cấu hình để chấm.")
    missing_train = sorted(ids - set(all_training))
    if missing_train:
        raise SystemExit(f"ID đánh giá không có trong split train: {missing_train}")

    prediction_rows = read_jsonl(predictions_path)
    predictions = {row["id"]: row for row in prediction_rows}
    prediction_ids = set(predictions)
    missing_predictions = sorted(ids - prediction_ids)
    unexpected_predictions = sorted(prediction_ids - ids)
    duplicates = sorted(
        sample_id for sample_id, count in Counter(row["id"] for row in prediction_rows).items()
        if count > 1
    )
    if missing_predictions or unexpected_predictions or duplicates:
        raise SystemExit(
            "ID trong file dự đoán không khớp với tập cần chấm. "
            f"Thiếu: {missing_predictions}; ngoài tập: {unexpected_predictions}; "
            f"trùng: {duplicates}"
        )

    training = {sample_id: all_training[sample_id] for sample_id in ids}
    catalog_path = dataset.split_dir / f"{dataset.split}_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    fuzzy_fields = {
        tool["name"]: {
            key for key, spec in tool.get("parameters", {}).get("properties", {}).items()
            if spec.get("x-match") == "fuzzy"
        }
        for app in catalog.get("apps", [])
        for tool in app.get("tools", [])
    }

    # Mỗi ToolEnv giữ state riêng, nên phát lại nhãn và dự đoán trên hai instance.
    total_score = 0.0
    total_time = 0
    over_limit = 0
    for sample_id in sorted(ids):
        label = training[sample_id]["label"]
        prediction = predictions[sample_id]
        gold_calls = label["tool_calls"]
        predicted_calls = prediction["tool_calls"]
        gold_env = dataset.tool_env(dataset.split, sample_id, root=dataset.root)
        predicted_env = dataset.tool_env(dataset.split, sample_id, root=dataset.root)
        gold_hash = replay_calls(gold_env, gold_calls)
        predicted_hash = replay_calls(predicted_env, predicted_calls)
        matched = match_calls(predicted_calls, gold_calls, fuzzy_fields)
        elapsed = int(prediction["time"])
        total_time += elapsed
        timed_out = elapsed > TIME_LIMIT_MS
        over_limit += timed_out

        # Điểm mẫu là F1 của tool call; sai state hoặc quá thời hạn thì nhận 0.
        if timed_out:
            score = 0.0
        elif not gold_calls:
            score = float(not predicted_calls)
        elif (matched == 0 or prediction.get("state_hash") != gold_hash
              or predicted_hash != gold_hash):
            score = 0.0
        else:
            score = 2 * matched / (len(predicted_calls) + len(gold_calls))
        total_score += score

    count = len(ids)
    print(f"Split: {dataset.split}")
    print(f"Score: {100 * total_score / count:.2f}/100")
    print(f"Samples: {count}")
    print(f"Average time: {total_time / count:.0f} ms/sample")
    print(f"Over {TIME_LIMIT_MS} ms: {over_limit}")


def evaluate_coverage(predictions_path: Path, dataset) -> None:
    expected_ids = {row["id"] for row in dataset.samples}
    prediction_rows = read_jsonl(predictions_path)
    prediction_counts = Counter(row["id"] for row in prediction_rows)
    prediction_ids = set(prediction_counts)
    missing = sorted(expected_ids - prediction_ids)
    duplicates = sorted(sample_id for sample_id, count in prediction_counts.items() if count > 1)
    unexpected = sorted(prediction_ids - expected_ids)
    total_time = sum(int(row["time"]) for row in prediction_rows)
    over_limit = sum(int(row["time"]) > TIME_LIMIT_MS for row in prediction_rows)
    count = len(expected_ids)
    if not count:
        raise SystemExit(f"Split '{dataset.split}' không có mẫu để đối chiếu.")
    print(f"Split: {dataset.split}")
    print(f"Coverage: {len(prediction_ids & expected_ids)}/{count}")
    print(f"Missing IDs: {len(missing)}")
    print(f"Duplicate IDs: {len(duplicates)}")
    print(f"Unexpected IDs: {len(unexpected)}")
    print(f"Average time: {total_time / max(len(prediction_rows), 1):.0f} ms/sample")
    print(f"Over {TIME_LIMIT_MS} ms: {over_limit}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chấm file dự đoán của split được chọn trong config.toml."
    )
    parser.add_argument(
        "predictions", type=Path, nargs="?",
        help="File dự đoán. Mặc định dùng paths.result_dir/prediction.submission_file_name.",
    )
    args = parser.parse_args()
    prediction_path = args.predictions or (
        Path(CONFIG["paths"]["result_dir"])
        / CONFIG["prediction"]["submission_file_name"]
    )
    if not prediction_path.is_file():
        raise SystemExit(f"Không tìm thấy file dự đoán: {prediction_path}")

    dataset = open_dataset(DATA_ROOT, CONFIG["prediction"]["split"])
    if CONFIG["prediction"]["eval_ids"] and dataset.split != "train":
        raise SystemExit("prediction.eval_ids chỉ dùng được với split train.")
    if dataset.split == "train":
        evaluate_train(
            prediction_path,
            dataset,
            CONFIG["prediction"]["eval_ids"] or None,
        )
    else:
        evaluate_coverage(prediction_path, dataset)


if __name__ == "__main__":
    main()
