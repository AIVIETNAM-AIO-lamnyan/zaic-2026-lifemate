#!/usr/bin/env python
"""Điểm vào dòng lệnh cho pipeline dự đoán LifeMate."""

from __future__ import annotations

from pathlib import Path

from src.lifemate_config import load_config
from src.zalo_ai.pipeline import run_prediction


def main() -> None:
    """Đọc cấu hình từ môi trường rồi chạy một lượt dự đoán."""
    config = load_config()
    output_path = (
        Path(config["paths"]["result_dir"])
        / config["prediction"]["submission_file_name"]
    )

    run_prediction(
        data_root=Path(config["paths"]["data_root"]),
        output_path=output_path,
        split=config["prediction"]["split"],
        eval_ids_path=config["prediction"]["eval_ids"] or None,
    )


if __name__ == "__main__":
    main()
