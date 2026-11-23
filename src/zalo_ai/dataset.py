"""Đọc dữ liệu cuộc thi và nạp ToolEnv của split đang chạy."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Dataset:
    """Dữ liệu và runtime công cụ đã được chọn cho một split."""

    root: Path
    split: str
    split_dir: Path
    samples: list[dict[str, Any]]
    catalog_tools: dict[str, dict[str, Any]]
    tool_env: Any


def _load_tool_env(split_dir: Path):
    """Nạp lớp ToolEnv từ thư viện đi kèm split dữ liệu."""
    # Mỗi split mang theo implementation ToolEnv tương ứng với bộ dữ liệu đó.
    sys.path.insert(0, str(split_dir))
    return importlib.import_module("zaic_tools").ToolEnv


def _read_samples(split_dir: Path, split: str) -> list[dict[str, Any]]:
    """Đọc các dòng JSONL theo đúng thứ tự trong file nguồn."""
    path = split_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu split: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def open_dataset(data_root: str | Path, split: str) -> Dataset:
    """Đọc dữ liệu trực tiếp từ thư mục được chọn trong config.toml."""
    root = Path(data_root).expanduser().resolve()
    split_dir = root / split
    return _build_dataset(root, split, split_dir)


def _build_dataset(root: Path, split: str, split_dir: Path) -> Dataset:
    """Đọc catalog, mẫu và runtime công cụ cho split đã chọn."""
    catalog_path = split_dir / f"{split}_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tools = {
        tool["name"]: tool
        for app in catalog.get("apps", [])
        for tool in app.get("tools", [])
    }
    return Dataset(
        root=root,
        split=split,
        split_dir=split_dir,
        samples=_read_samples(split_dir, split),
        catalog_tools=tools,
        tool_env=_load_tool_env(split_dir),
    )


def read_eval_ids(path: str | Path | None) -> set[str] | None:
    """Nạp ID đánh giá train, nếu đường dẫn được chỉ định."""
    if path is None or not str(path):
        return None
    ids = {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise ValueError(f"Danh sách ID đánh giá đang trống: {path}")
    return ids
