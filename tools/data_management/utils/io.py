from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text()) or {}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def write_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows)
    pd.DataFrame(data).to_parquet(path, index=False)
    return len(data)


def read_parquet_records(path: str | Path) -> list[dict[str, Any]]:
    return pd.read_parquet(path).to_dict(orient="records")


def read_partition_records(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    files = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
    rows: list[dict[str, Any]] = []
    for file_path in files:
        rows.extend(read_parquet_records(file_path))
    return rows
