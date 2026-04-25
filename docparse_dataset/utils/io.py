from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def read_parquet(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict(orient="records")


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_parquet_sharded(rows: list[dict], output_dir: Path, shard_size: int = 10000) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        chunk = rows[start : start + shard_size]
        pd.DataFrame(chunk).to_parquet(output_dir / f"part-{shard_index:05d}.parquet", index=False)
        total += len(chunk)
    return total


def expand_input_paths(paths: list[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(child for child in path.iterdir() if child.suffix in {".json", ".jsonl", ".parquet"}))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("no input source files found")
    return expanded
