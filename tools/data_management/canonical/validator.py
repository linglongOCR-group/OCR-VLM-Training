from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_canonical(canonical_root: str | Path, *, task: str | None = None, source: str | None = None) -> None:
    root = Path(canonical_root)
    if not root.is_dir():
        raise ValueError(f"canonical root does not exist: {root}")
    record_root = root / "records"
    if task:
        record_root = record_root / task
    files = sorted(record_root.glob(f"**/source={source}/part-*.parquet" if source else "**/part-*.parquet"))
    if not files:
        raise ValueError("no canonical record parquet files found")
    seen: set[str] = set()
    for file_path in files:
        frame = pd.read_parquet(file_path)
        for column in ("record_id", "task", "source_name", "document_id", "page_id", "image_asset_id", "target"):
            if column not in frame.columns:
                raise ValueError(f"{file_path} missing required column {column}")
        duplicates = set(frame["record_id"]) & seen
        if duplicates:
            raise ValueError(f"duplicate canonical record IDs: {sorted(duplicates)[:5]}")
        seen.update(frame["record_id"])
