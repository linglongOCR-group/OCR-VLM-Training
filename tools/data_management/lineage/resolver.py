from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class LineageResolver:
    def __init__(self, canonical_root: str | Path, view_root: str | Path | None = None) -> None:
        self.canonical_root = Path(canonical_root)
        self.view_root = Path(view_root) if view_root else None
        self._record_index: dict[str, Path] | None = None

    def _ensure_index(self) -> dict[str, Path]:
        if self._record_index is not None:
            return self._record_index
        index: dict[str, Path] = {}
        records_dir = self.canonical_root / "records"
        if records_dir.exists():
            for path in sorted(records_dir.glob("**/*.parquet")):
                table = pq.read_table(path, columns=["record_id"])
                for batch in table.to_batches():
                    for row in batch.to_pylist():
                        record_id = row.get("record_id")
                        if record_id:
                            index[record_id] = path
        self._record_index = index
        return self._record_index

    def trace_view_record(self, view_record_id: str) -> dict[str, Any]:
        if self.view_root is None:
            raise ValueError("view_root is required to trace view records")
        for path in sorted(self.view_root.glob("*.parquet")):
            frame = pd.read_parquet(path)
            matches = frame[frame["id"] == view_record_id]
            if not matches.empty:
                row = matches.iloc[0].to_dict()
                return {"view_record": row, "canonical_record": self.trace_canonical_record(row["canonical_record_id"])}
        raise KeyError(f"view record not found: {view_record_id}")

    def trace_canonical_record(self, canonical_record_id: str) -> dict[str, Any]:
        index = self._ensure_index()
        path = index.get(canonical_record_id)
        if path is None:
            raise KeyError(f"canonical record not found: {canonical_record_id}")
        frame = pd.read_parquet(path)
        matches = frame[frame["record_id"] == canonical_record_id]
        if matches.empty:
            raise KeyError(f"canonical record not found: {canonical_record_id}")
        return matches.iloc[0].to_dict()
