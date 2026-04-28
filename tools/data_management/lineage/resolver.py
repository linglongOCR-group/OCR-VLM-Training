from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class LineageResolver:
    def __init__(self, canonical_root: str | Path, view_root: str | Path | None = None) -> None:
        self.canonical_root = Path(canonical_root)
        self.view_root = Path(view_root) if view_root else None

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
        for path in sorted((self.canonical_root / "records").glob("**/*.parquet")):
            frame = pd.read_parquet(path)
            matches = frame[frame["record_id"] == canonical_record_id]
            if not matches.empty:
                row = matches.iloc[0].to_dict()
                return {"canonical_record": row}
        raise KeyError(f"canonical record not found: {canonical_record_id}")
