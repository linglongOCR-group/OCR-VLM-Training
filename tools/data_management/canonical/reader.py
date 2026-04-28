from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools.data_management.utils.io import read_partition_records


class CanonicalReader:
    def __init__(self, canonical_root: str | Path) -> None:
        self.canonical_root = Path(canonical_root)

    def read_task_records(self, task: str, source_name: str) -> list[dict[str, Any]]:
        return read_partition_records(self.canonical_root / "records" / task / f"source={source_name}")

    def read_asset_manifest(self) -> dict[str, dict[str, Any]]:
        path = self.canonical_root / "assets/manifests/asset_manifest.parquet"
        if path.is_file():
            rows = pd.read_parquet(path).to_dict(orient="records")
            return {row["asset_id"]: row for row in rows}
        files = sorted((self.canonical_root / "assets/manifests").glob("source=*/part-*.parquet"))
        if not files:
            return {}
        rows: list[dict[str, Any]] = []
        for file_path in files:
            rows.extend(pd.read_parquet(file_path).to_dict(orient="records"))
        return {row["asset_id"]: row for row in rows}
