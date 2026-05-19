from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from tools.data_management.utils.io import read_partition_records


class CanonicalReader:
    def __init__(self, canonical_root: str | Path) -> None:
        self.canonical_root = Path(canonical_root)

    def read_task_records(self, task: str, source_name: str) -> list[dict[str, Any]]:
        return read_partition_records(self.canonical_root / "records" / task / f"source={source_name}")

    def read_asset_manifest(
        self,
        *,
        asset_ids: Iterable[str] | None = None,
        source_names: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        selected_asset_ids = set(asset_ids) if asset_ids is not None else None
        path = self.canonical_root / "assets/manifests/asset_manifest.parquet"
        if path.is_file():
            rows = _read_manifest_rows(path, selected_asset_ids)
            return {row["asset_id"]: row for row in rows}
        if source_names is None:
            files = sorted((self.canonical_root / "assets/manifests").glob("source=*/part-*.parquet"))
        else:
            files = []
            for source_name in sorted(set(source_names)):
                files.extend(sorted((self.canonical_root / "assets/manifests" / f"source={source_name}").glob("part-*.parquet")))
        if not files:
            return {}
        rows: list[dict[str, Any]] = []
        for file_path in files:
            rows.extend(_read_manifest_rows(file_path, selected_asset_ids))
        return {row["asset_id"]: row for row in rows}


def _read_manifest_rows(path: Path, selected_asset_ids: set[str] | None) -> list[dict[str, Any]]:
    if selected_asset_ids is None:
        return pd.read_parquet(path).to_dict(orient="records")
    asset_id_column = pd.read_parquet(path, columns=["asset_id"])["asset_id"].tolist()
    mask = [asset_id in selected_asset_ids for asset_id in asset_id_column]
    if not any(mask):
        return []
    return pd.read_parquet(path).loc[mask].to_dict(orient="records")
