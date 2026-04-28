from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tools.data_management.schemas import AssetRecord, CanonicalDocument, CanonicalPage, CanonicalRegion, CanonicalTaskRecord
from tools.data_management.utils.io import write_json, write_parquet


@dataclass(slots=True)
class CanonicalWriteReport:
    source_name: str
    documents: int = 0
    pages: int = 0
    regions: int = 0
    task_records: dict[str, int] = field(default_factory=dict)
    assets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "documents": self.documents,
            "pages": self.pages,
            "regions": self.regions,
            "task_records": self.task_records,
            "assets": self.assets,
        }


class CanonicalWriter:
    def __init__(self, canonical_root: str | Path, *, overwrite_partitions: bool = True) -> None:
        self.canonical_root = Path(canonical_root)
        self.overwrite_partitions = overwrite_partitions

    def write_documents(self, records: Iterable[CanonicalDocument], source_name: str) -> int:
        return self._write_partition("entities/documents", source_name, [record.to_dict() for record in records])

    def write_pages(self, records: Iterable[CanonicalPage], source_name: str) -> int:
        return self._write_partition("entities/pages", source_name, [record.to_dict() for record in records])

    def write_regions(self, records: Iterable[CanonicalRegion], source_name: str) -> int:
        return self._write_partition("entities/regions", source_name, [record.to_dict() for record in records])

    def write_task_records(self, task: str, source_name: str, records: Iterable[CanonicalTaskRecord]) -> int:
        return self._write_partition(f"records/{task}", source_name, [record.to_dict() for record in records])

    def write_asset_manifest(self, records: Iterable[AssetRecord]) -> int:
        rows = [record.to_dict() for record in records]
        return write_parquet(self.canonical_root / "assets/manifests/asset_manifest.parquet", rows)

    def write_manifest(self, source_name: str, manifest: dict[str, Any]) -> None:
        write_json(self.canonical_root / "manifests/sources" / f"{source_name}.json", manifest)

    def _write_partition(self, relative: str, source_name: str, rows: list[dict[str, Any]]) -> int:
        output_dir = self.canonical_root / relative / f"source={source_name}"
        if self.overwrite_partitions and output_dir.exists():
            for stale in output_dir.glob("part-*.parquet"):
                stale.unlink()
        return write_parquet(output_dir / "part-00000.parquet", rows)
