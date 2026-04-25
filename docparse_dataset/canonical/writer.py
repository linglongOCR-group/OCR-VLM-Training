from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


class CanonicalWriter:
    """Persists normalized entities, records, manifests, and asset references."""

    def __init__(self, canonical_root: str | Path, overwrite_partition: bool = False):
        self.root = Path(canonical_root)
        self.overwrite = overwrite_partition

    def _partition_dir(self, *segments: str) -> Path:
        return self.root.joinpath(*segments)

    def _write_partition(self, records: Sequence[dict], partition_dir: Path, shard_size: int = 10000) -> int:
        partition_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            for stale in partition_dir.glob("part-*.parquet"):
                stale.unlink()
        total = 0
        for shard_index, start in enumerate(range(0, len(records), shard_size)):
            chunk = list(records[start : start + shard_size])
            pd.DataFrame(chunk).to_parquet(partition_dir / f"part-{shard_index:05d}.parquet", index=False)
            total += len(chunk)
        return total

    def write_documents(self, records: list[dict], source_name: str) -> int:
        return self._write_partition(records, self._partition_dir("entities", "documents", f"source={source_name}"))

    def write_pages(self, records: list[dict], source_name: str) -> int:
        return self._write_partition(records, self._partition_dir("entities", "pages", f"source={source_name}"))

    def write_regions(self, records: list[dict], source_name: str) -> int:
        return self._write_partition(records, self._partition_dir("entities", "regions", f"source={source_name}"))

    def write_task_records(self, task: str, source_name: str, records: list[dict]) -> int:
        return self._write_partition(records, self._partition_dir("records", task, f"source={source_name}"))

    def write_manifest(self, manifest: dict, name: str) -> None:
        manifest_dir = self.root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        path = manifest_dir / f"{name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


class _ShardWriter:
    """Streaming sharded Parquet writer."""

    def __init__(self, output_dir: Path, shard_size: int = 10000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict] = []
        self.shard_index = 0
        self.count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stale_part in self.output_dir.glob("part-*.parquet"):
            stale_part.unlink()

    def write(self, record: dict) -> None:
        self.buffer.append(record)
        self.count += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def write_many(self, records: Sequence[dict]) -> None:
        for record in records:
            self.write(record)

    def flush(self) -> None:
        if not self.buffer:
            return
        pd.DataFrame(self.buffer).to_parquet(self.output_dir / f"part-{self.shard_index:05d}.parquet", index=False)
        self.buffer = []
        self.shard_index += 1

    def close(self) -> None:
        self.flush()
