from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from concurrent.futures import Future, wait
from concurrent.futures import FIRST_COMPLETED


# ---------------------------------------------------------------------------
# Export report
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExportReport:
    dataset_name: str
    scanned_samples: int = 0
    skipped_samples: int = 0
    skipped_completed_samples: int = 0
    documents: int = 0
    pages: int = 0
    regions: int = 0
    task_records: dict[str, int] = field(default_factory=dict)
    assets: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
        *,
        overwrite: bool = True,
        id_column: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0
        self.count = 0
        self.id_column = id_column
        self.existing_ids: set[str] = set()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for stale in self.output_dir.glob("part-*.parquet"):
                stale.unlink()
        else:
            self._load_existing_state()

    def write_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            if self.id_column:
                row_id = row.get(self.id_column)
                if row_id is not None:
                    normalized_id = str(row_id)
                    if normalized_id in self.existing_ids:
                        continue
                    self.existing_ids.add(normalized_id)
            self.buffer.append(row)
            self.count += 1
            if len(self.buffer) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        pd.DataFrame(self.buffer).to_parquet(
            self.output_dir / f"part-{self.shard_index:05d}.parquet",
            index=False,
        )
        self.buffer = []
        self.shard_index += 1

    def close(self) -> None:
        self.flush()

    def _load_existing_state(self) -> None:
        part_files = sorted(self.output_dir.glob("part-*.parquet"))
        if not part_files:
            return
        self.shard_index = _next_shard_index(part_files)
        if self.id_column:
            for file_path in part_files:
                frame = pd.read_parquet(file_path, columns=[self.id_column])
                self.count += len(frame)
                self.existing_ids.update(str(value) for value in frame[self.id_column])
        else:
            self.count = sum(len(pd.read_parquet(fp)) for fp in part_files)


def _next_shard_index(part_files: Sequence[Path]) -> int:
    max_index = -1
    for file_path in part_files:
        try:
            max_index = max(max_index, int(file_path.stem.removeprefix("part-")))
        except ValueError:
            continue
    return max_index + 1


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------

def drain_completed_exports(in_flight: set[Future[list[dict[str, Any]]]]) -> Iterable[dict[str, Any]]:
    done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
    in_flight.clear()
    in_flight.update(pending)
    for future in done:
        yield from future.result()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def entity_dir(canonical_root: Path, entity: str, source_name: str) -> Path:
    return canonical_root / "entities" / entity / f"source={source_name}"


def records_dir(canonical_root: Path, task: str, source_name: str) -> Path:
    return canonical_root / "records" / task / f"source={source_name}"


def assets_manifest_dir(canonical_root: Path, source_name: str) -> Path:
    return canonical_root / "assets" / "manifests" / f"source={source_name}"


def assets_files_dir(canonical_root: Path, source_name: str) -> Path:
    return canonical_root / "assets" / "files" / f"source={source_name}"


def region_crop_relative_path(source_name: str, asset_id: str, extension: str = ".jpg") -> Path:
    safe_id = asset_id.replace(":", "_").replace("/", "_")
    return (
        Path("canonical") / "assets" / "files" / f"source={source_name}"
        / "region_crop" / safe_id[:2] / f"{safe_id}{extension}"
    )
