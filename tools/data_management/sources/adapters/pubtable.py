from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import shutil
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image

from tools.data_management.canonical.writer import CanonicalWriteReport, CanonicalWriter
from tools.data_management.config.resolver import resolve_path
from tools.data_management.paths import dataset_relative_path, infer_dataset_root_from_path
from tools.data_management.progress import ProgressReporter
from tools.data_management.schemas import (
    AssetRecord,
    CanonicalDocument,
    CanonicalPage,
    CanonicalRegion,
    CanonicalTaskRecord,
    stable_hash,
    stable_id,
)
from tools.data_management.sources.adapters.base import SourceAdapter
from tools.data_management.sources.adapters._shared import (
    ExportReport,
    ShardWriter,
    drain_completed_exports,
)
from tools.data_management.utils.io import read_yaml


DEFAULT_TASKS = ("table",)


@dataclass(slots=True)
class PubTableExportOptions:
    data_dir: Path
    dataset_name: str
    max_samples: int | None = None
    shard_size: int = 10000
    skip_errors: bool = False
    output_root: Path | None = None
    num_workers: int = 1
    worker_chunksize: int = 256
    max_in_flight: int | None = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.worker_chunksize <= 0:
            raise ValueError("worker_chunksize must be positive")
        if self.max_in_flight is not None and self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive when set")


class PubTableSourceAdapter(SourceAdapter):
    name = "pubtable"
    version = "1.0.0"

    def __init__(self, options: PubTableExportOptions) -> None:
        self.options = options

    @classmethod
    def from_profile(cls, path: str | Path, *, dataset_root: str | Path | None = None) -> "PubTableSourceAdapter":
        data = read_yaml(path)
        profile_path = Path(path)

        def resolve(val: str | None) -> Path | None:
            return resolve_path(val, base=profile_path.parent, dataset_root=Path(dataset_root)) if val else None
        export_config = data.get("export") or {}

        return cls(
            PubTableExportOptions(
                data_dir=resolve_path(data["data_dir"], base=profile_path.parent, dataset_root=Path(dataset_root)),
                dataset_name=data["dataset_name"],
                max_samples=data.get("max_samples"),
                shard_size=int(data.get("shard_size", 10000)),
                skip_errors=bool(data.get("skip_errors", False)),
                output_root=resolve(data.get("output_root") or data.get("canonical_output")),
                num_workers=int(export_config.get("num_workers", data.get("num_workers", 1))),
                worker_chunksize=int(export_config.get("worker_chunksize", data.get("worker_chunksize", 256))),
                max_in_flight=(
                    int(export_config["max_in_flight"])
                    if export_config.get("max_in_flight") is not None
                    else int(data["max_in_flight"])
                    if data.get("max_in_flight") is not None
                    else None
                ),
            )
        )

    def export(
        self,
        canonical_root: str | Path,
        *,
        tasks: list[str] | None = None,
        overwrite_partitions: bool = True,
        progress: ProgressReporter | None = None,
    ) -> CanonicalWriteReport:
        selected_tasks = set(tasks or DEFAULT_TASKS)
        canonical_root = Path(canonical_root)

        if overwrite_partitions:
            shutil.rmtree(
                canonical_root / "assets" / "files" / f"source={self.options.dataset_name}",
                ignore_errors=True,
            )
        if progress:
            progress.log(
                "export-source",
                phase="start",
                source=self.options.dataset_name,
                canonical_root=canonical_root,
                tasks=sorted(selected_tasks),
                num_workers=self.options.num_workers,
            )

        manifest_writer = CanonicalWriter(canonical_root, overwrite_partitions=overwrite_partitions)
        report = ExportReport(dataset_name=self.options.dataset_name)
        document_writer = ShardWriter(
            canonical_root / "entities/documents" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
            id_column="document_id",
        )
        page_writer = ShardWriter(
            canonical_root / "entities/pages" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
            id_column="page_id",
        )
        region_writer = ShardWriter(
            canonical_root / "entities/regions" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
            id_column="region_id",
        )
        task_writers = {
            task: ShardWriter(
                canonical_root / "records" / task / f"source={self.options.dataset_name}",
                self.options.shard_size,
                overwrite=overwrite_partitions,
                id_column="record_id",
            )
            for task in selected_tasks
        }
        asset_writer = ShardWriter(
            canonical_root / "assets/manifests" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
            id_column="asset_id",
        )

        for result in self._iter_export_results(selected_tasks, canonical_root):
            report.scanned_samples += 1
            if progress:
                progress.update(
                    "export-source",
                    report.scanned_samples,
                    total=self.options.max_samples,
                    scanned=report.scanned_samples,
                    file=result.get("file"),
                    workers=self.options.num_workers,
                )
            if result.get("error"):
                report.errors.append({"record": str(result["idx"]), "error": str(result["error"])})
                report.skipped_samples += 1
                if not self.options.skip_errors:
                    raise RuntimeError(f"failed to export record {result['idx']}: {result['error']}")
                continue

            exported = result["exported"]
            document_writer.write_many(exported["documents"])
            page_writer.write_many(exported["pages"])
            region_writer.write_many(exported["regions"])
            asset_writer.write_many(exported["assets"])
            for task, rows in exported["task_records"].items():
                task_writers[task].write_many(rows)

        for writer in [document_writer, page_writer, region_writer, asset_writer, *task_writers.values()]:
            writer.close()
            if progress and writer.count:
                progress.log("export-source", phase="wrote", path=writer.output_dir, rows=writer.count)

        report.documents = document_writer.count
        report.pages = page_writer.count
        report.regions = region_writer.count
        report.assets = asset_writer.count
        report.task_records = {task: writer.count for task, writer in sorted(task_writers.items()) if writer.count}
        manifest_writer.write_manifest(
            self.options.dataset_name,
            {
                "source_name": self.options.dataset_name,
                "adapter": self.name,
                "adapter_version": self.version,
                "report": asdict(report),
            },
        )
        if progress:
            progress.finish(
                "export-source",
                total=report.scanned_samples,
                source=self.options.dataset_name,
                documents=report.documents,
                records=sum(report.task_records.values()),
                assets=report.assets,
                skipped=report.skipped_samples,
            )
        return CanonicalWriteReport(
            source_name=self.options.dataset_name,
            documents=report.documents,
            pages=report.pages,
            regions=report.regions,
            task_records=report.task_records,
            assets=report.assets,
        )

    def _iter_export_results(self, selected_tasks: set[str], canonical_root: Path) -> Iterable[dict[str, Any]]:
        selected_task_tuple = tuple(sorted(selected_tasks))
        dataset_root = infer_dataset_root_from_path(canonical_root)
        rows = _iter_pubtable_rows(self.options.data_dir, self.options.max_samples)
        if self.options.num_workers == 1:
            for item in rows:
                yield _export_record_for_worker(item, self.options.dataset_name, selected_task_tuple, canonical_root, dataset_root)
            return

        batches = _iter_batches(rows, self.options.worker_chunksize)
        max_in_flight = self.options.max_in_flight or self.options.num_workers * 4
        max_in_flight = max(max_in_flight, self.options.num_workers)
        with ProcessPoolExecutor(max_workers=self.options.num_workers) as pool:
            in_flight: set[Future[list[dict[str, Any]]]] = set()
            for batch in batches:
                while len(in_flight) >= max_in_flight:
                    yield from drain_completed_exports(in_flight)
                in_flight.add(
                    pool.submit(
                        _export_record_batch_star,
                        (batch, self.options.dataset_name, selected_task_tuple, canonical_root, dataset_root),
                    )
                )
            while in_flight:
                yield from drain_completed_exports(in_flight)


def _export_record(
    row: dict[str, Any],
    idx: int,
    source_name: str,
    selected_tasks: set[str],
    canonical_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    task = "table"
    if task not in selected_tasks:
        return {"documents": [], "pages": [], "regions": [], "assets": [], "task_records": {task: []}}

    image_data = row.get("image", {})
    image_bytes = image_data.get("bytes") if isinstance(image_data, dict) else None
    if not image_bytes:
        raise ValueError(f"record {idx}: no image bytes")

    html_with_text = _join_html_with_text(row.get("html_with_text"))
    if not html_with_text:
        raise ValueError(f"record {idx}: no html_with_text")

    filename = Path(str(row.get("filename", f"table_{idx:06d}"))).name
    if not filename:
        filename = f"table_{idx:06d}"
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        image_format = (image.format or Path(filename).suffix.lstrip(".") or "unknown").lower()
    asset_path = canonical_root / "assets" / "files" / f"source={source_name}" / filename
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(image_bytes)
    asset_relpath = dataset_relative_path(asset_path, dataset_root)

    doc_hash = stable_hash(filename)
    doc_id = stable_id("doc", source_name, doc_hash)

    document = CanonicalDocument(
        document_id=doc_id,
        source_name=source_name,
        source_document_id=filename,
        document_path=filename,
        document_type="image",
        num_pages=1,
        metadata={"cols": width, "rows": height},
    )

    page_id = stable_id("page", source_name, doc_hash, "0000")
    page_asset_id = stable_id("asset", "page_render", source_name, doc_hash, "0000")

    page_asset = AssetRecord(
        asset_id=page_asset_id,
        asset_type="page_render",
        source_name=source_name,
        document_id=doc_id,
        page_id=page_id,
        region_id=None,
        task=task,
        path=asset_relpath,
        width=width,
        height=height,
        format=image_format,
        coordinate_space="canonical_page_pixel_xyxy",
    )

    page = CanonicalPage(
        page_id=page_id,
        document_id=doc_id,
        source_name=source_name,
        source_page_id=filename,
        page_index=0,
        page_image_asset_id=page_asset_id,
        width=width,
        height=height,
        attributes={"source": "pubtable1m"},
    )

    region_id = stable_id("region", source_name, doc_hash)
    crop_asset_id = stable_id("asset", "region_crop", source_name, doc_hash)

    crop_asset = AssetRecord(
        asset_id=crop_asset_id,
        asset_type="region_crop",
        source_name=source_name,
        document_id=doc_id,
        page_id=page_id,
        region_id=region_id,
        task=task,
        path=asset_relpath,
        width=width,
        height=height,
        format=image_format,
        parent_asset_id=page_asset_id,
        transform_spec_hash=stable_hash({"operation": "copy", "source": "pubtable1m"}),
        transform={"operation": "copy", "source": "pubtable1m"},
    )

    region = CanonicalRegion(
        region_id=region_id,
        page_id=page_id,
        document_id=doc_id,
        source_name=source_name,
        source_annotation_id=filename,
        bbox=[0, 0, width, height],
        rotation=0.0,
        element_type="table",
        category="table",
        reading_order=1,
        crop_asset_id=crop_asset_id,
        metadata={"filename": filename},
    )

    task_record = CanonicalTaskRecord(
        record_id=stable_id(task, source_name, region_id),
        task=task,
        source_name=source_name,
        document_id=doc_id,
        page_id=page_id,
        region_id=region_id,
        image_asset_id=crop_asset_id,
        target={"html": html_with_text},
        category="table",
        provenance={"filename": filename},
        metadata={"width": width, "height": height},
    )

    return {
        "documents": [document.to_dict()],
        "pages": [page.to_dict()],
        "regions": [region.to_dict()],
        "assets": [page_asset.to_dict(), crop_asset.to_dict()],
        "task_records": {task: [task_record.to_dict()]},
    }


def _join_html_with_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        html = "".join(str(item) for item in value).strip()
    else:
        html = str(value).strip()
    if html and "<table" not in html.lower():
        html = f"<table>{html}</table>"
    return html


def _iter_pubtable_rows(data_dir: Path, max_samples: int | None) -> Iterable[dict[str, Any]]:
    scanned = 0
    for file_path in sorted(data_dir.glob("*.parquet")):
        if max_samples is not None and scanned >= max_samples:
            break
        parquet_file = pq.ParquetFile(file_path)
        for row_group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group)
            for row in table.to_pylist():
                if max_samples is not None and scanned >= max_samples:
                    break
                scanned += 1
                yield {"idx": scanned, "file": file_path.name, "row": row}


def _iter_batches(items: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _export_record_for_worker(
    item: dict[str, Any],
    source_name: str,
    selected_tasks: tuple[str, ...],
    canonical_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    idx = int(item["idx"])
    try:
        exported = _export_record(
            item["row"],
            idx,
            source_name,
            set(selected_tasks),
            canonical_root,
            dataset_root,
        )
        return {"idx": idx, "file": item.get("file"), "exported": exported}
    except Exception as exc:
        return {"idx": idx, "file": item.get("file"), "error": str(exc)}


def _export_record_batch_star(args: tuple[list[dict[str, Any]], str, tuple[str, ...], Path, Path]) -> list[dict[str, Any]]:
    batch, source_name, selected_tasks, canonical_root, dataset_root = args
    return [
        _export_record_for_worker(item, source_name, selected_tasks, canonical_root, dataset_root)
        for item in batch
    ]
