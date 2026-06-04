from __future__ import annotations

import json
import os
import re
import shutil
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from PIL import Image

from tools.data_management.canonical.writer import CanonicalWriteReport, CanonicalWriter
from tools.data_management.config.resolver import resolve_path
from tools.data_management.paths import DATA_ROOT_ENV, dataset_relative_path, infer_dataset_root_from_path
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
from tools.data_management.sources.adapters._shared import (
    ExportReport,
    ShardWriter,
    drain_completed_exports,
    assets_files_dir,
    assets_manifest_dir,
    entity_dir,
    records_dir,
    region_crop_relative_path,
)
from tools.data_management.sources.adapters.base import SourceAdapter
from tools.data_management.utils.io import read_yaml


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_TASKS = ("layout", "text", "table", "formula", "diagram", "seal")
TEXT_TYPES = {
    "title",
    "text",
    "ref_text",
    "table_caption",
    "table_footnote",
    "image_caption",
    "image_footnote",
    "page_footnote",
    "header",
    "footer",
    "page_number",
    "list",
    "list_item",
    "code",
    "code_caption",
    "algorithm",
    "aside_text",
    "phonetic",
}
DIAGRAM_TYPES = {"image", "chart", "diagram"}
SEAL_TYPES = {"seal", "stamp"}


@dataclass(slots=True)
class MinerUExportOptions:
    mineru_root: Path
    source_image_root: Path | None
    dataset_name: str
    mineru_subdir: str = "vlm"
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    max_samples: int | None = None
    shard_size: int = 10000
    allow_unreadable_images: bool = False
    skip_errors: bool = False
    skip_completed: bool = False
    output_root: Path | None = None
    num_workers: int = 1
    worker_chunksize: int = 8
    max_in_flight: int | None = None

    def __post_init__(self) -> None:
        self.mineru_root = Path(self.mineru_root)
        self.source_image_root = Path(self.source_image_root) if self.source_image_root else None
        self.image_extensions = tuple(self.image_extensions)
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.worker_chunksize <= 0:
            raise ValueError("worker_chunksize must be positive")
        if self.max_in_flight is not None and self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive when set")


@dataclass(frozen=True, slots=True)
class MinerUSample:
    sample_id: str
    sample_dir: Path
    annot_dir: Path
    model_json: Path
    middle_json: Path | None
    markdown: Path | None
    content_v2_json: Path | None


class MinerUSourceAdapter(SourceAdapter):
    name = "mineru"
    version = "1.0.0"

    def __init__(self, options: MinerUExportOptions) -> None:
        self.options = options

    @classmethod
    def from_profile(cls, path: str | Path, *, dataset_root: str | Path | None = None) -> "MinerUSourceAdapter":
        data = read_yaml(path)
        profile_path = Path(path)
        root = Path(dataset_root or os.environ.get(DATA_ROOT_ENV) or profile_path.parent)
        export_config = data.get("export") or {}

        def resolve_profile_path(value: str | None) -> Path | None:
            if not value:
                return None
            return resolve_path(value, base=profile_path.parent, dataset_root=root)

        return cls(
            MinerUExportOptions(
                mineru_root=resolve_profile_path(data["mineru_root"]),
                source_image_root=resolve_profile_path(data.get("source_image_root") or data.get("source_path")),
                dataset_name=data["dataset_name"],
                mineru_subdir=data.get("mineru_subdir", "vlm"),
                image_extensions=tuple(data.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)),
                max_samples=int(data["max_samples"]) if data.get("max_samples") is not None else None,
                shard_size=int(data.get("shard_size", 10000)),
                allow_unreadable_images=bool(data.get("allow_unreadable_images", False)),
                skip_errors=bool(data.get("skip_errors", False)),
                skip_completed=bool(export_config.get("skip_completed", data.get("skip_completed", False))),
                output_root=resolve_profile_path(data.get("output_root") or data.get("canonical_output")),
                num_workers=int(export_config.get("num_workers", data.get("num_workers", 1))),
                worker_chunksize=int(export_config.get("worker_chunksize", data.get("worker_chunksize", 8))),
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
        if self.options.skip_completed:
            overwrite_partitions = False

        with _source_export_lock(canonical_root, self.options.dataset_name):
            completed_source_document_ids = (
                _load_completed_source_document_ids(canonical_root, self.options.dataset_name)
                if self.options.skip_completed
                else set()
            )
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
                    skip_completed=self.options.skip_completed,
                    completed=len(completed_source_document_ids),
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

            def handle_result(result: dict[str, Any]) -> None:
                report.scanned_samples += 1
                sample_id = str(result["sample_id"])
                skipped_completed = bool(result.get("skipped_completed"))
                if skipped_completed:
                    report.skipped_completed_samples += 1
                if progress:
                    progress.update(
                        "export-source",
                        report.scanned_samples,
                        total=self.options.max_samples,
                        scanned=report.scanned_samples,
                        skipped_completed=report.skipped_completed_samples,
                        sample=sample_id,
                    )
                if skipped_completed:
                    if progress and (
                        report.skipped_completed_samples == 1
                        or report.skipped_completed_samples % progress.log_every == 0
                    ):
                        progress.log(
                            "export-source",
                            phase="skip_completed",
                            skipped_completed=report.skipped_completed_samples,
                            sample=sample_id,
                        )
                    return
                if result.get("error"):
                    report.errors.append({"sample_id": sample_id, "error": str(result["error"])})
                    report.skipped_samples += 1
                    if progress:
                        progress.log("export-source", phase="skip", sample=sample_id, error=result["error"])
                    if not self.options.skip_errors:
                        raise RuntimeError(f"failed to export sample {sample_id}: {result['error']}")
                    return
                _write_exported_rows(
                    result["exported"],
                    document_writer=document_writer,
                    page_writer=page_writer,
                    region_writer=region_writer,
                    asset_writer=asset_writer,
                    task_writers=task_writers,
                )

            for result in self._iter_export_results(selected_tasks, canonical_root, completed_source_document_ids):
                handle_result(result)

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
                    skipped_completed=report.skipped_completed_samples,
                )
            return CanonicalWriteReport(
                source_name=self.options.dataset_name,
                documents=report.documents,
                pages=report.pages,
                regions=report.regions,
                task_records=report.task_records,
                assets=report.assets,
            )

    def _iter_export_results(
        self,
        selected_tasks: set[str],
        canonical_root: Path,
        completed_source_document_ids: set[str],
    ) -> Iterable[dict[str, Any]]:
        samples = _limited_samples(self.options)
        if self.options.num_workers == 1:
            for sample in samples:
                if sample.sample_id in completed_source_document_ids:
                    yield _skipped_completed_result(sample)
                    continue
                yield _export_sample_for_worker(self.options, tuple(sorted(selected_tasks)), canonical_root, sample)
            return

        max_in_flight = self.options.max_in_flight if self.options.max_in_flight is not None else self.options.num_workers * 4
        max_in_flight = max(max_in_flight, self.options.num_workers)
        task_tuple = tuple(sorted(selected_tasks))
        with ProcessPoolExecutor(max_workers=self.options.num_workers) as pool:
            in_flight: set[Future[list[dict[str, Any]]]] = set()

            def submit_chunk(chunk: list[MinerUSample]) -> None:
                in_flight.add(pool.submit(_export_sample_chunk_for_worker, self.options, task_tuple, canonical_root, chunk))

            chunk: list[MinerUSample] = []
            for sample in samples:
                if sample.sample_id in completed_source_document_ids:
                    yield _skipped_completed_result(sample)
                    continue
                chunk.append(sample)
                if len(chunk) >= self.options.worker_chunksize:
                    while len(in_flight) >= max_in_flight:
                        yield from drain_completed_exports(in_flight)
                    submit_chunk(chunk)
                    chunk = []
            if chunk:
                while len(in_flight) >= max_in_flight:
                    yield from drain_completed_exports(in_flight)
                submit_chunk(chunk)
            while in_flight:
                yield from drain_completed_exports(in_flight)

    def _export_sample(self, sample: MinerUSample, selected_tasks: set[str], canonical_root: Path) -> dict[str, Any]:
        return self._export_sample_records(sample, selected_tasks, canonical_root)

    def _export_sample_records(self, sample: MinerUSample, selected_tasks: set[str], canonical_root: Path) -> dict[str, Any]:
        if not sample.model_json.is_file():
            raise FileNotFoundError(f"missing model json: {sample.model_json}")
        model_pages = _read_json(sample.model_json)
        if not isinstance(model_pages, list):
            raise ValueError(f"model json must be a list: {sample.model_json}")
        middle_pages = _load_middle_pages(sample)
        markdown = _load_markdown(sample)
        image_path, image_accessible, image_unreadable = _resolve_source_image(sample.sample_id, self.options)
        dataset_root = infer_dataset_root_from_path(canonical_root)
        source_image_path = Path(image_path)
        source_image_relpath = dataset_relative_path(source_image_path, dataset_root) if source_image_path.is_absolute() else image_path
        document_hash = stable_hash(sample.sample_id)
        document_id = stable_id("doc", self.options.dataset_name, document_hash)
        document = CanonicalDocument(
            document_id=document_id,
            source_name=self.options.dataset_name,
            source_document_id=sample.sample_id,
            document_path=source_image_relpath,
            document_type="image",
            num_pages=max(len(model_pages), 1),
            metadata={
                "mineru_dir": str(sample.annot_dir),
                "image_accessible": image_accessible,
                "image_unreadable": image_unreadable,
            },
        )

        pages: list[CanonicalPage] = []
        regions: list[CanonicalRegion] = []
        task_records: dict[str, list[CanonicalTaskRecord]] = {task: [] for task in selected_tasks}
        assets: list[AssetRecord] = []

        with _RegionCropSaver(
            source_image_path,
            page_width=0,
            page_height=0,
            dataset_root=dataset_root,
            source_name=self.options.dataset_name,
        ) as crop_saver:
            for page_index, page_blocks in enumerate(model_pages):
                if not isinstance(page_blocks, list):
                    continue
                middle_page = middle_pages[page_index] if page_index < len(middle_pages) else None
                width, height = _page_size(middle_page)
                crop_saver.page_width = width
                crop_saver.page_height = height
                page_id = stable_id("page", self.options.dataset_name, document_hash, f"{page_index:04d}")
                page_asset_id = stable_id("asset", "page_render", self.options.dataset_name, document_hash, f"{page_index:04d}")
                page_source_id = sample.sample_id if len(model_pages) == 1 else f"{sample.sample_id}_p{page_index:04d}"
                page_asset = AssetRecord(
                    asset_id=page_asset_id,
                    asset_type="page_render",
                    source_name=self.options.dataset_name,
                    document_id=document_id,
                    page_id=page_id,
                    region_id=None,
                    task="layout",
                    path=source_image_relpath,
                    width=width,
                    height=height,
                    format=Path(image_path).suffix.lstrip(".") or "unknown",
                    coordinate_space="canonical_page_pixel_xyxy",
                )
                assets.append(page_asset)
                pages.append(
                    CanonicalPage(
                        page_id=page_id,
                        document_id=document_id,
                        source_name=self.options.dataset_name,
                        source_page_id=page_source_id,
                        page_index=page_index,
                        page_image_asset_id=page_asset_id,
                        width=width,
                        height=height,
                        attributes={"mineru_subdir": self.options.mineru_subdir},
                    )
                )

                middle_bboxes = _middle_bbox_by_index(middle_page)
                layout_elements: list[dict[str, Any]] = []
                for reading_order, block in enumerate((item for item in page_blocks if isinstance(item, dict)), start=1):
                    block_type = str(block.get("type") or "unknown")
                    bbox = _block_bbox(block, middle_bboxes, width, height)
                    source_annotation_id = str(block.get("index", reading_order))
                    region_hash = stable_hash(
                        {"sample": sample.sample_id, "page": page_index, "annotation": source_annotation_id, "bbox": bbox}
                    )
                    region_id = stable_id("region", self.options.dataset_name, document_hash, f"{page_index:04d}", region_hash)
                    crop_asset_id = stable_id("asset", "region_crop", self.options.dataset_name, region_hash)
                    crop_path, crop_width, crop_height = crop_saver.save(bbox, crop_asset_id)
                    assets.append(
                        AssetRecord(
                            asset_id=crop_asset_id,
                            asset_type="region_crop",
                            source_name=self.options.dataset_name,
                            document_id=document_id,
                            page_id=page_id,
                            region_id=region_id,
                            task=_task_for_region_type(block_type),
                            path=crop_path,
                            width=crop_width,
                            height=crop_height,
                            format="jpg",
                            parent_asset_id=page_asset_id,
                            transform_spec_hash=stable_hash({"operation": "crop", "bbox": bbox}),
                            transform={"operation": "crop", "bbox": bbox, "source": "source_image_bbox"},
                        )
                    )
                    regions.append(
                        CanonicalRegion(
                            region_id=region_id,
                            page_id=page_id,
                            document_id=document_id,
                            source_name=self.options.dataset_name,
                            source_annotation_id=source_annotation_id,
                            bbox=bbox,
                            rotation=float(block.get("angle") or 0.0),
                            element_type=block_type,
                            category=block_type,
                            reading_order=reading_order,
                            crop_asset_id=crop_asset_id,
                            metadata={"mineru_index": block.get("index"), "mineru_bbox": block.get("bbox")},
                        )
                    )
                    layout_elements.append(
                        {
                            "region_id": region_id,
                            "bbox": bbox,
                            "label": block_type,
                            "rotation": float(block.get("angle") or 0.0),
                            "reading_order": reading_order,
                        }
                    )
                    task = _task_for_region_type(block_type)
                    if task in selected_tasks and task != "layout":
                        target = _target_for_region(task, block, markdown)
                        if target:
                            task_records[task].append(
                                CanonicalTaskRecord(
                                    record_id=stable_id(task, self.options.dataset_name, region_id),
                                    task=task,
                                    source_name=self.options.dataset_name,
                                    document_id=document_id,
                                    page_id=page_id,
                                    region_id=region_id,
                                    image_asset_id=crop_asset_id,
                                    target=target,
                                    category=block_type,
                                    provenance={"source_annotation_id": source_annotation_id, "mineru_dir": str(sample.annot_dir)},
                                    metadata={"width": crop_width, "height": crop_height, "uses_page_image": False},
                                )
                            )
                if "layout" in selected_tasks and layout_elements:
                    task_records["layout"].append(
                        CanonicalTaskRecord(
                            record_id=stable_id("layout", self.options.dataset_name, page_id),
                            task="layout",
                            source_name=self.options.dataset_name,
                            document_id=document_id,
                            page_id=page_id,
                            region_id=None,
                            image_asset_id=page_asset_id,
                            target={"coordinate_space": "canonical_page_pixel_xyxy", "elements": layout_elements},
                            provenance={"mineru_dir": str(sample.annot_dir)},
                            metadata={"width": width, "height": height},
                        )
                    )

        return {"documents": [document], "pages": pages, "regions": regions, "task_records": task_records, "assets": assets}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _discover_samples(options: MinerUExportOptions) -> Iterable[MinerUSample]:
    for sample_dir in sorted(path for path in options.mineru_root.iterdir() if path.is_dir()):
        annot_dir = sample_dir / options.mineru_subdir
        if not annot_dir.is_dir():
            continue
        sample_id = sample_dir.name
        yield MinerUSample(
            sample_id=sample_id,
            sample_dir=sample_dir,
            annot_dir=annot_dir,
            model_json=annot_dir / f"{sample_id}_model.json",
            middle_json=_optional_file(annot_dir / f"{sample_id}_middle.json"),
            markdown=_optional_file(annot_dir / f"{sample_id}.md"),
            content_v2_json=_optional_file(annot_dir / f"{sample_id}_content_list_v2.json"),
        )


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _load_middle_pages(sample: MinerUSample) -> list[dict[str, Any]]:
    if sample.middle_json is None:
        return []
    middle = _read_json(sample.middle_json)
    pdf_info = middle.get("pdf_info") if isinstance(middle, dict) else None
    return pdf_info if isinstance(pdf_info, list) else []


def _load_markdown(sample: MinerUSample) -> str:
    return sample.markdown.read_text() if sample.markdown is not None else ""


def _page_size(middle_page: dict[str, Any] | None, fallback: Sequence[int] = (1, 1)) -> tuple[int, int]:
    if middle_page:
        raw = middle_page.get("page_size")
        if isinstance(raw, Sequence) and len(raw) >= 2:
            return max(int(raw[0]), 1), max(int(raw[1]), 1)
    return int(fallback[0]), int(fallback[1])


def _middle_bbox_by_index(middle_page: dict[str, Any] | None) -> dict[int, list[float]]:
    if not middle_page:
        return {}
    mapping: dict[int, list[float]] = {}
    for block in middle_page.get("para_blocks") or []:
        if isinstance(block, dict) and "index" in block and "bbox" in block:
            mapping[int(block["index"])] = [float(value) for value in block["bbox"]]
    return mapping


def _scale_bbox_if_normalized(bbox: Sequence[float], width: int, height: int) -> list[float]:
    values = [float(value) for value in bbox]
    if len(values) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    if all(0 <= value <= 1 for value in values):
        return [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return values


def _block_bbox(block: dict[str, Any], middle_bboxes: dict[int, list[float]], width: int, height: int) -> list[float]:
    index = block.get("index")
    if index is not None and int(index) in middle_bboxes:
        return middle_bboxes[int(index)]
    return _scale_bbox_if_normalized(block.get("bbox") or [], width, height)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        fragments: list[str] = []
        _collect_text(value, fragments)
        if fragments:
            return "\n".join(fragments)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _collect_text(value: Any, fragments: list[str]) -> None:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            fragments.append(content)
        for child in value.values():
            _collect_text(child, fragments)
    elif isinstance(value, list):
        for child in value:
            _collect_text(child, fragments)


def _target_for_region(task: str, block: dict[str, Any], markdown: str) -> dict[str, Any]:
    content = block.get("content")
    text = _content_text(content)
    if task == "table":
        return {"html": text} if text else {}
    if task == "formula":
        return {"latex": text} if text else {}
    if task == "diagram":
        return {"serializations": {"description": text}, "nodes": [], "edges": []} if text else {}
    if task == "seal":
        return {"horizontal_segments": [[text]], "circular_segments": []} if text else {}
    return {"text": text} if text else {}


def _limited_samples(options: MinerUExportOptions) -> Iterable[MinerUSample]:
    for index, sample in enumerate(_discover_samples(options)):
        if options.max_samples is not None and index >= options.max_samples:
            break
        yield sample



def _export_sample_chunk_for_worker(
    options: MinerUExportOptions,
    selected_tasks: tuple[str, ...],
    canonical_root: Path,
    samples: list[MinerUSample],
) -> list[dict[str, Any]]:
    return [_export_sample_for_worker(options, selected_tasks, canonical_root, sample) for sample in samples]


def _export_sample_for_worker(
    options: MinerUExportOptions,
    selected_tasks: tuple[str, ...],
    canonical_root: Path,
    sample: MinerUSample,
) -> dict[str, Any]:
    try:
        exported = MinerUSourceAdapter(options)._export_sample(sample, set(selected_tasks), Path(canonical_root))
        return {"sample_id": sample.sample_id, "exported": _exported_to_rows(exported), "error": None}
    except Exception as exc:
        return {
            "sample_id": sample.sample_id,
            "exported": None,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def _skipped_completed_result(sample: MinerUSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "exported": None,
        "error": None,
        "skipped_completed": True,
    }


def _exported_to_rows(exported: dict[str, Any]) -> dict[str, Any]:
    return {
        "documents": [record.to_dict() for record in exported["documents"]],
        "pages": [record.to_dict() for record in exported["pages"]],
        "regions": [record.to_dict() for record in exported["regions"]],
        "assets": [record.to_dict() for record in exported["assets"]],
        "task_records": {
            task: [record.to_dict() for record in rows]
            for task, rows in exported["task_records"].items()
        },
    }


def _load_completed_source_document_ids(canonical_root: Path, source_name: str) -> set[str]:
    output_dir = canonical_root / "entities/documents" / f"source={source_name}"
    completed: set[str] = set()
    for file_path in sorted(output_dir.glob("part-*.parquet")):
        frame = pd.read_parquet(file_path, columns=["source_document_id"])
        completed.update(str(value) for value in frame["source_document_id"])
    return completed


def _write_exported_rows(
    exported: dict[str, Any],
    *,
    document_writer: "ShardWriter",
    page_writer: "ShardWriter",
    region_writer: "ShardWriter",
    asset_writer: "ShardWriter",
    task_writers: dict[str, "ShardWriter"],
) -> None:
    document_writer.write_many(exported["documents"])
    page_writer.write_many(exported["pages"])
    region_writer.write_many(exported["regions"])
    asset_writer.write_many(exported["assets"])
    for task, rows in exported["task_records"].items():
        task_writers[task].write_many(rows)


@contextmanager
def _source_export_lock(canonical_root: Path, source_name: str) -> Iterable[None]:
    lock_dir = canonical_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"export-source={source_name}.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"source export is already running for {source_name}: {lock_path}") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class _RegionCropSaver:
    def __init__(
        self,
        source_image_path: Path,
        *,
        page_width: int,
        page_height: int,
        dataset_root: Path,
        source_name: str,
    ) -> None:
        self.source_image_path = source_image_path
        self.page_width = page_width
        self.page_height = page_height
        self.dataset_root = dataset_root
        self.source_name = source_name
        self.image: Image.Image | None = None

    def __enter__(self) -> "_RegionCropSaver":
        try:
            with Image.open(self.source_image_path) as raw_image:
                self.image = raw_image.convert("RGB")
        except Exception as exc:
            raise ValueError(f"cannot read source image for region crop: {self.source_image_path}") from exc
        return self

    def __exit__(self, exc_type: object, exc: object, traceback_obj: object) -> None:
        if self.image is not None:
            self.image.close()

    def save(self, bbox: list[float], asset_id: str) -> tuple[str, int, int]:
        if self.image is None:
            raise RuntimeError("region crop saver is not open")
        image_width, image_height = self.image.size
        crop_box = _bbox_to_image_crop_box(bbox, self.page_width, self.page_height, image_width, image_height)
        crop = self.image.crop(crop_box)
        relative_path = region_crop_relative_path(self.source_name, asset_id)
        output_path = self.dataset_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="JPEG", quality=95)
        return relative_path.as_posix(), crop.width, crop.height


def _bbox_to_image_crop_box(
    bbox: list[float],
    page_width: int,
    page_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    scale_x = image_width / float(max(page_width, 1))
    scale_y = image_height / float(max(page_height, 1))
    left = max(0, min(image_width - 1, int(x1 * scale_x)))
    top = max(0, min(image_height - 1, int(y1 * scale_y)))
    right = max(left + 1, min(image_width, int(x2 * scale_x)))
    bottom = max(top + 1, min(image_height, int(y2 * scale_y)))
    return left, top, right, bottom


def _task_for_region_type(region_type: str) -> str:
    if region_type == "table":
        return "table"
    if region_type in {"equation", "formula", "inline_formula"}:
        return "formula"
    if region_type in SEAL_TYPES:
        return "seal"
    if region_type in DIAGRAM_TYPES:
        return "diagram"
    if region_type in TEXT_TYPES:
        return "text"
    return "text"


def _resolve_source_image(sample_id: str, options: MinerUExportOptions) -> tuple[str, bool, bool]:
    if options.source_image_root is None:
        return sample_id, False, False
    for extension in options.image_extensions:
        candidate = options.source_image_root / f"{sample_id}{extension}"
        try:
            exists = candidate.exists()
            is_file = candidate.is_file() if exists else False
        except PermissionError:
            if options.allow_unreadable_images:
                return str(candidate), False, True
            raise
        if exists:
            return str(candidate), is_file, not is_file
    prefixed = _resolve_prefixed_source_image(sample_id, options)
    if prefixed is not None:
        return str(prefixed), True, False
    expected = options.source_image_root / f"{sample_id}{options.image_extensions[0]}"
    if options.allow_unreadable_images:
        return str(expected), False, False
    raise FileNotFoundError(f"source image not found for {sample_id}: {expected}")


def _resolve_prefixed_source_image(sample_id: str, options: MinerUExportOptions) -> Path | None:
    if options.source_image_root is None:
        return None

    candidates = _prefixed_image_candidates(sample_id, options)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]

    collision = re.fullmatch(r"(.+)_([2-9]\d*)", sample_id)
    if not collision:
        return None
    base_sample_id = collision.group(1)
    collision_index = int(collision.group(2)) - 1
    base_candidates = _prefixed_image_candidates(base_sample_id, options)
    if collision_index < len(base_candidates):
        return base_candidates[collision_index]
    return None


def _prefixed_image_candidates(sample_id: str, options: MinerUExportOptions) -> list[Path]:
    if options.source_image_root is None:
        return []
    candidates: list[Path] = []
    extensions = set(options.image_extensions)
    candidates.extend(
        path
        for path in options.source_image_root.iterdir()
        if path.is_file() and path.suffix in extensions and path.stem.startswith(sample_id)
    )
    return sorted(candidates, key=lambda path: path.name)
