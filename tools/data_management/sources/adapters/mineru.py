from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from tools.data_management.canonical.writer import CanonicalWriteReport, CanonicalWriter
from tools.data_management.config.resolver import resolve_path
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
    output_root: Path | None = None

    def __post_init__(self) -> None:
        self.mineru_root = Path(self.mineru_root)
        self.source_image_root = Path(self.source_image_root) if self.source_image_root else None
        self.image_extensions = tuple(self.image_extensions)
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


@dataclass(slots=True)
class MinerUExportReport:
    dataset_name: str
    scanned_samples: int = 0
    skipped_samples: int = 0
    documents: int = 0
    pages: int = 0
    regions: int = 0
    task_records: dict[str, int] = field(default_factory=dict)
    assets: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)


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
        root = Path(dataset_root or os.environ.get("OCR_DATASET_ROOT") or profile_path.parent)

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
                output_root=resolve_profile_path(data.get("output_root") or data.get("canonical_output")),
            )
        )

    def scan_documents(self) -> Iterable[dict]:
        for sample in _discover_samples(self.options):
            yield {"source_document_id": sample.sample_id, "sample_dir": str(sample.sample_dir)}

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
        if progress:
            progress.log(
                "export-source",
                phase="start",
                source=self.options.dataset_name,
                canonical_root=canonical_root,
                tasks=sorted(selected_tasks),
            )
        manifest_writer = CanonicalWriter(canonical_root, overwrite_partitions=overwrite_partitions)
        report = MinerUExportReport(dataset_name=self.options.dataset_name)
        document_writer = _ShardWriter(
            canonical_root / "entities/documents" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )
        page_writer = _ShardWriter(
            canonical_root / "entities/pages" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )
        region_writer = _ShardWriter(
            canonical_root / "entities/regions" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )
        task_writers = {
            task: _ShardWriter(
                canonical_root / "records" / task / f"source={self.options.dataset_name}",
                self.options.shard_size,
                overwrite=overwrite_partitions,
            )
            for task in selected_tasks
        }
        asset_writer = _ShardWriter(
            canonical_root / "assets/manifests" / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )

        for sample in _discover_samples(self.options):
            if self.options.max_samples is not None and report.scanned_samples >= self.options.max_samples:
                break
            report.scanned_samples += 1
            if progress:
                progress.update(
                    "export-source",
                    report.scanned_samples,
                    total=self.options.max_samples,
                    scanned=report.scanned_samples,
                    sample=sample.sample_id,
                )
            try:
                exported = self._export_sample(sample, selected_tasks)
            except Exception as exc:
                report.errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                report.skipped_samples += 1
                if progress:
                    progress.log("export-source", phase="skip", sample=sample.sample_id, error=exc)
                if not self.options.skip_errors:
                    raise
                continue
            document_writer.write_many(record.to_dict() for record in exported["documents"])
            page_writer.write_many(record.to_dict() for record in exported["pages"])
            region_writer.write_many(record.to_dict() for record in exported["regions"])
            asset_writer.write_many(record.to_dict() for record in exported["assets"])
            for task, rows in exported["task_records"].items():
                task_writers[task].write_many(record.to_dict() for record in rows)

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

    def _export_sample(self, sample: MinerUSample, selected_tasks: set[str]) -> dict[str, Any]:
        if not sample.model_json.is_file():
            raise FileNotFoundError(f"missing model json: {sample.model_json}")
        model_pages = _read_json(sample.model_json)
        if not isinstance(model_pages, list):
            raise ValueError(f"model json must be a list: {sample.model_json}")
        middle_pages = _load_middle_pages(sample)
        content_pages = _load_content_v2_pages(sample)
        markdown = _load_markdown(sample)
        image_path, image_accessible, image_unreadable = _resolve_source_image(sample.sample_id, self.options)
        document_hash = stable_hash(sample.sample_id)
        document_id = stable_id("doc", self.options.dataset_name, document_hash)
        document = CanonicalDocument(
            document_id=document_id,
            source_name=self.options.dataset_name,
            source_document_id=sample.sample_id,
            document_path=image_path,
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

        for page_index, page_blocks in enumerate(model_pages):
            if not isinstance(page_blocks, list):
                continue
            middle_page = middle_pages[page_index] if page_index < len(middle_pages) else None
            width, height = _page_size(middle_page)
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
                path=image_path,
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
            content_items = _content_v2_by_type_and_bbox(content_pages, page_index)
            layout_elements: list[dict[str, Any]] = []
            for reading_order, block in enumerate((item for item in page_blocks if isinstance(item, dict)), start=1):
                block_type = str(block.get("type") or "unknown")
                bbox = _block_bbox(block, middle_bboxes, width, height)
                source_annotation_id = str(block.get("index", reading_order))
                region_hash = stable_hash({"sample": sample.sample_id, "page": page_index, "annotation": source_annotation_id, "bbox": bbox})
                region_id = stable_id("region", self.options.dataset_name, document_hash, f"{page_index:04d}", region_hash)
                content_item = _find_content_v2_item(block, content_items, bbox)
                crop_path = _crop_uri(sample, content_item)
                crop_asset_id = None
                if crop_path:
                    crop_asset_id = stable_id("asset", "region_crop", self.options.dataset_name, region_hash)
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
                            width=width,
                            height=height,
                            format=Path(crop_path).suffix.lstrip(".") or "unknown",
                            parent_asset_id=page_asset_id,
                            transform_spec_hash=stable_hash({"operation": "crop", "bbox": bbox}),
                            transform={"operation": "crop", "bbox": bbox, "source": "mineru_content_v2"},
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
                    target = _target_for_region(task, block, content_item, markdown)
                    if target:
                        task_records[task].append(
                            CanonicalTaskRecord(
                                record_id=stable_id(task, self.options.dataset_name, region_id),
                                task=task,
                                source_name=self.options.dataset_name,
                                document_id=document_id,
                                page_id=page_id,
                                region_id=region_id,
                                image_asset_id=crop_asset_id or page_asset_id,
                                target=target,
                                category=block_type,
                                provenance={"source_annotation_id": source_annotation_id, "mineru_dir": str(sample.annot_dir)},
                                metadata={"width": width, "height": height, "uses_page_image": crop_asset_id is None},
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


def _load_content_v2_pages(sample: MinerUSample) -> list[list[dict[str, Any]]]:
    if sample.content_v2_json is None:
        return []
    data = _read_json(sample.content_v2_json)
    return data if isinstance(data, list) else []


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


def _content_v2_by_type_and_bbox(content_pages: list[list[dict[str, Any]]], page_idx: int) -> list[dict[str, Any]]:
    if page_idx >= len(content_pages) or not isinstance(content_pages[page_idx], list):
        return []
    return [item for item in content_pages[page_idx] if isinstance(item, dict)]


def _find_content_v2_item(block: dict[str, Any], items: list[dict[str, Any]], bbox: list[float]) -> dict[str, Any] | None:
    block_type = block.get("type")
    rounded_bbox = [round(value) for value in bbox]
    for item in items:
        item_bbox = item.get("bbox")
        if item.get("type") == block_type and isinstance(item_bbox, list) and [round(float(value)) for value in item_bbox] == rounded_bbox:
            return item
    for item in items:
        if item.get("type") == block_type:
            return item
    return None


def _crop_uri(sample: MinerUSample, content_item: dict[str, Any] | None) -> str | None:
    if not content_item:
        return None
    content = content_item.get("content")
    if not isinstance(content, dict):
        return None
    image_source = content.get("image_source")
    if not isinstance(image_source, dict) or not image_source.get("path"):
        return None
    return str(sample.annot_dir / image_source["path"])


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


def _target_for_region(task: str, block: dict[str, Any], content_item: dict[str, Any] | None, markdown: str) -> dict[str, Any]:
    content = block.get("content")
    if content is None and content_item is not None:
        content = content_item.get("content")
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
    expected = options.source_image_root / f"{sample_id}{options.image_extensions[0]}"
    if options.allow_unreadable_images:
        return str(expected), False, False
    raise FileNotFoundError(f"source image not found for {sample_id}: {expected}")


class _ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int, *, overwrite: bool) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0
        self.count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for stale in self.output_dir.glob("part-*.parquet"):
                stale.unlink()

    def write_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.buffer.append(row)
            self.count += 1
            if len(self.buffer) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        pd.DataFrame(self.buffer).to_parquet(self.output_dir / f"part-{self.shard_index:05d}.parquet", index=False)
        self.buffer = []
        self.shard_index += 1

    def close(self) -> None:
        self.flush()
