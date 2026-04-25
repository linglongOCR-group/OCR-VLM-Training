from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from docparse_dataset.sources.adapters.base import SourceAdapter
from docparse_dataset.canonical.schemas import DetectionRecord, PageRecord, RegionRecord
from docparse_dataset.canonical.writer import CanonicalWriter

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CONTENT_REGION_TYPES = {
    "title", "text", "ref_text", "equation", "table", "table_caption",
    "table_footnote", "image_caption", "image_footnote", "page_footnote",
    "header", "footer", "page_number", "list", "code", "code_caption",
    "algorithm", "aside_text", "seal", "diagram",
}


@dataclass(slots=True)
class MinerUExportOptions:
    mineru_root: Path
    source_image_root: Path | None
    output_root: Path
    dataset_name: str
    mineru_subdir: str = "vlm"
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    shard_size: int = 10000
    max_samples: int | None = None
    allow_unreadable_images: bool = False
    skip_errors: bool = False

    def __post_init__(self) -> None:
        self.mineru_root = Path(self.mineru_root)
        self.source_image_root = Path(self.source_image_root) if self.source_image_root else None
        self.output_root = Path(self.output_root)
        self.image_extensions = tuple(self.image_extensions)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


@dataclass(slots=True)
class MinerUExportReport:
    dataset_name: str
    scanned_samples: int = 0
    skipped_samples: int = 0
    page_records: int = 0
    detection_records: int = 0
    region_records: int = 0
    missing_model_json: int = 0
    missing_middle_json: int = 0
    missing_markdown: int = 0
    missing_images: int = 0
    unreadable_images: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _MinerUSample:
    sample_id: str
    sample_dir: Path
    annot_dir: Path
    model_json: Path
    middle_json: Path | None
    markdown: Path | None
    content_v2_json: Path | None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _discover_samples(mineru_root: Path, mineru_subdir: str) -> Iterable[_MinerUSample]:
    for sample_dir in sorted(path for path in mineru_root.iterdir() if path.is_dir()):
        annot_dir = sample_dir / mineru_subdir
        if not annot_dir.is_dir():
            continue
        sample_id = sample_dir.name
        model_json = annot_dir / f"{sample_id}_model.json"
        yield _MinerUSample(
            sample_id=sample_id,
            sample_dir=sample_dir,
            annot_dir=annot_dir,
            model_json=model_json,
            middle_json=_optional_file(annot_dir / f"{sample_id}_middle.json"),
            markdown=_optional_file(annot_dir / f"{sample_id}.md"),
            content_v2_json=_optional_file(annot_dir / f"{sample_id}_content_list_v2.json"),
        )


def _load_middle_pages(sample: _MinerUSample) -> list[dict[str, Any]]:
    if sample.middle_json is None:
        return []
    middle = _read_json(sample.middle_json)
    pdf_info = middle.get("pdf_info") if isinstance(middle, dict) else None
    return pdf_info if isinstance(pdf_info, list) else []


def _load_markdown(sample: _MinerUSample) -> str:
    return sample.markdown.read_text() if sample.markdown is not None else ""


def _load_content_v2_pages(sample: _MinerUSample) -> list[list[dict[str, Any]]]:
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
            mapping[int(block["index"])] = [float(v) for v in block["bbox"]]
    return mapping


def _scale_bbox_if_normalized(bbox: Sequence[float], width: int, height: int) -> list[float]:
    values = [float(v) for v in bbox]
    if len(values) != 4:
        return values
    if all(0 <= value <= 1 for value in values):
        return [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return values


def _block_bbox(block: dict[str, Any], middle_bboxes: dict[int, list[float]], width: int, height: int) -> list[float]:
    index = block.get("index")
    if index is not None and int(index) in middle_bboxes:
        return middle_bboxes[int(index)]
    return _scale_bbox_if_normalized(block.get("bbox") or [], width, height)


def _resolve_source_image(sample_id: str, options: MinerUExportOptions) -> tuple[str, bool, bool]:
    if options.source_image_root is None:
        return "", False, False
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


def _content_v2_by_type_and_bbox(content_pages: list[list[dict[str, Any]]], page_idx: int) -> list[dict[str, Any]]:
    if page_idx >= len(content_pages) or not isinstance(content_pages[page_idx], list):
        return []
    return [item for item in content_pages[page_idx] if isinstance(item, dict)]


def _find_content_v2_item(block: dict[str, Any], items: list[dict[str, Any]], bbox_px: list[float]) -> dict[str, Any] | None:
    block_type = block.get("type")
    rounded_bbox = [round(v) for v in bbox_px]
    for item in items:
        item_bbox = item.get("bbox")
        if item.get("type") == block_type and isinstance(item_bbox, list) and [round(float(v)) for v in item_bbox] == rounded_bbox:
            return item
    for item in items:
        if item.get("type") == block_type:
            return item
    return None


def _crop_uri(sample: _MinerUSample, content_item: dict[str, Any] | None) -> str | None:
    if not content_item:
        return None
    content = content_item.get("content")
    if not isinstance(content, dict):
        return None
    source = content.get("image_source")
    if not isinstance(source, dict) or not source.get("path"):
        return None
    return str(sample.annot_dir / source["path"])


def _canonical_block(
    block: dict[str, Any],
    middle_bboxes: dict[int, list[float]],
    width: int,
    height: int,
) -> dict[str, Any]:
    bbox_px = _block_bbox(block, middle_bboxes, width, height)
    return {
        "block_id": f"b{int(block.get('index', 0))}",
        "category": str(block.get("type") or "unknown"),
        "bbox_px": bbox_px,
        "angle_deg": float(block.get("angle") or 0.0),
        "content": _content_text(block.get("content")),
        "metadata": {
            "mineru_type": block.get("type"),
            "mineru_index": block.get("index"),
            "mineru_bbox": block.get("bbox"),
        },
    }


def _target_for_region(block: dict[str, Any], content_item: dict[str, Any] | None) -> dict[str, Any]:
    region_type = str(block.get("type") or "unknown")
    content = block.get("content")
    if content is None and content_item is not None:
        content = content_item.get("content")
    text = _content_text(content)
    if region_type == "equation":
        return {"latex": text}
    if region_type == "table":
        return {"html": text}
    if region_type in {"list", "algorithm"}:
        return {"json": text}
    return {"text": text}


def _build_records_for_sample(
    sample: _MinerUSample,
    options: MinerUExportOptions,
    report: MinerUExportReport,
) -> tuple[list[PageRecord], list[DetectionRecord], list[RegionRecord]]:
    if not sample.model_json.is_file():
        report.missing_model_json += 1
        report.skipped_samples += 1
        return [], [], []

    model_pages = _read_json(sample.model_json)
    if not isinstance(model_pages, list):
        raise ValueError(f"model json must be a list: {sample.model_json}")
    middle_pages = _load_middle_pages(sample)
    markdown = _load_markdown(sample)
    content_pages = _load_content_v2_pages(sample)
    if sample.middle_json is None:
        report.missing_middle_json += 1
    if sample.markdown is None:
        report.missing_markdown += 1

    image_uri, image_accessible, image_unreadable = _resolve_source_image(sample.sample_id, options)
    if not image_uri:
        report.missing_images += 1
    elif not image_accessible:
        report.missing_images += 1
    if image_unreadable:
        report.unreadable_images += 1

    page_records: list[PageRecord] = []
    detection_records: list[DetectionRecord] = []
    region_records: list[RegionRecord] = []

    for page_idx, page_blocks in enumerate(model_pages):
        if not isinstance(page_blocks, list):
            continue
        middle_page = middle_pages[page_idx] if page_idx < len(middle_pages) else None
        width, height = _page_size(middle_page)
        middle_bboxes = _middle_bbox_by_index(middle_page)
        content_items = _content_v2_by_type_and_bbox(content_pages, page_idx)
        sample_page_id = sample.sample_id if len(model_pages) == 1 else f"{sample.sample_id}_p{page_idx:04d}"
        page_ref = {
            "sample_id": sample_page_id,
            "dataset_name": options.dataset_name,
            "page_index": page_idx,
            "image_uri": image_uri,
        }

        blocks = [_canonical_block(block, middle_bboxes, width, height) for block in page_blocks if isinstance(block, dict)]

        page_records.append(
            PageRecord(
                sample_id=sample_page_id,
                image_uri=image_uri,
                width_px=width,
                height_px=height,
                task_type="page_markdown",
                source_type="mineru",
                provenance={"dataset_name": options.dataset_name, "mineru_dir": str(sample.annot_dir)},
                metadata={"image_accessible": image_accessible, "mineru_subdir": options.mineru_subdir},
                blocks=blocks,
                reading_order=[block["block_id"] for block in blocks],
                merged_markdown=markdown,
                doc_id=sample.sample_id,
                page_index=page_idx,
            )
        )

        detection_records.append(
            DetectionRecord(
                sample_id=sample_page_id,
                image_uri=image_uri,
                width_px=width,
                height_px=height,
                task_type="layout_detection",
                source_type="mineru",
                provenance={"dataset_name": options.dataset_name, "mineru_dir": str(sample.annot_dir)},
                metadata={"image_accessible": image_accessible, "mineru_subdir": options.mineru_subdir},
                instances=[
                    {
                        "instance_id": block["block_id"],
                        "category": block["category"],
                        "bbox_px": block["bbox_px"],
                        "angle_deg": block["angle_deg"],
                        "mineru_type": block["metadata"]["mineru_type"],
                        "content_present": bool(block["content"]),
                    }
                    for block in blocks
                ],
                source_page_ref=page_ref,
            )
        )

        for block in page_blocks:
            if not isinstance(block, dict):
                continue
            region_type = str(block.get("type") or "unknown")
            bbox_px = _block_bbox(block, middle_bboxes, width, height)
            content_item = _find_content_v2_item(block, content_items, bbox_px)
            crop_uri = _crop_uri(sample, content_item)
            targets = _target_for_region(block, content_item)
            has_target = any(bool(value) for value in targets.values())
            if not has_target and crop_uri is None:
                continue
            if not has_target and region_type not in CONTENT_REGION_TYPES:
                continue
            region_id = f"{sample_page_id}_{region_type}_{int(block.get('index', len(region_records))):04d}"
            region_records.append(
                RegionRecord(
                    sample_id=region_id,
                    image_uri=crop_uri or image_uri,
                    width_px=width,
                    height_px=height,
                    task_type=f"{region_type}_recognition",
                    source_type="mineru",
                    provenance={"dataset_name": options.dataset_name, "mineru_dir": str(sample.annot_dir)},
                    metadata={
                        "image_accessible": image_accessible if crop_uri is None else Path(crop_uri).is_file(),
                        "mineru_type": region_type,
                        "mineru_index": block.get("index"),
                        "uses_page_image": crop_uri is None,
                    },
                    region_type=region_type,
                    targets=targets,
                    source_page_ref=page_ref,
                    bbox_px=bbox_px,
                )
            )

    return page_records, detection_records, region_records


class MinerUAdapter(SourceAdapter):
    name = "mineru"
    version = "1.0.0"

    def __init__(self, options: MinerUExportOptions):
        self._options = options
        self._samples: list[_MinerUSample] | None = None

    def _ensure_scanned(self) -> None:
        if self._samples is None:
            self._samples = list(_discover_samples(self._options.mineru_root, self._options.mineru_subdir))
            if self._options.max_samples is not None:
                self._samples = self._samples[: self._options.max_samples]

    def scan_documents(self) -> Iterable[dict]:
        self._ensure_scanned()
        for sample in self._samples:
            yield {"sample_id": sample.sample_id, "sample_dir": str(sample.sample_dir)}

    def export_documents(self) -> Iterable[dict]:
        self._ensure_scanned()
        for sample in self._samples:
            yield {
                "document_id": f"doc:mineru:{sample.sample_id}",
                "source_name": "mineru",
                "source_document_id": sample.sample_id,
                "document_path": str(sample.sample_dir),
                "document_type": "pdf",
                "metadata": {"mineru_subdir": self._options.mineru_subdir},
                "schema_version": "1.0.0",
            }

    def export_pages(self) -> Iterable[dict]:
        self._ensure_scanned()
        for sample in self._samples:
            if not sample.model_json.is_file():
                continue
            model_pages = _read_json(sample.model_json)
            if not isinstance(model_pages, list):
                continue
            middle_pages = _load_middle_pages(sample)
            for page_idx, page_blocks in enumerate(model_pages):
                if not isinstance(page_blocks, list):
                    continue
                middle_page = middle_pages[page_idx] if page_idx < len(middle_pages) else None
                width, height = _page_size(middle_page)
                sample_page_id = sample.sample_id if len(model_pages) == 1 else f"{sample.sample_id}_p{page_idx:04d}"
                yield {
                    "page_id": f"page:mineru:{sample_page_id}",
                    "document_id": f"doc:mineru:{sample.sample_id}",
                    "source_name": "mineru",
                    "source_page_id": sample_page_id,
                    "page_index": page_idx,
                    "width": width,
                    "height": height,
                    "schema_version": "1.0.0",
                }

    def export_regions(self) -> Iterable[dict]:
        self._ensure_scanned()
        for sample in self._samples:
            if not sample.model_json.is_file():
                continue
            model_pages = _read_json(sample.model_json)
            if not isinstance(model_pages, list):
                continue
            middle_pages = _load_middle_pages(sample)
            content_pages = _load_content_v2_pages(sample)
            for page_idx, page_blocks in enumerate(model_pages):
                if not isinstance(page_blocks, list):
                    continue
                middle_page = middle_pages[page_idx] if page_idx < len(middle_pages) else None
                width, height = _page_size(middle_page)
                middle_bboxes = _middle_bbox_by_index(middle_page)
                content_items = _content_v2_by_type_and_bbox(content_pages, page_idx)
                sample_page_id = sample.sample_id if len(model_pages) == 1 else f"{sample.sample_id}_p{page_idx:04d}"
                for region_idx, block in enumerate(page_blocks):
                    if not isinstance(block, dict):
                        continue
                    region_type = str(block.get("type") or "unknown")
                    bbox_px = _block_bbox(block, middle_bboxes, width, height)
                    yield {
                        "region_id": f"region:mineru:{sample_page_id}:{region_idx:04d}",
                        "page_id": f"page:mineru:{sample_page_id}",
                        "document_id": f"doc:mineru:{sample.sample_id}",
                        "source_name": "mineru",
                        "bbox": bbox_px,
                        "element_type": region_type,
                        "schema_version": "1.0.0",
                    }

    def export_task_records(self, task: str) -> Iterable[dict]:
        self._ensure_scanned()
        for sample in self._samples:
            if not sample.model_json.is_file():
                continue
            model_pages = _read_json(sample.model_json)
            if not isinstance(model_pages, list):
                continue
            middle_pages = _load_middle_pages(sample)
            content_pages = _load_content_v2_pages(sample)
            for page_idx, page_blocks in enumerate(model_pages):
                if not isinstance(page_blocks, list):
                    continue
                middle_page = middle_pages[page_idx] if page_idx < len(middle_pages) else None
                width, height = _page_size(middle_page)
                middle_bboxes = _middle_bbox_by_index(middle_page)
                content_items = _content_v2_by_type_and_bbox(content_pages, page_idx)
                sample_page_id = sample.sample_id if len(model_pages) == 1 else f"{sample.sample_id}_p{page_idx:04d}"

                if task == "layout":
                    blocks = [_canonical_block(blk, middle_bboxes, width, height) for blk in page_blocks if isinstance(blk, dict)]
                    yield {
                        "record_id": f"layout:mineru:{sample_page_id}",
                        "task": "layout",
                        "source_name": "mineru",
                        "document_id": f"doc:mineru:{sample.sample_id}",
                        "page_id": f"page:mineru:{sample_page_id}",
                        "target": {
                            "coordinate_space": "canonical_page_pixel_xyxy",
                            "elements": [
                                {
                                    "region_id": f"region:mineru:{sample_page_id}:{i:04d}",
                                    "bbox": blk["bbox_px"],
                                    "label": blk["category"],
                                    "rotation": blk["angle_deg"],
                                }
                                for i, blk in enumerate(blocks)
                            ],
                        },
                        "schema_version": "1.0.0",
                    }

                elif task in ("text", "table", "formula", "diagram", "seal"):
                    for region_idx, block in enumerate(page_blocks):
                        if not isinstance(block, dict):
                            continue
                        region_type = str(block.get("type") or "unknown")
                        bbox_px = _block_bbox(block, middle_bboxes, width, height)
                        content_item = _find_content_v2_item(block, content_items, bbox_px)
                        targets = _target_for_region(block, content_item)
                        has_target = any(bool(v) for v in targets.values())
                        if not has_target:
                            continue
                        yield {
                            "record_id": f"{task}:mineru:{sample_page_id}:{region_idx:04d}",
                            "task": task,
                            "source_name": "mineru",
                            "document_id": f"doc:mineru:{sample.sample_id}",
                            "page_id": f"page:mineru:{sample_page_id}",
                            "region_id": f"region:mineru:{sample_page_id}:{region_idx:04d}",
                            "target": targets,
                            "schema_version": "1.0.0",
                        }

    def export_all(self, writer: CanonicalWriter, tasks: list[str] | None = None) -> MinerUExportReport:
        report = MinerUExportReport(dataset_name=self._options.dataset_name, options=asdict(self._options))
        self._ensure_scanned()
        tasks = tasks or ["layout", "text", "table", "formula", "diagram", "seal"]

        page_writer = writer
        page_buffer: list[dict] = []
        detection_buffer: list[dict] = []
        region_buffer: list[dict] = []
        task_buffers: dict[str, list[dict]] = {t: [] for t in tasks}

        for sample in self._samples:
            if self._options.max_samples is not None and report.scanned_samples >= self._options.max_samples:
                break
            report.scanned_samples += 1
            try:
                pages, detections, regions = _build_records_for_sample(sample, self._options, report)
            except Exception as exc:
                report.errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                report.skipped_samples += 1
                if not self._options.skip_errors:
                    raise
                continue

            page_buffer.extend(p.to_dict() for p in pages)
            detection_buffer.extend(d.to_dict() for d in detections)
            region_buffer.extend(r.to_dict() for r in regions)
            for task in tasks:
                for record in self.export_task_records_for_sample(sample, task, pages, detections, regions):
                    task_buffers[task].append(record)

        source_name = self._options.dataset_name.lower()
        report.page_records = writer.write_pages(page_buffer, source_name)
        report.detection_records = writer.write_pages(detection_buffer, source_name)
        report.region_records = writer.write_regions(region_buffer, source_name)
        for task in tasks:
            writer.write_task_records(task, source_name, task_buffers[task])

        writer.write_manifest(asdict(report), f"{self._options.dataset_name}_mineru_export_report")
        return report

    def export_task_records_for_sample(self, sample, task, pages, detections, regions):
        return []


# Legacy-compatible standalone export
def export_mineru_dataset(options: MinerUExportOptions) -> MinerUExportReport:
    adapter = MinerUAdapter(options)
    writer = CanonicalWriter(options.output_root)
    return adapter.export_all(writer)
