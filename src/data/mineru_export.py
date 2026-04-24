from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from src.data.schemas import DetectionRecord, PageRecord, RegionRecord


CANONICAL_RECORDS = ("page", "detection", "region")
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CONTENT_REGION_TYPES = {
    "title",
    "text",
    "ref_text",
    "equation",
    "table",
    "table_caption",
    "table_footnote",
    "image_caption",
    "image_footnote",
    "page_footnote",
    "header",
    "footer",
    "page_number",
    "list",
    "code",
    "code_caption",
    "algorithm",
    "aside_text",
    "seal",
    "diagram",
}


@dataclass(slots=True)
class MinerUExportOptions:
    mineru_root: Path
    source_image_root: Path | None
    output_root: Path
    dataset_name: str
    mineru_subdir: str = "vlm"
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    records: tuple[str, ...] = CANONICAL_RECORDS
    shard_size: int = 10000
    max_samples: int | None = None
    allow_unreadable_images: bool = False
    skip_errors: bool = False

    def __post_init__(self) -> None:
        self.mineru_root = Path(self.mineru_root)
        self.source_image_root = Path(self.source_image_root) if self.source_image_root else None
        self.output_root = Path(self.output_root)
        self.image_extensions = tuple(self.image_extensions)
        self.records = tuple(self.records)
        unknown_records = set(self.records) - set(CANONICAL_RECORDS)
        if unknown_records:
            raise ValueError(f"unsupported records: {sorted(unknown_records)}")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")


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
class MinerUSample:
    sample_id: str
    sample_dir: Path
    annot_dir: Path
    model_json: Path
    middle_json: Path | None
    markdown: Path | None
    content_v2_json: Path | None


def load_options_from_profile(path: str | Path) -> MinerUExportOptions:
    profile_path = Path(path)
    data = yaml.safe_load(profile_path.read_text()) or {}
    return MinerUExportOptions(
        mineru_root=Path(data["mineru_root"]),
        source_image_root=Path(data["source_image_root"]) if data.get("source_image_root") else None,
        output_root=Path(data["output_root"]),
        dataset_name=data["dataset_name"],
        mineru_subdir=data.get("mineru_subdir", "vlm"),
        image_extensions=tuple(data.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)),
        records=tuple(data.get("records", CANONICAL_RECORDS)),
        shard_size=int(data.get("shard_size", 10000)),
        max_samples=int(data["max_samples"]) if data.get("max_samples") is not None else None,
        allow_unreadable_images=bool(data.get("allow_unreadable_images", False)),
        skip_errors=bool(data.get("skip_errors", False)),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _discover_samples(options: MinerUExportOptions) -> Iterable[MinerUSample]:
    for sample_dir in sorted(path for path in options.mineru_root.iterdir() if path.is_dir()):
        annot_dir = sample_dir / options.mineru_subdir
        if not annot_dir.is_dir():
            continue
        sample_id = sample_dir.name
        model_json = annot_dir / f"{sample_id}_model.json"
        yield MinerUSample(
            sample_id=sample_id,
            sample_dir=sample_dir,
            annot_dir=annot_dir,
            model_json=model_json,
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


def _crop_uri(sample: MinerUSample, content_item: dict[str, Any] | None) -> str | None:
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
    sample: MinerUSample,
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
        if "page" in options.records:
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

        if "detection" in options.records:
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

        if "region" in options.records:
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


def _record_to_dict(record: PageRecord | DetectionRecord | RegionRecord) -> dict[str, Any]:
    return record.to_dict()


def _write_sharded(records: Sequence[PageRecord | DetectionRecord | RegionRecord], output_dir: Path, shard_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        rows = [_record_to_dict(record) for record in records[start : start + shard_size]]
        pd.DataFrame(rows).to_parquet(output_dir / f"part-{shard_index:05d}.parquet", index=False)


class _ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[PageRecord | DetectionRecord | RegionRecord] = []
        self.shard_index = 0
        self.count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stale_part in self.output_dir.glob("part-*.parquet"):
            stale_part.unlink()

    def write_many(self, records: Sequence[PageRecord | DetectionRecord | RegionRecord]) -> None:
        for record in records:
            self.buffer.append(record)
            self.count += 1
            if len(self.buffer) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        rows = [_record_to_dict(record) for record in self.buffer]
        pd.DataFrame(rows).to_parquet(self.output_dir / f"part-{self.shard_index:05d}.parquet", index=False)
        self.buffer = []
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


def _write_report(report: MinerUExportReport, output_root: Path) -> None:
    manifest_dir = output_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{report.dataset_name}_mineru_export_report.json"
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))


def export_mineru_dataset(options: MinerUExportOptions) -> MinerUExportReport:
    report = MinerUExportReport(dataset_name=options.dataset_name, options=asdict(options))
    page_writer = (
        _ShardWriter(options.output_root / "page_records" / options.dataset_name, options.shard_size)
        if "page" in options.records
        else None
    )
    detection_writer = (
        _ShardWriter(options.output_root / "detection_records" / options.dataset_name, options.shard_size)
        if "detection" in options.records
        else None
    )
    region_writer = (
        _ShardWriter(options.output_root / "region_records" / options.dataset_name, options.shard_size)
        if "region" in options.records
        else None
    )

    for sample in _discover_samples(options):
        if options.max_samples is not None and report.scanned_samples >= options.max_samples:
            break
        report.scanned_samples += 1
        try:
            pages, detections, regions = _build_records_for_sample(sample, options, report)
        except Exception as exc:
            report.errors.append({"sample_id": sample.sample_id, "error": str(exc)})
            report.skipped_samples += 1
            if not options.skip_errors:
                raise
            continue
        if page_writer is not None:
            page_writer.write_many(pages)
        if detection_writer is not None:
            detection_writer.write_many(detections)
        if region_writer is not None:
            region_writer.write_many(regions)

    for writer in (page_writer, detection_writer, region_writer):
        if writer is not None:
            writer.close()

    report.page_records = page_writer.count if page_writer is not None else 0
    report.detection_records = detection_writer.count if detection_writer is not None else 0
    report.region_records = region_writer.count if region_writer is not None else 0

    _write_report(report, options.output_root)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MinerU annotations to canonical OCR records.")
    parser.add_argument("--profile", help="YAML profile with MinerU export options.")
    parser.add_argument("--mineru-root")
    parser.add_argument("--source-image-root")
    parser.add_argument("--output-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--mineru-subdir", default="vlm")
    parser.add_argument("--image-extensions", nargs="+", default=list(DEFAULT_IMAGE_EXTENSIONS))
    parser.add_argument("--records", nargs="+", choices=CANONICAL_RECORDS, default=list(CANONICAL_RECORDS))
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-unreadable-images", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _options_from_args(args: argparse.Namespace) -> MinerUExportOptions:
    if args.profile:
        options = load_options_from_profile(args.profile)
        if args.max_samples is not None:
            options.max_samples = args.max_samples
        return options
    missing = [name for name in ("mineru_root", "output_root", "dataset_name") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required arguments without --profile: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    return MinerUExportOptions(
        mineru_root=Path(args.mineru_root),
        source_image_root=Path(args.source_image_root) if args.source_image_root else None,
        output_root=Path(args.output_root),
        dataset_name=args.dataset_name,
        mineru_subdir=args.mineru_subdir,
        image_extensions=tuple(args.image_extensions),
        records=tuple(args.records),
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        allow_unreadable_images=args.allow_unreadable_images,
        skip_errors=args.skip_errors,
    )


def main() -> None:
    report = export_mineru_dataset(_options_from_args(_parse_args()))
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
