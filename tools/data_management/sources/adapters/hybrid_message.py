from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
from tools.data_management.sources.adapters.base import SourceAdapter
from tools.data_management.utils.io import read_yaml

DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_TASKS = ("layout", "text", "table", "formula")

#
# Regex for parsing MinerU box tokens
#
BOX_TOKEN_RE = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"
    r"<\|ref_start\|>([^<]+)<\|ref_end\|>"
    r"(<\|rotate_(?:up|right|down|left)\|>)?"
)

ROTATION_MAP = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}

#
# Task detection from user message
#
TASK_PATTERNS = {
    "layout": "Layout Detection",
    "table": "Table Recognition",
    "formula": "Formula Recognition",
    "text": "Text Recognition",
}


def _detect_task(user_content: str) -> str:
    for task, pattern in TASK_PATTERNS.items():
        if pattern in user_content:
            return task
    return "text"


def _image_dimensions(path: str, *, task: str | None = None) -> tuple[int, int]:
    """Read image dimensions. Only reads the actual file for layout tasks
    where pixel-accurate bbox conversion is needed. Returns (1, 1) fallback
    for other tasks or on failure."""
    if task != "layout":
        return 1, 1
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return 1, 1


@dataclass(slots=True)
class HybridMessageExportOptions:
    data_file: Path
    dataset_name: str
    source_image_root: Path | None = None
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    image_materialization: str = "copy"
    max_samples: int | None = None
    shard_size: int = 5000
    allow_unreadable_images: bool = False
    skip_errors: bool = False
    output_root: Path | None = None

    def __post_init__(self) -> None:
        self.data_file = Path(self.data_file)
        if self.source_image_root is not None:
            self.source_image_root = Path(self.source_image_root)
        self.image_extensions = tuple(self.image_extensions)
        if self.image_materialization not in {"copy", "source_reference"}:
            raise ValueError("image_materialization must be copy or source_reference")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


@dataclass(slots=True)
class HybridMessageExportReport:
    dataset_name: str
    scanned_samples: int = 0
    skipped_samples: int = 0
    documents: int = 0
    pages: int = 0
    regions: int = 0
    task_records: dict[str, int] = field(default_factory=dict)
    assets: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)


class HybridMessageSourceAdapter(SourceAdapter):
    name = "hybrid_message"
    version = "1.0.0"

    def __init__(self, options: HybridMessageExportOptions) -> None:
        self.options = options

    @classmethod
    def from_profile(
        cls, path: str | Path, *, dataset_root: str | Path | None = None
    ) -> "HybridMessageSourceAdapter":
        data = read_yaml(path)
        profile_path = Path(path)
        root = Path(
            dataset_root or Path(os.environ.get(DATA_ROOT_ENV, profile_path.parent))
        )

        def resolve_profile_path(value: str | None) -> Path | None:
            if not value:
                return None
            return resolve_path(value, base=profile_path.parent, dataset_root=root)

        source_path = resolve_profile_path(data.get("source_path"))
        data_file_value = data.get("data_file")
        if data_file_value:
            data_file_path = Path(str(data_file_value))
            if data_file_path.is_absolute():
                data_file = data_file_path
            elif str(data_file_value).startswith(("sources/", "canonical", "views/")):
                data_file = resolve_path(data_file_value, base=profile_path.parent, dataset_root=root)
            elif source_path is not None:
                data_file = source_path / data_file_path
            else:
                data_file = profile_path.parent / data_file_path
        else:
            if source_path is None:
                raise ValueError("hybrid_message source profile requires source_path or data_file")
            data_file = source_path / "data.json"

        return cls(
            HybridMessageExportOptions(
                data_file=data_file,
                dataset_name=data["dataset_name"],
                source_image_root=resolve_profile_path(data.get("source_image_root") or data.get("source_path")),
                image_extensions=tuple(
                    data.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)
                ),
                image_materialization=str(data.get("image_materialization", "copy")),
                max_samples=int(data["max_samples"]) if data.get("max_samples") is not None else None,
                shard_size=int(data.get("shard_size", 5000)),
                allow_unreadable_images=bool(data.get("allow_unreadable_images", False)),
                skip_errors=bool(data.get("skip_errors", False)),
                output_root=resolve_profile_path(
                    data.get("output_root") or data.get("canonical_output")
                ),
            )
        )

    def scan_documents(self) -> Iterable[dict]:
        records = self._load_records()
        for idx, record in enumerate(records):
            images = record.get("images", [])
            img_path = images[0] if images else ""
            yield {
                "source_document_id": f"record_{idx:06d}",
                "image_path": img_path,
            }

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
            )
        manifest_writer = CanonicalWriter(
            canonical_root, overwrite_partitions=overwrite_partitions
        )
        report = HybridMessageExportReport(dataset_name=self.options.dataset_name)

        document_writer = _ShardWriter(
            canonical_root
            / "entities/documents"
            / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )
        page_writer = _ShardWriter(
            canonical_root
            / "entities/pages"
            / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )
        region_writer = _ShardWriter(
            canonical_root
            / "entities/regions"
            / f"source={self.options.dataset_name}",
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
            canonical_root
            / "assets/manifests"
            / f"source={self.options.dataset_name}",
            self.options.shard_size,
            overwrite=overwrite_partitions,
        )

        records = self._load_records()
        for idx, record in enumerate(records):
            if self.options.max_samples is not None and report.scanned_samples >= self.options.max_samples:
                break
            report.scanned_samples += 1
            if progress:
                progress.update(
                    "export-source",
                    report.scanned_samples,
                    total=min(len(records), self.options.max_samples or len(records)),
                    scanned=report.scanned_samples,
                    record=idx,
                )
            try:
                exported = self._export_record(idx, record, selected_tasks, canonical_root)
            except Exception as exc:
                report.errors.append(
                    {"record_index": str(idx), "error": str(exc)}
                )
                report.skipped_samples += 1
                if progress:
                    progress.log("export-source", phase="skip", record=idx, error=exc)
                if not self.options.skip_errors:
                    raise
                continue
            document_writer.write_many(
                record.to_dict() for record in exported["documents"]
            )
            page_writer.write_many(record.to_dict() for record in exported["pages"])
            region_writer.write_many(record.to_dict() for record in exported["regions"])
            asset_writer.write_many(record.to_dict() for record in exported["assets"])
            for task, rows in exported["task_records"].items():
                task_writers[task].write_many(record.to_dict() for record in rows)

        for writer in [
            document_writer,
            page_writer,
            region_writer,
            asset_writer,
            *task_writers.values(),
        ]:
            writer.close()
            if progress and writer.count:
                progress.log("export-source", phase="wrote", path=writer.output_dir, rows=writer.count)

        report.documents = document_writer.count
        report.pages = page_writer.count
        report.regions = region_writer.count
        report.assets = asset_writer.count
        report.task_records = {
            task: writer.count
            for task, writer in sorted(task_writers.items())
            if writer.count
        }
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

    def _load_records(self) -> list[dict[str, Any]]:
        with open(self.options.data_file, "r") as f:
            return json.load(f)

    def _export_record(
        self, idx: int, record: dict[str, Any], selected_tasks: set[str], canonical_root: Path
    ) -> dict[str, Any]:
        messages = record.get("messages", [])
        images = record.get("images", [])

        if not messages or len(messages) < 2:
            raise ValueError(f"record {idx}: need at least 2 messages (user + assistant)")
        if not images:
            raise ValueError(f"record {idx}: no image path")

        image_path = str(images[0])
        source_image_path = self._resolve_source_image_path(Path(image_path))
        dataset_root = infer_dataset_root_from_path(canonical_root)
        user_content = messages[0].get("content", "")
        assistant_content = messages[1].get("content", "")

        task = _detect_task(user_content)
        if task not in selected_tasks:
            return {"documents": [], "pages": [], "regions": [], "assets": [], "task_records": {}}

        # Get image dimensions
        width, height = _image_dimensions(image_path, task=task)

        # Generate IDs
        doc_hash = stable_hash(f"{self.options.dataset_name}:{idx:06d}")
        doc_id = stable_id("doc", self.options.dataset_name, doc_hash)
        page_id = stable_id("page", self.options.dataset_name, doc_hash, "0000")
        region_id = stable_id("region", self.options.dataset_name, doc_hash, "0000")

        # Asset for the page/image
        page_asset_id = stable_id(
            "asset", "page_render", self.options.dataset_name, doc_hash, "0000"
        )
        crop_asset_id = stable_id("asset", "region_crop", self.options.dataset_name, doc_hash, "0000")
        image_format = Path(image_path).suffix.lstrip(".") or "unknown"
        if self.options.image_materialization == "source_reference":
            crop_path, crop_width, crop_height, crop_format = _reference_region_image(
                source_image_path,
                dataset_root=dataset_root,
                extension=image_format,
            )
        else:
            crop_path, crop_width, crop_height, crop_format = _copy_region_image(
                source_image_path,
                dataset_root=dataset_root,
                source_name=self.options.dataset_name,
                asset_id=crop_asset_id,
                extension=image_format,
            )
        try:
            page_image_path = dataset_relative_path(source_image_path, dataset_root) if source_image_path.is_absolute() else image_path
        except ValueError:
            page_image_path = crop_path

        # Document
        document = CanonicalDocument(
            document_id=doc_id,
            source_name=self.options.dataset_name,
            source_document_id=f"record_{idx:06d}",
            document_path=page_image_path,
            document_type="image",
            num_pages=1,
            metadata={"record_index": idx, "task": task},
        )

        # Page
        page = CanonicalPage(
            page_id=page_id,
            document_id=doc_id,
            source_name=self.options.dataset_name,
            source_page_id=f"record_{idx:06d}",
            page_index=0,
            page_image_asset_id=page_asset_id,
            width=width,
            height=height,
            attributes={"adapter": "hybrid_message"},
        )

        # Asset (page render)
        page_asset = AssetRecord(
            asset_id=page_asset_id,
            asset_type="page_render",
            source_name=self.options.dataset_name,
            document_id=doc_id,
            page_id=page_id,
            region_id=None,
            task=task,
            path=page_image_path,
            width=width,
            height=height,
            format=image_format,
            coordinate_space="canonical_page_pixel_xyxy",
            transform={"source": "hybrid_message"},
        )

        crop_asset = AssetRecord(
            asset_id=crop_asset_id,
            asset_type="region_crop",
            source_name=self.options.dataset_name,
            document_id=doc_id,
            page_id=page_id,
            region_id=region_id,
            task=task,
            path=crop_path,
            width=crop_width,
            height=crop_height,
            format=crop_format,
            parent_asset_id=page_asset_id,
            transform_spec_hash=stable_hash({"operation": "copy", "source": image_path}),
            transform={"operation": "copy", "source": "hybrid_message_image"},
        )

        # Region (whole image for non-layout, or sentinel for layout)
        region = CanonicalRegion(
            region_id=region_id,
            page_id=page_id,
            document_id=doc_id,
            source_name=self.options.dataset_name,
            source_annotation_id=f"record_{idx:06d}",
            bbox=[0.0, 0.0, float(width), float(height)],
            rotation=0.0,
            element_type=task,
            category=task,
            reading_order=1,
            crop_asset_id=crop_asset_id,
            metadata={"record_index": idx},
        )

        # Task record with parsed target
        target = _parse_target(task, assistant_content, width, height)
        task_record = CanonicalTaskRecord(
            record_id=stable_id(task, self.options.dataset_name, region_id),
            task=task,
            source_name=self.options.dataset_name,
            document_id=doc_id,
            page_id=page_id,
            region_id=region_id,
            image_asset_id=crop_asset_id,
            target=target,
            category=task,
            provenance={"record_index": idx, "source_file": str(self.options.data_file)},
            metadata={"width": crop_width, "height": crop_height, "uses_page_image": False},
        )

        return {
            "documents": [document],
            "pages": [page],
            "regions": [region],
            "assets": [page_asset, crop_asset],
            "task_records": {task: [task_record]},
        }

    def _resolve_source_image_path(self, image_path: Path) -> Path:
        by_name = self.options.source_image_root / image_path.name if self.options.source_image_root is not None else None
        if self.options.image_materialization == "source_reference" and by_name is not None and by_name.is_file():
            return by_name
        if image_path.is_file():
            return image_path
        if by_name is not None and by_name.is_file():
            return by_name
        return image_path


def _parse_target(task: str, assistant_content: str, width: int, height: int) -> dict[str, Any]:
    """Parse assistant content into canonical target format based on task type."""
    if task == "layout":
        return _parse_layout_target(assistant_content, width, height)
    elif task == "table":
        return {"otsl": assistant_content.strip()}
    elif task == "formula":
        latex = assistant_content.strip()
        # Strip $$ wrappers if present
        if latex.startswith("$$") and latex.endswith("$$"):
            latex = latex[2:-2].strip()
        elif latex.startswith("$") and latex.endswith("$"):
            latex = latex[1:-1].strip()
        return {"latex": latex}
    elif task == "text":
        return {"text": assistant_content.strip()}
    else:
        return {"raw": assistant_content.strip()}


def _parse_layout_target(
    assistant_content: str, width: int, height: int
) -> dict[str, Any]:
    """
    Parse MinerU box tokens into structured layout elements.
    Box token coordinates are in grid (0-1000) format; convert to pixel coordinates.
    """
    elements: list[dict[str, Any]] = []
    for match in BOX_TOKEN_RE.finditer(assistant_content):
        gx1 = int(match.group(1))
        gy1 = int(match.group(2))
        gx2 = int(match.group(3))
        gy2 = int(match.group(4))
        label = match.group(5)
        rotate_token = match.group(6) or "<|rotate_up|>"
        rotation = ROTATION_MAP.get(rotate_token, 0)

        # Convert grid (0-1000) to pixel
        px1 = gx1 * width / 1000.0
        py1 = gy1 * height / 1000.0
        px2 = gx2 * width / 1000.0
        py2 = gy2 * height / 1000.0

        elements.append(
            {
                "bbox": [px1, py1, px2, py2],
                "label": label,
                "rotation": rotation,
                "reading_order": len(elements) + 1,
            }
        )

    return {
        "coordinate_space": "canonical_page_pixel_xyxy",
        "elements": elements,
    }


def _copy_region_image(
    source_image_path: Path,
    *,
    dataset_root: Path,
    source_name: str,
    asset_id: str,
    extension: str,
) -> tuple[str, int, int, str]:
    if not source_image_path.is_file():
        raise FileNotFoundError(f"source image does not exist: {source_image_path}")
    normalized_extension = _normalize_extension(extension)
    safe_asset_id = asset_id.replace(":", "_").replace("/", "_")
    relative_path = (
        Path("canonical")
        / "assets"
        / "files"
        / f"source={source_name}"
        / "region_crop"
        / safe_asset_id[:2]
        / f"{safe_asset_id}{normalized_extension}"
    )
    output_path = dataset_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_image_path, output_path)
    width, height = _copied_image_dimensions(output_path)
    return relative_path.as_posix(), width, height, normalized_extension.lstrip(".")


def _reference_region_image(
    source_image_path: Path,
    *,
    dataset_root: Path,
    extension: str,
) -> tuple[str, int, int, str]:
    if not source_image_path.is_file():
        raise FileNotFoundError(f"source image does not exist: {source_image_path}")
    relative_path = dataset_relative_path(source_image_path, dataset_root)
    normalized_extension = _normalize_extension(extension)
    return relative_path, 1, 1, normalized_extension.lstrip(".")


def _normalize_extension(extension: str | None) -> str:
    if not extension:
        return ".jpg"
    extension = extension.lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension


def _copied_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return 1, 1


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
        pd.DataFrame(self.buffer).to_parquet(
            self.output_dir / f"part-{self.shard_index:05d}.parquet", index=False
        )
        self.buffer = []
        self.shard_index += 1

    def close(self) -> None:
        self.flush()
