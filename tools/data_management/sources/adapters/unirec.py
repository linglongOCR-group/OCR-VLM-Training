from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from tools.data_management.canonical.writer import CanonicalWriteReport, CanonicalWriter
from tools.data_management.config.resolver import resolve_path
from tools.data_management.paths import DATA_ROOT_ENV, dataset_relative_path, dataset_root_from_env, infer_dataset_root_from_path
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
from tools.data_management.sources.adapters._shared import ExportReport, ShardWriter
from tools.data_management.sources.adapters.base import SourceAdapter
from tools.data_management.utils.io import read_yaml


DEFAULT_TASKS = ("text", "formula", "table")
NEWLINE_TOKENS = ("<|ln|>", "<|pn|>", "<|sn|>", "<<<change_line_token_wrap>>>")
TABLE_HINT_RE = re.compile(r"\b(table|tabular|tbl|cell|row|column|col)\b", re.IGNORECASE)
INLINE_FORMULA_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
ESCAPED_DOLLAR_FORMULA_RE = re.compile(r"\\\$(.*?)\\\$", re.DOTALL)
DISPLAY_FORMULA_RE = re.compile(r"^\s*(?:\\\[(?P<bracket>.*?)\\\]|\$\$(?P<dollar>.*?)\$\$)\s*$", re.DOTALL)
LATEX_SIGNAL_RE = re.compile(
    r"(\\[a-zA-Z]+|[_^{}]|\\frac|\\sum|\\int|\\mathrm|\\left|\\right|\\begin|\\end)"
)


@dataclass(slots=True)
class UniRecExportOptions:
    source_root: Path
    dataset_name: str
    max_samples: int | None = None
    shard_size: int = 10000
    skip_errors: bool = False
    output_root: Path | None = None
    allow_unreadable_images: bool = False

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root)
        if self.output_root is not None:
            self.output_root = Path(self.output_root)
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


class UniRecSourceAdapter(SourceAdapter):
    name = "unirec"
    version = "1.0.0"

    def __init__(self, options: UniRecExportOptions) -> None:
        self.options = options

    @classmethod
    def from_profile(cls, path: str | Path, *, dataset_root: str | Path | None = None) -> "UniRecSourceAdapter":
        data = read_yaml(path)
        profile_path = Path(path)
        root = Path(dataset_root or os.environ.get(DATA_ROOT_ENV, profile_path.parent))

        def resolve(value: str | None) -> Path | None:
            return resolve_path(value, base=profile_path.parent, dataset_root=root) if value else None

        source_root = resolve(data.get("source_path") or data.get("source_root"))
        if source_root is None:
            raise ValueError("unirec source profile requires source_path or source_root")
        return cls(
            UniRecExportOptions(
                source_root=source_root,
                dataset_name=data["dataset_name"],
                max_samples=int(data["max_samples"]) if data.get("max_samples") is not None else None,
                shard_size=int(data.get("shard_size", 10000)),
                skip_errors=bool(data.get("skip_errors", False)),
                allow_unreadable_images=bool(data.get("allow_unreadable_images", False)),
                output_root=resolve(data.get("output_root") or data.get("canonical_output")),
            )
        )

    def scan_documents(self) -> Iterable[dict[str, Any]]:
        for item in self._iter_source_records():
            record = item["record"]
            yield {
                "source_document_id": str(record.get("record_id") or item["idx"]),
                "subset_id": item["subset_id"],
                "image_path": str(record.get("image_path") or ""),
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
        if progress:
            progress.log(
                "export-source",
                phase="start",
                source=self.options.dataset_name,
                canonical_root=canonical_root,
                tasks=sorted(selected_tasks),
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

        for item in self._iter_source_records():
            if self.options.max_samples is not None and report.scanned_samples >= self.options.max_samples:
                break
            report.scanned_samples += 1
            if progress:
                progress.update(
                    "export-source",
                    report.scanned_samples,
                    total=self.options.max_samples,
                    scanned=report.scanned_samples,
                    subset=item["subset_id"],
                )
            try:
                exported = self._export_record(item, selected_tasks, canonical_root)
            except Exception as exc:
                report.errors.append({"record": str(item["idx"]), "error": str(exc)})
                report.skipped_samples += 1
                if not self.options.skip_errors:
                    raise
                continue
            if not exported["task_records"]:
                report.skipped_samples += 1
                continue
            task = next(iter(exported["task_records"]))
            document_writer.write_many(record.to_dict() for record in exported["documents"])
            page_writer.write_many(record.to_dict() for record in exported["pages"])
            region_writer.write_many(record.to_dict() for record in exported["regions"])
            asset_writer.write_many(record.to_dict() for record in exported["assets"])
            for task_name, rows in exported["task_records"].items():
                task_writers[task_name].write_many(record.to_dict() for record in rows)
            report.task_records[task] = report.task_records.get(task, 0) + 1

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

    def _iter_source_records(self) -> Iterable[dict[str, Any]]:
        idx = 0
        for annotation_file in sorted(self.options.source_root.glob("*/annotations/records.jsonl")):
            subset_id = annotation_file.parent.parent.name
            with annotation_file.open() as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    idx += 1
                    yield {
                        "idx": idx,
                        "subset_id": subset_id,
                        "annotation_file": annotation_file,
                        "line_number": line_number,
                        "record": json.loads(line),
                    }

    def _export_record(self, item: dict[str, Any], selected_tasks: set[str], canonical_root: Path) -> dict[str, Any]:
        record = item["record"]
        raw_label = str(record.get("label") or "")
        if bool(record.get("label_empty")) or not raw_label:
            return {"documents": [], "pages": [], "regions": [], "assets": [], "task_records": {}}

        task = classify_unirec_record(record)
        if task not in selected_tasks:
            return {"documents": [], "pages": [], "regions": [], "assets": [], "task_records": {}}

        image_path = self._resolve_image_path(record, item["subset_id"])
        width, height, image_format = _image_info(
            image_path,
            size_key=str(record.get("size_key") or ""),
            allow_unreadable=self.options.allow_unreadable_images,
        )
        dataset_root = _resolve_dataset_root(canonical_root, image_path)
        image_relpath = dataset_relative_path(image_path, dataset_root) if image_path.is_absolute() else image_path.as_posix()

        original_record_id = str(record.get("record_id") or f"record_{item['idx']:08d}")
        source_record_id = f"{item['subset_id']}:{original_record_id}"
        doc_hash = stable_hash(f"{self.options.dataset_name}:{source_record_id}")
        doc_id = stable_id("doc", self.options.dataset_name, doc_hash)
        page_id = stable_id("page", self.options.dataset_name, doc_hash, "0000")
        region_id = stable_id("region", self.options.dataset_name, doc_hash, "0000")
        page_asset_id = stable_id("asset", "page_render", self.options.dataset_name, doc_hash, "0000")
        crop_asset_id = stable_id("asset", "region_crop", self.options.dataset_name, doc_hash, "0000")

        metadata = {
            "record_id": original_record_id,
            "canonical_source_record_id": source_record_id,
            "subset_id": item["subset_id"],
            "source_index": record.get("source_index"),
            "original_file_name": record.get("original_file_name"),
            "size_key": record.get("size_key"),
            "document_types": record.get("document_types") or [],
            "unirec_category": record.get("category"),
            "unirec_type": record.get("type") or (record.get("raw") or {}).get("type"),
        }

        document = CanonicalDocument(
            document_id=doc_id,
            source_name=self.options.dataset_name,
            source_document_id=source_record_id,
            document_path=image_relpath,
            document_type="image",
            num_pages=1,
            language=record.get("language"),
            domain=record.get("category"),
            metadata=metadata,
        )
        page = CanonicalPage(
            page_id=page_id,
            document_id=doc_id,
            source_name=self.options.dataset_name,
            source_page_id=source_record_id,
            page_index=0,
            page_image_asset_id=page_asset_id,
            width=width,
            height=height,
            attributes={"adapter": "unirec", **metadata},
        )
        page_asset = AssetRecord(
            asset_id=page_asset_id,
            asset_type="page_render",
            source_name=self.options.dataset_name,
            document_id=doc_id,
            page_id=page_id,
            region_id=None,
            task=task,
            path=image_relpath,
            width=width,
            height=height,
            format=image_format,
            coordinate_space="canonical_page_pixel_xyxy",
            transform={"operation": "source_reference", "source": "unirec"},
        )
        crop_asset = AssetRecord(
            asset_id=crop_asset_id,
            asset_type="region_crop",
            source_name=self.options.dataset_name,
            document_id=doc_id,
            page_id=page_id,
            region_id=region_id,
            task=task,
            path=image_relpath,
            width=width,
            height=height,
            format=image_format,
            parent_asset_id=page_asset_id,
            transform_spec_hash=stable_hash({"operation": "source_reference", "source": image_relpath}),
            transform={"operation": "source_reference", "source": "unirec_region_crop"},
        )
        region = CanonicalRegion(
            region_id=region_id,
            page_id=page_id,
            document_id=doc_id,
            source_name=self.options.dataset_name,
            source_annotation_id=source_record_id,
            bbox=[0.0, 0.0, float(width), float(height)],
            rotation=0.0,
            element_type=task,
            category=task,
            reading_order=1,
            crop_asset_id=crop_asset_id,
            metadata=metadata,
        )
        task_record = CanonicalTaskRecord(
            record_id=stable_id(task, self.options.dataset_name, region_id),
            task=task,
            source_name=self.options.dataset_name,
            document_id=doc_id,
            page_id=page_id,
            region_id=region_id,
            image_asset_id=crop_asset_id,
            target=_target_for_task(task, raw_label),
            category=task,
            language=record.get("language"),
            provenance={
                "annotation_file": str(item["annotation_file"]),
                "line_number": item["line_number"],
                "source_record_id": source_record_id,
            },
            metadata=metadata,
        )
        return {
            "documents": [document],
            "pages": [page],
            "regions": [region],
            "assets": [page_asset, crop_asset],
            "task_records": {task: [task_record]},
        }

    def _resolve_image_path(self, record: dict[str, Any], subset_id: str) -> Path:
        raw_path = Path(str(record.get("image_path") or record.get("original_file_name") or ""))
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(self.options.source_root / raw_path)
            category = str(record.get("category") or "")
            parts = raw_path.parts
            prefix = ("subsets", category, subset_id)
            if len(parts) >= 3 and parts[:3] == prefix:
                candidates.append(self.options.source_root / subset_id / Path(*parts[3:]))
            if len(parts) >= 1:
                candidates.append(self.options.source_root / subset_id / "images" / raw_path.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]


def classify_unirec_record(record: dict[str, Any]) -> str:
    label = clean_unirec_text(str(record.get("label") or ""))
    if _has_table_hint(record) or _looks_like_table_label(label):
        return "table"
    if _is_standalone_formula(label):
        return "formula"
    return "text"


def clean_unirec_text(value: str) -> str:
    for token in NEWLINE_TOKENS:
        value = value.replace(token, "")
    value = INLINE_FORMULA_RE.sub(lambda match: f"${match.group(1).strip()}$", value)
    value = ESCAPED_DOLLAR_FORMULA_RE.sub(lambda match: f"${match.group(1).strip()}$", value)
    return value.replace(r"\(", "").replace(r"\)", "").strip()


def strip_formula_wrappers(value: str) -> str:
    cleaned = clean_unirec_text(value)
    display_match = DISPLAY_FORMULA_RE.match(cleaned)
    if display_match:
        return (display_match.group("bracket") or display_match.group("dollar") or "").strip()
    if cleaned.startswith("$") and cleaned.endswith("$") and cleaned.count("$") == 2:
        return cleaned[1:-1].strip()
    if cleaned.startswith(r"\["):
        cleaned = cleaned[2:].strip()
    if cleaned.endswith(r"\]"):
        cleaned = cleaned[:-2].strip()
    if cleaned.startswith("$$"):
        cleaned = cleaned[2:].strip()
    if cleaned.endswith("$$"):
        cleaned = cleaned[:-2].strip()
    return cleaned.strip()


def _target_for_task(task: str, label: str) -> dict[str, Any]:
    if task == "formula":
        return {"latex": strip_formula_wrappers(label), "display": True}
    cleaned = clean_unirec_text(label)
    if task == "table":
        return {"text": cleaned}
    return {"text": cleaned}


def _has_table_hint(record: dict[str, Any]) -> bool:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    fields = [
        record.get("category"),
        record.get("subset_id"),
        record.get("type"),
        raw.get("type"),
        record.get("annotation_source"),
        record.get("original_file_name"),
        *(record.get("document_types") or []),
    ]
    return any(TABLE_HINT_RE.search(str(value or "")) for value in fields)


def _looks_like_table_label(label: str) -> bool:
    lowered = label.lower()
    if "<table" in lowered or "<tr" in lowered or "<td" in lowered:
        return True
    lines = [line for line in label.splitlines() if line.strip()]
    if len(lines) >= 2 and sum(line.count("\t") for line in lines) >= 2:
        return True
    if len(lines) >= 2 and sum(line.count("|") for line in lines) >= 4:
        return True
    return False


def _is_standalone_formula(label: str) -> bool:
    if DISPLAY_FORMULA_RE.match(label):
        return True
    if not LATEX_SIGNAL_RE.search(label):
        return False
    text_without_math = re.sub(r"\\\((.*?)\\\)", "", label)
    text_without_math = DISPLAY_FORMULA_RE.sub("", text_without_math)
    ascii_words = re.findall(r"[A-Za-z]{2,}", text_without_math)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text_without_math)
    return len(ascii_words) <= 2 and len(cjk_chars) == 0


def _image_info(path: Path, *, size_key: str, allow_unreadable: bool) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = (image.format or path.suffix.lstrip(".") or "unknown").lower()
            return int(width), int(height), image_format
    except Exception:
        if not allow_unreadable:
            raise
        width, height = _parse_size_key(size_key)
        return width, height, path.suffix.lstrip(".").lower() or "unknown"


def _parse_size_key(size_key: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)_(\d+)\s*$", size_key)
    if not match:
        return 1, 1
    return max(1, int(match.group(1))), max(1, int(match.group(2)))


def _resolve_dataset_root(canonical_root: Path, image_path: Path) -> Path:
    env_root = dataset_root_from_env(required=False)
    if env_root is not None and image_path.is_absolute():
        try:
            image_path.resolve().relative_to(env_root.resolve())
            return env_root
        except ValueError:
            pass
    return infer_dataset_root_from_path(canonical_root)
