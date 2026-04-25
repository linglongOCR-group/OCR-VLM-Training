from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import BaseLoader, Environment

from docparse_dataset.canonical.schemas import BaseRecord, DetectionRecord, PageRecord, RegionRecord, validate_record
from docparse_dataset.registry.serializer_registry import SerializerRegistry
from docparse_dataset.utils.io import read_json, read_jsonl, read_parquet, write_parquet_sharded, expand_input_paths

TARGET_PRIORITY = ("text", "markdown", "html", "latex", "json", "tree_json")
_JINJA_ENV = Environment(
    loader=BaseLoader(),
    variable_start_string="{{",
    variable_end_string="}}",
    keep_trailing_newline=True,
)


def image_ref(image_uri: str) -> dict[str, str]:
    return {"image": image_uri}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _target_from_record(record: BaseRecord) -> str:
    if isinstance(record, PageRecord):
        if record.merged_markdown:
            return record.merged_markdown
        return _json_dumps({"blocks": record.blocks, "reading_order": record.reading_order})

    if isinstance(record, RegionRecord):
        for key in TARGET_PRIORITY:
            if key in record.targets:
                value = record.targets[key]
                return value if isinstance(value, str) else _json_dumps(value)
        first_key = sorted(record.targets)[0]
        value = record.targets[first_key]
        return value if isinstance(value, str) else _json_dumps(value)

    if isinstance(record, DetectionRecord):
        return _json_dumps(record.instances)

    raise TypeError(f"unsupported record type: {type(record).__name__}")


def _extra_info(record: BaseRecord) -> dict[str, Any]:
    data = {
        "sample_id": record.sample_id,
        "record_type": record.record_type,
        "task_type": record.task_type,
        "source_type": record.source_type,
        "provenance": record.provenance or None,
        "difficulty": record.difficulty,
        "is_synthetic": record.is_synthetic,
        "metadata": record.metadata or None,
        "image_uri": record.image_uri,
        "width_px": record.width_px,
        "height_px": record.height_px,
    }
    if isinstance(record, RegionRecord):
        data["region_type"] = record.region_type
        data["source_page_ref"] = record.source_page_ref
        data["bbox_px"] = record.bbox_px
    if isinstance(record, DetectionRecord):
        data["source_page_ref"] = record.source_page_ref
    if isinstance(record, PageRecord):
        data["doc_id"] = record.doc_id
        data["page_index"] = record.page_index
    return data


class ViewBuilder:
    """Materializes SFT or RLVR training records from canonical records."""

    def __init__(
        self,
        prompt_config: dict | None = None,
        serializer_registry: SerializerRegistry | None = None,
        image_policy: dict | None = None,
    ):
        self._prompt_config = prompt_config or {}
        self._serializers = serializer_registry
        self._image_policy = image_policy or {}

    def build(self, view_config: dict) -> dict[str, list[dict]]:
        records = self._load_partitions(view_config)
        records = self._apply_selection(records, view_config)
        records = self._assign_splits(records, view_config)
        return self._materialize(records, view_config)

    def _load_partitions(self, view_config: dict) -> list[BaseRecord]:
        include = view_config.get("include", [])
        canonical_root = Path(view_config.get("canonical_root", "canonical"))
        records: list[BaseRecord] = []
        for entry in include:
            task = entry["task"]
            for source in entry.get("sources", []):
                partition_dir = canonical_root / "records" / task / f"source={source}"
                if partition_dir.is_dir():
                    for parquet_file in sorted(partition_dir.glob("part-*.parquet")):
                        for row in read_parquet(parquet_file):
                            records.append(validate_record(row))
        return records

    def _apply_selection(self, records: list[BaseRecord], view_config: dict) -> list[BaseRecord]:
        include = view_config.get("include", [])
        exclude = view_config.get("exclude", [])
        selected: list[BaseRecord] = []
        for record in records:
            if self._should_include(record, include) and not self._should_include(record, exclude):
                selected.append(record)
        return selected

    def _should_include(self, record: BaseRecord, rules: list[dict]) -> bool:
        if not rules:
            return True
        for rule in rules:
            task = rule.get("task")
            sources = rule.get("sources", [])
            if task and record.task_type and task not in record.task_type:
                continue
            if sources and record.source_type not in sources and record.metadata.get("dataset_name") not in sources:
                continue
            where = rule.get("where", {})
            if where:
                if not self._matches_where(record, where):
                    continue
            return True
        return False

    def _matches_where(self, record: BaseRecord, where: dict) -> bool:
        for field, condition in where.items():
            record_value = getattr(record, field, None)
            if record_value is None:
                record_value = record.metadata.get(field) if record.metadata else None
            if isinstance(condition, dict):
                if "in" in condition and record_value not in condition["in"]:
                    return False
                if "not_contains" in condition:
                    if isinstance(record_value, list):
                        if any(v in record_value for v in condition["not_contains"]):
                            return False
            elif record_value != condition:
                return False
        return True

    def _assign_splits(self, records: list[BaseRecord], view_config: dict) -> list[BaseRecord]:
        split_policy = view_config.get("split_policy", {})
        split_level = split_policy.get("level", "record")
        train_ratio = split_policy.get("train_ratio", 0.98)
        val_ratio = split_policy.get("val_ratio", 0.01)
        seed = split_policy.get("seed", 42)

        import random as _random
        rng = _random.Random(seed)

        train_docs: set[str] = set()
        val_docs: set[str] = set()
        if split_level == "document":
            doc_ids = list({getattr(r, "doc_id", r.sample_id) for r in records})
            rng.shuffle(doc_ids)
            n_train = int(len(doc_ids) * train_ratio)
            n_val = int(len(doc_ids) * val_ratio)
            train_docs = set(doc_ids[:n_train])
            val_docs = set(doc_ids[n_train : n_train + n_val])

        for record in records:
            if split_level == "document":
                doc = getattr(record, "doc_id", record.sample_id) or record.sample_id
                if doc in train_docs:
                    split = "train"
                elif doc in val_docs:
                    split = "val"
                else:
                    split = "test"
            else:
                r = rng.random()
                if r < train_ratio:
                    split = "train"
                elif r < train_ratio + val_ratio:
                    split = "val"
                else:
                    split = "test"
            record.metadata["split"] = split

        return records

    def _materialize(self, records: list[BaseRecord], view_config: dict) -> dict[str, list[dict]]:
        stage = view_config.get("stage", "sft")
        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

        for record in records:
            split = record.metadata.get("split", "train")
            if stage == "sft":
                row = self._to_sft_row(record, view_config)
            elif stage == "rlvr":
                row = self._to_rlvr_row(record, view_config)
            else:
                raise ValueError(f"unsupported stage: {stage}")
            splits[split].append(row)

        return splits

    def _render_prompt(self, record: BaseRecord) -> str:
        image_placeholder = self._prompt_config.get("image_placeholder", "<image>")
        if isinstance(record, PageRecord):
            rules = self._prompt_config.get("page_rules", [])
            default = self._prompt_config.get("page_default", f"{image_placeholder}\nAnalyze the page image and return the requested OCR result.")
            for rule in rules:
                match = rule.get("match", {})
                task_types = match.get("task_type", [])
                if record.task_type in task_types:
                    return _JINJA_ENV.from_string(rule["prompt"]).render(image_placeholder=image_placeholder)
            return _JINJA_ENV.from_string(default).render(image_placeholder=image_placeholder)

        elif isinstance(record, RegionRecord):
            rules = self._prompt_config.get("region_rules", [])
            default = self._prompt_config.get("region_default", f"{image_placeholder}\nRecognize the content in the image.")
            render_kwargs = {"image_placeholder": image_placeholder, "region_type": record.region_type.replace("_", " ")}
            for rule in rules:
                match = rule.get("match", {})
                task_contains = match.get("task_type_contains")
                if task_contains and task_contains in record.task_type:
                    return _JINJA_ENV.from_string(rule["prompt"]).render(**render_kwargs)
                region_type = match.get("region_type")
                if region_type and record.region_type == region_type:
                    return _JINJA_ENV.from_string(rule["prompt"]).render(**render_kwargs)
            return _JINJA_ENV.from_string(default).render(**render_kwargs)

        elif isinstance(record, DetectionRecord):
            default = self._prompt_config.get("detection_default", f"{image_placeholder}\nDetect the requested document objects.")
            return _JINJA_ENV.from_string(default).render(image_placeholder=image_placeholder)

        return f"<image>\nAnalyze the image."

    def _serialize_label(self, record: BaseRecord, view_config: dict) -> str:
        target_format = view_config.get("target_serialization", {}).get(record.task_type, "")
        if target_format and self._serializers:
            serializer = self._serializers.get(target_format)
            if serializer:
                ctx = {"image_transform": self._image_policy.get("transform", {}).get(record.task_type, {})}
                return serializer.serialize(record.to_dict(), ctx)
        return _target_from_record(record)

    def _to_sft_row(self, record: BaseRecord, view_config: dict) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": self._render_prompt(record)},
                {"role": "assistant", "content": self._serialize_label(record, view_config)},
            ],
            "images": [image_ref(record.image_uri)],
            "sample_id": record.sample_id,
            "task_type": record.task_type,
            "extra_info": _extra_info(record),
        }

    def _to_rlvr_row(self, record: BaseRecord, view_config: dict) -> dict[str, Any]:
        label = self._serialize_label(record, view_config)
        reward_profile = view_config.get("reward_profile", {})
        profile_id = reward_profile.get("default", "normalized_levenshtein_v1")
        if isinstance(reward_profile.get("by_task"), dict):
            profile_id = reward_profile["by_task"].get(record.task_type, profile_id)

        return {
            "data_source": f"ocr_vlm:{record.record_type}:{record.task_type}",
            "prompt": [{"role": "user", "content": self._render_prompt(record)}],
            "images": [image_ref(record.image_uri)],
            "reward_model": {"style": "rule", "ground_truth": label},
            "reward_profile_id": profile_id,
            "reward_payload": {"label": label},
            "extra_info": _extra_info(record),
        }

    def write_parquet(self, splits: dict[str, list[dict]], output_dir: str | Path, shard_size: int = 10000) -> dict[str, int]:
        output_dir = Path(output_dir)
        counts: dict[str, int] = {}
        for split_name, rows in splits.items():
            if rows:
                counts[split_name] = write_parquet_sharded(rows, output_dir / split_name, shard_size)
        return counts


# Backward-compatible standalone functions using ViewBuilder defaults
def to_grpo_row(record: BaseRecord | dict[str, Any], *, profile: str | None = None) -> dict[str, Any]:
    record = validate_record(record)
    return {
        "data_source": f"ocr_vlm:{record.record_type}:{record.task_type}",
        "prompt": [{"role": "user", "content": _prompt_for_record(record, profile=profile)}],
        "images": [image_ref(record.image_uri)],
        "reward_model": {"style": "rule", "ground_truth": _target_from_record(record)},
        "extra_info": _extra_info(record),
    }


def to_sft_row(record: BaseRecord | dict[str, Any], *, profile: str | None = None) -> dict[str, Any]:
    record = validate_record(record)
    return {
        "messages": [
            {"role": "user", "content": _prompt_for_record(record, profile=profile)},
            {"role": "assistant", "content": _target_from_record(record)},
        ],
        "images": [image_ref(record.image_uri)],
        "sample_id": record.sample_id,
        "task_type": record.task_type,
        "extra_info": _extra_info(record),
    }


def to_layout_row(record: BaseRecord | dict[str, Any]) -> dict[str, Any]:
    record = validate_record(record)
    if isinstance(record, PageRecord):
        instances = record.blocks
    elif isinstance(record, DetectionRecord):
        instances = record.instances
    else:
        raise TypeError("layout views accept PageRecord or DetectionRecord")
    return {
        "sample_id": record.sample_id,
        "task": "layout_or_detection",
        "image_uri": record.image_uri,
        "images": [image_ref(record.image_uri)],
        "width_px": record.width_px,
        "height_px": record.height_px,
        "instances": instances,
        "extra_info": _extra_info(record),
    }


# Legacy prompt helpers
from pathlib import Path as _Path
import yaml as _yaml
from dataclasses import dataclass as _dataclass, field as _field

TEMPLATES_DIR = _Path(__file__).resolve().parent.parent.parent / "configs" / "prompt_templates"


@_dataclass
class _PromptConfig:
    image_placeholder: str = "<image>"
    page_rules: list[dict[str, Any]] = _field(default_factory=list)
    page_default: str = ""
    region_rules: list[dict[str, Any]] = _field(default_factory=list)
    region_default: str = ""
    detection_default: str = ""
    formats: dict[str, Any] = _field(default_factory=dict)
    sampling: dict[str, Any] = _field(default_factory=dict)


_default_prompt_config: _PromptConfig | None = None


def load_prompt_config(profile: str | _Path | None = None) -> _PromptConfig:
    profile = profile or "default"
    path = _Path(profile)
    if not path.is_file():
        path = TEMPLATES_DIR / f"{profile}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    raw = _yaml.safe_load(path.read_text()) or {}
    cfg = _PromptConfig()
    cfg.image_placeholder = raw.get("image_placeholder", "<image>")
    page = raw.get("page", {})
    cfg.page_default = page.get("default", "")
    cfg.page_rules = page.get("rules", [])
    region = raw.get("region", {})
    cfg.region_default = region.get("default", "")
    cfg.region_rules = region.get("rules", [])
    detection = raw.get("detection", {})
    cfg.detection_default = detection.get("default", "")
    cfg.formats = raw.get("formats", {})
    cfg.sampling = raw.get("sampling", {})
    return cfg


def get_default_config() -> _PromptConfig:
    global _default_prompt_config
    if _default_prompt_config is None:
        _default_prompt_config = load_prompt_config("default")
    return _default_prompt_config


def _matches_rule(rule: dict[str, Any], record: BaseRecord) -> bool:
    match_spec = rule.get("match", {})
    task_types = match_spec.get("task_type")
    if task_types and isinstance(record, PageRecord) and record.task_type in task_types:
        return True
    task_contains = match_spec.get("task_type_contains")
    if task_contains and task_contains in record.task_type:
        return True
    region_type = match_spec.get("region_type")
    if region_type and isinstance(record, RegionRecord) and record.region_type == region_type:
        return True
    return False


def resolve_prompt(config: _PromptConfig, record: BaseRecord) -> str:
    render_kwargs: dict[str, str] = {"image_placeholder": config.image_placeholder}
    if isinstance(record, PageRecord):
        rules, default = config.page_rules, config.page_default
    elif isinstance(record, RegionRecord):
        rules, default = config.region_rules, config.region_default
        render_kwargs["region_type"] = record.region_type.replace("_", " ")
    elif isinstance(record, DetectionRecord):
        rules, default = [], config.detection_default
    else:
        raise TypeError(f"unsupported record type: {type(record).__name__}")
    for rule in rules:
        if _matches_rule(rule, record):
            return _JINJA_ENV.from_string(rule["prompt"]).render(**render_kwargs)
    return _JINJA_ENV.from_string(default).render(**render_kwargs)


def _prompt_for_record(record: BaseRecord, *, profile: str | None = None) -> str:
    config = load_prompt_config(profile) if profile else get_default_config()
    return resolve_prompt(config, record)


VIEW_EXPORTERS = {
    "grpo": to_grpo_row,
    "sft": to_sft_row,
    "layout": to_layout_row,
}


def read_source_records(paths: list[str | Path]) -> list[BaseRecord]:
    records: list[BaseRecord] = []
    for path in expand_input_paths(paths):
        if path.suffix == ".jsonl":
            rows = read_jsonl(path)
        elif path.suffix == ".json":
            rows = read_json(path)
        elif path.suffix == ".parquet":
            rows = read_parquet(path)
        else:
            raise ValueError(f"unsupported source format: {path}")
        records.extend(validate_record(row) for row in rows)
    return records


def export_view(records: list[BaseRecord], view: str, output: str | Path) -> int:
    if view not in VIEW_EXPORTERS:
        raise ValueError(f"unsupported view: {view}")
    exporter = VIEW_EXPORTERS[view]
    rows = [exporter(record) for record in records]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def export_view_sharded(
    paths: list[str | Path],
    view: str,
    output_dir: str | Path,
    shard_size: int = 10000,
    max_records: int | None = None,
) -> int:
    if view not in VIEW_EXPORTERS:
        raise ValueError(f"unsupported view: {view}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when set")

    exporter = VIEW_EXPORTERS[view]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for stale_part in output_path.glob("part-*.parquet"):
        stale_part.unlink()

    buffer: list[dict] = []
    total = 0
    shard_index = 0

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        pd.DataFrame(buffer).to_parquet(output_path / f"part-{shard_index:05d}.parquet", index=False)
        buffer = []
        shard_index += 1

    for path in expand_input_paths(paths):
        for record in read_source_records([path]):
            buffer.append(exporter(record))
            total += 1
            if len(buffer) >= shard_size:
                flush()
            if max_records is not None and total >= max_records:
                flush()
                return total

    flush()
    return total
