from __future__ import annotations

import json
from typing import Any

from src.data.prompts import get_default_config, load_prompt_config, resolve_prompt
from src.data.schemas import BaseRecord, DetectionRecord, PageRecord, RegionRecord, validate_record


TARGET_PRIORITY = ("text", "markdown", "html", "latex", "json", "tree_json")


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


def _prompt_for_record(record: BaseRecord, *, profile: str | None = None) -> str:
    config = load_prompt_config(profile) if profile else get_default_config()
    return resolve_prompt(config, record)


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
