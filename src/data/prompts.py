from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment

from src.data.schemas import BaseRecord, DetectionRecord, PageRecord, RegionRecord

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompt_templates"

_JINJA_ENV = Environment(
    loader=BaseLoader(),
    variable_start_string="{{",
    variable_end_string="}}",
    keep_trailing_newline=True,
)


@dataclass
class PromptConfig:
    image_placeholder: str = "<image>"
    page_rules: list[dict[str, Any]] = field(default_factory=list)
    page_default: str = ""
    region_rules: list[dict[str, Any]] = field(default_factory=list)
    region_default: str = ""
    detection_default: str = ""
    formats: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)


def load_prompt_config(profile: str | Path | None = None) -> PromptConfig:
    profile = profile or "default"
    path = Path(profile)
    if not path.is_file():
        path = TEMPLATES_DIR / f"{profile}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return _parse_raw_config(raw)


def _parse_raw_config(raw: dict[str, Any]) -> PromptConfig:
    cfg = PromptConfig()
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


def _render(template: str, **kwargs: str) -> str:
    return _JINJA_ENV.from_string(template).render(**kwargs)


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


def resolve_prompt(config: PromptConfig, record: BaseRecord) -> str:
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
            return _render(rule["prompt"], **render_kwargs)
    return _render(default, **render_kwargs)


_default_config: PromptConfig | None = None


def get_default_config() -> PromptConfig:
    global _default_config
    if _default_config is None:
        _default_config = load_prompt_config("default")
    return _default_config
