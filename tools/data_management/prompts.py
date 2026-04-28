from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment


TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "configs" / "data" / "prompts"

_JINJA_ENV = Environment(
    loader=BaseLoader(),
    variable_start_string="{{",
    variable_end_string="}}",
    keep_trailing_newline=True,
)


@dataclass(slots=True)
class PromptConfig:
    image_placeholder: str = "<image>"
    system_prompt: str = "You are a helpful assistant."
    by_task: dict[str, str] = field(default_factory=dict)
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
    by_task = raw.get("task") or {}
    if not by_task:
        by_task = _legacy_prompt_map(raw)
    return PromptConfig(
        image_placeholder=raw.get("image_placeholder", "<image>"),
        system_prompt=raw.get("system_prompt", "You are a helpful assistant."),
        by_task=dict(by_task),
        formats=raw.get("formats", {}) or {},
        sampling=raw.get("sampling", {}) or {},
    )


def _legacy_prompt_map(raw: dict[str, Any]) -> dict[str, str]:
    page = raw.get("page", {}) or {}
    region = raw.get("region", {}) or {}
    detection = raw.get("detection", {}) or {}
    table_prompt = _first_rule_prompt(region, "table") or "{{ image_placeholder }}\nTable Recognition:"
    formula_prompt = _first_rule_prompt(region, "formula") or "{{ image_placeholder }}\nFormula Recognition:"
    seal_prompt = _first_rule_prompt(region, "seal") or "{{ image_placeholder }}\nSeal Recognition:"
    text_prompt = _first_rule_prompt(region, "text") or "{{ image_placeholder }}\nText Recognition:"
    return {
        "layout": _first_rule_prompt(page, "layout_detection") or detection.get("default") or "\nLayout Detection:",
        "table": table_prompt,
        "formula": formula_prompt,
        "text": text_prompt,
        "diagram": "{{ image_placeholder }}\nImage Analysis:",
        "seal": seal_prompt,
        "page": page.get("default") or "{{ image_placeholder }}\nConvert the page image into markdown.",
    }


def _first_rule_prompt(section: dict[str, Any], token: str) -> str | None:
    for rule in section.get("rules", []) or []:
        match = rule.get("match", {}) or {}
        if token in str(match.get("task_type", "")) or token in str(match.get("task_type_contains", "")):
            return rule.get("prompt")
        if match.get("region_type") == token:
            return rule.get("prompt")
    return None


def render_prompt(template: str, *, task: str, record: dict[str, Any], image_placeholder: str = "<image>") -> str:
    return _JINJA_ENV.from_string(template).render(
        task=task,
        record=record,
        image_placeholder=image_placeholder,
        source_name=record.get("source_name"),
        category=record.get("category"),
    )


def resolve_prompt(config: PromptConfig, task: str, record: dict[str, Any]) -> tuple[str, str]:
    template_id = f"{task}_default_v1"
    template = config.by_task.get(task) or config.by_task.get("default")
    if not template:
        defaults = {
            "layout": "\nLayout Detection:",
            "table": "{{ image_placeholder }}\nTable Recognition:",
            "formula": "{{ image_placeholder }}\nFormula Recognition:",
            "text": "{{ image_placeholder }}\nText Recognition:",
            "diagram": "{{ image_placeholder }}\nImage Analysis:",
            "seal": "{{ image_placeholder }}\nSeal Recognition:",
        }
        template = defaults.get(task, "{{ image_placeholder }}\nRecognize the document content.")
    return render_prompt(template, task=task, record=record, image_placeholder=config.image_placeholder), template_id
