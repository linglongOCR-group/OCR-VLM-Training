from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.data_management.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSING_CONFIG = PROJECT_ROOT / "configs" / "data" / "processing.yaml"


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    config_path: Path | None
    dataset_root: Path
    source_root: Path
    canonical_root: Path
    view_root: Path
    registries: dict[str, Any]
    execution: dict[str, Any]


def load_processing_config(path: str | Path | None = None, *, require_dataset_root: bool = True) -> ProcessingConfig:
    config_path = _resolve_optional_config(path)
    raw = read_yaml(config_path) if config_path else {}
    paths = raw.get("paths") or {}
    dataset_root = _resolve_dataset_root(paths.get("dataset_root"), required=require_dataset_root)
    return ProcessingConfig(
        config_path=config_path,
        dataset_root=dataset_root,
        source_root=_resolve_under_dataset(paths.get("source_root"), dataset_root, "sources"),
        canonical_root=_resolve_under_dataset(paths.get("canonical_root"), dataset_root, "canonical"),
        view_root=_resolve_under_dataset(paths.get("view_root"), dataset_root, "views"),
        registries=raw.get("registries") or {},
        execution=raw.get("execution") or {},
    )


def resolve_source_config(path: str | Path, processing: ProcessingConfig) -> Path:
    return resolve_path(path, base=PROJECT_ROOT, dataset_root=processing.dataset_root)


def resolve_profile_path(value: str | Path | None, *, processing: ProcessingConfig, default: Path) -> Path:
    if value:
        return resolve_path(value, base=PROJECT_ROOT, dataset_root=processing.dataset_root)
    return default


def resolve_path(value: str | Path, *, base: Path, dataset_root: Path) -> Path:
    text = _expand_env(str(value))
    path = Path(text)
    if path.is_absolute():
        return path
    if text.startswith("sources/") or text.startswith("canonical") or text.startswith("views/"):
        return dataset_root / path
    return base / path


def import_from_dotted_path(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"invalid dotted import path: {path}")
    return getattr(importlib.import_module(module_name), attr)


def _resolve_optional_config(path: str | Path | None) -> Path | None:
    if path:
        return Path(path)
    if DEFAULT_PROCESSING_CONFIG.is_file():
        return DEFAULT_PROCESSING_CONFIG
    return None


def _resolve_dataset_root(value: Any, *, required: bool) -> Path:
    candidate = _expand_env(str(value)) if value else os.environ.get("OCR_DATASET_ROOT")
    if not candidate:
        if not required:
            return PROJECT_ROOT
        raise ValueError("dataset root is required; set OCR_DATASET_ROOT or paths.dataset_root")
    return Path(candidate).expanduser()


def _resolve_under_dataset(value: Any, dataset_root: Path, default_name: str) -> Path:
    if value in {None, ""}:
        return dataset_root / default_name
    text = _expand_env(str(value))
    path = Path(text)
    return path if path.is_absolute() else dataset_root / path


def _expand_env(value: str) -> str:
    if value.startswith("${oc.env:") and value.endswith("}"):
        env_name = value[len("${oc.env:") : -1]
        return os.environ.get(env_name, "")
    return os.path.expandvars(value)
