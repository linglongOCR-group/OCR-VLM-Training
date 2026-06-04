from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "OCR_DATA_ROOT"


def dataset_root_from_env(*, required: bool = True) -> Path | None:
    value = os.environ.get(DATA_ROOT_ENV)
    if not value:
        if required:
            raise ValueError(f"dataset root is required; set {DATA_ROOT_ENV}")
        return None
    return Path(value).expanduser()


def infer_dataset_root_from_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.name in {"canonical", "views", "sources"}:
        return resolved.parent
    if resolved.parent.name in {"canonical", "views", "sources"}:
        return resolved.parent.parent
    return resolved.parent


def dataset_relative_path(path: str | Path, dataset_root: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        validate_relative_path(str(candidate))
        return candidate.as_posix()
    root = Path(dataset_root)
    try:
        relative = candidate.absolute().relative_to(root.absolute())
        return relative.as_posix()
    except ValueError:
        pass
    try:
        relative = candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is not under {DATA_ROOT_ENV}: {candidate}") from exc
    return relative.as_posix()


def resolve_dataset_path(path: str | Path, dataset_root: str | Path | None = None) -> Path:
    text = str(path)
    validate_relative_path(text)
    root = Path(dataset_root) if dataset_root is not None else dataset_root_from_env(required=True)
    if root is None:
        raise ValueError(f"dataset root is required; set {DATA_ROOT_ENV}")
    return root / text


def validate_relative_path(path: str) -> None:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"image paths must be relative to {DATA_ROOT_ENV}: {path}")
    if not path:
        raise ValueError("image path must be non-empty")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"image path traversal is not allowed: {path}")
