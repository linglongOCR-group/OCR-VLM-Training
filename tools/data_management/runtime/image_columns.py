from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.data_management.paths import DATA_ROOT_ENV, resolve_dataset_path, validate_relative_path


def resolve_runtime_images(
    row: dict[str, Any],
    image_assets_dir: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    images_bytes = _as_list(row.get("images_bytes"))
    if images_bytes:
        return [{"bytes": bytes(image)} for image in images_bytes if image]

    images_path = _as_list(row.get("images_path"))
    if not images_path:
        return _normalize_nested_images(row.get("images"), data_root=data_root)

    root = data_root or os.environ.get(DATA_ROOT_ENV) or image_assets_dir
    if not root:
        raise ValueError(f"{DATA_ROOT_ENV} is required when resolving referenced images")
    resolved = []
    for image_reference in images_path:
        if not isinstance(image_reference, str):
            raise ValueError(f"images_path entries must be strings, got {image_reference!r}")
        image_path = resolve_dataset_path(image_reference, root)
        if not image_path.is_file():
            raise FileNotFoundError(f"images_path asset does not exist: {image_path}")
        resolved.append({"image": str(image_path)})
    return resolved


def _normalize_nested_images(value: Any, *, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    images = _as_list(value)
    if not images:
        return []
    root = data_root or os.environ.get(DATA_ROOT_ENV)
    if not root:
        raise ValueError(f"{DATA_ROOT_ENV} is required when resolving referenced images")
    normalized = []
    for image in images:
        if not isinstance(image, dict) or not image.get("image"):
            raise ValueError(f"images entries must be VERL image dictionaries, got {image!r}")
        image_reference = image["image"]
        if not isinstance(image_reference, str):
            raise ValueError(f"images entries must contain string image paths, got {image!r}")
        validate_relative_path(image_reference)
        image_path = resolve_dataset_path(image_reference, root)
        if not image_path.is_file():
            raise FileNotFoundError(f"nested image asset does not exist: {image_path}")
        normalized_image = dict(image)
        normalized_image["image"] = str(image_path)
        normalized.append(normalized_image)
    return normalized


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, bytes | bytearray | memoryview):
        return [bytes(value)]
    if hasattr(value, "as_py"):
        value = value.as_py()
        if value is None:
            return []
        if isinstance(value, bytes | bytearray | memoryview):
            return [bytes(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    return [value]

