from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_runtime_images(row: dict[str, Any], image_assets_dir: str | Path | None = None) -> list[dict[str, Any]]:
    images_bytes = _as_list(row.get("images_bytes"))
    if images_bytes:
        return [{"bytes": bytes(image)} for image in images_bytes if image]

    images_path = _as_list(row.get("images_path"))
    if not images_path:
        return []

    assets_dir = image_assets_dir or os.environ.get("VIEW_IMAGE_ASSETS_DIR")
    if not assets_dir:
        raise ValueError("image_assets_dir is required when resolving images_path")

    base = Path(assets_dir)
    resolved = []
    for image_name in images_path:
        if not _is_valid_filename(image_name):
            raise ValueError(f"images_path entries must be filenames, got {image_name!r}")
        resolved.append({"image": str(base / str(image_name))})
    return resolved


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


def _is_valid_filename(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and path.name == value
