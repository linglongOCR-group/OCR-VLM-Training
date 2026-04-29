from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_view(view_root: str | Path, *, require_images: bool = False) -> None:
    root = Path(view_root)
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise ValueError(f"no view parquet files found under {root}")
    seen_docs_by_split: dict[str, set[str]] = {}
    for path in files:
        frame = pd.read_parquet(path)
        split = path.stem
        for column in ("id", "stage", "task", "image_path", "prompt", "label", "canonical_record_id", "split"):
            if column not in frame.columns:
                raise ValueError(f"{path} missing required column {column}")
        is_rlvr = (frame["stage"] == "rlvr").any() if "stage" in frame.columns else False
        if is_rlvr:
            for column in ("images", "data_source", "extra_info"):
                if column not in frame.columns:
                    raise ValueError(f"{path} rlvr view missing required column {column}")
        for row_index, row in frame.iterrows():
            prompt = row["prompt"]
            if prompt is None or (hasattr(prompt, "__len__") and len(prompt) == 0):
                raise ValueError(f"{path}:{row_index} prompt is empty")
            if not row["label"]:
                raise ValueError(f"{path}:{row_index} label is empty")
            if row["split"] != split:
                raise ValueError(f"{path}:{row_index} split column does not match file split")
            if row["stage"] == "rlvr" and not row.get("reward_profile_id"):
                raise ValueError(f"{path}:{row_index} rlvr record missing reward_profile_id")
            images = _as_list(row.get("images")) if "images" in frame.columns else []
            if images:
                placeholders = _image_placeholders(row["prompt"])
                if placeholders != len(images):
                    raise ValueError(
                        f"{path}:{row_index} has {placeholders} '<image>' placeholders but {len(images)} images"
                    )
                invalid = [image for image in images if not _is_valid_image_reference(image)]
                if invalid:
                    raise ValueError(f"{path}:{row_index} has invalid image references")
            if require_images and not _has_embedded_image(images) and not Path(str(row["image_path"])).exists():
                raise ValueError(f"{path}:{row_index} image does not exist: {row['image_path']}")
            document_id = row.get("document_id")
            if document_id:
                seen_docs_by_split.setdefault(str(document_id), set()).add(split)
    leaked = {doc: splits for doc, splits in seen_docs_by_split.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"document split leakage detected: {list(leaked)[:5]}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    return [value]


def _image_placeholders(prompt: Any) -> int:
    messages = _as_list(prompt)
    if not messages:
        return 0
    if all(isinstance(message, dict) for message in messages):
        return sum(
            message.get("content", "").count("<image>")
            for message in messages
            if isinstance(message.get("content", ""), str)
        )
    if isinstance(messages[0], str):
        return messages[0].count("<image>")
    return 0


def _has_embedded_image(images: list[Any]) -> bool:
    return any(isinstance(image, dict) and bool(image.get("bytes")) for image in images)


def _is_valid_image_reference(image: Any) -> bool:
    if not isinstance(image, dict):
        return False
    has_bytes = bool(image.get("bytes"))
    has_path = bool(image.get("image"))
    return has_bytes != has_path
