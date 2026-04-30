from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def validate_view(view_root: str | Path, *, require_images: bool = False, image_assets_dir: str | Path | None = None) -> None:
    root = Path(view_root)
    assets_dir = Path(image_assets_dir) if image_assets_dir is not None else root / "assets"
    files = _view_parquet_files(root)
    if not files:
        raise ValueError(f"no view parquet files found under {root}")
    seen_docs_by_split: dict[str, set[str]] = {}
    for path in files:
        parquet_file = pq.ParquetFile(path)
        columns = set(parquet_file.schema_arrow.names)
        split = _split_name(root, path)
        for column in ("id", "stage", "task", "image_path", "prompt", "label", "canonical_record_id", "split"):
            if column not in columns:
                raise ValueError(f"{path} missing required column {column}")
        if "images" in columns:
            raise ValueError(f"{path} has legacy images column; expected images_bytes or images_path")
        if require_images and "images_bytes" not in columns and "images_path" not in columns:
            raise ValueError(f"{path} missing required image column images_bytes or images_path")
        read_columns = [
            column
            for column in (
                "stage",
                "image_path",
                "prompt",
                "label",
                "split",
                "images_bytes",
                "images_path",
                "document_id",
                "reward_profile_id",
                "data_source",
                "extra_info",
            )
            if column in columns
        ]
        for row_index, row in enumerate(_iter_rows(parquet_file, columns=read_columns)):
            prompt = row["prompt"]
            if prompt is None or (hasattr(prompt, "__len__") and len(prompt) == 0):
                raise ValueError(f"{path}:{row_index} prompt is empty")
            if not row["label"]:
                raise ValueError(f"{path}:{row_index} label is empty")
            if row["split"] != split:
                raise ValueError(f"{path}:{row_index} split column does not match file split")
            if row["stage"] == "rlvr":
                for column in ("data_source", "extra_info"):
                    if column not in columns:
                        raise ValueError(f"{path} rlvr view missing required column {column}")
                if not row.get("reward_profile_id"):
                    raise ValueError(f"{path}:{row_index} rlvr record missing reward_profile_id")
            images_bytes = _as_list(row.get("images_bytes")) if "images_bytes" in columns else []
            images_path = _as_list(row.get("images_path")) if "images_path" in columns else []
            if images_bytes and images_path:
                raise ValueError(f"{path}:{row_index} has both images_bytes and images_path")
            images = images_bytes or images_path
            if images:
                placeholders = _image_placeholders(row["prompt"])
                if placeholders != len(images):
                    raise ValueError(
                        f"{path}:{row_index} has {placeholders} '<image>' placeholders but {len(images)} images"
                    )
                if images_bytes:
                    invalid_bytes = [image for image in images_bytes if not isinstance(image, bytes | bytearray | memoryview) or not image]
                    if invalid_bytes:
                        raise ValueError(f"{path}:{row_index} has invalid embedded images")
                else:
                    invalid_paths = [image for image in images_path if not _is_valid_image_filename(image)]
                    if invalid_paths:
                        raise ValueError(f"{path}:{row_index} has invalid image path references")
            elif require_images:
                raise ValueError(f"{path}:{row_index} missing image data")
            if require_images and images_path:
                missing = [image for image in images_path if not (assets_dir / str(image)).is_file()]
                if missing:
                    raise ValueError(f"{path}:{row_index} image asset does not exist: {missing[0]}")
            document_id = row.get("document_id")
            if document_id:
                seen_docs_by_split.setdefault(str(document_id), set()).add(split)
    leaked = {doc: splits for doc, splits in seen_docs_by_split.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"document split leakage detected: {list(leaked)[:5]}")


def _view_parquet_files(root: Path) -> list[Path]:
    files = list(root.glob("*.parquet"))
    for split in ("train", "val", "test"):
        files.extend((root / split).glob("part-*.parquet"))
    return sorted(files)


def _split_name(root: Path, path: Path) -> str:
    if path.parent == root:
        return path.stem
    return path.parent.name


def _iter_rows(parquet_file: pq.ParquetFile, *, columns: list[str] | None = None, batch_size: int = 1024):
    for row_group in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=columns)
        for batch in table.to_batches(max_chunksize=batch_size):
            yield from batch.to_pylist()


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


def _is_valid_image_filename(image: Any) -> bool:
    if not isinstance(image, str) or not image:
        return False
    path = Path(image)
    return not path.is_absolute() and path.name == image
