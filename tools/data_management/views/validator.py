from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from tools.data_management.paths import dataset_root_from_env, infer_dataset_root_from_path, resolve_dataset_path, validate_relative_path


def validate_view(
    view_root: str | Path,
    *,
    require_images: bool = False,
    image_assets_dir: str | Path | None = None,
    max_aspect_ratio: float | None = 200.0,
    num_workers: int = 1,
    worker_batch_size: int = 1024,
) -> None:
    root = Path(view_root)
    data_root = dataset_root_from_env(required=False) or infer_dataset_root_from_path(root)
    assets_dir = Path(image_assets_dir) if image_assets_dir is not None else None
    if max_aspect_ratio is not None and max_aspect_ratio <= 0:
        raise ValueError(f"max_aspect_ratio must be positive, got {max_aspect_ratio}")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if worker_batch_size < 1:
        raise ValueError("worker_batch_size must be at least 1")
    files = _view_parquet_files(root)
    if not files:
        raise ValueError(f"no view parquet files found under {root}")
    tasks = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        columns = set(parquet_file.schema_arrow.names)
        split = _split_name(root, path)
        for column in ("id", "stage", "task", "image_path", "prompt", "label", "canonical_record_id", "split"):
            if column not in columns:
                raise ValueError(f"{path} missing required column {column}")
        if require_images and "images" not in columns and "images_bytes" not in columns and "images_path" not in columns:
            raise ValueError(f"{path} missing required image column images, images_bytes, or images_path")
        read_columns = [
            column
            for column in (
                "stage",
                "image_path",
                "prompt",
                "label",
                "split",
                "images",
                "images_bytes",
                "images_path",
                "document_id",
                "reward_profile_id",
                "data_source",
                "extra_info",
            )
            if column in columns
        ]
        row_offset = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            tasks.append(
                (
                    path,
                    split,
                    tuple(columns),
                    tuple(read_columns),
                    row_group_index,
                    row_offset,
                    data_root,
                    assets_dir,
                    require_images,
                    max_aspect_ratio,
                    worker_batch_size,
                )
            )
            row_offset += parquet_file.metadata.row_group(row_group_index).num_rows
    seen_docs_by_split: dict[str, set[str]] = {}
    if num_workers == 1:
        for task in tasks:
            _merge_seen_docs(seen_docs_by_split, _validate_view_row_group(task))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_validate_view_row_group, task) for task in tasks]
            for future in as_completed(futures):
                _merge_seen_docs(seen_docs_by_split, future.result())
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
        yield from _iter_row_group_rows(parquet_file, row_group, columns=columns, batch_size=batch_size)


def _iter_row_group_rows(
    parquet_file: pq.ParquetFile,
    row_group: int,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    batch_size: int = 1024,
):
    table = parquet_file.read_row_group(row_group, columns=list(columns) if columns is not None else None)
    for batch in table.to_batches(max_chunksize=batch_size):
        yield from batch.to_pylist()


def _validate_view_row_group(task: tuple[Any, ...]) -> dict[str, set[str]]:
    (
        path,
        split,
        columns,
        read_columns,
        row_group_index,
        row_offset,
        data_root,
        assets_dir,
        require_images,
        max_aspect_ratio,
        worker_batch_size,
    ) = task
    path = Path(path)
    data_root = Path(data_root)
    assets_dir = Path(assets_dir) if assets_dir is not None else None
    columns = set(columns)
    parquet_file = pq.ParquetFile(path)
    seen_docs_by_split: dict[str, set[str]] = {}
    for local_index, row in enumerate(
        _iter_row_group_rows(parquet_file, row_group_index, columns=read_columns, batch_size=worker_batch_size)
    ):
        document_id = _validate_view_row(
            path,
            row_offset + local_index,
            row,
            columns,
            split,
            data_root,
            assets_dir,
            require_images,
            max_aspect_ratio,
        )
        if document_id:
            seen_docs_by_split.setdefault(str(document_id), set()).add(split)
    return seen_docs_by_split


def _validate_view_row(
    path: Path,
    row_index: int,
    row: dict[str, Any],
    columns: set[str],
    split: str,
    data_root: Path,
    assets_dir: Path | None,
    require_images: bool,
    max_aspect_ratio: float | None,
) -> Any | None:
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
    images_nested = _as_list(row.get("images")) if "images" in columns else []
    images_bytes = _as_list(row.get("images_bytes")) if "images_bytes" in columns else []
    images_path = _as_list(row.get("images_path")) if "images_path" in columns else []
    carriers = [carrier for carrier in (images_nested, images_bytes, images_path) if carrier]
    if len(carriers) > 1:
        raise ValueError(f"{path}:{row_index} has multiple image carriers")
    images = images_nested or images_bytes or images_path
    if images:
        placeholders = _image_placeholders(row["prompt"])
        if placeholders != len(images):
            raise ValueError(f"{path}:{row_index} has {placeholders} '<image>' placeholders but {len(images)} images")
        if images_bytes:
            invalid_bytes = [image for image in images_bytes if not isinstance(image, bytes | bytearray | memoryview) or not image]
            if invalid_bytes:
                raise ValueError(f"{path}:{row_index} has invalid embedded images")
        elif images_path:
            invalid_paths = [image for image in images_path if not _is_valid_image_reference(image)]
            if invalid_paths:
                raise ValueError(f"{path}:{row_index} has invalid image path references")
        else:
            invalid_images = [image for image in images_nested if not _is_valid_nested_image(image)]
            if invalid_images:
                raise ValueError(f"{path}:{row_index} has invalid nested image references")
    elif require_images:
        raise ValueError(f"{path}:{row_index} missing image data")
    if require_images and images_path:
        missing = [image for image in images_path if not _resolve_image_reference(str(image), data_root, assets_dir).is_file()]
        if missing:
            raise ValueError(f"{path}:{row_index} image asset does not exist: {missing[0]}")
    if require_images and images_nested:
        missing = [
            image["image"]
            for image in images_nested
            if not _resolve_image_reference(image["image"], data_root, assets_dir).is_file()
        ]
        if missing:
            raise ValueError(f"{path}:{row_index} image asset does not exist: {missing[0]}")
    if max_aspect_ratio is not None and images:
        if images_bytes:
            for image_index, image in enumerate(images_bytes):
                _validate_image_aspect_ratio(
                    path,
                    row_index,
                    image_index,
                    _image_size_from_bytes(image, path=path, row_index=row_index, image_index=image_index),
                    max_aspect_ratio,
                )
        else:
            references = images_path or [image["image"] for image in images_nested]
            for image_index, image in enumerate(references):
                _validate_image_aspect_ratio(
                    path,
                    row_index,
                    image_index,
                    _image_size_from_path(
                        _resolve_image_reference(str(image), data_root, assets_dir),
                        path=path,
                        row_index=row_index,
                        image_index=image_index,
                    ),
                    max_aspect_ratio,
                )
    return row.get("document_id")


def _merge_seen_docs(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for document_id, splits in source.items():
        target.setdefault(document_id, set()).update(splits)


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


def _is_valid_image_reference(image: Any) -> bool:
    if not isinstance(image, str) or not image:
        return False
    try:
        validate_relative_path(image)
    except ValueError:
        return False
    return True


def _is_valid_nested_image(image: Any) -> bool:
    return isinstance(image, dict) and _is_valid_image_reference(image.get("image"))


def _resolve_image_reference(image: str, data_root: Path, assets_dir: Path | None) -> Path:
    if assets_dir is not None and Path(image).name == image:
        return assets_dir / image
    return resolve_dataset_path(image, data_root)


def _image_size_from_bytes(image: bytes | bytearray | memoryview, *, path: Path, row_index: int, image_index: int) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(bytes(image))) as decoded:
            return decoded.size
    except Exception as exc:
        raise ValueError(f"{path}:{row_index} image {image_index} cannot be decoded for aspect ratio check") from exc


def _image_size_from_path(image_path: Path, *, path: Path, row_index: int, image_index: int) -> tuple[int, int]:
    if not image_path.is_file():
        raise ValueError(f"{path}:{row_index} image asset does not exist: {image_path}")
    try:
        with Image.open(image_path) as decoded:
            return decoded.size
    except Exception as exc:
        raise ValueError(f"{path}:{row_index} image {image_index} cannot be decoded for aspect ratio check: {image_path}") from exc


def _validate_image_aspect_ratio(
    path: Path,
    row_index: int,
    image_index: int,
    size: tuple[int, int],
    max_aspect_ratio: float,
) -> None:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}:{row_index} image {image_index} has invalid dimensions {width}x{height}")
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio >= max_aspect_ratio:
        raise ValueError(
            f"{path}:{row_index} image {image_index} aspect ratio must be less than "
            f"{max_aspect_ratio:g}, got {aspect_ratio:g} ({width}x{height})"
        )
