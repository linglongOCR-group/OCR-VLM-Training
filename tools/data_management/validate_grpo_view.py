from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from tools.data_management.paths import dataset_root_from_env, infer_dataset_root_from_path, resolve_dataset_path, validate_relative_path


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
    # Normalize numpy arrays to lists.
    messages = _as_list(prompt)
    if not messages:
        return 0
    # Chat-format: list of message dicts with "content" keys.
    if all(isinstance(m, dict) for m in messages):
        count = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                count += content.count("<image>")
        return count
    # Plain string prompt (first element is a string).
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


def _iter_rows(path: Path, columns: list[str], *, max_rows: int, batch_size: int = 1024):
    parquet_file = pq.ParquetFile(path)
    missing = [column for column in columns if column not in parquet_file.schema_arrow.names]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    yielded = 0
    for row_group in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=columns)
        for batch in table.to_batches(max_chunksize=batch_size):
            for row in batch.to_pylist():
                yield yielded, row
                yielded += 1
                if max_rows > 0 and yielded >= max_rows:
                    return


def _iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files = list(path.glob("*.parquet"))
            for split in ("train", "val", "test"):
                files.extend((path / split).glob("part-*.parquet"))
            yield from sorted(files)
        else:
            yield path


def validate_grpo_view(
    paths: Iterable[str | Path],
    max_rows_per_file: int = 1000,
    *,
    image_assets_dir: str | Path | None = None,
) -> None:
    for path in _iter_files(paths):
        data_root = dataset_root_from_env(required=False) or infer_dataset_root_from_path(_view_root_for_file(path))
        assets_dir = Path(image_assets_dir) if image_assets_dir is not None else None
        parquet_file = pq.ParquetFile(path)
        columns = set(parquet_file.schema_arrow.names)
        required_columns = ["prompt", "extra_info", "data_source"]
        missing_columns = [column for column in required_columns if column not in columns]
        if missing_columns:
            raise ValueError(f"{path} missing required columns: {missing_columns}")
        read_columns = required_columns + [
            column for column in ("images", "images_bytes", "images_path") if column in columns
        ]
        for row_index, row in _iter_rows(
            path,
            read_columns,
            max_rows=max_rows_per_file,
        ):
            if not row.get("data_source"):
                raise ValueError(f"{path}:{row_index} missing data_source in rlvr view")
            images_nested = _as_list(row.get("images"))
            images_bytes = _as_list(row.get("images_bytes"))
            images_path = _as_list(row.get("images_path"))
            carriers = [carrier for carrier in (images_nested, images_bytes, images_path) if carrier]
            if len(carriers) > 1:
                raise ValueError(f"{path}:{row_index} has multiple image carriers")
            images = images_nested or images_bytes or images_path
            placeholders = _image_placeholders(row["prompt"])
            if placeholders != len(images):
                extra_info = row.get("extra_info") or {}
                sample_id = extra_info.get("sample_id", "<unknown>") if isinstance(extra_info, dict) else "<unknown>"
                raise ValueError(
                    f"{path}:{row_index} sample_id={sample_id} has {placeholders} '<image>' placeholders "
                    f"but {len(images)} images. Re-export the GRPO view with the current tools.data_management.views exporter."
                )
            if images_bytes:
                invalid_bytes = [image for image in images_bytes if not isinstance(image, bytes | bytearray | memoryview) or not image]
                if invalid_bytes:
                    raise ValueError(f"{path}:{row_index} has invalid embedded images")
            elif images_path:
                invalid_paths = [image for image in images_path if not _is_valid_image_reference(image)]
                if invalid_paths:
                    raise ValueError(f"{path}:{row_index} has invalid image path references: {invalid_paths[:1]}")
                missing = [
                    image
                    for image in images_path
                    if not _resolve_image_reference(str(image), data_root, assets_dir).is_file()
                ]
                if missing:
                    raise ValueError(f"{path}:{row_index} image asset does not exist: {missing[0]}")
            elif images_nested:
                invalid_images = [image for image in images_nested if not _is_valid_nested_image(image)]
                if invalid_images:
                    raise ValueError(f"{path}:{row_index} has invalid nested image references: {invalid_images[:1]}")
                missing = [
                    image["image"]
                    for image in images_nested
                    if not _resolve_image_reference(image["image"], data_root, assets_dir).is_file()
                ]
                if missing:
                    raise ValueError(f"{path}:{row_index} image asset does not exist: {missing[0]}")
            else:
                raise ValueError(f"{path}:{row_index} missing image data")


def _default_assets_dir(path: Path) -> Path:
    if path.parent.name in {"train", "val", "test"}:
        return path.parent.parent / "assets"
    return path.parent / "assets"


def _view_root_for_file(path: Path) -> Path:
    if path.parent.name in {"train", "val", "test"}:
        return path.parent.parent
    return path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GRPO parquet prompt/image alignment.")
    parser.add_argument("paths", nargs="+", help="GRPO parquet files or directories to validate.")
    parser.add_argument("--max-rows-per-file", type=int, default=1000)
    parser.add_argument("--image-assets-dir")
    args = parser.parse_args()
    validate_grpo_view(args.paths, max_rows_per_file=args.max_rows_per_file, image_assets_dir=args.image_assets_dir)


if __name__ == "__main__":
    main()
