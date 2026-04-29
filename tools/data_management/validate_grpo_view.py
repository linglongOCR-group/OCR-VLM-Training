from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
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


def _image_count(images: Any) -> int:
    return len(_as_list(images))


def _invalid_images(images: Any) -> list[Any]:
    invalid = []
    for image in _as_list(images):
        if not isinstance(image, dict):
            invalid.append(image)
            continue
        has_bytes = bool(image.get("bytes"))
        has_path = bool(image.get("image"))
        if has_bytes == has_path:
            invalid.append(image)
    return invalid


def _iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from sorted(path.glob("*.parquet"))
        else:
            yield path


def validate_grpo_view(paths: Iterable[str | Path], max_rows_per_file: int = 1000) -> None:
    for path in _iter_files(paths):
        frame = pd.read_parquet(path, columns=["prompt", "images", "extra_info", "data_source"])
        if max_rows_per_file > 0:
            frame = frame.head(max_rows_per_file)
        for row_index, row in frame.iterrows():
            if not row.get("data_source"):
                raise ValueError(f"{path}:{row_index} missing data_source in rlvr view")
            placeholders = _image_placeholders(row["prompt"])
            images = _image_count(row["images"])
            if placeholders != images:
                extra_info = row.get("extra_info") or {}
                sample_id = extra_info.get("sample_id", "<unknown>") if isinstance(extra_info, dict) else "<unknown>"
                raise ValueError(
                    f"{path}:{row_index} sample_id={sample_id} has {placeholders} '<image>' placeholders "
                    f"but {images} images. Re-export the GRPO view with the current tools.data_management.views exporter."
                )
            if invalid := _invalid_images(row["images"]):
                raise ValueError(f"{path}:{row_index} has invalid image references: {invalid[:1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GRPO parquet prompt/image alignment.")
    parser.add_argument("paths", nargs="+", help="GRPO parquet files or directories to validate.")
    parser.add_argument("--max-rows-per-file", type=int, default=1000)
    args = parser.parse_args()
    validate_grpo_view(args.paths, max_rows_per_file=args.max_rows_per_file)


if __name__ == "__main__":
    main()
