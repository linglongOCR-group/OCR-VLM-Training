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
    return list(value)


def _image_placeholders(prompt: Any) -> int:
    count = 0
    for message in _as_list(prompt):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            count += content.count("<image>")
    return count


def _image_count(images: Any) -> int:
    return len(_as_list(images))


def _iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from sorted(path.glob("*.parquet"))
        else:
            yield path


def validate_grpo_view(paths: Iterable[str | Path], max_rows_per_file: int = 1000) -> None:
    for path in _iter_files(paths):
        frame = pd.read_parquet(path, columns=["prompt", "images", "extra_info"])
        if max_rows_per_file > 0:
            frame = frame.head(max_rows_per_file)
        for row_index, row in frame.iterrows():
            placeholders = _image_placeholders(row["prompt"])
            images = _image_count(row["images"])
            if placeholders != images:
                extra_info = row.get("extra_info") or {}
                sample_id = extra_info.get("sample_id", "<unknown>") if isinstance(extra_info, dict) else "<unknown>"
                raise ValueError(
                    f"{path}:{row_index} sample_id={sample_id} has {placeholders} '<image>' placeholders "
                    f"but {images} images. Re-export the GRPO view with the current src.data.views exporter."
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GRPO parquet prompt/image alignment.")
    parser.add_argument("paths", nargs="+", help="GRPO parquet files or directories to validate.")
    parser.add_argument("--max-rows-per-file", type=int, default=1000)
    args = parser.parse_args()
    validate_grpo_view(args.paths, max_rows_per_file=args.max_rows_per_file)


if __name__ == "__main__":
    main()
