from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.schemas import BaseRecord, validate_record
from src.data.views import to_grpo_row, to_layout_row, to_sft_row


VIEW_EXPORTERS = {
    "grpo": to_grpo_row,
    "sft": to_sft_row,
    "layout": to_layout_row,
}


def _read_json(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    raise ValueError(f"{path} must contain a JSON object or array")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def expand_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(child for child in path.iterdir() if child.suffix in {".json", ".jsonl", ".parquet"}))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("no input source files found")
    return expanded


def read_source_records(paths: Iterable[str | Path]) -> list[BaseRecord]:
    records: list[BaseRecord] = []
    for path in expand_input_paths(paths):
        if path.suffix == ".jsonl":
            rows = _read_jsonl(path)
        elif path.suffix == ".json":
            rows = _read_json(path)
        elif path.suffix == ".parquet":
            rows = pd.read_parquet(path).to_dict(orient="records")
        else:
            raise ValueError(f"unsupported source format: {path}")
        records.extend(validate_record(row) for row in rows)
    return records


def export_view(records: Iterable[BaseRecord], view: str, output: str | Path) -> int:
    if view not in VIEW_EXPORTERS:
        raise ValueError(f"unsupported view: {view}")
    exporter = VIEW_EXPORTERS[view]
    rows = [exporter(record) for record in records]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def export_view_sharded(
    paths: Iterable[str | Path],
    view: str,
    output_dir: str | Path,
    shard_size: int = 10000,
    max_records: int | None = None,
) -> int:
    if view not in VIEW_EXPORTERS:
        raise ValueError(f"unsupported view: {view}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when set")

    exporter = VIEW_EXPORTERS[view]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for stale_part in output_path.glob("part-*.parquet"):
        stale_part.unlink()

    buffer: list[dict] = []
    total = 0
    shard_index = 0

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        pd.DataFrame(buffer).to_parquet(output_path / f"part-{shard_index:05d}.parquet", index=False)
        buffer = []
        shard_index += 1

    for path in expand_input_paths(paths):
        for record in read_source_records([path]):
            buffer.append(exporter(record))
            total += 1
            if len(buffer) >= shard_size:
                flush()
            if max_records is not None and total >= max_records:
                flush()
                return total

    flush()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical OCR records to VERL-compatible parquet views.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSON, JSONL, or parquet source record files.")
    parser.add_argument("--output", help="Output parquet path.")
    parser.add_argument("--output-dir", help="Output directory for sharded parquet view export.")
    parser.add_argument("--view", choices=sorted(VIEW_EXPORTERS), required=True, help="Training view to export.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Rows per parquet shard when using --output-dir.")
    parser.add_argument("--max-records", type=int, help="Maximum records to export when using --output-dir.")
    args = parser.parse_args()

    if bool(args.output) == bool(args.output_dir):
        raise SystemExit("exactly one of --output or --output-dir is required")

    if args.output_dir:
        count = export_view_sharded(
            args.input,
            args.view,
            args.output_dir,
            shard_size=args.shard_size,
            max_records=args.max_records,
        )
        print(f"exported {count} rows to {args.output_dir}")
        return

    count = export_view(read_source_records(args.input), args.view, args.output)
    print(f"exported {count} rows to {args.output}")


if __name__ == "__main__":
    main()
