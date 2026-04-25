#!/usr/bin/env python3
"""docds — CLI for docparse_dataset: Source → Canonical → View processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docparse_dataset.config.loader import load_yaml
from docparse_dataset.sources.adapters.mineru import (
    MinerUExportOptions,
    export_mineru_dataset,
)
from docparse_dataset.views.builder import (
    export_view_sharded,
    VIEW_EXPORTERS,
)


def cmd_export_source(args: argparse.Namespace) -> None:
    """Export a source dataset to canonical format."""
    if args.profile:
        config = load_yaml(args.profile)
        options = MinerUExportOptions(
            mineru_root=Path(config["mineru_root"]),
            source_image_root=Path(config["source_image_root"]) if config.get("source_image_root") else None,
            output_root=Path(config["output_root"]),
            dataset_name=config["dataset_name"],
            mineru_subdir=config.get("mineru_subdir", "vlm"),
            image_extensions=tuple(config.get("image_extensions", (".jpg", ".jpeg", ".png", ".webp"))),
            shard_size=int(config.get("shard_size", 10000)),
            max_samples=int(config["max_samples"]) if config.get("max_samples") is not None else None,
            allow_unreadable_images=bool(config.get("allow_unreadable_images", False)),
            skip_errors=bool(config.get("skip_errors", False)),
        )
    else:
        options = MinerUExportOptions(
            mineru_root=Path(args.mineru_root),
            source_image_root=Path(args.source_image_root) if args.source_image_root else None,
            output_root=Path(args.output_root),
            dataset_name=args.dataset_name,
            mineru_subdir=args.mineru_subdir,
            max_samples=args.max_samples,
            allow_unreadable_images=args.allow_unreadable_images,
            skip_errors=args.skip_errors,
        )
        if args.tasks:
            tasks = [t.strip() for t in args.tasks.split(",")]
        else:
            tasks = None

    report = export_mineru_dataset(options)
    print(json.dumps(
        {k: v for k, v in report.__dict__.items() if k != "options"},
        ensure_ascii=False, indent=2, default=str,
    ))


def cmd_validate_canonical(args: argparse.Namespace) -> None:
    """Validate canonical records."""
    canonical_root = Path(args.canonical_root) if args.canonical_root else Path("canonical")
    task = args.task
    source = args.source

    record_dir = canonical_root / "records"
    if task:
        record_dir = record_dir / task
    if source:
        record_dir = record_dir / f"source={source}"

    if not record_dir.is_dir():
        print(f"canonical directory not found: {record_dir}", file=sys.stderr)
        sys.exit(1)

    issues: list[str] = []
    parquet_files = sorted(record_dir.rglob("part-*.parquet"))
    if not parquet_files:
        print("no canonical parquet files found", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    seen_ids: set[str] = set()
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        for col in ["record_id", "task", "source_name", "document_id", "page_id"]:
            if col not in df.columns:
                issues.append(f"{pq_file}: missing required column '{col}'")
        if "record_id" in df.columns:
            for rid in df["record_id"]:
                if rid in seen_ids:
                    issues.append(f"{pq_file}: duplicate record_id '{rid}'")
                seen_ids.add(rid)

    if issues:
        print(f"validation found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    print(f"canonical records valid: {len(parquet_files)} file(s), {len(seen_ids)} record(s)")


def cmd_build_view(args: argparse.Namespace) -> None:
    """Build a training view from canonical records."""
    view_config = load_yaml(args.view_config)
    view_name = view_config.get("name", Path(args.view_config).stem)
    view_root = Path(args.view_root) if args.view_root else Path("views") / view_name

    # For now, build views from pre-exported canonical parquet files
    canonical_root = view_config.get("canonical_root", "canonical")
    train_files: list[Path] = []
    for entry in view_config.get("include", []):
        task = entry["task"]
        for source in entry.get("sources", []):
            partition_dir = Path(canonical_root) / "records" / task / f"source={source}"
            if partition_dir.is_dir():
                train_files.extend(sorted(partition_dir.glob("part-*.parquet")))

    if not train_files:
        print("no canonical records found matching view config", file=sys.stderr)
        sys.exit(1)

    input_paths: list[str | Path] = [str(p) for p in train_files]

    view_type = view_config.get("stage", "sft")
    if view_type not in VIEW_EXPORTERS:
        view_type = "sft" if view_type == "sft" else "grpo"

    splits = args.split.split(",") if args.split else ["train", "val", "test"]
    dry_run = args.dry_run

    for split_name in splits:
        if dry_run:
            print(f"[dry-run] would write {split_name} records to {view_root / split_name}")
            continue
        count = export_view_sharded(
            input_paths,
            view_type,
            view_root / split_name,
            shard_size=args.shard_size,
            max_records=args.max_records,
        )
        print(f"exported {count} {split_name} rows to {view_root / split_name}")


def cmd_validate_view(args: argparse.Namespace) -> None:
    """Validate a training view."""
    view_name = args.view_name
    view_root = Path(args.view_root) if args.view_root else Path("views") / view_name

    if not view_root.is_dir():
        print(f"view directory not found: {view_root}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    issues: list[str] = []

    for split_name in ("train", "val", "test"):
        split_dir = view_root / split_name
        if not split_dir.is_dir():
            continue
        for pq_file in sorted(split_dir.glob("part-*.parquet")):
            df = pd.read_parquet(pq_file)
            for col in ["prompt", "images"]:
                if col not in df.columns:
                    issues.append(f"{pq_file}: missing required column '{col}'")
                    continue
            for idx, row in df.iterrows():
                prompt = row.get("prompt")
                images = row.get("images")
                if prompt is not None and images is not None:
                    prompt_str = json.dumps(prompt) if not isinstance(prompt, str) else prompt
                    image_count = len(images) if isinstance(images, list) else 0
                    placeholder_count = prompt_str.count("<image>")
                    if placeholder_count != image_count:
                        extra_info = row.get("extra_info") or {}
                        sample_id = extra_info.get("sample_id", "<unknown>") if isinstance(extra_info, dict) else "<unknown>"
                        issues.append(f"{pq_file}:{idx} sample_id={sample_id}: {placeholder_count} <image> placeholders but {image_count} images")

    if issues:
        print(f"validation found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    print(f"view '{view_name}' is valid")


def cmd_reward_smoke_test(args: argparse.Namespace) -> None:
    """Smoke-test reward computation using ground truth as prediction."""
    from docparse_dataset.rewards.levenshtein import NormalizedLevenshteinReward

    view_name = args.view_name
    view_root = Path(args.view_root) if args.view_root else Path("views") / view_name
    limit = args.limit or 100

    import pandas as pd
    adapter = NormalizedLevenshteinReward()

    total = 0
    perfect = 0
    errors = 0

    for split_name in ("train", "val", "test"):
        split_dir = view_root / split_name
        if not split_dir.is_dir():
            continue
        for pq_file in sorted(split_dir.glob("part-*.parquet")):
            df = pd.read_parquet(pq_file)
            for _, row in df.iterrows():
                if total >= limit:
                    break
                try:
                    label = row.get("reward_model", {}).get("ground_truth", "")
                    result = adapter.score(label, {"label": label or ""})
                    total += 1
                    if result.normalized_score >= 0.99:
                        perfect += 1
                except Exception as exc:
                    errors += 1
                    print(f"error scoring {pq_file}: {exc}")

    print(f"reward smoke test: {total} samples, {perfect} perfect (>=0.99), {errors} errors")


def cmd_inspect_view(args: argparse.Namespace) -> None:
    """Inspect view records."""
    view_name = args.view_name
    view_root = Path(args.view_root) if args.view_root else Path("views") / view_name
    limit = args.limit or 10

    import pandas as pd

    shown = 0
    for split_name in ("train", "val", "test"):
        split_dir = view_root / split_name
        if not split_dir.is_dir():
            continue
        for pq_file in sorted(split_dir.glob("part-*.parquet")):
            df = pd.read_parquet(pq_file)
            for _, row in df.iterrows():
                if shown >= limit:
                    return
                shown += 1
                print(f"--- record {shown} ({split_name}) ---")
                print(f"  source: {row.get('extra_info', {}).get('source_type', '?')}")
                print(f"  task:   {row.get('extra_info', {}).get('task_type', '?')}")
                prompt = row.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0].get("content", "") if prompt else ""
                print(f"  prompt: {str(prompt)[:120]}...")
                label = row.get("reward_model", {}).get("ground_truth", "") or ""
                if not label:
                    messages = row.get("messages", [])
                    if messages:
                        label = messages[-1].get("content", "") if messages else ""
                print(f"  label:  {str(label)[:120]}...")
                reward_profile = row.get("reward_profile_id")
                if reward_profile:
                    print(f"  reward: {reward_profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="docds — docparse_dataset CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export-source
    export_parser = subparsers.add_parser("export-source", help="Export source dataset to canonical format")
    export_parser.add_argument("--profile", help="YAML profile with export options")
    export_parser.add_argument("--mineru-root", help="Path to MinerU output root")
    export_parser.add_argument("--source-image-root", help="Path to source images")
    export_parser.add_argument("--output-root", help="Canonical output directory")
    export_parser.add_argument("--dataset-name", help="Dataset name")
    export_parser.add_argument("--mineru-subdir", default="vlm")
    export_parser.add_argument("--tasks", help="Comma-separated task list")
    export_parser.add_argument("--max-samples", type=int)
    export_parser.add_argument("--allow-unreadable-images", action="store_true")
    export_parser.add_argument("--skip-errors", action="store_true")
    export_parser.add_argument("--num-workers", type=int, default=1)

    # validate-canonical
    validate_canonical = subparsers.add_parser("validate-canonical", help="Validate canonical records")
    validate_canonical.add_argument("--task", help="Filter by task")
    validate_canonical.add_argument("--source", help="Filter by source")
    validate_canonical.add_argument("--canonical-root", help="Canonical root directory")

    # build-view
    build_view = subparsers.add_parser("build-view", help="Build a training view")
    build_view.add_argument("view_config", help="Path to view YAML config")
    build_view.add_argument("--view-root", help="View output root")
    build_view.add_argument("--split", default="train,val,test", help="Splits to build (comma-separated)")
    build_view.add_argument("--num-workers", type=int, default=1)
    build_view.add_argument("--shard-size", type=int, default=10000)
    build_view.add_argument("--max-records", type=int)
    build_view.add_argument("--overwrite", action="store_true")
    build_view.add_argument("--dry-run", action="store_true")

    # validate-view
    validate_view = subparsers.add_parser("validate-view", help="Validate a training view")
    validate_view.add_argument("view_name", help="View name")
    validate_view.add_argument("--view-root", help="View root directory")

    # inspect-view
    inspect_view = subparsers.add_parser("inspect-view", help="Inspect view records")
    inspect_view.add_argument("view_name", help="View name")
    inspect_view.add_argument("--view-root", help="View root directory")
    inspect_view.add_argument("--task")
    inspect_view.add_argument("--limit", type=int, default=10)

    # reward-smoke-test
    reward_smoke = subparsers.add_parser("reward-smoke-test", help="Smoke-test reward computation")
    reward_smoke.add_argument("--view", dest="view_name", help="View name")
    reward_smoke.add_argument("--view-root", help="View root directory")
    reward_smoke.add_argument("--limit", type=int, default=100)

    # score-predictions
    score_pred = subparsers.add_parser("score-predictions", help="Score model predictions")
    score_pred.add_argument("--view", dest="view_name", help="View name")
    score_pred.add_argument("--predictions", help="Predictions parquet file")
    score_pred.add_argument("--output", help="Output scores parquet file")

    args = parser.parse_args()

    command_map = {
        "export-source": cmd_export_source,
        "validate-canonical": cmd_validate_canonical,
        "build-view": cmd_build_view,
        "validate-view": cmd_validate_view,
        "inspect-view": cmd_inspect_view,
        "reward-smoke-test": cmd_reward_smoke_test,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"command not implemented: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
