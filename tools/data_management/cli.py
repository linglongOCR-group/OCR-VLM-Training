from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tools.data_management.canonical.validator import validate_canonical
from tools.data_management.config.resolver import load_processing_config, resolve_source_config
from tools.data_management.lineage.resolver import LineageResolver
from tools.data_management.progress import ProgressReporter
from tools.data_management.registry.configured import configured_source_registry
from tools.data_management.views import ViewBuilder, reward_smoke_test, score_predictions, validate_view


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docds", description="Document dataset processing CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_source = subparsers.add_parser("export-source")
    export_source.add_argument("source")
    export_source.add_argument("--source-config", required=True)
    export_source.add_argument("--config")
    export_source.add_argument("--canonical-root")
    export_source.add_argument("--tasks")
    export_source.add_argument("--max-samples", type=int)
    export_source.add_argument("--skip-errors", action="store_true")
    export_source.add_argument("--skip-completed", action="store_true")
    export_source.add_argument("--allow-unreadable-images", action="store_true")
    export_source.add_argument("--overwrite-partitions", dest="overwrite_partitions", action="store_true", default=None)
    export_source.add_argument("--no-overwrite-partitions", dest="overwrite_partitions", action="store_false")
    export_source.add_argument("--num-workers", type=int)
    export_source.add_argument("--worker-chunksize", type=int)
    export_source.add_argument("--max-in-flight", type=int)
    _add_progress_args(export_source)

    validate_canonical_cmd = subparsers.add_parser("validate-canonical")
    validate_canonical_cmd.add_argument("--config")
    validate_canonical_cmd.add_argument("--canonical-root")
    validate_canonical_cmd.add_argument("--task")
    validate_canonical_cmd.add_argument("--source")

    build_view = subparsers.add_parser("build-view")
    build_view.add_argument("view_config")
    build_view.add_argument("--config")
    build_view.add_argument("--overwrite", action="store_true", default=False)
    build_view.add_argument("--num-workers", type=int)
    build_view.add_argument("--worker-batch-size", type=int)
    build_view.add_argument("--schema-sample-size", type=int)
    build_view.add_argument("--skip-failure", action="store_true", default=False)
    _add_progress_args(build_view)

    validate_view_cmd = subparsers.add_parser("validate-view")
    validate_view_cmd.add_argument("view_root")
    validate_view_cmd.add_argument("--config")
    validate_view_cmd.add_argument("--require-images", action="store_true")
    validate_view_cmd.add_argument("--image-assets-dir")
    validate_view_cmd.add_argument("--max-aspect-ratio", type=float, default=200.0)
    validate_view_cmd.add_argument("--num-workers", type=int, default=1)
    validate_view_cmd.add_argument("--worker-batch-size", type=int, default=1024)

    reward_smoke = subparsers.add_parser("reward-smoke-test")
    reward_smoke.add_argument("--view", required=True)
    reward_smoke.add_argument("--config")
    reward_smoke.add_argument("--limit", type=int, default=100)

    score = subparsers.add_parser("score-predictions")
    score.add_argument("--view", required=True)
    score.add_argument("--config")
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)

    trace = subparsers.add_parser("trace")
    trace.add_argument("--config")
    trace.add_argument("--canonical-root")
    trace.add_argument("--view-root")
    trace.add_argument("--view-record-id")
    trace.add_argument("--canonical-record-id")

    args = parser.parse_args(argv)
    if args.command == "export-source":
        processing = load_processing_config(args.config)
        source_config = resolve_source_config(args.source_config, processing)
        adapter_class = configured_source_registry(processing).get(args.source)
        if not hasattr(adapter_class, "from_profile"):
            raise SystemExit(f"source adapter {args.source} does not implement from_profile")
        adapter = adapter_class.from_profile(source_config, dataset_root=processing.dataset_root)
        if args.max_samples is not None:
            adapter.options.max_samples = args.max_samples
        if args.skip_errors:
            adapter.options.skip_errors = True
        if args.skip_completed:
            adapter.options.skip_completed = True
        if args.allow_unreadable_images:
            adapter.options.allow_unreadable_images = True
        if args.num_workers is not None:
            adapter.options.num_workers = args.num_workers
        if args.worker_chunksize is not None:
            adapter.options.worker_chunksize = args.worker_chunksize
        if args.max_in_flight is not None:
            adapter.options.max_in_flight = args.max_in_flight
        adapter.options.__post_init__()
        canonical_root = (
            Path(args.canonical_root)
            if args.canonical_root
            else adapter.options.output_root
            or processing.canonical_root
        )
        tasks = args.tasks.split(",") if args.tasks else None
        progress = _make_progress(args, root=Path(canonical_root).parent)
        overwrite_partitions = args.overwrite_partitions
        if overwrite_partitions is None:
            overwrite_partitions = not getattr(adapter.options, "skip_completed", False)
        report = adapter.export(
            canonical_root,
            tasks=tasks,
            overwrite_partitions=overwrite_partitions,
            progress=progress,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-canonical":
        processing = load_processing_config(args.config)
        canonical_root = Path(args.canonical_root) if args.canonical_root else processing.canonical_root
        validate_canonical(canonical_root, task=args.task, source=args.source)
        print("canonical ok")
        return

    if args.command == "build-view":
        builder = ViewBuilder.from_config_path(args.view_config, processing_config=args.config)
        progress = _make_progress(args, root=_common_parent(builder.canonical_root, builder.view_root))
        report = builder.build(
            args.view_config,
            overwrite=args.overwrite,
            progress=progress,
            num_workers=args.num_workers,
            worker_batch_size=args.worker_batch_size,
            schema_sample_size=args.schema_sample_size,
            skip_failure=args.skip_failure,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-view":
        processing = load_processing_config(
            args.config,
            require_dataset_root=_view_arg_requires_dataset_root(args.view_root),
        )
        validate_view(
            _resolve_view_arg(args.view_root, processing),
            require_images=args.require_images,
            image_assets_dir=args.image_assets_dir,
            max_aspect_ratio=args.max_aspect_ratio,
            num_workers=args.num_workers,
            worker_batch_size=args.worker_batch_size,
        )
        print("view ok")
        return

    if args.command == "reward-smoke-test":
        processing = load_processing_config(args.config)
        view_root = _resolve_view_arg(args.view, processing)
        print(json.dumps(reward_smoke_test(view_root, limit=args.limit), ensure_ascii=False, indent=2))
        return

    if args.command == "score-predictions":
        processing = load_processing_config(args.config)
        view_root = _resolve_view_arg(args.view, processing)
        count = score_predictions(view_root, args.predictions, args.output)
        print(f"scored {count} predictions")
        return

    if args.command == "trace":
        processing = load_processing_config(args.config)
        canonical_root = Path(args.canonical_root) if args.canonical_root else processing.canonical_root
        view_root = Path(args.view_root) if args.view_root else processing.view_root
        resolver = LineageResolver(canonical_root, view_root)
        if args.view_record_id:
            payload = resolver.trace_view_record(args.view_record_id)
        elif args.canonical_record_id:
            payload = resolver.trace_canonical_record(args.canonical_record_id)
        else:
            raise SystemExit("trace requires --view-record-id or --canonical-record-id")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return


def _resolve_view_arg(value: str, processing) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    if value.startswith("views/"):
        return processing.dataset_root / path
    return processing.view_root / path


def _view_arg_requires_dataset_root(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and not path.exists()


def _add_progress_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true", default=None)
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")


def _make_progress(args: argparse.Namespace, *, root: Path | None) -> ProgressReporter:
    if args.quiet:
        return ProgressReporter.disabled()
    enabled = bool(args.progress) if args.progress is not None else sys.stderr.isatty()
    return ProgressReporter(enabled=enabled, log_every=args.log_every, root=root)


def _common_parent(*paths: Path) -> Path:
    return Path(os.path.commonpath([str(path.resolve()) for path in paths]))


if __name__ == "__main__":
    main()
