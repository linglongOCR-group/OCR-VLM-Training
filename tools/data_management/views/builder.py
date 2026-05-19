from __future__ import annotations

import hashlib
import io
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tools.data_management.canonical.reader import CanonicalReader
from tools.data_management.config.resolver import load_processing_config, resolve_path
from tools.data_management.paths import dataset_relative_path, infer_dataset_root_from_path, resolve_dataset_path
from tools.data_management.progress import ProgressReporter
from tools.data_management.prompts import load_prompt_config, resolve_prompt
from tools.data_management.otsl import html_to_otsl
from tools.data_management.registry.configured import configured_reward_registry, configured_serializer_registry
from tools.data_management.registry.reward_registry import default_reward_registry
from tools.data_management.registry.serializer_registry import default_serializer_registry
from tools.data_management.schemas import ViewRecord, stable_hash, stable_id, to_plain
from tools.data_management.utils.io import read_yaml, write_json


_WORKER_BUILDER: "ViewBuilder | None" = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_CONTEXT: dict[str, Any] | None = None
_WORKER_VIEW_NAME: str | None = None
_WORKER_STAGE: str | None = None


@dataclass(slots=True)
class ViewBuildReport:
    view_name: str
    stage: str
    total_records: int
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "stage": self.stage,
            "total_records": self.total_records,
            "split_counts": self.split_counts,
        }


class _SplitParquetWriter:
    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema,
        *,
        batch_size: int = 512,
        rows_per_shard: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.batch_size = min(batch_size, rows_per_shard) if rows_per_shard else batch_size
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        self.counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.temp_paths: dict[str, Path] = {}
        self.shard_indices: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.rows_in_current_shard: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.shard_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.temp_dirs: dict[str, Path] = {}
        if rows_per_shard:
            for split in self.buffers:
                temp_dir = output_dir / f"{split}.tmp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                self.temp_dirs[split] = temp_dir
        else:
            self.temp_paths = {
                split: output_dir / f"{split}.parquet.tmp" for split in self.buffers
            }

    def write(self, row: dict[str, Any]) -> None:
        split = row["split"]
        if split not in self.buffers:
            raise ValueError(f"unsupported split: {split}")
        buffer = self.buffers[split]
        buffer.append(row)
        if len(buffer) >= self.batch_size:
            self._flush(split)

    def finish(self) -> dict[str, int]:
        for split in self.buffers:
            self._flush(split)
        for writer in self.writers.values():
            writer.close()
        self.writers = {}
        if self.rows_per_shard:
            for split, count in self.counts.items():
                final_path = self.output_dir / f"{split}.parquet"
                final_dir = self.output_dir / split
                temp_dir = self.temp_dirs[split]
                final_path.unlink(missing_ok=True)
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                if count:
                    temp_dir.replace(final_dir)
                else:
                    shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            for split, count in self.counts.items():
                final_path = self.output_dir / f"{split}.parquet"
                final_dir = self.output_dir / split
                temp_path = self.temp_paths[split]
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                if count:
                    temp_path.replace(final_path)
                else:
                    temp_path.unlink(missing_ok=True)
                    final_path.unlink(missing_ok=True)
        return dict(self.counts)

    def abort(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers = {}
        if self.rows_per_shard:
            for path in self.temp_dirs.values():
                shutil.rmtree(path, ignore_errors=True)
        else:
            for path in self.temp_paths.values():
                path.unlink(missing_ok=True)

    def _flush(self, split: str) -> None:
        buffer = self.buffers[split]
        while buffer:
            capacity = self._remaining_capacity(split)
            chunk = buffer[:capacity]
            del buffer[:capacity]
            self._write_chunk(split, chunk)

    def _write_chunk(self, split: str, rows: list[dict[str, Any]]) -> None:
        table = pa.Table.from_pylist(rows, schema=self.schema)
        writer = self.writers.get(split)
        if writer is None:
            temp_path = self._next_temp_path(split)
            temp_path.unlink(missing_ok=True)
            writer = pq.ParquetWriter(temp_path, self.schema)
            self.writers[split] = writer
        writer.write_table(table)
        self.counts[split] += len(rows)
        self.rows_in_current_shard[split] += len(rows)
        if self.rows_per_shard and self.rows_in_current_shard[split] >= self.rows_per_shard:
            writer.close()
            del self.writers[split]
            self.rows_in_current_shard[split] = 0

    def _remaining_capacity(self, split: str) -> int:
        if not self.rows_per_shard:
            return len(self.buffers[split])
        remaining = self.rows_per_shard - self.rows_in_current_shard[split]
        return max(1, remaining)

    def _next_temp_path(self, split: str) -> Path:
        if not self.rows_per_shard:
            return self.temp_paths[split]
        shard_index = self.shard_indices[split]
        self.shard_indices[split] += 1
        self.shard_counts[split] += 1
        temp_dir = self.temp_dirs[split]
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"part-{shard_index:05d}.parquet"

    def asset_dirs(self, split: str) -> tuple[Path, Path]:
        if self.rows_per_shard:
            return self.temp_dirs[split] / "assets", self.output_dir / split / "assets"
        assets_dir = self.output_dir / "assets"
        return assets_dir, assets_dir


class ViewBuilder:
    def __init__(self, canonical_root: str | Path, view_root: str | Path, *, processing_config: Any | None = None) -> None:
        self.canonical_root = Path(canonical_root)
        self.view_root = Path(view_root)
        self.processing_config = processing_config
        self.dataset_root = self._resolve_dataset_root(processing_config)
        self.canonical_reader = CanonicalReader(self.canonical_root)
        self.serializers = (
            configured_serializer_registry(processing_config) if processing_config else default_serializer_registry()
        )
        self.rewards = configured_reward_registry(processing_config) if processing_config else default_reward_registry()

    def _resolve_dataset_root(self, processing_config: Any | None) -> Path:
        if processing_config is not None:
            try:
                self.canonical_root.resolve().relative_to(processing_config.dataset_root.resolve())
                return processing_config.dataset_root
            except ValueError:
                pass
        return infer_dataset_root_from_path(self.canonical_root)

    @classmethod
    def from_config_path(cls, config_path: str | Path, *, processing_config: str | Path | None = None) -> "ViewBuilder":
        config_path = Path(config_path)
        config = read_yaml(config_path)
        paths = config.get("paths") or {}
        canonical_root = paths.get("canonical_root") or config.get("canonical_root")
        view_root = paths.get("view_root") or config.get("view_root") or config_path.parent
        processing = load_processing_config(processing_config, require_dataset_root=canonical_root is None)
        canonical_root = (
            resolve_path(canonical_root, base=config_path.parent, dataset_root=processing.dataset_root)
            if canonical_root
            else processing.canonical_root
        )
        view_root = resolve_path(view_root, base=config_path.parent, dataset_root=processing.dataset_root)
        return cls(canonical_root, view_root, processing_config=processing)

    def build(
        self,
        view_config: dict[str, Any] | str | Path,
        *,
        overwrite: bool = True,
        progress: ProgressReporter | None = None,
        num_workers: int | None = None,
        worker_batch_size: int | None = None,
        schema_sample_size: int | None = None,
    ) -> ViewBuildReport:
        if isinstance(view_config, str | Path):
            config = read_yaml(view_config)
            config.setdefault("_config_path", str(view_config))
        else:
            config = dict(view_config)
        view_name = config.get("name") or Path(config.get("_config_path", self.view_root)).parent.name
        stage = config.get("stage") or config.get("training_stage") or "sft"
        if stage not in {"sft", "rlvr", "eval"}:
            raise ValueError("view stage must be sft, rlvr, or eval")
        if progress:
            progress.log("build-view", phase="start", view=view_name, stage=stage, view_root=self.view_root)
        records = self._load_selected_records(config)
        records = self._apply_excludes(records, config.get("exclude") or [])
        records = self._apply_samples(records, config.get("sample") or [])
        execution = self._resolve_execution(
            config,
            num_workers=num_workers,
            worker_batch_size=worker_batch_size,
            schema_sample_size=schema_sample_size,
        )
        records, label_filter_counts = self._apply_label_filters(
            records,
            progress=progress,
            num_workers=execution["num_workers"],
            worker_batch_size=execution["worker_batch_size"],
        )
        materialize_ctx = self._build_materialize_context(config, records)
        records, filter_counts = self._apply_image_filters(records, materialize_ctx, progress=progress)
        filter_counts = {**label_filter_counts, **filter_counts}
        if not records:
            raise ValueError("view image filters produced no records")
        split_map = self._assign_splits(records, config.get("split_policy") or {})
        rows_per_shard = _resolve_rows_per_shard(config)
        if progress:
            progress.log(
                "build-view",
                phase="selected",
                total=len(records),
                image_mode=materialize_ctx["image_materialization"]["mode"],
                rows_per_shard=rows_per_shard,
                num_workers=execution["num_workers"],
                worker_batch_size=execution["worker_batch_size"],
                schema_sample_size=execution["schema_sample_size"],
            )
        view_assets_dir = self.view_root / "assets"
        if overwrite:
            try:
                shutil.rmtree(view_assets_dir)
            except FileNotFoundError:
                pass
        if _uses_root_view_assets(materialize_ctx["image_materialization"]):
            view_assets_dir.mkdir(parents=True, exist_ok=True)
        materialize_ctx["view_assets_dir"] = view_assets_dir
        output_dir = self.view_root
        output_dir.mkdir(parents=True, exist_ok=True)
        schema = self._infer_view_schema(
            records,
            config,
            materialize_ctx,
            view_name=view_name,
            stage=stage,
            split_map=split_map,
            progress=progress,
            schema_sample_size=execution["schema_sample_size"],
        )
        writer = _SplitParquetWriter(output_dir, schema, rows_per_shard=rows_per_shard)
        materialize_ctx["asset_dir_resolver"] = writer.asset_dirs
        try:
            self._write_materialized_rows(
                records,
                config,
                materialize_ctx,
                writer,
                view_name=view_name,
                stage=stage,
                split_map=split_map,
                progress=progress,
                num_workers=execution["num_workers"],
                worker_batch_size=execution["worker_batch_size"],
            )
            split_counts = writer.finish()
        except Exception:
            writer.abort()
            raise
        stats = {"split_counts": split_counts, "total_records": sum(split_counts.values())}
        filtered_records = sum(filter_counts.values())
        if filtered_records:
            stats["filtered_records"] = filtered_records
            stats["filter_counts"] = filter_counts
        if writer.rows_per_shard:
            stats["shard_counts"] = writer.shard_counts
        write_json(output_dir / "stats.json", stats)
        if "_config_path" in config:
            target_config = output_dir / "view.yaml"
            source_config = Path(config["_config_path"])
            if source_config.resolve() != target_config.resolve():
                target_config.write_text(source_config.read_text())
        if progress:
            progress.finish(
                "build-view",
                total=sum(split_counts.values()),
                view=view_name,
                view_root=output_dir,
                split_counts=split_counts,
            )
        return ViewBuildReport(view_name=view_name, stage=stage, total_records=sum(split_counts.values()), split_counts=split_counts)

    def _resolve_execution(
        self,
        config: dict[str, Any],
        *,
        num_workers: int | None,
        worker_batch_size: int | None,
        schema_sample_size: int | None,
    ) -> dict[str, int]:
        view_build_config = {}
        if self.processing_config is not None:
            view_build_config.update(((self.processing_config.execution or {}).get("view_build") or {}))
        view_build_config.update(((config.get("execution") or {}).get("view_build") or {}))
        resolved_workers = num_workers if num_workers is not None else view_build_config.get("num_workers", 1)
        resolved_batch_size = (
            worker_batch_size if worker_batch_size is not None else view_build_config.get("worker_batch_size", 256)
        )
        resolved_schema_sample_size = (
            schema_sample_size
            if schema_sample_size is not None
            else view_build_config.get("schema_sample_size", 4096)
        )
        resolved_workers = int(resolved_workers)
        resolved_batch_size = int(resolved_batch_size)
        resolved_schema_sample_size = int(resolved_schema_sample_size)
        if resolved_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if resolved_batch_size < 1:
            raise ValueError("worker_batch_size must be at least 1")
        if resolved_schema_sample_size < 1:
            raise ValueError("schema_sample_size must be at least 1")
        return {
            "num_workers": resolved_workers,
            "worker_batch_size": resolved_batch_size,
            "schema_sample_size": resolved_schema_sample_size,
        }

    def _write_materialized_rows(
        self,
        records: list[dict[str, Any]],
        config: dict[str, Any],
        materialize_ctx: dict[str, Any],
        writer: _SplitParquetWriter,
        *,
        view_name: str,
        stage: str,
        split_map: dict[str, str],
        progress: ProgressReporter | None,
        num_workers: int,
        worker_batch_size: int,
    ) -> None:
        if num_workers == 1:
            for index, record in enumerate(records, start=1):
                row = self._materialize(
                    record,
                    config,
                    materialize_ctx,
                    view_name=view_name,
                    stage=stage,
                    split=split_map[record["record_id"]],
                )
                writer.write(row)
                if progress:
                    progress.update("build-view", index, total=len(records), phase="materialize")
            return

        worker_ctx = dict(materialize_ctx)
        worker_ctx.pop("asset_dir_resolver", None)
        processed = 0
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_view_worker,
            initargs=(self.processing_config, config, worker_ctx, view_name, stage),
        ) as executor:
            batches = _iter_materialize_batches(records, split_map, worker_batch_size)
            for rows in executor.map(_materialize_record_batch, batches):
                for row in rows:
                    writer.write(row)
                processed += len(rows)
                if progress:
                    progress.update("build-view", processed, total=len(records), phase="materialize")

    def _infer_view_schema(
        self,
        records: list[dict[str, Any]],
        config: dict[str, Any],
        materialize_ctx: dict[str, Any],
        *,
        view_name: str,
        stage: str,
        split_map: dict[str, str],
        progress: ProgressReporter | None = None,
        schema_sample_size: int = 4096,
    ) -> pa.Schema:
        """Infer a stable parquet schema without loading image bytes."""
        schema_ctx = {**materialize_ctx, "schema_inference": True}
        schemas: list[pa.Schema] = []
        batch: list[dict[str, Any]] = []
        sample_records = _schema_sample_records(records, schema_sample_size)
        for index, record in enumerate(sample_records, start=1):
            row = self._materialize(
                record,
                config,
                schema_ctx,
                view_name=view_name,
                stage=stage,
                split=split_map[record["record_id"]],
            )
            batch.append(row)
            if len(batch) >= 4096:
                schemas.append(pa.Table.from_pylist(batch).schema.remove_metadata())
                batch = []
            if progress:
                progress.update("build-view", index, total=len(sample_records), phase="infer-schema")
        if batch:
            schemas.append(pa.Table.from_pylist(batch).schema.remove_metadata())
        if not schemas:
            raise ValueError("view selection produced no records")
        return pa.unify_schemas(schemas)

    def _load_selected_records(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        sample_key_sets = self._compute_sample_key_sets(config)
        for include in config.get("include") or []:
            task = include["task"]
            for source in include.get("sources") or []:
                source_rows = self._read_task_records_with_samples(task, source, sample_key_sets)
                rows.extend(_filter_rows(source_rows, include.get("where") or {}))
        if not rows:
            raise ValueError("view selection produced no records")
        return [to_plain(row) for row in rows]

    def _apply_label_filters(
        self,
        records: list[dict[str, Any]],
        *,
        progress: ProgressReporter | None = None,
        num_workers: int = 1,
        worker_batch_size: int = 256,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        keep_flags: list[bool | None] = [None] * len(records)
        html_checks: list[tuple[int, str]] = []
        for index, record in enumerate(records):
            target = record.get("target") or {}
            if (
                record.get("task") == "table"
                and isinstance(target, dict)
                and not target.get("enhanced_otsl")
                and not target.get("otsl")
                and target.get("html")
            ):
                html_checks.append((index, str(target["html"])))
            else:
                keep_flags[index] = _record_has_serializable_label(record)

        if html_checks:
            processed = 0
            if num_workers == 1:
                for batch in _iter_batches(html_checks, worker_batch_size):
                    for index, keep in _html_label_check_batch(batch):
                        keep_flags[index] = keep
                    processed += len(batch)
                    if progress:
                        progress.update(
                            "build-view",
                            processed,
                            total=len(html_checks),
                            phase="label-filter-html",
                        )
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for batch_result in executor.map(
                        _html_label_check_batch,
                        _iter_batches(html_checks, worker_batch_size),
                    ):
                        for index, keep in batch_result:
                            keep_flags[index] = keep
                        processed += len(batch_result)
                        if progress:
                            progress.update(
                                "build-view",
                                processed,
                                total=len(html_checks),
                                phase="label-filter-html",
                            )

        kept = [record for record, keep in zip(records, keep_flags, strict=True) if keep]
        empty_label_count = len(records) - len(kept)
        counts = {"empty_label": empty_label_count} if empty_label_count else {}
        if progress and empty_label_count:
            progress.log(
                "build-view",
                phase="label-filter",
                input=len(records),
                kept=len(kept),
                dropped=empty_label_count,
                empty_label=empty_label_count,
            )
        return kept, counts

    def _compute_sample_key_sets(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        samples = config.get("sample") or []
        if isinstance(samples, dict):
            samples = [samples]
        if not samples:
            return []

        includes = config.get("include") or []
        plans: list[dict[str, Any]] = []
        for sample in samples:
            count = sample.get("count")
            if count is None:
                continue
            count = int(count)
            if count < 0:
                raise ValueError("sample count must be non-negative")
            level = sample.get("level", "document")
            key_field = "record_id" if level == "record" else "page_id" if level == "page" else "document_id"
            sources = set(sample.get("sources") or [])
            tasks = set(sample.get("tasks") or [])
            where = sample.get("where") or {}
            seed = str(sample.get("seed", 42))

            matched_keys: set[str] = set()
            for include in includes:
                task = include["task"]
                if tasks and task not in tasks:
                    continue
                for source in include.get("sources") or []:
                    if sources and source not in sources:
                        continue
                    matched_keys.update(
                        self._read_sample_candidate_keys(task, source, key_field, include.get("where") or {}, where)
                    )

            selected_keys = set(
                sorted(
                    matched_keys,
                    key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
                )[:count]
            )
            plans.append({**sample, "key_field": key_field, "selected_keys": selected_keys})
        return plans

    def _read_sample_candidate_keys(
        self, task: str, source: str, key_field: str, include_where: dict[str, Any], sample_where: dict[str, Any]
    ) -> set[str]:
        # Fast path for the common case: sample by document/page/record without predicates.
        if not include_where and not sample_where:
            root = self.canonical_root / "records" / task / f"source={source}"
            files = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
            keys: set[str] = set()
            for file_path in files:
                if file_path.exists():
                    df = pd.read_parquet(file_path, columns=[key_field])
                    keys.update(df[key_field].astype(str).tolist())
            return keys

        rows = self.canonical_reader.read_task_records(task, source)
        rows = _filter_rows(rows, include_where)
        rows = _filter_rows(rows, sample_where)
        return {str(row[key_field]) for row in rows}

    def _read_task_records_with_samples(
        self, task: str, source: str, sample_key_sets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        applicable = []
        for sample in sample_key_sets:
            sources = set(sample.get("sources") or [])
            tasks = set(sample.get("tasks") or [])
            if sources and source not in sources:
                continue
            if tasks and task not in tasks:
                continue
            applicable.append(sample)
        if not applicable:
            return self.canonical_reader.read_task_records(task, source)

        root = self.canonical_root / "records" / task / f"source={source}"
        files = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        rows: list[dict[str, Any]] = []
        for file_path in files:
            if not file_path.exists():
                continue
            df = pd.read_parquet(file_path)
            mask = pd.Series(True, index=df.index)
            for sample in applicable:
                key_field = sample["key_field"]
                selected_keys = sample["selected_keys"]
                key_mask = [str(value) in selected_keys for value in df[key_field].tolist()]
                mask &= pd.Series(key_mask, index=df.index)
            rows.extend(df.loc[mask].to_dict(orient="records"))
        return rows

    def _apply_excludes(self, records: list[dict[str, Any]], excludes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not excludes:
            return records
        kept = []
        for record in records:
            drop = False
            for exclude in excludes:
                if exclude.get("task") not in {None, record.get("task")}:
                    continue
                sources = exclude.get("sources")
                if sources and record.get("source_name") not in sources:
                    continue
                if _matches_where(record, exclude.get("where") or {}):
                    drop = True
                    break
            if not drop:
                kept.append(record)
        return kept

    def _apply_image_filters(
        self,
        records: list[dict[str, Any]],
        materialize_ctx: dict[str, Any],
        *,
        progress: ProgressReporter | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        image_filter = materialize_ctx.get("image_filter") or {}
        max_aspect_ratio = image_filter.get("max_aspect_ratio")
        if max_aspect_ratio is None:
            return records, {}

        asset_manifest = materialize_ctx["asset_manifest"]
        image_transform_config = materialize_ctx.get("image_transform_config") or {}
        kept: list[dict[str, Any]] = []
        counts = {"aspect_ratio": 0, "invalid_dimensions": 0}
        for record in records:
            asset = asset_manifest.get(record["image_asset_id"], {})
            width = _first_positive_dimension(asset.get("width"), (record.get("metadata") or {}).get("width"))
            height = _first_positive_dimension(asset.get("height"), (record.get("metadata") or {}).get("height"))
            if width is None or height is None:
                counts["invalid_dimensions"] += 1
                continue

            image_transform = _resolve_image_transform(
                image_transform_config,
                record["task"],
                width,
                height,
            )
            effective_width = _first_positive_dimension(image_transform.get("output_width"), width)
            effective_height = _first_positive_dimension(image_transform.get("output_height"), height)
            if effective_width is None or effective_height is None:
                counts["invalid_dimensions"] += 1
                continue

            aspect_ratio = max(effective_width / effective_height, effective_height / effective_width)
            if aspect_ratio > max_aspect_ratio:
                counts["aspect_ratio"] += 1
                continue
            kept.append(record)

        if progress:
            progress.log(
                "build-view",
                phase="aspect-filter",
                input=len(records),
                kept=len(kept),
                dropped=sum(counts.values()),
                max_aspect_ratio=max_aspect_ratio,
                aspect_ratio=counts["aspect_ratio"],
                invalid_dimensions=counts["invalid_dimensions"],
            )
        return kept, counts

    def _apply_samples(self, records: list[dict[str, Any]], samples: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
        if not samples:
            return records
        if isinstance(samples, dict):
            samples = [samples]
        kept_records = records
        for sample in samples:
            count = sample.get("count")
            if count is None:
                continue
            count = int(count)
            if count < 0:
                raise ValueError("sample count must be non-negative")
            level = sample.get("level", "document")
            key_field = "record_id" if level == "record" else "page_id" if level == "page" else "document_id"
            sources = set(sample.get("sources") or [])
            tasks = set(sample.get("tasks") or [])
            where = sample.get("where") or {}
            seed = str(sample.get("seed", 42))

            matched_keys: set[str] = set()
            for record in kept_records:
                if sources and record.get("source_name") not in sources:
                    continue
                if tasks and record.get("task") not in tasks:
                    continue
                if where and not _matches_where(record, where):
                    continue
                matched_keys.add(str(record[key_field]))

            selected_keys = set(
                sorted(
                    matched_keys,
                    key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
                )[:count]
            )

            next_records: list[dict[str, Any]] = []
            for record in kept_records:
                applies = True
                if sources and record.get("source_name") not in sources:
                    applies = False
                if tasks and record.get("task") not in tasks:
                    applies = False
                if where and not _matches_where(record, where):
                    applies = False
                if not applies or str(record[key_field]) in selected_keys:
                    next_records.append(record)
            kept_records = next_records
        return kept_records

    def _assign_splits(self, records: list[dict[str, Any]], split_policy: dict[str, Any]) -> dict[str, str]:
        level = split_policy.get("level", "document")
        seed = str(split_policy.get("seed", 42))
        train_ratio = float(split_policy.get("train_ratio", 0.98))
        val_ratio = float(split_policy.get("val_ratio", 0.01))
        split_by_key: dict[str, str] = {}
        split_map: dict[str, str] = {}
        for record in records:
            key = record["record_id"] if level == "record" else record["page_id"] if level == "page" else record["document_id"]
            if key not in split_by_key:
                value = int(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
                if value < train_ratio:
                    split_by_key[key] = "train"
                elif value < train_ratio + val_ratio:
                    split_by_key[key] = "val"
                else:
                    split_by_key[key] = "test"
            split_map[record["record_id"]] = split_by_key[key]
        return split_map

    def _build_materialize_context(self, config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
        """Pre-load shared resources that _materialize needs for every record.

        Caching these avoids reloading the full asset manifest (640K+ records
        from disk) and re-parsing prompt configs once per record.
        """
        asset_ids = {str(record["image_asset_id"]) for record in records}
        source_names = {str(record["source_name"]) for record in records}
        return {
            "asset_manifest": self.canonical_reader.read_asset_manifest(asset_ids=asset_ids, source_names=source_names),
            "prompt_config": load_prompt_config(config.get("prompt_profile") or config.get("model_family") or "default"),
            "image_transform_config": config.get("image_transform") or {},
            "image_materialization": _resolve_image_materialization(config),
            "image_filter": _resolve_image_filter(config),
            "view_name": config.get("name"),
            "dataset_root": self.dataset_root,
        }

    def _materialize(self, record: dict[str, Any], config: dict[str, Any], ctx: dict[str, Any], *, view_name: str, stage: str, split: str) -> dict[str, Any]:
        task = record["task"]
        target_format = (config.get("target_serialization") or {}).get(task) or _default_serializer_for_task(task)
        serializer = self.serializers.get(target_format)
        asset = ctx["asset_manifest"].get(record["image_asset_id"], {})
        prompt_config = ctx["prompt_config"]
        prompt_text, prompt_template_id = resolve_prompt(prompt_config, task, record)
        prompt = [
            {"role": "system", "content": prompt_config.system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        asset_width = int((asset or {}).get("width") or (record.get("metadata") or {}).get("width") or 1)
        asset_height = int((asset or {}).get("height") or (record.get("metadata") or {}).get("height") or 1)
        image_transform = _resolve_image_transform(
            ctx.get("image_transform_config") or {}, task, asset_width, asset_height
        )
        context = {
            "model_family": config.get("model_family"),
            "target_format": target_format,
            "width": asset_width,
            "height": asset_height,
            "image_transform": image_transform,
        }
        label = serializer.serialize(record, context)
        canonical_image_path = str(asset.get("path") or record["image_asset_id"])
        resolved_image_path = resolve_dataset_path(canonical_image_path, ctx["dataset_root"])
        content_hash = stable_hash(record["record_id"] + prompt_template_id)
        view_id = stable_id("view", view_name, content_hash)
        view_asset_id = stable_id("view_asset", view_name, task, content_hash)
        image_materialization = ctx["image_materialization"]
        image_mode = image_materialization["mode"]
        runtime_image_data = None
        runtime_image_transformed = False
        should_materialize_image_data = image_mode == "embedded" or bool(image_transform)
        if not ctx.get("schema_inference") and should_materialize_image_data:
            runtime_image_data = _transform_and_encode(str(resolved_image_path), image_transform)
            runtime_image_transformed = runtime_image_data is not None
            if runtime_image_data is None:
                runtime_image_data = _read_image_file_bytes(str(resolved_image_path))
        view_image_path = canonical_image_path
        extension = ".png" if runtime_image_transformed else _image_file_extension(canonical_image_path)
        image_filename = f"{view_asset_id}{extension}"
        images_column = None
        images_bytes_column = None
        images_path_column = None
        if image_mode == "embedded":
            if runtime_image_data is None:
                if ctx.get("schema_inference"):
                    images_bytes_column = [b""]
                else:
                    raise ValueError(f"embedded view cannot read image bytes for record {record['record_id']}")
            else:
                images_bytes_column = [runtime_image_data]
        elif image_mode == "source_reference":
            if image_transform:
                if ctx.get("schema_inference"):
                    view_image_path = dataset_relative_path(ctx["view_assets_dir"] / image_filename, ctx["dataset_root"])
                    images_path_column = [view_image_path]
                else:
                    if runtime_image_data is None:
                        raise ValueError(f"source_reference view cannot transform image for record {record['record_id']}")
                    view_image_path = _save_view_asset(
                        runtime_image_data,
                        view_asset_id,
                        task,
                        ctx["view_assets_dir"],
                        dataset_root=ctx["dataset_root"],
                        extension=extension,
                    )
                    images_path_column = [view_image_path]
            else:
                view_image_path = canonical_image_path
                images_path_column = [view_image_path]
        elif image_mode == "nested_reference":
            if image_transform:
                if ctx.get("schema_inference"):
                    view_image_path = dataset_relative_path(ctx["view_assets_dir"] / image_filename, ctx["dataset_root"])
                else:
                    if runtime_image_data is None:
                        raise ValueError(f"nested_reference view cannot transform image for record {record['record_id']}")
                    view_image_path = _save_view_asset(
                        runtime_image_data,
                        view_asset_id,
                        task,
                        ctx["view_assets_dir"],
                        dataset_root=ctx["dataset_root"],
                        extension=extension,
                    )
            else:
                view_image_path = canonical_image_path
            images_column = [{"image": view_image_path}]
        else:
            raise ValueError(f"unsupported image materialization mode: {image_mode}")
        messages = None
        if stage == "sft":
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": label},
            ]

        view_record = ViewRecord(
            id=view_id,
            stage=stage,  # type: ignore[arg-type]
            task=task,
            image_path=view_image_path,
            prompt=prompt,
            label=label,
            source_name=record["source_name"],
            document_id=record["document_id"],
            page_id=record["page_id"],
            region_id=record.get("region_id"),
            canonical_record_id=record["record_id"],
            canonical_image_asset_id=record["image_asset_id"],
            target_format=target_format,
            prompt_template_id=prompt_template_id,
            split=split,  # type: ignore[arg-type]
            view_image_asset_id=view_asset_id,
            images=images_column,
            images_bytes=images_bytes_column,
            images_path=images_path_column,
            data_source=task,
            extra_info={"sample_id": view_id, "task_type": task},
            messages=messages,
            metadata={"canonical_target": record.get("target"), "category": record.get("category")},
        )
        if stage == "rlvr":
            reward_profile_id = _reward_profile_for_task(config.get("reward_profile") or {}, task)
            reward = self.rewards.get(reward_profile_id)
            payload = reward.prepare_payload(record, {"label": label, "id": view_record.id}, config.get("reward_payload") or {})
            view_record.reward_profile_id = reward_profile_id
            view_record.reward_payload = payload
            view_record.reward_model = {"style": "rule", "ground_truth": label}
            view_record.answer_key = label
            view_record.verifier_metadata = {"payload_materialization": "inline"}
        return view_record.to_dict()


def _default_serializer_for_task(task: str) -> str:
    defaults = {
        "layout": "mineru_layout_box_v1",
        "table": "enhanced_otsl_v1",
        "formula": "latex_plain_v1",
        "text": "plain_text_v1",
    }
    try:
        return defaults[task]
    except KeyError as exc:
        raise KeyError(f"no default serializer for task {task}") from exc


def _record_has_serializable_label(record: dict[str, Any]) -> bool:
    target = record.get("target") or {}
    if not isinstance(target, dict):
        return False
    task = record.get("task")
    if task == "text":
        return bool(str(target.get("text", "")))
    if task == "formula":
        return bool(str(target.get("latex", "")))
    if task == "table":
        if target.get("enhanced_otsl"):
            return bool(str(target["enhanced_otsl"]))
        if target.get("otsl"):
            return bool(str(target["otsl"]))
        if target.get("html"):
            return bool(html_to_otsl(str(target["html"])))
        return False
    return True


def _reward_profile_for_task(config: dict[str, Any], task: str) -> str:
    by_task = config.get("by_task") or {}
    return by_task.get(task) or config.get("default") or "normalized_levenshtein_v1"


def _first_positive_dimension(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            dimension = int(value)
        except (TypeError, ValueError):
            continue
        if dimension > 0:
            return dimension
    return None


def _resolve_image_materialization(config: dict[str, Any]) -> dict[str, Any]:
    materialization = ((config.get("image_policy") or {}).get("materialization") or {})
    mode = materialization.get("mode") or "embedded"
    if mode not in {"embedded", "source_reference", "nested_reference"}:
        raise ValueError(f"unsupported image materialization mode: {mode}")
    return {"mode": mode}


def _resolve_image_filter(config: dict[str, Any]) -> dict[str, float]:
    image_filter = ((config.get("image_policy") or {}).get("filter") or {})
    value = image_filter.get("max_aspect_ratio")
    if value in (None, False, 0, "0"):
        return {}
    max_aspect_ratio = float(value)
    if max_aspect_ratio <= 0:
        return {}
    return {"max_aspect_ratio": max_aspect_ratio}


def _uses_root_view_assets(image_materialization: dict[str, Any]) -> bool:
    del image_materialization
    return False


def _resolve_rows_per_shard(config: dict[str, Any]) -> int | None:
    policy = config.get("shard_policy") or config.get("sharding") or {}
    value = policy.get("rows_per_shard", policy.get("shard_size"))
    if value in (None, False, 0, "0"):
        return None
    rows_per_shard = int(value)
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    return rows_per_shard


def _iter_materialize_batches(
    records: list[dict[str, Any]],
    split_map: dict[str, str],
    batch_size: int,
):
    batch = []
    for record in records:
        batch.append((record, split_map[record["record_id"]]))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _html_label_check_batch(batch: list[tuple[int, str]]) -> list[tuple[int, bool]]:
    return [(index, bool(html_to_otsl(html))) for index, html in batch]


def _schema_sample_records(records: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    sampled: list[dict[str, Any]] = []
    for record in records:
        key = (str(record.get("task", "")), str(record.get("source_name", "")))
        count = counts.get(key, 0)
        if count >= sample_size:
            continue
        sampled.append(record)
        counts[key] = count + 1
    return sampled


def _init_view_worker(
    processing_config: Any | None,
    config: dict[str, Any],
    materialize_ctx: dict[str, Any],
    view_name: str,
    stage: str,
) -> None:
    global _WORKER_BUILDER, _WORKER_CONFIG, _WORKER_CONTEXT, _WORKER_VIEW_NAME, _WORKER_STAGE
    _WORKER_BUILDER = ViewBuilder(".", ".", processing_config=processing_config)
    _WORKER_CONFIG = config
    _WORKER_CONTEXT = materialize_ctx
    _WORKER_VIEW_NAME = view_name
    _WORKER_STAGE = stage


def _materialize_record_batch(batch: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    if (
        _WORKER_BUILDER is None
        or _WORKER_CONFIG is None
        or _WORKER_CONTEXT is None
        or _WORKER_VIEW_NAME is None
        or _WORKER_STAGE is None
    ):
        raise RuntimeError("view materialization worker is not initialized")
    return [
        _WORKER_BUILDER._materialize(
            record,
            _WORKER_CONFIG,
            _WORKER_CONTEXT,
            view_name=_WORKER_VIEW_NAME,
            stage=_WORKER_STAGE,
            split=split,
        )
        for record, split in batch
    ]


def _filter_rows(rows: list[dict[str, Any]], where: dict[str, Any]) -> list[dict[str, Any]]:
    if not where:
        return rows
    return [row for row in rows if _matches_where(row, where)]


def _matches_where(row: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, condition in where.items():
        value = row.get(key)
        if isinstance(condition, dict):
            if "in" in condition and value not in condition["in"]:
                return False
            if "not_contains" in condition:
                values = value or []
                if isinstance(values, str):
                    values = [values]
                if any(item in values for item in condition["not_contains"]):
                    return False
        elif value != condition:
            return False
    return True


def _resolve_image_transform(
    image_transform_config: dict[str, Any], task: str, width: int, height: int
) -> dict[str, Any]:
    """Compute image_transform context for a task based on view config.

    Supports per-task transforms:
      image_transform:
        layout:
          pad_to_square: true   # pad to square preserving aspect ratio, then uniform resize
          resize_to: 1036
        # other tasks: no transform (identity)

    When pad_to_square is True the image is letterboxed to a square of
    max(width, height) then uniformly resized.  When False the image is
    stretched non-uniformly to resize_to × resize_to (matching MinerU's
    layout-detection pre-processing).
    """
    task_config = image_transform_config.get(task) or {}
    if not task_config:
        return {}

    resize_to = task_config.get("resize_to")
    pad_to_square = task_config.get("pad_to_square", False) if resize_to else False

    if not resize_to or width <= 0 or height <= 0:
        return task_config  # passthrough raw config if no resize or no dims

    if pad_to_square:
        max_dim = max(width, height)
        scale_x = scale_y = float(resize_to) / float(max_dim)
        if width >= height:
            pad_left = 0.0
            pad_top = (max_dim - height) * scale_y / 2.0
        else:
            pad_left = (max_dim - width) * scale_x / 2.0
            pad_top = 0.0
    else:
        scale_x = float(resize_to) / float(width)
        scale_y = float(resize_to) / float(height)
        pad_left = 0.0
        pad_top = 0.0

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "output_width": int(resize_to),
        "output_height": int(resize_to),
        "pad_to_square": pad_to_square,
    }


def _read_image_file_bytes(image_path: str | None) -> bytes | None:
    """Read a raw image file into bytes for inline Parquet storage."""
    if not image_path:
        return None
    try:
        return Path(image_path).read_bytes()
    except Exception:
        return None


def _asset_dirs_for_row(ctx: dict[str, Any], split: str) -> tuple[Path, Path]:
    resolver = ctx.get("asset_dir_resolver")
    if resolver:
        return resolver(split)
    assets_dir = ctx["view_assets_dir"]
    return assets_dir, assets_dir


def _save_view_asset(
    image_data: bytes,
    asset_id: str,
    task: str,
    assets_dir: Path,
    *,
    dataset_root: Path,
    return_assets_dir: Path | None = None,
    extension: str = ".png",
) -> str:
    """Write image bytes to ``{assets_dir}/{asset_id}{extension}``, returning a dataset-relative path."""
    del task, return_assets_dir
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_path = assets_dir / f"{asset_id}{_normalize_extension(extension)}"
    asset_path.write_bytes(image_data)
    return dataset_relative_path(asset_path, dataset_root)


def _image_file_extension(image_path: str | None) -> str:
    if not image_path:
        return ".png"
    extension = Path(str(image_path)).suffix
    return _normalize_extension(extension)


def _normalize_extension(extension: str | None) -> str:
    if not extension:
        return ".png"
    extension = extension.lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension


def _transform_and_encode(image_path: str | None, image_transform: dict[str, Any]) -> bytes | None:
    """Read an image, apply pad-to-square + resize, return PNG bytes.

    Returns None if image_transform is empty (identity) or the image cannot be read.
    The returned bytes are a PNG-encoded image ready for Parquet storage.
    """
    if not image_transform or not image_path:
        return None

    output_width = int(image_transform.get("output_width", 0))
    output_height = int(image_transform.get("output_height", 0))
    if output_width <= 0 or output_height <= 0:
        return None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    orig_w, orig_h = img.size
    pad_to_square = image_transform.get("pad_to_square", False)

    if pad_to_square:
        max_dim = max(orig_w, orig_h)
        canvas = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        paste_x = (max_dim - orig_w) // 2
        paste_y = (max_dim - orig_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas

    # Resize to output dimensions (LANCZOS for quality)
    img = img.resize((output_width, output_height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
