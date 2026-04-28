from __future__ import annotations

import hashlib
import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from tools.data_management.canonical.reader import CanonicalReader
from tools.data_management.config.resolver import load_processing_config, resolve_path
from tools.data_management.prompts import load_prompt_config, resolve_prompt
from tools.data_management.registry.configured import configured_reward_registry, configured_serializer_registry
from tools.data_management.registry.reward_registry import default_reward_registry
from tools.data_management.registry.serializer_registry import default_serializer_registry
from tools.data_management.schemas import ViewRecord, stable_hash, stable_id, to_plain
from tools.data_management.utils.io import read_yaml, write_json, write_parquet


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


class ViewBuilder:
    def __init__(self, canonical_root: str | Path, view_root: str | Path, *, processing_config: Any | None = None) -> None:
        self.canonical_root = Path(canonical_root)
        self.view_root = Path(view_root)
        self.canonical_reader = CanonicalReader(self.canonical_root)
        self.serializers = (
            configured_serializer_registry(processing_config) if processing_config else default_serializer_registry()
        )
        self.rewards = configured_reward_registry(processing_config) if processing_config else default_reward_registry()

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

    def build(self, view_config: dict[str, Any] | str | Path, *, overwrite: bool = True) -> ViewBuildReport:
        if isinstance(view_config, str | Path):
            config = read_yaml(view_config)
            config.setdefault("_config_path", str(view_config))
        else:
            config = dict(view_config)
        view_name = config.get("name") or Path(config.get("_config_path", self.view_root)).parent.name
        stage = config.get("stage") or config.get("training_stage") or "sft"
        if stage not in {"sft", "rlvr", "eval"}:
            raise ValueError("view stage must be sft, rlvr, or eval")
        records = self._load_selected_records(config)
        records = self._apply_excludes(records, config.get("exclude") or [])
        split_map = self._assign_splits(records, config.get("split_policy") or {})
        materialize_ctx = self._build_materialize_context(config)
        view_assets_dir = self.view_root / "assets"
        if overwrite:
            try:
                shutil.rmtree(view_assets_dir)
            except FileNotFoundError:
                pass
        view_assets_dir.mkdir(parents=True, exist_ok=True)
        materialize_ctx["view_assets_dir"] = view_assets_dir
        rows = [
            self._materialize(record, config, materialize_ctx, view_name=view_name, stage=stage, split=split_map[record["record_id"]])
            for record in records
        ]
        output_dir = self.view_root
        output_dir.mkdir(parents=True, exist_ok=True)
        buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for row in rows:
            buckets[row["split"]].append(row)
        split_counts: dict[str, int] = {}
        for split, split_rows in buckets.items():
            split_counts[split] = write_parquet(output_dir / f"{split}.parquet", split_rows) if split_rows else 0
        write_json(output_dir / "stats.json", {"split_counts": split_counts, "total_records": len(rows)})
        if "_config_path" in config:
            target_config = output_dir / "view.yaml"
            source_config = Path(config["_config_path"])
            if source_config.resolve() != target_config.resolve():
                target_config.write_text(source_config.read_text())
        return ViewBuildReport(view_name=view_name, stage=stage, total_records=len(rows), split_counts=split_counts)

    def _load_selected_records(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for include in config.get("include") or []:
            task = include["task"]
            for source in include.get("sources") or []:
                source_rows = self.canonical_reader.read_task_records(task, source)
                rows.extend(_filter_rows(source_rows, include.get("where") or {}))
        if not rows:
            raise ValueError("view selection produced no records")
        return [to_plain(row) for row in rows]

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

    def _build_materialize_context(self, config: dict[str, Any]) -> dict[str, Any]:
        """Pre-load shared resources that _materialize needs for every record.

        Caching these avoids reloading the full asset manifest (640K+ records
        from disk) and re-parsing prompt configs once per record.
        """
        return {
            "asset_manifest": self.canonical_reader.read_asset_manifest(),
            "prompt_config": load_prompt_config(config.get("prompt_profile") or config.get("model_family") or "default"),
            "image_transform_config": config.get("image_transform") or {},
            "view_name": config.get("name"),
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
        image_bytes = _transform_and_encode(asset.get("path"), image_transform)
        canonical_image_path = str(asset.get("path") or record["image_asset_id"])
        content_hash = stable_hash(record["record_id"] + prompt_template_id)
        view_id = stable_id("view", view_name, content_hash)
        view_asset_id = stable_id("view_asset", view_name, task, content_hash)
        if image_transform and image_bytes is not None:
            view_asset_path = _save_view_asset(
                image_bytes, view_asset_id, task,
                ctx["view_assets_dir"]
            )
            view_image_path = view_asset_path
        else:
            raw_bytes = _read_image_bytes(asset.get("path"))
            if raw_bytes is not None:
                view_image_path = _save_view_asset(
                    raw_bytes, view_asset_id, task,
                    ctx["view_assets_dir"]
                )
            else:
                view_image_path = canonical_image_path
        images_column = [{"image": view_image_path}]

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
            image_bytes=image_bytes,
            images=images_column,
            data_source=task,
            extra_info={"sample_id": view_id, "task_type": task},
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


def _reward_profile_for_task(config: dict[str, Any], task: str) -> str:
    by_task = config.get("by_task") or {}
    return by_task.get(task) or config.get("default") or "normalized_levenshtein_v1"


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


def _read_image_bytes(image_path: str | None) -> bytes | None:
    """Read a raw image file into bytes for inline Parquet storage."""
    if not image_path:
        return None
    try:
        return Path(image_path).read_bytes()
    except Exception:
        return None


def _save_view_asset(image_bytes: bytes, asset_id: str, task: str, assets_dir: Path) -> str:
    """Write image bytes to ``{assets_dir}/{task}/{asset_id}.png``, returning the absolute path."""
    task_dir = assets_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    asset_path = task_dir / f"{asset_id}.png"
    asset_path.write_bytes(image_bytes)
    return str(asset_path)


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
