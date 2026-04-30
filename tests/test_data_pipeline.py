import json
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from tools.data_management.canonical import validate_canonical
from tools.data_management.cli import main as docds_main
from tools.data_management.config import load_processing_config
from tools.data_management.progress import ProgressReporter
from tools.data_management.prompts import load_prompt_config, resolve_prompt
from tools.data_management.serializers.layout_mineru import MinerULayoutSerializer
from tools.data_management.sources.adapters.mineru import MinerUExportOptions, MinerUSourceAdapter
from tools.data_management.validate_grpo_view import validate_grpo_view
from tools.data_management.views import ViewBuilder, reward_smoke_test, score_predictions, validate_view


def _minimal_png(width: int = 100, height: int = 200) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _decode_png(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes))


def _as_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    if hasattr(value, "as_py"):
        return value.as_py()
    return bytes(value)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
        if value is None:
            return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_image_bytes(value) -> bytes:
    return _as_bytes(_as_list(value)[0])


def _write_fake_mineru_dataset(root: Path, source_root: Path, sample: str = "paper_001") -> None:
    vlm_dir = root / sample / "vlm"
    images_dir = vlm_dir / "images"
    images_dir.mkdir(parents=True)
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / f"{sample}.jpg").write_bytes(_minimal_png(100, 200))
    (images_dir / "table_crop.png").write_bytes(_minimal_png(50, 50))

    model = [
        [
            {"type": "title", "index": 0, "bbox": [0.1, 0.1, 0.9, 0.2], "angle": 0, "content": "A Title"},
            {"type": "table", "index": 1, "bbox": [0.2, 0.3, 0.8, 0.6], "angle": 0, "content": "<table><tr><td>A</td></tr></table>"},
            {"type": "equation", "index": 2, "bbox": [0.2, 0.7, 0.8, 0.8], "angle": 0, "content": "x^2"},
        ]
    ]
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [1000, 2000],
                "para_blocks": [
                    {"index": 0, "type": "title", "bbox": [100, 200, 900, 400]},
                    {"index": 1, "type": "table", "bbox": [200, 600, 800, 1200]},
                    {"index": 2, "type": "equation", "bbox": [200, 1400, 800, 1600]},
                ],
            }
        ],
    }
    content_v2 = [
        [
            {
                "type": "table",
                "bbox": [200, 600, 800, 1200],
                "content": {"image_source": {"path": "images/table_crop.png"}},
            }
        ]
    ]
    (vlm_dir / f"{sample}_model.json").write_text(json.dumps(model))
    (vlm_dir / f"{sample}_middle.json").write_text(json.dumps(middle))
    (vlm_dir / f"{sample}_content_list_v2.json").write_text(json.dumps(content_v2))
    (vlm_dir / f"{sample}.md").write_text("# A Title\n")


def _export_fake_canonical(tmp_path: Path) -> Path:
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    canonical_root = tmp_path / "canonical"
    _write_fake_mineru_dataset(mineru_root, source_root)
    report = MinerUSourceAdapter(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            dataset_name="FakeMinerU",
            allow_unreadable_images=True,
        )
    ).export(canonical_root)
    assert report.documents == 1
    assert report.pages == 1
    assert report.regions == 3
    assert report.task_records == {"formula": 1, "layout": 1, "table": 1, "text": 1}
    return canonical_root


def test_mineru_adapter_writes_spec_partitions(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)

    validate_canonical(canonical_root, source="FakeMinerU")

    assert (canonical_root / "entities/documents/source=FakeMinerU/part-00000.parquet").is_file()
    assert (canonical_root / "records/layout/source=FakeMinerU/part-00000.parquet").is_file()
    table = pd.read_parquet(canonical_root / "records/table/source=FakeMinerU/part-00000.parquet").iloc[0]
    assets = pd.read_parquet(canonical_root / "assets/manifests/source=FakeMinerU/part-00000.parquet")

    assert table["task"] == "table"
    assert table["target"]["html"].startswith("<table>")
    assert table["image_asset_id"] in set(assets["asset_id"])
    layout = pd.read_parquet(canonical_root / "records/layout/source=FakeMinerU/part-00000.parquet").iloc[0]
    assert layout["target"]["coordinate_space"] == "canonical_page_pixel_xyxy"
    assert list(layout["target"]["elements"][0]["bbox"]) == [100, 200, 900, 400]


def test_layout_serializer_outputs_mineru_1000_grid():
    record = {
        "task": "layout",
        "target": {"elements": [{"bbox": [100, 200, 500, 700], "label": "table", "rotation": 0}]},
        "metadata": {"width": 1000, "height": 1000},
    }

    label = MinerULayoutSerializer().serialize(record, {"width": 1000, "height": 1000, "image_transform": {}})

    assert label == "<|box_start|>100 200 500 700<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>"


def test_layout_serializer_floors_relative_coordinates_for_mineru_view():
    record = {
        "task": "layout",
        "target": {"elements": [{"bbox": [1, 1, 2, 2], "label": "text", "rotation": 90}]},
        "metadata": {"width": 3, "height": 3},
    }

    label = MinerULayoutSerializer().serialize(record, {"width": 3, "height": 3, "image_transform": {}})

    assert label == "<|box_start|>333 333 666 666<|box_end|><|ref_start|>text<|ref_end|><|rotate_right|>"


def test_layout_serializer_non_uniform_stretch_to_1036():
    """Non-uniform stretch to 1036x1036 matches MinerU's layout pre-processing."""
    record = {
        "task": "layout",
        "target": {"elements": [{"bbox": [100, 200, 500, 700], "label": "table", "rotation": 0}]},
        "metadata": {"width": 1000, "height": 2000},
    }
    transform = {
        "scale_x": 1036.0 / 1000.0,
        "scale_y": 1036.0 / 2000.0,
        "pad_left": 0.0,
        "pad_top": 0.0,
        "output_width": 1036,
        "output_height": 1036,
        "pad_to_square": False,
    }

    label = MinerULayoutSerializer().serialize(record, {"width": 1000, "height": 2000, "image_transform": transform})

    assert label == (
        "<|box_start|>100 100 500 350<|box_end|>"
        "<|ref_start|>table<|ref_end|>"
        "<|rotate_up|>"
    )


def test_prompt_config_resolves_task_prompt():
    config = load_prompt_config("mineru2.5")
    prompt, template_id = resolve_prompt(config, "table", {"source_name": "FakeMinerU"})

    assert prompt == "<image>\nTable Recognition:"
    assert template_id == "table_default_v1"
    assert config.formats["table_recognition"]["raw_target"] == "otsl"
    assert config.formats["layout_detection"]["coordinate_grid"] == 1000


def test_processing_config_derives_dataset_subdirectories(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("OCR_DATASET_ROOT", str(dataset_root))

    config = load_processing_config()

    assert config.dataset_root == dataset_root
    assert config.source_root == dataset_root / "sources"
    assert config.canonical_root == dataset_root / "canonical"
    assert config.view_root == dataset_root / "views"


def test_source_profile_resolves_dataset_relative_paths(tmp_path):
    mineru_root = tmp_path / "mineru-output"
    profile_path = tmp_path / "docbank.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "DocBank_500K",
                "adapter": "mineru",
                "mineru_root": str(mineru_root),
                "source_path": "sources/DocBank_500K",
                "canonical_output": "canonical",
            }
        )
    )

    adapter = MinerUSourceAdapter.from_profile(profile_path, dataset_root=tmp_path / "dataset")

    assert adapter.options.mineru_root == mineru_root
    assert adapter.options.source_image_root == tmp_path / "dataset" / "sources" / "DocBank_500K"
    assert adapter.options.output_root == tmp_path / "dataset" / "canonical"


def test_build_sft_and_rlvr_views(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_rlvr"
    config = {
        "name": "mineru25_rlvr",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "layout", "sources": ["FakeMinerU"]},
            {"task": "table", "sources": ["FakeMinerU"]},
            {"task": "formula", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "target_serialization": {
            "layout": "mineru_layout_box_v1",
            "table": "enhanced_otsl_v1",
            "formula": "latex_plain_v1",
            "text": "plain_text_v1",
        },
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    report = ViewBuilder(canonical_root, view_root).build(config)

    assert report.total_records == 4
    validate_view(view_root)
    train = pd.read_parquet(view_root / "train.parquet")
    assert set(train["stage"]) == {"rlvr"}
    assert set(train["reward_profile_id"]) == {"normalized_levenshtein_v1"}
    assert all(row["label"] == row["answer_key"] == row["reward_payload"]["label"] for _, row in train.iterrows())
    validate_grpo_view([view_root])
    # Portable multimodal columns
    for column in ("images_bytes", "images_path", "data_source", "extra_info"):
        assert column in train.columns, f"missing column {column}"
    assert "images" not in train.columns
    for _, row in train.iterrows():
        assert row["data_source"] == row["task"]
        assert row["extra_info"]["sample_id"] == row["id"]
        assert row["extra_info"]["task_type"] == row["task"]
        assert len(_as_list(row["images_bytes"])) == 1
        assert not _as_list(row["images_path"])
        assert _decode_png(_first_image_bytes(row["images_bytes"])).size[0] > 0
        assert row["image_path"]
    smoke = reward_smoke_test(view_root)
    assert smoke["min_score"] == 1.0


def test_build_sft_view_embeds_images_and_messages(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_sft"
    config = {
        "name": "mineru25_sft",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "text", "sources": ["FakeMinerU"]}],
        "target_serialization": {"text": "plain_text_v1"},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root)
    train = pd.read_parquet(view_root / "train.parquet")
    row = train.iloc[0]

    assert "images" not in train.columns
    assert len(_as_list(row["images_bytes"])) == 1
    assert not _as_list(row["images_path"])
    assert _decode_png(_first_image_bytes(row["images_bytes"])).size[0] > 0
    assert len(row["messages"]) == 2
    assert row["messages"][0]["role"] == "user"
    assert "<image>" in row["messages"][0]["content"]
    assert row["messages"][1]["role"] == "assistant"
    assert row["messages"][1]["content"] == row["label"]


def test_view_image_bytes_with_transform(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_transform_view"
    config = {
        "name": "test_transform_view",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "layout", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "target_serialization": {
            "layout": "mineru_layout_box_v1",
            "text": "plain_text_v1",
        },
        "image_transform": {
            "layout": {"pad_to_square": True, "resize_to": 128},
        },
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root)
    train = pd.read_parquet(view_root / "train.parquet")

    layout_rows = train[train["task"] == "layout"]
    text_rows = train[train["task"] == "text"]

    for _, row in layout_rows.iterrows():
        assert row["view_image_asset_id"] is not None
        assert "images" not in train.columns
        assert len(_as_list(row["images_bytes"])) == 1
        assert _decode_png(_first_image_bytes(row["images_bytes"])).size == (128, 128)
        assert "<|box_start|>300 100 700 200<|box_end|>" in row["label"]
        assert row["image_path"]

    for _, row in text_rows.iterrows():
        assert row["view_image_asset_id"] is not None
        assert len(_as_list(row["images_bytes"])) == 1
        assert _decode_png(_first_image_bytes(row["images_bytes"])).size[0] > 0
        assert "images" not in train.columns
        assert row["image_path"]


def test_view_image_assets_with_source_reference_policy(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_source_reference_view"
    config = {
        "name": "test_source_reference_view",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "layout", "sources": ["FakeMinerU"]}],
        "target_serialization": {"layout": "mineru_layout_box_v1"},
        "image_policy": {"materialization": {"mode": "source_reference"}},
        "image_transform": {"layout": {"pad_to_square": True, "resize_to": 128}},
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root, require_images=True)
    validate_grpo_view([view_root])
    train = pd.read_parquet(view_root / "train.parquet")
    row = train.iloc[0]

    images_path = _as_list(row["images_path"])
    assert len(images_path) == 1
    assert not _as_list(row["images_bytes"])
    assert "images" not in train.columns
    assert Path(images_path[0]).name == images_path[0]
    assert not Path(images_path[0]).is_absolute()
    assert "/" not in images_path[0]
    assert (view_root / "assets" / images_path[0]).is_file()
    assert row["image_path"] == images_path[0]

    # View asset files exist on disk
    assets_dir = view_root / "assets"
    assert assets_dir.is_dir()
    assert len(list(assets_dir.glob("*.png"))) > 0


def test_parallel_view_build_matches_single_process_output(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    base_config = {
        "name": "mineru25_parallel",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "include": [
            {"task": "layout", "sources": ["FakeMinerU"]},
            {"task": "table", "sources": ["FakeMinerU"]},
            {"task": "formula", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }
    single_root = tmp_path / "views" / "single"
    parallel_root = tmp_path / "views" / "parallel"

    ViewBuilder(canonical_root, single_root).build({**base_config, "paths": {"canonical_root": str(canonical_root), "view_root": str(single_root)}})
    ViewBuilder(canonical_root, parallel_root).build(
        {**base_config, "paths": {"canonical_root": str(canonical_root), "view_root": str(parallel_root)}},
        num_workers=2,
        worker_batch_size=1,
    )

    single = pd.read_parquet(single_root / "train.parquet").sort_values("id").reset_index(drop=True)
    parallel = pd.read_parquet(parallel_root / "train.parquet").sort_values("id").reset_index(drop=True)
    assert parallel["id"].tolist() == single["id"].tolist()
    assert parallel["label"].tolist() == single["label"].tolist()
    assert parallel["canonical_record_id"].tolist() == single["canonical_record_id"].tolist()
    assert [_first_image_bytes(value) for value in parallel["images_bytes"]] == [
        _first_image_bytes(value) for value in single["images_bytes"]
    ]
    validate_view(parallel_root)


def test_parallel_source_reference_build_writes_reachable_assets(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "parallel_source_reference"
    config = {
        "name": "parallel_source_reference",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "layout", "sources": ["FakeMinerU"]}],
        "target_serialization": {"layout": "mineru_layout_box_v1"},
        "image_policy": {"materialization": {"mode": "source_reference"}},
        "image_transform": {"layout": {"pad_to_square": True, "resize_to": 128}},
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
    }

    ViewBuilder(canonical_root, view_root).build(config, num_workers=2, worker_batch_size=1)

    train = pd.read_parquet(view_root / "train.parquet")
    image_name = _as_list(train.iloc[0]["images_path"])[0]
    assert Path(image_name).name == image_name
    assert (view_root / "assets" / image_name).is_file()
    validate_view(view_root, require_images=True)


def test_cli_build_view_accepts_worker_controls(tmp_path, capsys):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "parallel_cli"
    config_path = tmp_path / "parallel_view.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "parallel_cli",
                "stage": "rlvr",
                "model_family": "mineru2.5",
                "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
                "include": [{"task": "text", "sources": ["FakeMinerU"]}],
                "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
                "reward_profile": {"default": "normalized_levenshtein_v1"},
            }
        )
    )

    docds_main(["build-view", str(config_path), "--num-workers", "2", "--worker-batch-size", "1"])

    captured = capsys.readouterr()
    assert json.loads(captured.out)["total_records"] == 1
    validate_view(view_root)


def test_view_build_worker_controls_can_come_from_config(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "parallel_config"
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, root=tmp_path)

    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "parallel_config",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
            "include": [{"task": "text", "sources": ["FakeMinerU"]}],
            "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
            "execution": {"view_build": {"num_workers": 2, "worker_batch_size": 1}},
        },
        progress=progress,
    )

    logs = stream.getvalue()
    assert "num_workers=2" in logs
    assert "worker_batch_size=1" in logs
    validate_view(view_root)


def test_view_build_schema_inference_uses_bounded_sample(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "schema_sample"
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, root=tmp_path)

    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "schema_sample",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
            "include": [
                {"task": "layout", "sources": ["FakeMinerU"]},
                {"task": "table", "sources": ["FakeMinerU"]},
                {"task": "formula", "sources": ["FakeMinerU"]},
                {"task": "text", "sources": ["FakeMinerU"]},
            ],
            "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
        },
        schema_sample_size=1,
        progress=progress,
    )

    logs = stream.getvalue()
    assert "phase=infer-schema" in logs
    assert "current=4" in logs
    validate_view(view_root)


def test_cli_build_view_accepts_schema_sample_size(tmp_path, capsys):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "schema_sample_cli"
    config_path = tmp_path / "schema_sample_view.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "schema_sample_cli",
                "stage": "rlvr",
                "model_family": "mineru2.5",
                "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
                "include": [{"task": "text", "sources": ["FakeMinerU"]}],
                "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
                "reward_profile": {"default": "normalized_levenshtein_v1"},
            }
        )
    )

    docds_main(["build-view", str(config_path), "--schema-sample-size", "1"])

    captured = capsys.readouterr()
    assert json.loads(captured.out)["total_records"] == 1
    validate_view(view_root)


def test_view_build_schema_sample_size_can_come_from_config(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "schema_sample_config"
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, root=tmp_path)

    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "schema_sample_config",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
            "include": [
                {"task": "layout", "sources": ["FakeMinerU"]},
                {"task": "table", "sources": ["FakeMinerU"]},
            ],
            "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
            "execution": {"view_build": {"schema_sample_size": 1}},
        },
        progress=progress,
    )

    logs = stream.getvalue()
    assert "schema_sample_size=1" in logs
    validate_view(view_root)


def test_legacy_image_materialization_modes_are_rejected(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    for mode in ("embedded_bytes", "cached", "path", "runtime"):
        config = {
            "name": f"test_legacy_{mode}",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "paths": {"canonical_root": str(canonical_root), "view_root": str(tmp_path / "views" / mode)},
            "include": [{"task": "text", "sources": ["FakeMinerU"]}],
            "target_serialization": {"text": "plain_text_v1"},
            "image_policy": {"materialization": {"mode": mode}},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
            "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        }
        try:
            ViewBuilder(canonical_root, tmp_path / "views" / mode).build(config)
        except ValueError as exc:
            assert "unsupported image materialization mode" in str(exc)
        else:
            raise AssertionError(f"legacy mode {mode} was accepted")

def test_view_builder_samples_documents_by_source_across_tasks(tmp_path):
    builder = ViewBuilder(tmp_path / "canonical", tmp_path / "views")
    records = []
    for doc_idx in range(5):
        for task in ("text", "table"):
            records.append(
                {
                    "record_id": f"docbank-{doc_idx}-{task}",
                    "source_name": "DocBank_500K",
                    "task": task,
                    "document_id": f"docbank-{doc_idx}",
                    "page_id": f"docbank-{doc_idx}-p0",
                }
            )
    records.append(
        {
            "record_id": "hybrid-0-text",
            "source_name": "MinerU_Hybrid_4_23",
            "task": "text",
            "document_id": "hybrid-0",
            "page_id": "hybrid-0-p0",
        }
    )

    sampled = builder._apply_samples(
        records,
        [{"sources": ["DocBank_500K"], "level": "document", "count": 2, "seed": 123}],
    )

    docbank_docs = {row["document_id"] for row in sampled if row["source_name"] == "DocBank_500K"}
    assert len(docbank_docs) == 2
    assert sum(row["source_name"] == "DocBank_500K" for row in sampled) == 4
    assert any(row["source_name"] == "MinerU_Hybrid_4_23" for row in sampled)


def test_view_builder_shards_split_outputs(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_sft_sharded"
    config = {
        "name": "mineru25_sft_sharded",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "layout", "sources": ["FakeMinerU"]},
            {"task": "table", "sources": ["FakeMinerU"]},
            {"task": "formula", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "shard_policy": {"rows_per_shard": 2},
        "reward_profile": {"default": "normalized_levenshtein_v1"},
    }

    report = ViewBuilder(canonical_root, view_root).build(config)

    assert report.split_counts == {"train": 4, "val": 0, "test": 0}
    assert not (view_root / "train.parquet").exists()
    shard_paths = sorted((view_root / "train").glob("part-*.parquet"))
    assert [path.name for path in shard_paths] == ["part-00000.parquet", "part-00001.parquet"]
    assert [len(pd.read_parquet(path)) for path in shard_paths] == [2, 2]
    assert [len(pd.read_parquet(path, dtype_backend="pyarrow")) for path in shard_paths] == [2, 2]
    validate_view(view_root, require_images=True)


def test_sharded_source_reference_view_uses_root_assets_and_filename_refs(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_sft_source_reference_sharded"
    config = {
        "name": "mineru25_sft_source_reference_sharded",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "text", "sources": ["FakeMinerU"]}],
        "target_serialization": {"text": "plain_text_v1"},
        "image_policy": {"materialization": {"mode": "source_reference"}},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "shard_policy": {"rows_per_shard": 1},
    }

    ViewBuilder(canonical_root, view_root).build(config)

    shard_path = view_root / "train" / "part-00000.parquet"
    train = pd.read_parquet(shard_path)
    image_name = _as_list(train.iloc[0]["images_path"])[0]
    assert Path(image_name).name == image_name
    assert (view_root / "assets" / image_name).is_file()
    assert train.iloc[0]["image_path"] == image_name
    assert not (view_root / "train" / "assets").exists()
    validate_view(view_root, require_images=True)


def test_cli_validate_view_resolves_name_from_dataset_root(monkeypatch, tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "mineru25_rlvr"
    monkeypatch.setenv("OCR_DATASET_ROOT", str(dataset_root))
    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "mineru25_rlvr",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "include": [{"task": "text", "sources": ["FakeMinerU"]}],
            "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
        }
    )

    docds_main(["validate-view", "mineru25_rlvr"])


def test_cli_validate_view_absolute_path_does_not_require_dataset_root(monkeypatch, tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "absolute_path_view"
    monkeypatch.delenv("OCR_DATASET_ROOT", raising=False)
    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "absolute_path_view",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "include": [{"task": "text", "sources": ["FakeMinerU"]}],
            "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
        }
    )

    docds_main(["validate-view", str(view_root), "--require-images"])


def test_score_predictions_cli_path(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_rlvr"
    config_path = tmp_path / "view.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "mineru25_rlvr",
                "stage": "rlvr",
                "model_family": "mineru2.5",
                "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
                "include": [{"task": "text", "sources": ["FakeMinerU"]}],
                "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
                "reward_profile": {"default": "normalized_levenshtein_v1"},
            }
        )
    )
    docds_main(["build-view", str(config_path)])
    train = pd.read_parquet(view_root / "train.parquet")
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame([{"id": train.iloc[0]["id"], "prediction": train.iloc[0]["label"]}]).to_parquet(predictions_path)
    output_path = tmp_path / "scores.parquet"

    count = score_predictions(view_root, predictions_path, output_path)
    scores = pd.read_parquet(output_path)

    assert count == 1
    assert scores.iloc[0]["normalized_score"] == 1.0


def test_canonical_export_progress_logs_relative_paths(tmp_path):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    canonical_root = tmp_path / "canonical"
    _write_fake_mineru_dataset(mineru_root, source_root)
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, root=tmp_path)

    MinerUSourceAdapter(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            dataset_name="FakeMinerU",
            allow_unreadable_images=True,
        )
    ).export(canonical_root, progress=progress)

    logs = stream.getvalue()
    assert "export-source" in logs
    assert "FakeMinerU" in logs
    assert "scanned=1" in logs
    assert "canonical/entities/documents/source=FakeMinerU" in logs


def test_view_build_progress_logs_relative_paths(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_rlvr"
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, root=tmp_path)

    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "mineru25_rlvr",
            "stage": "rlvr",
            "model_family": "mineru2.5",
            "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
            "include": [{"task": "text", "sources": ["FakeMinerU"]}],
            "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
            "reward_profile": {"default": "normalized_levenshtein_v1"},
        },
        progress=progress,
    )

    logs = stream.getvalue()
    assert "build-view" in logs
    assert "views/mineru25_rlvr" in logs
    assert "materialize" in logs
    assert "total=1" in logs


def test_cli_build_view_progress_uses_stderr_and_keeps_stdout_json(tmp_path, capsys):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_rlvr"
    config_path = tmp_path / "view.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "mineru25_rlvr",
                "stage": "rlvr",
                "model_family": "mineru2.5",
                "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
                "include": [{"task": "text", "sources": ["FakeMinerU"]}],
                "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
                "reward_profile": {"default": "normalized_levenshtein_v1"},
            }
        )
    )

    docds_main(["build-view", str(config_path), "--progress", "--log-every", "1"])

    captured = capsys.readouterr()
    assert json.loads(captured.out)["total_records"] == 1
    assert "build-view" in captured.err
    assert "materialize" in captured.err


def test_cli_export_source_progress_uses_stderr_and_keeps_stdout_json(monkeypatch, tmp_path, capsys):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    canonical_root = tmp_path / "canonical"
    monkeypatch.setenv("OCR_DATASET_ROOT", str(tmp_path))
    _write_fake_mineru_dataset(mineru_root, source_root)
    config_path = tmp_path / "source.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "FakeMinerU",
                "adapter": "mineru",
                "mineru_root": str(mineru_root),
                "source_image_root": str(source_root),
            }
        )
    )

    docds_main(
        [
            "export-source",
            "mineru",
            "--source-config",
            str(config_path),
            "--canonical-root",
            str(canonical_root),
            "--progress",
            "--log-every",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out)["documents"] == 1
    assert "export-source" in captured.err
    assert "canonical/entities/documents/source=FakeMinerU" in captured.err
