import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from tools.data_management.canonical import validate_canonical
from tools.data_management.cli import main as docds_main
from tools.data_management.config import load_processing_config
from tools.data_management.prompts import load_prompt_config, resolve_prompt
from tools.data_management.serializers.layout_mineru import MinerULayoutSerializer
from tools.data_management.sources.adapters.mineru import MinerUExportOptions, MinerUSourceAdapter
from tools.data_management.views import ViewBuilder, reward_smoke_test, score_predictions, validate_view


def _minimal_png(width: int = 100, height: int = 200) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


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
    # VERL compatibility columns
    for column in ("images", "data_source", "extra_info"):
        assert column in train.columns, f"missing column {column}"
    for _, row in train.iterrows():
        assert row["data_source"] == row["task"]
        assert row["extra_info"]["sample_id"] == row["id"]
        assert row["extra_info"]["task_type"] == row["task"]
        assert hasattr(row["images"], "__len__")
        assert len(row["images"]) == 1
        assert isinstance(row["images"][0], dict)
        assert "image" in row["images"][0]
    smoke = reward_smoke_test(view_root)
    assert smoke["min_score"] == 1.0


def test_view_image_assets_with_transform(tmp_path):
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
        assert row["images"][0]["image"].startswith(str(view_root / "assets" / "layout"))
        assert Path(row["image_path"]).exists()
        assert Path(row["image_path"]).stem == row["view_image_asset_id"]

    for _, row in text_rows.iterrows():
        assert row["view_image_asset_id"] is not None
        assert isinstance(row["images"][0], dict)
        assert "image" in row["images"][0]
        assert row["image_path"] == row["images"][0]["image"]

    # View asset files exist on disk
    assets_dir = view_root / "assets" / "layout"
    assert assets_dir.is_dir()
    assert len(list(assets_dir.glob("*.png"))) > 0


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
