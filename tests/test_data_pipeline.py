import json
import shutil
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from tools.data_management.canonical import validate_canonical
from tools.data_management.cli import main as docds_main
from tools.data_management.config import load_processing_config
from tools.data_management.progress import ProgressReporter
from tools.data_management.prompts import load_prompt_config, resolve_prompt
from tools.data_management.serializers.layout_mineru import MinerULayoutSerializer
from tools.data_management.serializers.table_text import TableTextSerializer
from tools.data_management.sources.adapters.hybrid_message import HybridMessageExportOptions, HybridMessageSourceAdapter
from tools.data_management.sources.adapters.mineru import MinerUExportOptions, MinerUSourceAdapter
from tools.data_management.sources.adapters.pubtable import PubTableExportOptions, PubTableSourceAdapter
from tools.data_management.sources.adapters.unirec import UniRecExportOptions, UniRecSourceAdapter
from tools.data_management.validate_grpo_view import validate_grpo_view
from tools.data_management.views import ViewBuilder, reward_smoke_test, score_predictions, validate_view
from tools.data_management.views.builder import _SplitParquetWriter


@pytest.fixture(autouse=True)
def _isolate_ocr_data_root(monkeypatch):
    monkeypatch.delenv("OCR_DATA_ROOT", raising=False)


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


def _write_fake_pubtable_dataset(root: Path, samples: list[tuple[str, int, int]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "filename": [filename for filename, _, _ in samples],
            "image": pa.array(
                [{"bytes": _minimal_png(width, height), "path": filename} for filename, width, height in samples],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "otsl": [["<table>", "</table>"] for _ in samples],
            "html_with_text": [
                ["<tr>", "<td>", filename, "</td>", "</tr>"]
                for filename, _, _ in samples
            ],
            "cols": [999 for _ in samples],
            "rows": [888 for _ in samples],
        }
    )
    pq.write_table(table, root / "part-00000.parquet")


def _write_fake_unirec_subset(root: Path, subset: str, rows: list[dict]) -> None:
    annotation_dir = root / subset / "annotations"
    image_dir = root / subset / "images"
    metadata_dir = root / subset / "metadata"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (annotation_dir / "records.jsonl").open("w") as f:
        for idx, row in enumerate(rows):
            shard = row.get("shard", f"{idx:02x}")
            filename = row["original_file_name"]
            image_path = image_dir / shard / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(_minimal_png(*row.get("image_size", (48, 16))))
            payload = {
                "record_id": row["record_id"],
                "subset_id": subset,
                "category": row["category"],
                "language": row.get("language", "mixed"),
                "document_types": row.get("document_types", []),
                "annotation_source": "fixture",
                "source_index": idx,
                "size_key": row.get("size_key", "48_16"),
                "original_file_name": filename,
                "image_path": f"subsets/{row['category']}/{subset}/images/{shard}/{filename}",
                "label": row["label"],
                "label_empty": row.get("label_empty", False),
                "raw": {"file_name": filename, "label": row["label"]},
            }
            if row.get("type"):
                payload["type"] = row["type"]
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (metadata_dir / "asset_export.json").write_text(
        json.dumps({"subset_id": subset, "asset_dir": f"subsets/{rows[0]['category']}/{subset}/images"})
    )


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


def _set_asset_dimensions(canonical_root: Path, asset_id: str, *, width: int, height: int) -> None:
    manifest_path = canonical_root / "assets/manifests/source=FakeMinerU/part-00000.parquet"
    assets = pd.read_parquet(manifest_path)
    mask = assets["asset_id"] == asset_id
    assert mask.any()
    assets.loc[mask, "width"] = width
    assets.loc[mask, "height"] = height
    assets.to_parquet(manifest_path, index=False)


class _FailingOnBrokenTextSerializer:
    name = "test_failing_text_v1"
    version = "1.0.0"
    task = "text"

    def serialize(self, canonical_record: dict, context: dict) -> str:
        del context
        target = canonical_record.get("target") or {}
        text = str(target.get("text", ""))
        if text == "BROKEN_SERIALIZER_RECORD":
            raise ValueError("test serializer rejected BROKEN_SERIALIZER_RECORD")
        return text


def _fake_view_config(canonical_root: Path, view_root: Path, *, include: list[dict], **overrides) -> dict:
    config = {
        "name": view_root.name,
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": include,
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
        "reward_profile": {"default": "normalized_levenshtein_v1"},
    }
    config.update(overrides)
    return config


def _write_view_config(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config))
    return path


def _append_text_record_with_missing_image(canonical_root: Path) -> str:
    text_path = canonical_root / "records/text/source=FakeMinerU/part-00000.parquet"
    texts = pd.read_parquet(text_path)
    bad = texts.iloc[0].copy()
    bad["record_id"] = "fake-text-missing-image"
    bad["region_id"] = "fake-text-missing-image-region"
    bad["image_asset_id"] = "missing-image-asset"
    pd.concat([texts, pd.DataFrame([bad])], ignore_index=True).to_parquet(text_path, index=False)
    return str(bad["record_id"])


def _replace_text_record_image_asset(canonical_root: Path, image_asset_id: str) -> str:
    text_path = canonical_root / "records/text/source=FakeMinerU/part-00000.parquet"
    texts = pd.read_parquet(text_path)
    texts.at[0, "image_asset_id"] = image_asset_id
    texts.to_parquet(text_path, index=False)
    return str(texts.iloc[0]["record_id"])


def _replace_text_target(canonical_root: Path, target: dict) -> str:
    text_path = canonical_root / "records/text/source=FakeMinerU/part-00000.parquet"
    texts = pd.read_parquet(text_path)
    texts.at[0, "target"] = target
    texts.to_parquet(text_path, index=False)
    return str(texts.iloc[0]["record_id"])


def test_mineru_parallel_export_matches_serial_output(tmp_path):
    serial_dataset_root = tmp_path / "serial"
    serial_mineru_root = serial_dataset_root / "mineru"
    serial_source_root = serial_dataset_root / "source"
    _write_fake_mineru_dataset(serial_mineru_root, serial_source_root, sample="paper_001")
    _write_fake_mineru_dataset(serial_mineru_root, serial_source_root, sample="paper_002")

    parallel_dataset_root = tmp_path / "parallel"
    parallel_mineru_root = parallel_dataset_root / "mineru"
    parallel_source_root = parallel_dataset_root / "source"
    _write_fake_mineru_dataset(parallel_mineru_root, parallel_source_root, sample="paper_001")
    _write_fake_mineru_dataset(parallel_mineru_root, parallel_source_root, sample="paper_002")

    serial_root = serial_dataset_root / "canonical"
    parallel_root = parallel_dataset_root / "canonical"
    base_options = {
        "dataset_name": "FakeMinerU",
        "allow_unreadable_images": True,
    }

    serial_report = MinerUSourceAdapter(
        MinerUExportOptions(**base_options, mineru_root=serial_mineru_root, source_image_root=serial_source_root)
    ).export(serial_root)
    parallel_report = MinerUSourceAdapter(
        MinerUExportOptions(
            **base_options,
            mineru_root=parallel_mineru_root,
            source_image_root=parallel_source_root,
            num_workers=2,
            worker_chunksize=1,
            max_in_flight=2,
        )
    ).export(parallel_root)

    assert parallel_report.to_dict() == serial_report.to_dict()
    validate_canonical(parallel_root, source="FakeMinerU")

    def normalize_temp_root(value, root: Path):
        if isinstance(value, str):
            return value.replace(str(root), "<dataset_root>")
        if hasattr(value, "tolist"):
            return normalize_temp_root(value.tolist(), root)
        if isinstance(value, dict):
            return {key: normalize_temp_root(item, root) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize_temp_root(item, root) for item in value]
        return value

    for relative in [
        "entities/documents/source=FakeMinerU/part-00000.parquet",
        "entities/pages/source=FakeMinerU/part-00000.parquet",
        "entities/regions/source=FakeMinerU/part-00000.parquet",
        "assets/manifests/source=FakeMinerU/part-00000.parquet",
        "records/text/source=FakeMinerU/part-00000.parquet",
        "records/table/source=FakeMinerU/part-00000.parquet",
        "records/formula/source=FakeMinerU/part-00000.parquet",
        "records/layout/source=FakeMinerU/part-00000.parquet",
    ]:
        serial_frame = pd.read_parquet(serial_root / relative)
        parallel_frame = pd.read_parquet(parallel_root / relative)
        sort_column = next(
            column
            for column in ("document_id", "page_id", "region_id", "asset_id", "record_id")
            if column in serial_frame.columns
        )
        serial_frame = serial_frame.sort_values(sort_column).reset_index(drop=True)
        parallel_frame = parallel_frame.sort_values(sort_column).reset_index(drop=True)
        assert normalize_temp_root(parallel_frame.to_dict(orient="records"), parallel_dataset_root) == normalize_temp_root(
            serial_frame.to_dict(orient="records"),
            serial_dataset_root,
        )


def test_pubtable_export_materializes_embedded_images_with_dataset_relative_paths(tmp_path):
    data_dir = tmp_path / "sources" / "pubtable" / "data"
    _write_fake_pubtable_dataset(data_dir, [("PMC2147049_table_0.jpg", 37, 19)])
    canonical_root = tmp_path / "canonical"

    report = PubTableSourceAdapter(
        PubTableExportOptions(data_dir=data_dir, dataset_name="PubTableFake")
    ).export(canonical_root)

    assert report.documents == 1
    assert report.pages == 1
    assert report.regions == 1
    assert report.assets == 2
    assets = pd.read_parquet(canonical_root / "assets/manifests/source=PubTableFake/part-00000.parquet")
    pages = pd.read_parquet(canonical_root / "entities/pages/source=PubTableFake/part-00000.parquet")
    regions = pd.read_parquet(canonical_root / "entities/regions/source=PubTableFake/part-00000.parquet")
    table = pd.read_parquet(canonical_root / "records/table/source=PubTableFake/part-00000.parquet").iloc[0]

    expected_path = "canonical/assets/files/source=PubTableFake/PMC2147049_table_0.jpg"
    assert set(assets["path"]) == {expected_path}
    assert (tmp_path / expected_path).is_file()
    assert _decode_png((tmp_path / expected_path).read_bytes()).size == (37, 19)
    assert set(assets["width"]) == {37}
    assert set(assets["height"]) == {19}
    assert pages.iloc[0]["width"] == 37
    assert pages.iloc[0]["height"] == 19
    assert list(regions.iloc[0]["bbox"]) == [0, 0, 37, 19]
    assert table["target"] == {
        "html": "<table><tr><td>PMC2147049_table_0.jpg</td></tr></table>"
    }


def test_pubtable_parallel_export_matches_serial_output(tmp_path):
    samples = [("table_001.jpg", 31, 17), ("table_002.jpg", 29, 13), ("table_003.jpg", 23, 11)]
    serial_data_dir = tmp_path / "serial" / "sources" / "pubtable" / "data"
    parallel_data_dir = tmp_path / "parallel" / "sources" / "pubtable" / "data"
    _write_fake_pubtable_dataset(serial_data_dir, samples)
    _write_fake_pubtable_dataset(parallel_data_dir, samples)

    serial_root = tmp_path / "serial" / "canonical"
    parallel_root = tmp_path / "parallel" / "canonical"
    base_options = {"dataset_name": "PubTableFake", "shard_size": 2}
    serial_report = PubTableSourceAdapter(
        PubTableExportOptions(**base_options, data_dir=serial_data_dir)
    ).export(serial_root)
    parallel_report = PubTableSourceAdapter(
        PubTableExportOptions(
            **base_options,
            data_dir=parallel_data_dir,
            num_workers=2,
            worker_chunksize=1,
        )
    ).export(parallel_root)

    assert parallel_report.to_dict() == serial_report.to_dict()
    validate_canonical(parallel_root, source="PubTableFake")

    def normalize_value(value):
        if hasattr(value, "tolist"):
            return normalize_value(value.tolist())
        if isinstance(value, dict):
            return {key: normalize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        return value

    for relative in [
        "entities/documents/source=PubTableFake",
        "entities/pages/source=PubTableFake",
        "entities/regions/source=PubTableFake",
        "assets/manifests/source=PubTableFake",
        "records/table/source=PubTableFake",
    ]:
        serial_frame = pd.concat(pd.read_parquet(path) for path in sorted((serial_root / relative).glob("part-*.parquet")))
        parallel_frame = pd.concat(
            pd.read_parquet(path) for path in sorted((parallel_root / relative).glob("part-*.parquet"))
        )
        sort_column = next(
            column
            for column in ("document_id", "page_id", "region_id", "asset_id", "record_id")
            if column in serial_frame.columns
        )
        serial_frame = serial_frame.sort_values(sort_column).reset_index(drop=True)
        parallel_frame = parallel_frame.sort_values(sort_column).reset_index(drop=True)
        assert normalize_value(parallel_frame.to_dict(orient="records")) == normalize_value(
            serial_frame.to_dict(orient="records")
        )


def test_mineru_export_skip_completed_appends_missing_samples(tmp_path):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    canonical_root = tmp_path / "canonical"
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_001")
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_002")

    MinerUSourceAdapter(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            dataset_name="FakeMinerU",
            max_samples=1,
            allow_unreadable_images=True,
        )
    ).export(canonical_root)

    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, root=tmp_path)
    report = MinerUSourceAdapter(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            dataset_name="FakeMinerU",
            allow_unreadable_images=True,
            skip_completed=True,
            num_workers=2,
            worker_chunksize=1,
            max_in_flight=2,
        )
    ).export(canonical_root, overwrite_partitions=False, progress=progress)

    assert report.documents == 2
    validate_canonical(canonical_root, source="FakeMinerU")
    documents = pd.concat(
        pd.read_parquet(path)
        for path in sorted((canonical_root / "entities/documents/source=FakeMinerU").glob("part-*.parquet"))
    )
    assert sorted(documents["source_document_id"]) == ["paper_001", "paper_002"]
    assert documents["document_id"].is_unique
    assert "phase=skip_completed" in stream.getvalue()


def test_progress_update_logs_monitoring_metrics():
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, force_tty=False)

    progress.update("export-source", 5, total=10, workers=2, skipped_completed=1)

    logs = stream.getvalue()
    assert "elapsed_s=" in logs
    assert "rate_per_s=" in logs
    assert "pct=50.0" in logs
    assert "eta_s=" in logs
    assert "workers=2" in logs
    assert "skipped_completed=1" in logs


def test_hybrid_message_adapter_copies_text_images_to_canonical_region_crops(tmp_path):
    dataset_root = tmp_path
    source_image = dataset_root / "external" / "line.jpg"
    source_image.parent.mkdir()
    source_image.write_bytes(_minimal_png(32, 12))
    data_file = dataset_root / "sources" / "UniRec_990K" / "data.json"
    data_file.parent.mkdir(parents=True)
    data_file.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "<image>\nText Recognition:"},
                        {"role": "assistant", "content": "hello"},
                    ],
                    "images": [str(source_image)],
                }
            ]
        )
    )
    canonical_root = dataset_root / "canonical"

    report = HybridMessageSourceAdapter(
        HybridMessageExportOptions(
            data_file=data_file,
            dataset_name="UniRec_990K",
            allow_unreadable_images=False,
        )
    ).export(canonical_root)

    assert report.task_records == {"text": 1}
    assets = pd.read_parquet(canonical_root / "assets/manifests/source=UniRec_990K/part-00000.parquet")
    regions = pd.read_parquet(canonical_root / "entities/regions/source=UniRec_990K/part-00000.parquet")
    text = pd.read_parquet(canonical_root / "records/text/source=UniRec_990K/part-00000.parquet").iloc[0]

    crop_assets = assets[assets["asset_type"] == "region_crop"]
    assert len(crop_assets) == 1
    crop_asset = crop_assets.iloc[0]
    assert not Path(crop_asset["path"]).is_absolute()
    assert crop_asset["path"].startswith("canonical/assets/files/source=UniRec_990K/region_crop/")
    assert (dataset_root / crop_asset["path"]).is_file()
    assert _decode_png((dataset_root / crop_asset["path"]).read_bytes()).size == (32, 12)
    assert text["image_asset_id"] == crop_asset["asset_id"]
    assert text["metadata"]["uses_page_image"] is False
    assert regions.iloc[0]["crop_asset_id"] == crop_asset["asset_id"]


def test_hybrid_message_from_profile_uses_declared_data_file_and_mirrored_image_root(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "HaoY-Syn-Formula"
    mirrored_image = source_root / "images" / "formula_001.png"
    mirrored_image.parent.mkdir(parents=True)
    mirrored_image.write_bytes(_minimal_png(48, 16))
    data_file = source_root / "1022340.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "<image>Formula Recognition"},
                        {"role": "assistant", "content": r"x^2 + y^2 = z^2"},
                    ],
                    "images": ["/old/mount/Formula/images/formula_001.png"],
                }
            ]
        )
    )
    profile = tmp_path / "haoy_formula.yaml"
    profile.write_text(
        "\n".join(
            [
                "dataset_name: HaoY-Syn-Formula",
                "source_path: sources/HaoY-Syn-Formula",
                "source_image_root: sources/HaoY-Syn-Formula/images",
                "data_file: 1022340.json",
                "adapter: hybrid_message",
                "export_tasks: [formula]",
            ]
        )
    )
    canonical_root = dataset_root / "canonical"

    adapter = HybridMessageSourceAdapter.from_profile(profile, dataset_root=dataset_root)
    report = adapter.export(canonical_root, tasks=["formula"])

    assert report.task_records == {"formula": 1}
    formula = pd.read_parquet(canonical_root / "records/formula/source=HaoY-Syn-Formula/part-00000.parquet").iloc[0]
    assets = pd.read_parquet(canonical_root / "assets/manifests/source=HaoY-Syn-Formula/part-00000.parquet")
    crop_asset = assets[assets["asset_type"] == "region_crop"].iloc[0]
    assert formula["target"]["latex"] == r"x^2 + y^2 = z^2"
    assert (dataset_root / crop_asset["path"]).is_file()
    assert _decode_png((dataset_root / crop_asset["path"]).read_bytes()).size == (48, 16)


def test_hybrid_message_source_reference_image_materialization_reuses_source_images(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "HaoY-Syn-Formula"
    source_image = source_root / "images" / "formula_001.png"
    source_image.parent.mkdir(parents=True)
    source_image.write_bytes(_minimal_png(48, 16))
    data_file = source_root / "1022340.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "<image>Formula Recognition"},
                        {"role": "assistant", "content": r"x^2 + y^2 = z^2"},
                    ],
                    "images": [str(source_image)],
                }
            ]
        )
    )
    profile = tmp_path / "haoy_formula.yaml"
    profile.write_text(
        "\n".join(
            [
                "dataset_name: HaoY-Syn-Formula",
                "source_path: sources/HaoY-Syn-Formula",
                "source_image_root: sources/HaoY-Syn-Formula/images",
                "data_file: 1022340.json",
                "image_materialization: source_reference",
                "adapter: hybrid_message",
                "export_tasks: [formula]",
            ]
        )
    )
    canonical_root = dataset_root / "canonical"

    adapter = HybridMessageSourceAdapter.from_profile(profile, dataset_root=dataset_root)
    adapter.export(canonical_root, tasks=["formula"])

    assets = pd.read_parquet(canonical_root / "assets/manifests/source=HaoY-Syn-Formula/part-00000.parquet")
    crop_asset = assets[assets["asset_type"] == "region_crop"].iloc[0]
    assert crop_asset["path"] == "sources/HaoY-Syn-Formula/images/formula_001.png"
    assert not (canonical_root / "assets" / "files" / "source=HaoY-Syn-Formula").exists()


def test_unirec_adapter_exports_region_records_and_cleans_text_labels(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "UniRec40M_english"
    _write_fake_unirec_subset(
        source_root,
        "hiertext",
        [
            {
                "record_id": "hiertext_001",
                "category": "english",
                "language": "en",
                "type": "para",
                "original_file_name": "hiertext_001.jpg",
                "image_size": (40, 20),
                "label": (
                    r"Area is \(x^2\)<|ln|>today<|pn|>now<<<change_line_token_wrap>>>"
                    r"done and \(broken. Escaped formula is \$y+1\$"
                ),
            }
        ],
    )
    canonical_root = dataset_root / "canonical"

    report = UniRecSourceAdapter(
        UniRecExportOptions(source_root=source_root, dataset_name="UniRec40M_english")
    ).export(canonical_root, tasks=["text", "formula", "table"])

    assert report.task_records == {"text": 1}
    assert report.documents == 1
    assert report.pages == 1
    assert report.regions == 1
    assert report.assets == 2
    text = pd.read_parquet(canonical_root / "records/text/source=UniRec40M_english/part-00000.parquet").iloc[0]
    regions = pd.read_parquet(canonical_root / "entities/regions/source=UniRec40M_english/part-00000.parquet")
    assets = pd.read_parquet(canonical_root / "assets/manifests/source=UniRec40M_english/part-00000.parquet")
    crop_asset = assets[assets["asset_type"] == "region_crop"].iloc[0]

    assert text["target"] == {"text": "Area is $x^2$todaynowdone and broken. Escaped formula is $y+1$"}
    assert r"\$" not in text["target"]["text"]
    assert text["task"] == "text"
    assert text["image_asset_id"] == crop_asset["asset_id"]
    assert crop_asset["path"] == "sources/UniRec40M_english/hiertext/images/00/hiertext_001.jpg"
    assert list(regions.iloc[0]["bbox"]) == [0.0, 0.0, 40.0, 20.0]
    assert regions.iloc[0]["crop_asset_id"] == crop_asset["asset_id"]


def test_unirec_adapter_classifies_and_strips_standalone_formula_wrappers(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "UniRec40M_formula"
    _write_fake_unirec_subset(
        source_root,
        "latex_aug_formula",
        [
            {
                "record_id": "latex_001",
                "category": "formula",
                "language": "en",
                "original_file_name": "formula_001.jpg",
                "image_size": (32, 12),
                "label": r"\[ x^2 + y^2 = z^2 \]",
            },
            {
                "record_id": "latex_002",
                "category": "formula",
                "language": "en",
                "original_file_name": "formula_002.jpg",
                "image_size": (30, 10),
                "label": r"plain words and \(x\)",
            },
            {
                "record_id": "latex_003",
                "category": "formula",
                "language": "en",
                "original_file_name": "formula_003.jpg",
                "image_size": (30, 10),
                "label": r"\[\sin 0 = 1.",
            },
        ],
    )
    canonical_root = dataset_root / "canonical"

    report = UniRecSourceAdapter(
        UniRecExportOptions(source_root=source_root, dataset_name="UniRec40M_formula")
    ).export(canonical_root, tasks=["text", "formula"])

    assert report.task_records == {"formula": 2, "text": 1}
    formula = pd.read_parquet(canonical_root / "records/formula/source=UniRec40M_formula/part-00000.parquet")
    text = pd.read_parquet(canonical_root / "records/text/source=UniRec40M_formula/part-00000.parquet")

    latex_values = {row["latex"] for row in formula["target"]}
    assert latex_values == {"x^2 + y^2 = z^2", r"\sin 0 = 1."}
    assert set(formula["category"]) == {"formula"}
    assert text.iloc[0]["target"] == {"text": "plain words and $x$"}
    assert text.iloc[0]["task"] == "text"


def test_unirec_adapter_detects_table_records_from_metadata(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "UniRec40M_finance_report"
    _write_fake_unirec_subset(
        source_root,
        "dfcf_finance_pdf",
        [
            {
                "record_id": "finance_table_001",
                "category": "finance_report",
                "language": "zh",
                "document_types": ["Chinese", "table"],
                "original_file_name": "report_table_001.jpg",
                "image_size": (50, 30),
                "label": "项目\t公式\n收入\t" + r"\$x^2\$",
            }
        ],
    )
    canonical_root = dataset_root / "canonical"

    report = UniRecSourceAdapter(
        UniRecExportOptions(source_root=source_root, dataset_name="UniRec40M_finance_report")
    ).export(canonical_root, tasks=["text", "formula", "table"])

    assert report.task_records == {"table": 1}
    table = pd.read_parquet(canonical_root / "records/table/source=UniRec40M_finance_report/part-00000.parquet").iloc[0]
    assert table["target"] == {"text": "项目\t公式\n收入\t$x^2$"}
    assert r"\$" not in table["target"]["text"]


def test_unirec_adapter_preserves_duplicate_record_ids_across_subsets(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "UniRec40M_english"
    for subset in ("subset_a", "subset_b"):
        _write_fake_unirec_subset(
            source_root,
            subset,
            [
                {
                    "record_id": "duplicate_001",
                    "category": "english",
                    "language": "en",
                    "original_file_name": f"{subset}.jpg",
                    "image_size": (40, 20),
                    "label": f"text from {subset}",
                }
            ],
        )
    canonical_root = dataset_root / "canonical"

    report = UniRecSourceAdapter(
        UniRecExportOptions(source_root=source_root, dataset_name="UniRec40M_english")
    ).export(canonical_root, tasks=["text"])

    assert report.documents == 2
    assert report.task_records == {"text": 2}
    docs = pd.read_parquet(canonical_root / "entities/documents/source=UniRec40M_english/part-00000.parquet")
    text = pd.read_parquet(canonical_root / "records/text/source=UniRec40M_english/part-00000.parquet")
    assert len(set(docs["document_id"])) == 2
    assert sorted(row["text"] for row in text["target"]) == ["text from subset_a", "text from subset_b"]


def test_unirec_nested_reference_view_keeps_text_formula_and_table_rows(tmp_path):
    dataset_root = tmp_path
    source_root = dataset_root / "sources" / "UniRec40M_mixed"
    _write_fake_unirec_subset(
        source_root,
        "mixed",
        [
            {
                "record_id": "text_001",
                "category": "english",
                "language": "en",
                "original_file_name": "text.jpg",
                "image_size": (40, 20),
                "label": "plain text",
            },
            {
                "record_id": "formula_001",
                "category": "formula",
                "language": "en",
                "original_file_name": "formula.jpg",
                "image_size": (40, 20),
                "label": r"\[ x + y \]",
            },
            {
                "record_id": "table_001",
                "category": "finance_report",
                "language": "zh",
                "document_types": ["table"],
                "original_file_name": "table.jpg",
                "image_size": (40, 20),
                "label": "项目\t公式\n收入\t" + r"\$x^2\$",
            },
        ],
    )
    canonical_root = dataset_root / "canonical"
    UniRecSourceAdapter(
        UniRecExportOptions(source_root=source_root, dataset_name="UniRec40M_mixed")
    ).export(canonical_root, tasks=["text", "formula", "table"])
    view_root = dataset_root / "views" / "unirec_mixed"
    config = {
        "name": "unirec_mixed",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "text", "sources": ["UniRec40M_mixed"]},
            {"task": "formula", "sources": ["UniRec40M_mixed"]},
            {"task": "table", "sources": ["UniRec40M_mixed"]},
        ],
        "target_serialization": {"text": "plain_text_v1", "formula": "latex_plain_v1", "table": "table_text_v1"},
        "image_policy": {"materialization": {"mode": "nested_reference"}},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "shard_policy": {"rows_per_shard": 10},
    }

    report = ViewBuilder(canonical_root, view_root).build(config)

    assert report.total_records == 3
    train = pd.read_parquet(view_root / "train" / "part-00000.parquet")
    assert sorted(train["task"].tolist()) == ["formula", "table", "text"]
    labels_by_task = dict(zip(train["task"], train["label"], strict=True))
    assert labels_by_task["table"] == "项目\t公式\n收入\t$x^2$"
    assert r"\$" not in labels_by_task["table"]
    assert train["images"].notna().all()


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


def test_mineru_adapter_crops_regions_into_canonical_assets(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    dataset_root = canonical_root.parent

    assets = pd.read_parquet(canonical_root / "assets/manifests/source=FakeMinerU/part-00000.parquet")
    regions = pd.read_parquet(canonical_root / "entities/regions/source=FakeMinerU/part-00000.parquet")
    records = {
        task: pd.read_parquet(canonical_root / f"records/{task}/source=FakeMinerU/part-00000.parquet")
        for task in ("text", "table", "formula")
    }

    assert not any(Path(path).is_absolute() for path in assets["path"])
    assert not any(str(path).startswith("mineru/") for path in assets["path"])

    crop_assets = assets[assets["asset_type"] == "region_crop"]
    assert len(crop_assets) == 3
    for _, crop_asset in crop_assets.iterrows():
        crop_path = dataset_root / crop_asset["path"]
        assert crop_path.is_file()
        assert crop_path.is_relative_to(canonical_root / "assets" / "files")
        assert crop_asset["parent_asset_id"]
        assert crop_asset["transform"]["source"] == "source_image_bbox"

    for _, region in regions.iterrows():
        assert region["crop_asset_id"] in set(crop_assets["asset_id"])

    table = records["table"].iloc[0]
    table_asset = crop_assets[crop_assets["asset_id"] == table["image_asset_id"]].iloc[0]
    assert _decode_png((dataset_root / table_asset["path"]).read_bytes()).size == (60, 60)
    assert table["metadata"]["uses_page_image"] is False

    for frame in records.values():
        assert set(frame["image_asset_id"]).issubset(set(crop_assets["asset_id"]))


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


def test_table_text_serializer_returns_plain_table_text():
    record = {
        "task": "table",
        "target": {"text": "项目\t金额\n收入\t100"},
    }

    label = TableTextSerializer().serialize(record, {})

    assert label == "项目\t金额\n收入\t100"


def test_table_text_serializer_rejects_missing_text():
    record = {"task": "table", "target": {"html": "<table></table>"}}

    with pytest.raises(ValueError, match="requires text target"):
        TableTextSerializer().serialize(record, {})


def test_prompt_config_resolves_task_prompt():
    config = load_prompt_config("mineru2.5")
    prompt, template_id = resolve_prompt(config, "table", {"source_name": "FakeMinerU"})

    assert prompt == "<image>\nTable Recognition:"
    assert template_id == "table_default_v1"
    assert config.formats["table_recognition"]["raw_target"] == "otsl"
    assert config.formats["layout_detection"]["coordinate_grid"] == 1000


def test_processing_config_derives_dataset_subdirectories(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    monkeypatch.setenv("OCR_DATA_ROOT", str(dataset_root))

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


def test_cli_build_view_skip_failure_keeps_default_fail_fast_and_writes_successes(tmp_path, capsys):
    canonical_root = _export_fake_canonical(tmp_path)
    bad_record_id = _append_text_record_with_missing_image(canonical_root)
    include = [{"task": "text", "sources": ["FakeMinerU"]}]

    fail_root = tmp_path / "views" / "skip_failure_default_fails"
    fail_config = _write_view_config(
        tmp_path / "skip_failure_default_fails.yaml",
        _fake_view_config(canonical_root, fail_root, include=include),
    )
    with pytest.raises(ValueError, match=f"embedded view cannot read image bytes for record {bad_record_id}"):
        docds_main(["build-view", str(fail_config), "--schema-sample-size", "1"])

    skip_root = tmp_path / "views" / "skip_failure_cli"
    skip_config = _write_view_config(
        tmp_path / "skip_failure_cli.yaml",
        _fake_view_config(canonical_root, skip_root, include=include),
    )

    docds_main(["build-view", str(skip_config), "--skip-failure", "--schema-sample-size", "1"])

    captured = capsys.readouterr()
    assert json.loads(captured.out)["total_records"] == 1
    train = pd.read_parquet(skip_root / "train.parquet")
    assert train["canonical_record_id"].tolist() != [bad_record_id]
    assert train["task"].tolist() == ["text"]
    validate_view(skip_root)


def test_view_build_skip_failure_reports_final_materialization_diagnostics(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    bad_record_id = _append_text_record_with_missing_image(canonical_root)
    view_root = tmp_path / "views" / "skip_failure_stats"

    report = ViewBuilder(canonical_root, view_root).build(
        _fake_view_config(
            canonical_root,
            view_root,
            include=[{"task": "text", "sources": ["FakeMinerU"]}],
        ),
        skip_failure=True,
        schema_sample_size=1,
    )

    assert report.total_records == 1
    stats = json.loads((view_root / "stats.json").read_text())
    skipped = stats["skipped_materialization_failures"]
    assert skipped["total"] == 1
    assert skipped["counts"]["by_phase"] == {"materialize": 1}
    assert skipped["counts"]["by_task"] == {"text": 1}
    assert skipped["counts"]["by_source"] == {"FakeMinerU": 1}
    assert skipped["counts"]["by_exception_type"] == {"ValueError": 1}
    assert len(skipped["examples"]) <= 10
    example = skipped["examples"][0]
    assert example["phase"] == "materialize"
    assert example["record_id"] == bad_record_id
    assert example["source_name"] == "FakeMinerU"
    assert example["task"] == "text"
    assert example["target_format"] == "plain_text_v1"
    assert example["target_keys"] == ["text"]
    assert example["exception_type"] == "ValueError"
    assert "embedded view cannot read image bytes" in example["message"]
    assert "A Title" in example["target_preview"]


def test_view_build_skip_failure_fails_when_final_materialization_skips_every_record(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    _replace_text_record_image_asset(canonical_root, "missing-image-asset")
    view_root = tmp_path / "views" / "skip_failure_no_rows"

    with pytest.raises(ValueError, match="no records were materialized"):
        ViewBuilder(canonical_root, view_root).build(
            _fake_view_config(
                canonical_root,
                view_root,
                include=[{"task": "text", "sources": ["FakeMinerU"]}],
            ),
            skip_failure=True,
            schema_sample_size=1,
        )

    assert not (view_root / "train.parquet").exists()
    assert not (view_root / "stats.json").exists()


def test_view_build_skip_failure_schema_inference_skips_failed_samples_and_fails_when_all_fail(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    _replace_text_target(canonical_root, {"text": "BROKEN_SERIALIZER_RECORD"})
    config = _fake_view_config(
        canonical_root,
        tmp_path / "views" / "schema_skip_has_good_sample",
        include=[
            {"task": "text", "sources": ["FakeMinerU"]},
            {"task": "formula", "sources": ["FakeMinerU"]},
        ],
        target_serialization={"text": "test_failing_text_v1", "formula": "latex_plain_v1"},
    )
    builder = ViewBuilder(canonical_root, Path(config["paths"]["view_root"]))
    builder.serializers.register("test_failing_text_v1", _FailingOnBrokenTextSerializer())

    report = builder.build(config, skip_failure=True, schema_sample_size=1)

    assert report.total_records == 1
    train = pd.read_parquet(Path(config["paths"]["view_root"]) / "train.parquet")
    assert train["task"].tolist() == ["formula"]
    stats = json.loads((Path(config["paths"]["view_root"]) / "stats.json").read_text())
    assert stats["skipped_materialization_failures"]["counts"]["by_phase"] == {
        "infer-schema": 1,
        "materialize": 1,
    }

    all_fail_root = tmp_path / "views" / "schema_skip_all_fail"
    all_fail_config = _fake_view_config(
        canonical_root,
        all_fail_root,
        include=[{"task": "text", "sources": ["FakeMinerU"]}],
        target_serialization={"text": "test_failing_text_v1"},
    )
    all_fail_builder = ViewBuilder(canonical_root, all_fail_root)
    all_fail_builder.serializers.register("test_failing_text_v1", _FailingOnBrokenTextSerializer())
    with pytest.raises(ValueError, match="no valid records.*schema inference"):
        all_fail_builder.build(all_fail_config, skip_failure=True, schema_sample_size=1)


def test_enhanced_otsl_table_filter_drops_text_only_table_before_materialization(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    table_path = canonical_root / "records/table/source=FakeMinerU/part-00000.parquet"
    tables = pd.read_parquet(table_path)
    text_only_table_id = str(tables.iloc[0]["record_id"])
    tables.at[0, "target"] = {"text": "plain table text is not enhanced OTSL input"}
    tables.to_parquet(table_path, index=False)
    view_root = tmp_path / "views" / "enhanced_otsl_filters_text_only"

    report = ViewBuilder(canonical_root, view_root).build(
        _fake_view_config(
            canonical_root,
            view_root,
            include=[
                {"task": "table", "sources": ["FakeMinerU"]},
                {"task": "text", "sources": ["FakeMinerU"]},
            ],
            target_serialization={"table": "enhanced_otsl_v1", "text": "plain_text_v1"},
        )
    )

    assert report.total_records == 1
    train = pd.read_parquet(view_root / "train.parquet")
    assert train["task"].tolist() == ["text"]
    assert text_only_table_id not in set(train["canonical_record_id"])
    stats = json.loads((view_root / "stats.json").read_text())
    assert stats["filter_counts"]["empty_label"] == 1
    assert "skipped_materialization_failures" not in stats


def test_table_text_filter_drops_html_only_table_before_materialization(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    table_path = canonical_root / "records/table/source=FakeMinerU/part-00000.parquet"
    tables = pd.read_parquet(table_path)
    html_only_table_id = str(tables.iloc[0]["record_id"])
    tables.at[0, "target"] = {"html": "<table><tr><td>A</td></tr></table>"}
    tables.to_parquet(table_path, index=False)
    view_root = tmp_path / "views" / "table_text_filters_html_only"

    report = ViewBuilder(canonical_root, view_root).build(
        _fake_view_config(
            canonical_root,
            view_root,
            include=[
                {"task": "table", "sources": ["FakeMinerU"]},
                {"task": "text", "sources": ["FakeMinerU"]},
            ],
            target_serialization={"table": "table_text_v1", "text": "plain_text_v1"},
        )
    )

    assert report.total_records == 1
    train = pd.read_parquet(view_root / "train.parquet")
    assert train["task"].tolist() == ["text"]
    assert html_only_table_id not in set(train["canonical_record_id"])
    stats = json.loads((view_root / "stats.json").read_text())
    assert stats["filter_counts"]["empty_label"] == 1
    assert "skipped_materialization_failures" not in stats


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
    assert not Path(images_path[0]).is_absolute()
    assert (canonical_root.parent / images_path[0]).is_file()
    assert not (view_root / "assets").exists()
    assert row["image_path"] == images_path[0]


def test_view_image_assets_with_nested_reference_policy(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_nested_reference_view"
    config = {
        "name": "test_nested_reference_view",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "layout", "sources": ["FakeMinerU"]}],
        "target_serialization": {"layout": "mineru_layout_box_v1"},
        "image_policy": {"materialization": {"mode": "nested_reference"}},
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root, require_images=True)
    validate_grpo_view([view_root])
    train = pd.read_parquet(view_root / "train.parquet")
    row = train.iloc[0]

    images = _as_list(row["images"])
    assert len(images) == 1
    assert not _as_list(row["images_bytes"])
    assert not _as_list(row["images_path"])
    assert not Path(images[0]["image"]).is_absolute()
    assert (canonical_root.parent / images[0]["image"]).is_file()
    assert row["image_path"] == images[0]["image"]
    assert not (view_root / "assets").exists()


def test_view_aspect_ratio_filter_drops_records_above_threshold(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_aspect_ratio_filter"
    text_record = pd.read_parquet(canonical_root / "records/text/source=FakeMinerU/part-00000.parquet").iloc[0]
    _set_asset_dimensions(canonical_root, text_record["image_asset_id"], width=201, height=1)
    config = {
        "name": "test_aspect_ratio_filter",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "layout", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "image_policy": {"materialization": {"mode": "nested_reference"}, "filter": {"max_aspect_ratio": 200}},
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root, require_images=True)
    train = pd.read_parquet(view_root / "train.parquet")
    stats = json.loads((view_root / "stats.json").read_text())

    assert train["task"].tolist() == ["layout"]
    assert stats["total_records"] == 1
    assert stats["filtered_records"] == 1
    assert stats["filter_counts"] == {"aspect_ratio": 1, "invalid_dimensions": 0}


def test_view_aspect_ratio_filter_keeps_boundary_ratio(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_aspect_ratio_filter_boundary"
    text_record = pd.read_parquet(canonical_root / "records/text/source=FakeMinerU/part-00000.parquet").iloc[0]
    _set_asset_dimensions(canonical_root, text_record["image_asset_id"], width=200, height=1)
    config = {
        "name": "test_aspect_ratio_filter_boundary",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "text", "sources": ["FakeMinerU"]}],
        "image_policy": {"materialization": {"mode": "nested_reference"}, "filter": {"max_aspect_ratio": 200}},
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    train = pd.read_parquet(view_root / "train.parquet")
    stats = json.loads((view_root / "stats.json").read_text())

    assert train["task"].tolist() == ["text"]
    assert "filtered_records" not in stats


def test_view_aspect_ratio_filter_is_disabled_by_default(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_aspect_ratio_filter_disabled"
    text_record = pd.read_parquet(canonical_root / "records/text/source=FakeMinerU/part-00000.parquet").iloc[0]
    _set_asset_dimensions(canonical_root, text_record["image_asset_id"], width=1000, height=1)
    config = {
        "name": "test_aspect_ratio_filter_disabled",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "text", "sources": ["FakeMinerU"]}],
        "image_policy": {"materialization": {"mode": "nested_reference"}},
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    train = pd.read_parquet(view_root / "train.parquet")
    stats = json.loads((view_root / "stats.json").read_text())

    assert train["task"].tolist() == ["text"]
    assert "filtered_records" not in stats


def test_view_aspect_ratio_filter_uses_transformed_dimensions(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_aspect_ratio_filter_transform"
    layout_record = pd.read_parquet(canonical_root / "records/layout/source=FakeMinerU/part-00000.parquet").iloc[0]
    _set_asset_dimensions(canonical_root, layout_record["image_asset_id"], width=1000, height=1)
    config = {
        "name": "test_aspect_ratio_filter_transform",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "layout", "sources": ["FakeMinerU"]}],
        "image_policy": {"materialization": {"mode": "nested_reference"}, "filter": {"max_aspect_ratio": 200}},
        "image_transform": {"layout": {"pad_to_square": True, "resize_to": 128}},
        "split_policy": {"level": "record", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root, require_images=True)
    train = pd.read_parquet(view_root / "train.parquet")
    stats = json.loads((view_root / "stats.json").read_text())

    assert train["task"].tolist() == ["layout"]
    assert _as_list(train.iloc[0]["images"])[0]["image"].startswith("views/test_aspect_ratio_filter_transform/assets/")
    assert "filtered_records" not in stats


def test_validate_view_rejects_image_aspect_ratio_at_limit(tmp_path):
    view_root = tmp_path / "views" / "invalid_aspect_ratio"
    view_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id": "row-1",
                "stage": "sft",
                "task": "text",
                "image_path": "canonical/assets/files/source=Fake/image.png",
                "prompt": ["<image>"],
                "label": "text",
                "canonical_record_id": "record-1",
                "split": "train",
                "images_bytes": [_minimal_png(200, 1)],
            }
        ]
    ).to_parquet(view_root / "train.parquet", index=False)

    with pytest.raises(ValueError, match="aspect ratio"):
        validate_view(view_root, require_images=True)


def test_validate_view_parallel_rejects_image_aspect_ratio_at_limit(tmp_path):
    view_root = tmp_path / "views" / "parallel_invalid_aspect_ratio"
    split_root = view_root / "train"
    split_root.mkdir(parents=True)
    base_row = {
        "stage": "sft",
        "task": "text",
        "image_path": "canonical/assets/files/source=Fake/image.png",
        "prompt": ["<image>"],
        "label": "text",
        "split": "train",
        "images_bytes": [_minimal_png(10, 10)],
    }
    pd.DataFrame(
        [
            {"id": "row-1", "canonical_record_id": "record-1", **base_row},
            {
                "id": "row-2",
                "canonical_record_id": "record-2",
                **base_row,
                "images_bytes": [_minimal_png(200, 1)],
            },
        ]
    ).to_parquet(split_root / "part-00000.parquet", index=False, row_group_size=1)

    with pytest.raises(ValueError, match="aspect ratio"):
        validate_view(view_root, require_images=True, num_workers=2, worker_batch_size=1)


def test_split_parquet_writer_recreates_missing_temp_dir(tmp_path):
    schema = pd.DataFrame(
        [
            {
                "id": "row-1",
                "split": "train",
            }
        ]
    ).to_parquet(tmp_path / "schema_sample.parquet", index=False)
    del schema
    sample = pd.read_parquet(tmp_path / "schema_sample.parquet")
    writer = _SplitParquetWriter(
        tmp_path / "view",
        pa.Schema.from_pandas(sample),
        rows_per_shard=1,
    )

    shutil.rmtree(tmp_path / "view" / "train.tmp")
    writer.write({"id": "row-1", "split": "train"})
    split_counts = writer.finish()

    assert split_counts["train"] == 1
    assert (tmp_path / "view" / "train" / "part-00000.parquet").is_file()


def test_source_reference_transform_writes_view_asset_relative_path(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_source_reference_transform"
    config = {
        "name": "test_source_reference_transform",
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
    train = pd.read_parquet(view_root / "train.parquet")
    image_path = _as_list(train.iloc[0]["images_path"])[0]

    assert image_path.startswith("views/test_source_reference_transform/assets/")
    assert not Path(image_path).is_absolute()
    assert (canonical_root.parent / image_path).is_file()
    assert train.iloc[0]["image_path"] == image_path


def test_nested_reference_transform_writes_view_asset_relative_path(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "test_nested_reference_transform"
    config = {
        "name": "test_nested_reference_transform",
        "stage": "rlvr",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "layout", "sources": ["FakeMinerU"]}],
        "target_serialization": {"layout": "mineru_layout_box_v1"},
        "image_policy": {"materialization": {"mode": "nested_reference"}},
        "image_transform": {"layout": {"pad_to_square": True, "resize_to": 128}},
        "reward_profile": {"default": "normalized_levenshtein_v1"},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
    }

    ViewBuilder(canonical_root, view_root).build(config)
    validate_view(view_root, require_images=True)
    train = pd.read_parquet(view_root / "train.parquet")
    image_path = _as_list(train.iloc[0]["images"])[0]["image"]

    assert image_path.startswith("views/test_nested_reference_transform/assets/")
    assert not Path(image_path).is_absolute()
    assert (canonical_root.parent / image_path).is_file()
    assert train.iloc[0]["image_path"] == image_path


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
    image_path = _as_list(train.iloc[0]["images_path"])[0]
    assert image_path.startswith("views/parallel_source_reference/assets/")
    assert (canonical_root.parent / image_path).is_file()
    validate_view(view_root, require_images=True)


def test_parallel_build_skip_failure_reports_worker_failure_and_keeps_same_batch_success(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    bad_record_id = _append_text_record_with_missing_image(canonical_root)
    view_root = tmp_path / "views" / "parallel_skip_failure"

    report = ViewBuilder(canonical_root, view_root).build(
        _fake_view_config(
            canonical_root,
            view_root,
            include=[{"task": "text", "sources": ["FakeMinerU"]}],
        ),
        skip_failure=True,
        schema_sample_size=1,
        num_workers=2,
        worker_batch_size=2,
    )

    assert report.total_records == 1
    train = pd.read_parquet(view_root / "train.parquet")
    assert train["canonical_record_id"].tolist() != [bad_record_id]
    assert train["task"].tolist() == ["text"]
    stats = json.loads((view_root / "stats.json").read_text())
    skipped = stats["skipped_materialization_failures"]
    assert skipped["total"] == 1
    assert skipped["counts"]["by_phase"] == {"materialize": 1}
    assert skipped["examples"][0]["record_id"] == bad_record_id
    validate_view(view_root)


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


def test_view_builder_drops_empty_label_records(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    formula_path = canonical_root / "records/formula/source=FakeMinerU/part-00000.parquet"
    formulas = pd.read_parquet(formula_path)
    formulas.at[0, "target"] = {"latex": ""}
    formulas.to_parquet(formula_path, index=False)
    table_path = canonical_root / "records/table/source=FakeMinerU/part-00000.parquet"
    tables = pd.read_parquet(table_path)
    tables.at[0, "target"] = {"html": "<table></table>"}
    tables.to_parquet(table_path, index=False)
    view_root = tmp_path / "views" / "empty_label_filter"
    config = {
        "name": "empty_label_filter",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "formula", "sources": ["FakeMinerU"]},
            {"task": "table", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "target_serialization": {"formula": "latex_plain_v1", "table": "enhanced_otsl_v1", "text": "plain_text_v1"},
        "image_policy": {"materialization": {"mode": "embedded"}},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "shard_policy": {"rows_per_shard": 10},
    }

    report = ViewBuilder(canonical_root, view_root).build(config)

    assert report.total_records == 1
    train = pd.read_parquet(view_root / "train" / "part-00000.parquet")
    assert train["task"].tolist() == ["text"]
    stats = json.loads((view_root / "stats.json").read_text())
    assert stats["filter_counts"]["empty_label"] == 2


def test_sft_view_builder_drops_labels_with_reserved_media_tokens(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    text_path = canonical_root / "records/text/source=FakeMinerU/part-00000.parquet"
    texts = pd.read_parquet(text_path)
    texts.at[0, "target"] = {"text": "<video>"}
    texts.to_parquet(text_path, index=False)
    view_root = tmp_path / "views" / "reserved_media_token_filter"
    config = {
        "name": "reserved_media_token_filter",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [
            {"task": "formula", "sources": ["FakeMinerU"]},
            {"task": "text", "sources": ["FakeMinerU"]},
        ],
        "target_serialization": {"formula": "latex_plain_v1", "text": "plain_text_v1"},
        "image_policy": {"materialization": {"mode": "embedded"}},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "shard_policy": {"rows_per_shard": 10},
    }

    report = ViewBuilder(canonical_root, view_root).build(config)

    assert report.total_records == 1
    train = pd.read_parquet(view_root / "train" / "part-00000.parquet")
    assert train["task"].tolist() == ["formula"]
    stats = json.loads((view_root / "stats.json").read_text())
    assert stats["filter_counts"]["reserved_media_token_label"] == 1


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
    image_path = _as_list(train.iloc[0]["images_path"])[0]
    assert not Path(image_path).is_absolute()
    assert (canonical_root.parent / image_path).is_file()
    assert train.iloc[0]["image_path"] == image_path
    assert not (view_root / "assets").exists()
    assert not (view_root / "train" / "assets").exists()
    validate_view(view_root, require_images=True)


def test_sharded_nested_reference_view_uses_nested_images_without_view_assets(tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    view_root = tmp_path / "views" / "mineru25_sft_nested_reference_sharded"
    config = {
        "name": "mineru25_sft_nested_reference_sharded",
        "stage": "sft",
        "model_family": "mineru2.5",
        "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
        "include": [{"task": "text", "sources": ["FakeMinerU"]}],
        "target_serialization": {"text": "plain_text_v1"},
        "image_policy": {"materialization": {"mode": "nested_reference"}},
        "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "shard_policy": {"rows_per_shard": 1},
    }

    ViewBuilder(canonical_root, view_root).build(config)

    shard_path = view_root / "train" / "part-00000.parquet"
    train = pd.read_parquet(shard_path)
    image = _as_list(train.iloc[0]["images"])[0]["image"]
    assert not Path(image).is_absolute()
    assert (canonical_root.parent / image).is_file()
    assert train.iloc[0]["image_path"] == image
    assert not _as_list(train.iloc[0]["images_bytes"])
    assert not _as_list(train.iloc[0]["images_path"])
    assert not (view_root / "assets").exists()
    assert not (view_root / "train" / "assets").exists()
    validate_view(view_root, require_images=True)


def test_cli_validate_view_resolves_name_from_dataset_root(monkeypatch, tmp_path):
    canonical_root = _export_fake_canonical(tmp_path)
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "mineru25_rlvr"
    monkeypatch.setenv("OCR_DATA_ROOT", str(dataset_root))
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
    monkeypatch.delenv("OCR_DATA_ROOT", raising=False)
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


def test_cli_validate_view_accepts_worker_controls(monkeypatch, tmp_path, capsys):
    canonical_root = _export_fake_canonical(tmp_path)
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "parallel_validate_view"
    monkeypatch.setenv("OCR_DATA_ROOT", str(dataset_root))
    ViewBuilder(canonical_root, view_root).build(
        {
            "name": "parallel_validate_view",
            "stage": "sft",
            "model_family": "mineru2.5",
            "paths": {"canonical_root": str(canonical_root), "view_root": str(view_root)},
            "include": [{"task": "text", "sources": ["FakeMinerU"]}],
            "target_serialization": {"text": "plain_text_v1"},
            "image_policy": {"materialization": {"mode": "embedded"}},
            "split_policy": {"level": "document", "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        }
    )

    docds_main(["validate-view", "parallel_validate_view", "--require-images", "--num-workers", "2", "--worker-batch-size", "1"])

    captured = capsys.readouterr()
    assert "view ok" in captured.out


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
    monkeypatch.setenv("OCR_DATA_ROOT", str(tmp_path))
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


def test_cli_export_source_accepts_parallel_worker_options(monkeypatch, tmp_path, capsys):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    canonical_root = tmp_path / "canonical"
    monkeypatch.setenv("OCR_DATA_ROOT", str(tmp_path))
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_001")
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_002")
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
            "--num-workers",
            "2",
            "--worker-chunksize",
            "1",
            "--max-in-flight",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out)["documents"] == 2
    validate_canonical(canonical_root, source="FakeMinerU")
