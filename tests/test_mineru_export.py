import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.export_views import export_view, read_source_records
from src.data.mineru_export import (
    MinerUExportOptions,
    export_mineru_dataset,
    load_options_from_profile,
)
from src.data import mineru_export


def _write_fake_mineru_dataset(root: Path, source_root: Path, sample: str = "paper_001") -> Path:
    vlm_dir = root / sample / "vlm"
    images_dir = vlm_dir / "images"
    images_dir.mkdir(parents=True)
    (source_root / f"{sample}.jpg").write_bytes(b"not-a-real-image")
    (images_dir / "table_crop.jpg").write_bytes(b"crop")

    model = [
        [
            {"type": "title", "bbox": [0.1, 0.1, 0.9, 0.2], "angle": 0, "content": "A Title"},
            {"type": "table", "bbox": [0.2, 0.3, 0.8, 0.6], "angle": 0, "content": "<table><tr><td>A</td></tr></table>"},
            {
                "type": "list",
                "bbox": [0.2, 0.62, 0.8, 0.68],
                "angle": 0,
                "content": {"items": [{"content": "first"}, {"content": "second"}]},
            },
            {"type": "image", "bbox": [0.2, 0.7, 0.8, 0.9], "angle": 0, "content": None},
        ]
    ]
    middle = {
        "_backend": "vlm",
        "_version_name": "3.0.9",
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [1000, 2000],
                "para_blocks": [
                    {"index": 0, "type": "title", "bbox": [100, 200, 900, 400]},
                    {"index": 1, "type": "table", "bbox": [200, 600, 800, 1200]},
                    {"index": 2, "type": "list", "bbox": [200, 1240, 800, 1360]},
                    {"index": 3, "type": "image", "bbox": [200, 1400, 800, 1800]},
                ],
            }
        ],
    }
    content_v2 = [
        [
            {
                "type": "table",
                "bbox": [200, 600, 800, 1200],
                "content": {"table_body": [{"type": "text", "content": "A"}]},
            },
            {
                "type": "image",
                "bbox": [200, 1400, 800, 1800],
                "content": {"image_source": {"path": "images/table_crop.jpg"}},
            },
        ]
    ]
    (vlm_dir / f"{sample}_model.json").write_text(json.dumps(model))
    (vlm_dir / f"{sample}_middle.json").write_text(json.dumps(middle))
    (vlm_dir / f"{sample}_content_list_v2.json").write_text(json.dumps(content_v2))
    (vlm_dir / f"{sample}.md").write_text("# A Title\n\n<table><tr><td>A</td></tr></table>\n")
    return vlm_dir


def test_export_mineru_dataset_writes_canonical_parquets(tmp_path):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    output_root = tmp_path / "canonical"
    source_root.mkdir()
    _write_fake_mineru_dataset(mineru_root, source_root)

    report = export_mineru_dataset(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            output_root=output_root,
            dataset_name="FakeMinerU",
            shard_size=1,
            allow_unreadable_images=True,
        )
    )

    assert report.scanned_samples == 1
    assert report.page_records == 1
    assert report.detection_records == 1
    assert report.region_records == 4

    page_df = pd.concat(pd.read_parquet(p) for p in sorted((output_root / "page_records/FakeMinerU").glob("*.parquet")))
    detection_df = pd.concat(
        pd.read_parquet(p) for p in sorted((output_root / "detection_records/FakeMinerU").glob("*.parquet"))
    )
    region_df = pd.concat(
        pd.read_parquet(p) for p in sorted((output_root / "region_records/FakeMinerU").glob("*.parquet"))
    )

    assert page_df.iloc[0]["record_type"] == "page"
    assert page_df.iloc[0]["merged_markdown"].startswith("# A Title")
    assert page_df.iloc[0]["width_px"] == 1000
    assert detection_df.iloc[0]["instances"][1]["category"] == "table"
    assert list(detection_df.iloc[0]["instances"][1]["bbox_px"]) == [200, 600, 800, 1200]
    assert set(region_df["region_type"]) == {"title", "table", "list", "image"}
    assert all(isinstance(value, str) for targets in region_df["targets"] for value in targets.values())

    manifest = json.loads((output_root / "_manifests/FakeMinerU_mineru_export_report.json").read_text())
    assert manifest["dataset_name"] == "FakeMinerU"
    assert manifest["page_records"] == 1

    sft_output = tmp_path / "views" / "page_sft.parquet"
    page_records = read_source_records(sorted((output_root / "page_records/FakeMinerU").glob("*.parquet")))
    exported = export_view(page_records, "sft", sft_output)
    sft_df = pd.read_parquet(sft_output)

    assert exported == 1
    assert sft_df.iloc[0]["messages"][1]["content"].startswith("# A Title")


def test_export_mineru_dataset_strict_mode_fails_on_missing_image(tmp_path):
    mineru_root = tmp_path / "mineru"
    output_root = tmp_path / "canonical"
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_002")
    (source_root / "paper_002.jpg").unlink()

    with pytest.raises(FileNotFoundError):
        export_mineru_dataset(
            MinerUExportOptions(
                mineru_root=mineru_root,
                source_image_root=source_root,
                output_root=output_root,
                dataset_name="StrictFakeMinerU",
                allow_unreadable_images=False,
            )
        )


def test_permission_denied_image_resolution_is_unreadable_in_permissive_mode(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    options = MinerUExportOptions(
        mineru_root=tmp_path / "mineru",
        source_image_root=source_root,
        output_root=tmp_path / "canonical",
        dataset_name="PermissionFakeMinerU",
        allow_unreadable_images=True,
    )
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path.name == "paper_003.jpg":
            raise PermissionError("denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    uri, accessible, unreadable = mineru_export._resolve_source_image("paper_003", options)

    assert uri.endswith("paper_003.jpg")
    assert accessible is False
    assert unreadable is True


def test_load_options_from_profile(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "mineru_root": str(tmp_path / "mineru"),
                "source_image_root": str(tmp_path / "source"),
                "output_root": str(tmp_path / "canonical"),
                "dataset_name": "ProfileDataset",
                "mineru_subdir": "hybrid_auto",
                "records": ["page", "region"],
                "shard_size": 7,
                "max_samples": 3,
                "allow_unreadable_images": True,
            }
        )
    )

    options = load_options_from_profile(profile)

    assert options.dataset_name == "ProfileDataset"
    assert options.mineru_subdir == "hybrid_auto"
    assert options.records == ("page", "region")
    assert options.shard_size == 7
    assert options.max_samples == 3


def test_export_mineru_dataset_respects_max_samples(tmp_path):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    output_root = tmp_path / "canonical"
    source_root.mkdir()
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_a")
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_b")

    report = export_mineru_dataset(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            output_root=output_root,
            dataset_name="LimitedFakeMinerU",
            max_samples=1,
            allow_unreadable_images=True,
        )
    )

    assert report.scanned_samples == 1
    assert report.page_records == 1


def test_allow_unreadable_images_does_not_hide_parse_errors(tmp_path):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    output_root = tmp_path / "canonical"
    source_root.mkdir()
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_bad")
    (mineru_root / "paper_bad/vlm/paper_bad_model.json").write_text(json.dumps({"not": "a page list"}))

    with pytest.raises(ValueError):
        export_mineru_dataset(
            MinerUExportOptions(
                mineru_root=mineru_root,
                source_image_root=source_root,
                output_root=output_root,
                dataset_name="BadFakeMinerU",
                allow_unreadable_images=True,
            )
        )


def test_skip_errors_records_malformed_samples(tmp_path):
    mineru_root = tmp_path / "mineru"
    source_root = tmp_path / "source"
    output_root = tmp_path / "canonical"
    source_root.mkdir()
    _write_fake_mineru_dataset(mineru_root, source_root, sample="paper_bad")
    (mineru_root / "paper_bad/vlm/paper_bad_model.json").write_text(json.dumps({"not": "a page list"}))

    report = export_mineru_dataset(
        MinerUExportOptions(
            mineru_root=mineru_root,
            source_image_root=source_root,
            output_root=output_root,
            dataset_name="SkipBadFakeMinerU",
            skip_errors=True,
        )
    )

    assert report.scanned_samples == 1
    assert report.skipped_samples == 1
    assert report.errors[0]["sample_id"] == "paper_bad"
