import json

import pandas as pd

from src.data.export_views import export_view_sharded, expand_input_paths
from src.data.schemas import DetectionRecord, PageRecord, RegionRecord, validate_record
from src.data.validate_grpo_view import validate_grpo_view
from src.data.views import to_grpo_row, to_layout_row, to_sft_row


def test_region_record_does_not_require_page_fields():
    record = RegionRecord(
        sample_id="tbl_001",
        image_uri="images/tbl_001.png",
        width_px=912,
        height_px=436,
        task_type="table_recognition",
        region_type="table",
        targets={"html": "<table><tr><td>A</td></tr></table>"},
    )

    row = to_grpo_row(record)

    assert row["data_source"] == "ocr_vlm:region:table_recognition"
    assert row["prompt"][0]["role"] == "user"
    assert row["prompt"][0]["content"].startswith("<image>\n")
    assert row["images"] == [{"image": "images/tbl_001.png"}]
    assert row["reward_model"]["ground_truth"] == "<table><tr><td>A</td></tr></table>"
    assert row["extra_info"]["record_type"] == "region"
    assert row["extra_info"]["source_page_ref"] is None


def test_page_record_exports_sft_messages_with_page_target():
    record = PageRecord(
        sample_id="docA_p003",
        image_uri="images/docA_p003.png",
        width_px=2480,
        height_px=3508,
        task_type="page_markdown",
        blocks=[{"block_id": "b1", "category": "text", "bbox_px": [0, 0, 10, 10]}],
        reading_order=["b1"],
        merged_markdown="# Invoice\n",
    )

    row = to_sft_row(record)

    assert row["messages"] == [
        {"role": "user", "content": "<image>\nConvert the page image into markdown."},
        {"role": "assistant", "content": "# Invoice\n"},
    ]
    assert row["images"] == [{"image": "images/docA_p003.png"}]
    assert row["extra_info"]["record_type"] == "page"


def test_detection_record_exports_layout_instances_without_markdown():
    record = DetectionRecord(
        sample_id="det_formula_001",
        image_uri="images/det_formula_001.png",
        width_px=1024,
        height_px=1024,
        task_type="formula_detection",
        instances=[
            {
                "instance_id": "i1",
                "category": "formula",
                "bbox_px": [100, 220, 640, 330],
            }
        ],
    )

    row = to_layout_row(record)

    assert row["task"] == "layout_or_detection"
    assert row["extra_info"]["record_type"] == "detection"
    assert row["instances"][0]["category"] == "formula"
    assert "merged_markdown" not in row


def test_validate_record_round_trips_dicts_from_jsonl():
    raw = {
        "record_type": "region",
        "sample_id": "text_001",
        "image_uri": "images/text_001.png",
        "width_px": 300,
        "height_px": 64,
        "task_type": "text_recognition",
        "region_type": "text",
        "targets": {"text": "hello"},
    }

    record = validate_record(raw)

    assert isinstance(record, RegionRecord)
    assert to_grpo_row(record)["reward_model"]["ground_truth"] == "hello"


def test_view_rows_are_parquet_serializable(tmp_path):
    record = RegionRecord(
        sample_id="formula_001",
        image_uri="images/formula_001.png",
        width_px=512,
        height_px=128,
        task_type="formula_recognition",
        region_type="formula",
        targets={"latex": "x^2"},
    )
    output = tmp_path / "view.parquet"

    pd.DataFrame([to_grpo_row(record)]).to_parquet(output)
    loaded = pd.read_parquet(output)

    assert json.loads(json.dumps(loaded.iloc[0]["prompt"].tolist()))[0]["role"] == "user"


def test_validate_grpo_view_rejects_missing_image_placeholder(tmp_path):
    output = tmp_path / "bad_grpo.parquet"
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Recognize the text."}],
                "images": [{"image": "images/text.png"}],
                "extra_info": {"sample_id": "text_001"},
            }
        ]
    ).to_parquet(output, index=False)

    try:
        validate_grpo_view([output])
    except ValueError as exc:
        assert "text_001" in str(exc)
        assert "Re-export the GRPO view" in str(exc)
    else:
        raise AssertionError("missing <image> placeholder should fail validation")


def test_expand_input_paths_accepts_directories(tmp_path):
    input_dir = tmp_path / "records"
    input_dir.mkdir()
    (input_dir / "a.parquet").write_bytes(b"x")
    (input_dir / "b.jsonl").write_text("{}\n")

    paths = expand_input_paths([input_dir])

    assert paths == [input_dir / "a.parquet", input_dir / "b.jsonl"]


def test_export_view_sharded_streams_multiple_parquet_inputs(tmp_path):
    input_dir = tmp_path / "records"
    input_dir.mkdir()
    rows = [
        RegionRecord(
            sample_id=f"formula_{idx:03d}",
            image_uri=f"images/formula_{idx:03d}.png",
            width_px=512,
            height_px=128,
            task_type="formula_recognition",
            source_type="fixture",
            provenance={"dataset_name": "fixture"},
            metadata={"split": "train"},
            region_type="formula",
            targets={"latex": f"x^{idx}"},
        ).to_dict()
        for idx in range(3)
    ]
    pd.DataFrame(rows[:2]).to_parquet(input_dir / "part-00000.parquet", index=False)
    pd.DataFrame(rows[2:]).to_parquet(input_dir / "part-00001.parquet", index=False)

    output_dir = tmp_path / "views"
    exported = export_view_sharded([input_dir], "grpo", output_dir, shard_size=2)

    shards = sorted(output_dir.glob("part-*.parquet"))
    loaded = pd.concat(pd.read_parquet(path) for path in shards)

    assert exported == 3
    assert [path.name for path in shards] == ["part-00000.parquet", "part-00001.parquet"]
    assert list(loaded["data_source"]) == [
        "ocr_vlm:region:formula_recognition",
        "ocr_vlm:region:formula_recognition",
        "ocr_vlm:region:formula_recognition",
    ]


def test_export_view_sharded_honors_max_records(tmp_path):
    input_dir = tmp_path / "records"
    input_dir.mkdir()
    rows = [
        RegionRecord(
            sample_id=f"text_{idx:03d}",
            image_uri=f"images/text_{idx:03d}.png",
            width_px=512,
            height_px=128,
            task_type="text_recognition",
            source_type="fixture",
            provenance={"dataset_name": "fixture"},
            metadata={"split": "train"},
            region_type="text",
            targets={"text": f"value {idx}"},
        ).to_dict()
        for idx in range(5)
    ]
    pd.DataFrame(rows).to_parquet(input_dir / "part-00000.parquet", index=False)

    output_dir = tmp_path / "views"
    exported = export_view_sharded([input_dir], "grpo", output_dir, shard_size=2, max_records=3)

    shards = sorted(output_dir.glob("part-*.parquet"))
    loaded = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)

    assert exported == 3
    assert [path.name for path in shards] == ["part-00000.parquet", "part-00001.parquet"]
    assert list(loaded["extra_info"].map(lambda item: item["sample_id"])) == ["text_000", "text_001", "text_002"]
