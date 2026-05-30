from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tools.data_management.runtime.image_columns import resolve_runtime_images
from tools.data_management.runtime.verl_multimodal_dataset import OcrMultiTurnSFTDataset, OcrRLHFDataset


def test_runtime_images_prefers_embedded_bytes(tmp_path):
    image_file = tmp_path / "ignored.png"
    image_file.write_bytes(b"not-used")

    images = resolve_runtime_images(
        {"images_bytes": [b"abc"], "images_path": [image_file.name]},
        image_assets_dir=tmp_path,
    )

    assert images == [{"bytes": b"abc"}]


def test_runtime_images_resolves_relative_paths_against_data_root(tmp_path, monkeypatch):
    image_file = tmp_path / "canonical" / "assets" / "files" / "asset.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(b"png")
    monkeypatch.setenv("OCR_DATA_ROOT", str(tmp_path))

    images = resolve_runtime_images({"images_path": ["canonical/assets/files/asset.png"]})

    assert images == [{"image": str(image_file)}]


def test_runtime_images_resolves_nested_reference_relative_paths(tmp_path, monkeypatch):
    image_file = tmp_path / "views" / "view" / "assets" / "asset.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(b"png")
    monkeypatch.setenv("OCR_DATA_ROOT", str(tmp_path))

    images = resolve_runtime_images({"images": [{"image": "views/view/assets/asset.png"}]})

    assert images == [{"image": str(image_file)}]


def test_runtime_images_rejects_missing_reference_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.png"):
        resolve_runtime_images({"images_path": ["missing.png"]}, data_root=tmp_path)


def test_runtime_images_rejects_path_mode_without_data_root(monkeypatch):
    monkeypatch.delenv("OCR_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="OCR_DATA_ROOT"):
        resolve_runtime_images({"images_path": ["asset.png"]})


def test_runtime_images_rejects_absolute_or_traversal_paths(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        resolve_runtime_images({"images_path": [str(tmp_path / "asset.png")]}, data_root=tmp_path)

    with pytest.raises(ValueError, match="traversal"):
        resolve_runtime_images({"images_path": ["../asset.png"]}, data_root=tmp_path)


def test_ocr_rlhf_dataset_restores_fresh_runtime_images_after_embedded_build(tmp_path):
    dataset = object.__new__(OcrRLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "runtime_images"
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.data_root = tmp_path

    image = _png_bytes()
    row = {
        "prompt": [{"role": "user", "content": "Read this <image>."}],
        "images_bytes": [image],
    }

    messages = dataset._build_messages(row)

    assert row["runtime_images"] == [{"bytes": image}]
    assert "image" not in row["runtime_images"][0]
    assert row["prompt"][0]["content"] == "Read this <image>."
    assert messages[0]["content"][1]["type"] == "image"

    messages = dataset._build_messages(row)
    assert messages[0]["content"][1]["type"] == "image"


def test_ocr_rlhf_dataset_resolves_reference_assets_for_verl_filter(tmp_path):
    image_file = tmp_path / "canonical" / "assets" / "files" / "asset.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(_png_bytes())

    dataset = object.__new__(OcrRLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "runtime_images"
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.data_root = tmp_path
    row = {
        "prompt": [{"role": "user", "content": "Read this <image>."}],
        "images_path": ["canonical/assets/files/asset.png"],
    }

    messages = dataset._build_messages(row)

    assert row["runtime_images"] == [{"image": str(image_file)}]
    assert row["prompt"][0]["content"] == "Read this <image>."
    assert messages[0]["content"][1] == {"type": "image", "image": str(image_file)}


def test_ocr_sft_dataset_builds_messages_without_mutating_source_columns(tmp_path):
    dataset = object.__new__(OcrMultiTurnSFTDataset)
    dataset.messages_key = "messages"
    dataset.image_key = "runtime_images"
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.image_patch_size = 14
    dataset.data_root = tmp_path

    image = _png_bytes()
    row = {
        "messages": [
            {"role": "user", "content": "Read this <image>."},
            {"role": "assistant", "content": "done"},
        ],
        "images_bytes": [image],
    }

    messages = dataset._build_messages(row)

    assert "runtime_images" not in row
    assert row["messages"][0]["content"] == "Read this <image>."
    assert messages[0]["content"][1]["type"] == "image"


def test_ocr_sft_dataset_preserves_literal_media_tokens_in_assistant_text(tmp_path):
    dataset = object.__new__(OcrMultiTurnSFTDataset)
    dataset.messages_key = "messages"
    dataset.image_key = "runtime_images"
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.image_patch_size = 14
    dataset.data_root = tmp_path

    image = _png_bytes()
    row = {
        "messages": [
            {"role": "user", "content": "Read this <image>."},
            {"role": "assistant", "content": "<video>"},
        ],
        "images_bytes": [image],
    }

    messages = dataset._build_messages(row)

    assert messages[0]["content"][1]["type"] == "image"
    assert messages[1]["content"] == [{"type": "text", "text": "<video>"}]


def test_ocr_sft_dataset_does_not_rewrite_existing_sentinel_text(tmp_path):
    dataset = object.__new__(OcrMultiTurnSFTDataset)
    dataset.messages_key = "messages"
    dataset.image_key = "runtime_images"
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.image_patch_size = 14
    dataset.data_root = tmp_path

    image = _png_bytes()
    row = {
        "messages": [
            {"role": "user", "content": "Read this <image>."},
            {"role": "assistant", "content": "__ocr_literal_video_token__"},
        ],
        "images_bytes": [image],
    }

    messages = dataset._build_messages(row)

    assert messages[1]["content"] == [{"type": "text", "text": "__ocr_literal_video_token__"}]


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()
