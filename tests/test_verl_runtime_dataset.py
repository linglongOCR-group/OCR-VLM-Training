from pathlib import Path

import pytest

from tools.data_management.runtime.image_columns import resolve_runtime_images


def test_runtime_images_prefers_embedded_bytes(tmp_path):
    image_file = tmp_path / "ignored.png"
    image_file.write_bytes(b"not-used")

    images = resolve_runtime_images(
        {"images_bytes": [b"abc"], "images_path": [image_file.name]},
        image_assets_dir=tmp_path,
    )

    assert images == [{"bytes": b"abc"}]


def test_runtime_images_resolves_filename_paths_against_assets_dir(tmp_path):
    image_file = tmp_path / "asset.png"
    image_file.write_bytes(b"png")

    images = resolve_runtime_images({"images_path": ["asset.png"]}, image_assets_dir=tmp_path)

    assert images == [{"image": str(image_file)}]


def test_runtime_images_rejects_path_mode_without_assets_dir():
    with pytest.raises(ValueError, match="image_assets_dir"):
        resolve_runtime_images({"images_path": ["asset.png"]})


def test_runtime_images_rejects_absolute_or_nested_paths(tmp_path):
    with pytest.raises(ValueError, match="filename"):
        resolve_runtime_images({"images_path": ["nested/asset.png"]}, image_assets_dir=tmp_path)

    with pytest.raises(ValueError, match="filename"):
        resolve_runtime_images({"images_path": [str(tmp_path / "asset.png")]}, image_assets_dir=tmp_path)
