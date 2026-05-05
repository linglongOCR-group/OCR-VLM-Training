from pathlib import Path

from PIL import Image

from tools.data_management.sources.adapters.mineru import MinerUExportOptions, _resolve_source_image


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)


def test_resolve_source_image_uses_unique_prefixed_image_for_truncated_sample_id(tmp_path):
    image_root = tmp_path / "images"
    _write_image(image_root / "long_document_name_page_001.png")
    options = MinerUExportOptions(
        mineru_root=tmp_path / "annotations",
        source_image_root=image_root,
        dataset_name="FakeMinerU",
    )

    image_path, accessible, unreadable = _resolve_source_image("long_document_name", options)

    assert image_path == str(image_root / "long_document_name_page_001.png")
    assert accessible is True
    assert unreadable is False


def test_resolve_source_image_uses_exact_literal_filename_before_prefix_fallback(tmp_path):
    image_root = tmp_path / "images"
    _write_image(image_root / "book_en_[HTML5 Canvas]_page_208.png")
    options = MinerUExportOptions(
        mineru_root=tmp_path / "annotations",
        source_image_root=image_root,
        dataset_name="FakeMinerU",
    )

    image_path, accessible, unreadable = _resolve_source_image("book_en_[HTML5 Canvas]_page_208", options)

    assert image_path == str(image_root / "book_en_[HTML5 Canvas]_page_208.png")
    assert accessible is True
    assert unreadable is False


def test_resolve_source_image_uses_collision_suffix_as_one_based_prefixed_index(tmp_path):
    image_root = tmp_path / "images"
    _write_image(image_root / "long_document_name_page_010.png")
    _write_image(image_root / "long_document_name_page_020.png")
    options = MinerUExportOptions(
        mineru_root=tmp_path / "annotations",
        source_image_root=image_root,
        dataset_name="FakeMinerU",
    )

    first_path, _, _ = _resolve_source_image("long_document_name", options)
    second_path, _, _ = _resolve_source_image("long_document_name_2", options)

    assert first_path == str(image_root / "long_document_name_page_010.png")
    assert second_path == str(image_root / "long_document_name_page_020.png")
