from src.data.prompts import PromptConfig, load_prompt_config, resolve_prompt
from src.data.schemas import DetectionRecord, PageRecord, RegionRecord


def test_load_default_profile_returns_config():
    config = load_prompt_config("default")
    assert isinstance(config, PromptConfig)
    assert config.image_placeholder == "<image>"


def test_default_page_markdown_prompt():
    config = load_prompt_config("default")
    record = PageRecord(
        sample_id="p1", image_uri="img.png", width_px=100, height_px=100,
        task_type="page_markdown",
    )
    assert resolve_prompt(config, record) == "<image>\nConvert the page image into markdown."


def test_default_page_generic_prompt():
    config = load_prompt_config("default")
    record = PageRecord(
        sample_id="p1", image_uri="img.png", width_px=100, height_px=100,
        task_type="page_analysis",
    )
    assert resolve_prompt(config, record) == "<image>\nAnalyze the page image and return the requested OCR result."


def test_default_region_table_prompt():
    config = load_prompt_config("default")
    record = RegionRecord(
        sample_id="r1", image_uri="img.png", width_px=100, height_px=100,
        task_type="table_recognition", region_type="table",
        targets={"html": "<table></table>"},
    )
    assert resolve_prompt(config, record) == "<image>\nRecognize the table in the image and return the requested table representation."


def test_default_region_formula_prompt():
    config = load_prompt_config("default")
    record = RegionRecord(
        sample_id="r1", image_uri="img.png", width_px=100, height_px=100,
        task_type="formula_recognition", region_type="formula",
        targets={"latex": "x^2"},
    )
    assert resolve_prompt(config, record) == "<image>\nRecognize the formula in the image and return LaTeX."


def test_default_region_seal_prompt():
    config = load_prompt_config("default")
    record = RegionRecord(
        sample_id="r1", image_uri="img.png", width_px=100, height_px=100,
        task_type="seal_recognition", region_type="seal",
        targets={"text": "stamp"},
    )
    assert resolve_prompt(config, record) == "<image>\nRecognize the seal or stamp content in the image."


def test_default_region_generic_prompt_substitutes_region_type():
    config = load_prompt_config("default")
    record = RegionRecord(
        sample_id="r1", image_uri="img.png", width_px=100, height_px=100,
        task_type="text_recognition", region_type="text",
        targets={"text": "hello"},
    )
    assert resolve_prompt(config, record) == "<image>\nRecognize the text content in the image."


def test_default_detection_prompt():
    config = load_prompt_config("default")
    record = DetectionRecord(
        sample_id="d1", image_uri="img.png", width_px=100, height_px=100,
        task_type="formula_detection",
        instances=[{"category": "formula", "bbox_px": [0, 0, 50, 50]}],
    )
    assert resolve_prompt(config, record) == "<image>\nDetect the requested document objects and return a JSON list of instances."


def test_custom_profile_from_file(tmp_path):
    profile = tmp_path / "custom.yaml"
    profile.write_text(
        'image_placeholder: "<image>"\n'
        'page:\n'
        '  default: "{{ image_placeholder }}\\nCustom page prompt."\n'
        '  rules: []\n'
        'region:\n'
        '  default: "{{ image_placeholder }}\\nCustom region prompt."\n'
        '  rules: []\n'
        'detection:\n'
        '  default: "{{ image_placeholder }}\\nCustom detection prompt."\n'
    )
    config = load_prompt_config(str(profile))
    record = PageRecord(
        sample_id="p1", image_uri="img.png", width_px=100, height_px=100,
        task_type="page_markdown",
    )
    assert resolve_prompt(config, record) == "<image>\nCustom page prompt."


def test_backward_compat_prompt_for_record():
    """Existing _prompt_for_record behavior preserved with no profile arg."""
    from src.data.views import _prompt_for_record

    page_md = PageRecord(
        sample_id="p1", image_uri="img.png", width_px=100, height_px=100,
        task_type="page_markdown",
    )
    assert _prompt_for_record(page_md) == "<image>\nConvert the page image into markdown."

    region_text = RegionRecord(
        sample_id="r1", image_uri="img.png", width_px=100, height_px=100,
        task_type="text_recognition", region_type="text",
        targets={"text": "hello"},
    )
    assert _prompt_for_record(region_text) == "<image>\nRecognize the text content in the image."


def test_load_mineru25_profile():
    config = load_prompt_config("mineru2.5")
    assert isinstance(config, PromptConfig)
    assert config.image_placeholder == "<image>"


def test_mineru25_table_prompt_mentions_otsl():
    config = load_prompt_config("mineru2.5")
    record = RegionRecord(
        sample_id="r1", image_uri="img.png", width_px=100, height_px=100,
        task_type="table_recognition", region_type="table",
        targets={"html": "<table></table>"},
    )
    prompt = resolve_prompt(config, record)
    assert "OTSL" in prompt


def test_mineru25_layout_prompt():
    config = load_prompt_config("mineru2.5")
    record = PageRecord(
        sample_id="p1", image_uri="img.png", width_px=100, height_px=100,
        task_type="page_layout",
    )
    prompt = resolve_prompt(config, record)
    assert prompt == "\nLayout Detection:"


def test_mineru25_format_config_accessible():
    config = load_prompt_config("mineru2.5")
    assert "layout_detection" in config.formats
    assert config.formats["layout_detection"]["tokens"]["box_start"] == "<|box_start|>"
    assert "table_recognition" in config.formats
    assert config.formats["table_recognition"]["format"] == "otsl"


def test_mineru25_sampling_config():
    config = load_prompt_config("mineru2.5")
    assert "layout_detection" in config.sampling
    assert config.sampling["layout_detection"]["temperature"] == 0.0


def test_mineru25_detection_prompt_differs_from_default():
    default_config = load_prompt_config("default")
    mineru_config = load_prompt_config("mineru2.5")
    record = DetectionRecord(
        sample_id="d1", image_uri="img.png", width_px=100, height_px=100,
        task_type="formula_detection",
        instances=[{"category": "formula", "bbox_px": [0, 0, 50, 50]}],
    )
    assert resolve_prompt(default_config, record) != resolve_prompt(mineru_config, record)
