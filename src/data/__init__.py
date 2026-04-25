"""Data source schemas and training-view exporters.

Backward-compatible re-exports from docparse_dataset.
"""

from docparse_dataset.canonical.schemas import (
    DetectionRecord,
    PageRecord,
    RegionRecord,
    validate_record,
)
from docparse_dataset.views.builder import (
    VIEW_EXPORTERS,
    export_view,
    export_view_sharded,
    expand_input_paths,
    load_prompt_config,
    read_source_records,
    resolve_prompt,
    to_grpo_row,
    to_layout_row,
    to_sft_row,
    get_default_config,
)


class PromptConfig:
    """Backward-compatible PromptConfig class."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


__all__ = [
    "DetectionRecord",
    "PageRecord",
    "PromptConfig",
    "RegionRecord",
    "VIEW_EXPORTERS",
    "export_view",
    "export_view_sharded",
    "expand_input_paths",
    "get_default_config",
    "load_prompt_config",
    "read_source_records",
    "resolve_prompt",
    "to_grpo_row",
    "to_layout_row",
    "to_sft_row",
    "validate_record",
]
