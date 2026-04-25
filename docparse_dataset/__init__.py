"""docparse_dataset — Source → Canonical → View dataset processing for OCR VLM training."""

from docparse_dataset.sources.adapters.base import SourceAdapter
from docparse_dataset.serializers.base import TargetSerializer
from docparse_dataset.rewards.base import RewardAdapter, RewardResult
from docparse_dataset.canonical.schemas import (
    BaseRecord,
    DetectionRecord,
    DocumentRecord,
    PageRecord,
    RegionRecord,
    validate_record,
)
from docparse_dataset.views.builder import ViewBuilder

__all__ = [
    "BaseRecord",
    "DetectionRecord",
    "DocumentRecord",
    "PageRecord",
    "RegionRecord",
    "RewardAdapter",
    "RewardResult",
    "SourceAdapter",
    "TargetSerializer",
    "ViewBuilder",
    "validate_record",
]
