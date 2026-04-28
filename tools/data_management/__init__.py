"""Spec-aligned dataset processing module."""

from tools.data_management.canonical import CanonicalReader, CanonicalWriter, validate_canonical
from tools.data_management.prompts import PromptConfig, load_prompt_config, resolve_prompt
from tools.data_management.schemas import (
    AssetRecord,
    CanonicalDocument,
    CanonicalPage,
    CanonicalRegion,
    CanonicalTaskRecord,
    RewardResult,
    ViewRecord,
)
from tools.data_management.views import ViewBuilder, reward_smoke_test, score_predictions, validate_view

__all__ = [
    "AssetRecord",
    "CanonicalDocument",
    "CanonicalPage",
    "CanonicalReader",
    "CanonicalRegion",
    "CanonicalTaskRecord",
    "CanonicalWriter",
    "PromptConfig",
    "RewardResult",
    "ViewBuilder",
    "ViewRecord",
    "load_prompt_config",
    "resolve_prompt",
    "reward_smoke_test",
    "score_predictions",
    "validate_canonical",
    "validate_view",
]
