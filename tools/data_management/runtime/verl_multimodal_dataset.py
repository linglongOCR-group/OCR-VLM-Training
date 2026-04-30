from __future__ import annotations

from typing import Any

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.rl_dataset import RLHFDataset

from tools.data_management.runtime.image_columns import resolve_runtime_images


class OcrRLHFDataset(RLHFDataset):
    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        self.image_assets_dir = config.get("image_assets_dir", None) if config is not None else None
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict[str, Any]):
        images = resolve_runtime_images(example, self.image_assets_dir)
        example[self.image_key] = images
        messages = super()._build_messages(example)
        example[self.image_key] = images
        return messages


class OcrMultiTurnSFTDataset(MultiTurnSFTDataset):
    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        self.image_assets_dir = config.get("image_assets_dir", None) if config is not None else None
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict[str, Any]):
        example[self.image_key] = resolve_runtime_images(example, self.image_assets_dir)
        return super()._build_messages(example)
