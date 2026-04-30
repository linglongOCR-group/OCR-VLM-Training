from __future__ import annotations

import copy
from pathlib import Path
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
        working = _example_with_runtime_images(
            example=example,
            messages_key=self.prompt_key,
            image_key=self.image_key,
            image_assets_dir=self.image_assets_dir,
        )
        messages = super()._build_messages(working)
        example[self.image_key] = resolve_runtime_images(example, self.image_assets_dir)
        return messages


class OcrMultiTurnSFTDataset(MultiTurnSFTDataset):
    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        self.image_assets_dir = config.get("image_assets_dir", None) if config is not None else None
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict[str, Any]):
        working = _example_with_runtime_images(
            example=example,
            messages_key=self.messages_key,
            image_key=self.image_key,
            image_assets_dir=self.image_assets_dir,
        )
        return super()._build_messages(working)


def _example_with_runtime_images(
    *,
    example: dict[str, Any],
    messages_key: str,
    image_key: str,
    image_assets_dir: str | Path | None,
) -> dict[str, Any]:
    working = dict(example)
    if messages_key in working:
        working[messages_key] = copy.deepcopy(working[messages_key])
    working[image_key] = resolve_runtime_images(example, image_assets_dir)
    return working
