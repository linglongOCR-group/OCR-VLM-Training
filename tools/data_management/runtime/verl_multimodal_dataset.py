from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.rl_dataset import RLHFDataset

from tools.data_management.runtime.image_columns import resolve_runtime_images


_LITERAL_MEDIA_TOKENS = ("<image>", "<video>")


class OcrRLHFDataset(RLHFDataset):
    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        self.data_root = config.get("data_root", None) if config is not None else None
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict[str, Any]):
        working = _example_with_runtime_images(
            example=example,
            messages_key=self.prompt_key,
            image_key=self.image_key,
            data_root=self.data_root,
        )
        messages = super()._build_messages(working)
        example[self.image_key] = resolve_runtime_images(example, data_root=self.data_root)
        return messages


class OcrMultiTurnSFTDataset(MultiTurnSFTDataset):
    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        self.data_root = config.get("data_root", None) if config is not None else None
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict[str, Any]):
        sentinels = _literal_media_token_sentinels()
        working = _example_with_runtime_images(
            example=example,
            messages_key=self.messages_key,
            image_key=self.image_key,
            data_root=self.data_root,
            escape_non_user_media_tokens=True,
            media_token_sentinels=sentinels,
        )
        messages = super()._build_messages(working)
        _restore_literal_media_tokens(messages, sentinels)
        return messages


def _example_with_runtime_images(
    *,
    example: dict[str, Any],
    messages_key: str,
    image_key: str,
    data_root: str | Path | None,
    escape_non_user_media_tokens: bool = False,
    media_token_sentinels: dict[str, str] | None = None,
) -> dict[str, Any]:
    working = dict(example)
    if messages_key in working:
        working[messages_key] = copy.deepcopy(working[messages_key])
        if escape_non_user_media_tokens:
            _escape_non_user_media_tokens(working[messages_key], media_token_sentinels or _literal_media_token_sentinels())
    working[image_key] = resolve_runtime_images(example, data_root=data_root)
    return working


def _literal_media_token_sentinels() -> dict[str, str]:
    nonce = uuid.uuid4().hex
    return {token: f"__ocr_literal_media_token_{index}_{nonce}__" for index, token in enumerate(_LITERAL_MEDIA_TOKENS)}


def _escape_non_user_media_tokens(messages: Any, sentinels: dict[str, str]) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            for token, sentinel in sentinels.items():
                content = content.replace(token, sentinel)
            message["content"] = content


def _restore_literal_media_tokens(messages: Any, sentinels: dict[str, str]) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _restore_literal_media_token_text(content, sentinels)
        elif isinstance(content, list):
            for segment in content:
                if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                    segment["text"] = _restore_literal_media_token_text(segment["text"], sentinels)


def _restore_literal_media_token_text(text: str, sentinels: dict[str, str]) -> str:
    for token, sentinel in sentinels.items():
        text = text.replace(sentinel, token)
    return text
