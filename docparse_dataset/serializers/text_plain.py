from __future__ import annotations

from docparse_dataset.serializers.base import TargetSerializer


class PlainTextSerializer(TargetSerializer):
    """Converts canonical text target into plain text string."""

    name = "plain_text_v1"
    version = "1.0.0"
    task = "text"

    def serialize(self, canonical_record: dict, context: dict | None = None) -> str:
        target = canonical_record.get("target", {})
        return target.get("text", "")

    def validate(self, canonical_record: dict) -> None:
        target = canonical_record.get("target", {})
        if not isinstance(target, dict):
            raise ValueError("text target must be a dict")
