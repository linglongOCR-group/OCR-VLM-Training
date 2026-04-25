from __future__ import annotations

from docparse_dataset.serializers.base import TargetSerializer


class LatexSerializer(TargetSerializer):
    """Converts canonical formula target into LaTeX string."""

    name = "latex_plain_v1"
    version = "1.0.0"
    task = "formula"

    def serialize(self, canonical_record: dict, context: dict | None = None) -> str:
        target = canonical_record.get("target", {})
        return target.get("latex", "")

    def validate(self, canonical_record: dict) -> None:
        target = canonical_record.get("target", {})
        if not isinstance(target, dict):
            raise ValueError("formula target must be a dict")
        if "latex" not in target:
            raise ValueError("formula target must contain 'latex'")
