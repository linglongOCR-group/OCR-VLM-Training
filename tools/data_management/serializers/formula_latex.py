from __future__ import annotations

from typing import Any

from tools.data_management.serializers.base import TargetSerializer


class LatexPlainSerializer(TargetSerializer):
    name = "latex_plain_v1"
    version = "1.0.0"
    task = "formula"

    def serialize(self, canonical_record: dict[str, Any], context: dict[str, Any]) -> str:
        self.validate(canonical_record)
        target = canonical_record.get("target") or {}
        if "latex" not in target:
            raise ValueError("formula record requires latex target")
        return str(target["latex"])
