from __future__ import annotations

from typing import Any

from tools.data_management.serializers.base import TargetSerializer


class TableTextSerializer(TargetSerializer):
    name = "table_text_v1"
    version = "1.0.0"
    task = "table"

    def serialize(self, canonical_record: dict[str, Any], context: dict[str, Any]) -> str:
        self.validate(canonical_record)
        target = canonical_record.get("target") or {}
        if "text" not in target:
            raise ValueError("table text record requires text target")
        return str(target["text"])
