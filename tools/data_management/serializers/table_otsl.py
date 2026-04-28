from __future__ import annotations

from typing import Any

from tools.data_management.otsl import html_to_otsl
from tools.data_management.serializers.base import TargetSerializer


class EnhancedOTSLSerializer(TargetSerializer):
    name = "enhanced_otsl_v1"
    version = "1.0.0"
    task = "table"

    def serialize(self, canonical_record: dict[str, Any], context: dict[str, Any]) -> str:
        self.validate(canonical_record)
        target = canonical_record.get("target") or {}
        if target.get("enhanced_otsl"):
            return str(target["enhanced_otsl"])
        if target.get("otsl"):
            return str(target["otsl"])
        if target.get("html"):
            return html_to_otsl(str(target["html"]))
        raise ValueError("table record requires enhanced_otsl, otsl, or html target")
