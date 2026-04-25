from __future__ import annotations

import json
from typing import Any

from docparse_dataset.serializers.base import TargetSerializer


class MinerULayoutSerializer(TargetSerializer):
    """Converts canonical layout target into MinerU box string format."""

    name = "mineru_layout_box_v1"
    version = "1.0.0"
    task = "layout"

    def serialize(self, canonical_record: dict, context: dict | None = None) -> str:
        target = canonical_record.get("target", {})
        elements = target.get("elements", [])
        context = context or {}
        image_transform = context.get("image_transform", {})
        scale = image_transform.get("scale", 1.0)
        pad_left = image_transform.get("pad_left", 0)
        pad_top = image_transform.get("pad_top", 0)

        parts: list[str] = []
        for element in elements:
            bbox = element.get("bbox", [0, 0, 0, 0])
            if scale != 1.0 or pad_left or pad_top:
                x1 = int(bbox[0] * scale + pad_left)
                y1 = int(bbox[1] * scale + pad_top)
                x2 = int(bbox[2] * scale + pad_left)
                y2 = int(bbox[3] * scale + pad_top)
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            parts.append(f"<|box_start|>{x1} {y1} {x2} {y2}<|box_end|>")

            label = element.get("label", "unknown")
            parts.append(f"<|ref_start|>{label}<|ref_end|>")

            rotation = int(element.get("rotation", 0))
            if rotation == 0:
                parts.append("<|rotate_up|>")
            elif rotation == 90:
                parts.append("<|rotate_right|>")
            elif rotation == 180:
                parts.append("<|rotate_down|>")
            elif rotation == 270:
                parts.append("<|rotate_left|>")
            else:
                parts.append("<|rotate_up|>")

        return "".join(parts)

    def validate(self, canonical_record: dict) -> None:
        target = canonical_record.get("target", {})
        if not isinstance(target, dict):
            raise ValueError("canonical record target must be a dict")
        elements = target.get("elements")
        if not isinstance(elements, list):
            raise ValueError("layout target must contain 'elements' list")
