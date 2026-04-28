from __future__ import annotations

import math
from typing import Any

from tools.data_management.serializers.base import TargetSerializer


ROTATION_TOKENS = {
    0: "<|rotate_up|>",
    90: "<|rotate_right|>",
    180: "<|rotate_down|>",
    270: "<|rotate_left|>",
}


class MinerULayoutSerializer(TargetSerializer):
    name = "mineru_layout_box_v1"
    version = "1.0.0"
    task = "layout"

    def serialize(self, canonical_record: dict[str, Any], context: dict[str, Any]) -> str:
        self.validate(canonical_record)
        target = canonical_record.get("target") or {}
        elements = target.get("elements") or []
        width = float(context.get("width") or canonical_record.get("metadata", {}).get("width") or 1)
        height = float(context.get("height") or canonical_record.get("metadata", {}).get("height") or 1)
        transform = context.get("image_transform") or {}
        scale_x = float(transform.get("scale_x") or transform.get("scale") or 1.0)
        scale_y = float(transform.get("scale_y") or transform.get("scale") or 1.0)
        pad_left = float(transform.get("pad_left", 0.0))
        pad_top = float(transform.get("pad_top", 0.0))
        output_width = float(transform.get("output_width") or width * scale_x + pad_left)
        output_height = float(transform.get("output_height") or height * scale_y + pad_top)

        parts: list[str] = []
        for element in elements:
            x1, y1, x2, y2 = [float(value) for value in element["bbox"]]
            x1 = x1 * scale_x + pad_left
            y1 = y1 * scale_y + pad_top
            x2 = x2 * scale_x + pad_left
            y2 = y2 * scale_y + pad_top
            coords = [
                _to_grid(x1, output_width),
                _to_grid(y1, output_height),
                _to_grid(x2, output_width),
                _to_grid(y2, output_height),
            ]
            rotation = int(element.get("rotation") or 0)
            parts.append(
                f"<|box_start|>{coords[0]} {coords[1]} {coords[2]} {coords[3]}<|box_end|>"
                f"<|ref_start|>{element.get('label') or element.get('category') or 'unknown'}<|ref_end|>"
                f"{ROTATION_TOKENS.get(rotation, '<|rotate_up|>')}"
            )
        return "".join(parts)


def _to_grid(value: float, size: float) -> int:
    if size <= 0:
        return 0
    return max(0, min(1000, math.floor(1000 * value / size)))
