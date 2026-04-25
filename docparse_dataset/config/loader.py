from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def resolve_paths(config: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """Resolve relative paths in config against a base directory."""
    base = Path(base_dir)
    resolved: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("./"):
            resolved[key] = str(base / value)
        elif isinstance(value, dict):
            resolved[key] = resolve_paths(value, base_dir)
        else:
            resolved[key] = value
    return resolved
