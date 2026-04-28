from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.data_management.utils.io import read_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    data = read_yaml(path)
    data.setdefault("_config_path", str(Path(path)))
    return data
