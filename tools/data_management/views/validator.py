from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_view(view_root: str | Path, *, require_images: bool = False) -> None:
    root = Path(view_root)
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise ValueError(f"no view parquet files found under {root}")
    seen_docs_by_split: dict[str, set[str]] = {}
    for path in files:
        frame = pd.read_parquet(path)
        split = path.stem
        for column in ("id", "stage", "task", "image_path", "prompt", "label", "canonical_record_id", "split"):
            if column not in frame.columns:
                raise ValueError(f"{path} missing required column {column}")
        is_rlvr = (frame["stage"] == "rlvr").any() if "stage" in frame.columns else False
        if is_rlvr:
            for column in ("images", "data_source", "extra_info"):
                if column not in frame.columns:
                    raise ValueError(f"{path} rlvr view missing required column {column}")
        for row_index, row in frame.iterrows():
            prompt = row["prompt"]
            if prompt is None or (hasattr(prompt, "__len__") and len(prompt) == 0):
                raise ValueError(f"{path}:{row_index} prompt is empty")
            if not row["label"]:
                raise ValueError(f"{path}:{row_index} label is empty")
            if row["split"] != split:
                raise ValueError(f"{path}:{row_index} split column does not match file split")
            if row["stage"] == "rlvr" and not row.get("reward_profile_id"):
                raise ValueError(f"{path}:{row_index} rlvr record missing reward_profile_id")
            if require_images and not Path(str(row["image_path"])).exists():
                raise ValueError(f"{path}:{row_index} image does not exist: {row['image_path']}")
            document_id = row.get("document_id")
            if document_id:
                seen_docs_by_split.setdefault(str(document_id), set()).add(split)
    leaked = {doc: splits for doc, splits in seen_docs_by_split.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"document split leakage detected: {list(leaked)[:5]}")
