from __future__ import annotations

import os


def apply_npu_defaults() -> None:
    defaults = {
        "HCCL_CONNECT_TIMEOUT": "1500",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
