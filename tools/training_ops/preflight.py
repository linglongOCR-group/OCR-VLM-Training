from __future__ import annotations

import shlex

from tools.training_ops.config import RunContext
from tools.training_ops.executor import Executor


def run_preflight(context: RunContext, executor: Executor) -> dict:
    results = {}
    for node in context.selected_nodes:
        checks = []
        checks.append(executor.host(node, "preflight-ssh", "true").record.to_dict())
        checks.append(
            executor.host(
                node,
                "preflight-container",
                f"docker ps --format '{{{{.Names}}}}' | grep -Fx {shlex.quote(node.container)}",
            ).record.to_dict()
        )
        checks.append(
            executor.container(
                node,
                "preflight-runtime",
                "npu-smi info >/dev/null 2>&1; python - <<'PY'\nimport torch\nprint('python runtime ok')\nPY",
            ).record.to_dict()
        )
        paths = context.paths
        path_checks = [paths.get("model_path"), paths.get("train_file") or paths.get("train_files"), paths.get("val_file") or paths.get("val_files"), paths.get("ckpts_dir"), paths.get("ocr_data_root")]
        for value in [str(path) for path in path_checks if path]:
            quoted = shlex.quote(value)
            checks.append(executor.container(node, "preflight-path", f"test -e {quoted} || test -d {quoted}").record.to_dict())
        checks.append(
            executor.container(
                node,
                "preflight-active-processes",
                "ps -ef | grep -E 'ray|verl.trainer|sft_trainer' | grep -v grep || true",
                allow_failure=True,
            ).record.to_dict()
        )
        results[node.name] = checks
    return {"run_id": context.run_id, "warnings": ["active process checks are warning-only"], "nodes": results}
