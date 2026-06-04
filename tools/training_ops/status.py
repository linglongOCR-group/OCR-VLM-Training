from __future__ import annotations

import json
from pathlib import Path

from tools.training_ops.config import RunContext
from tools.training_ops.executor import Executor
from tools.training_ops.ray import build_ray_status_commands


def collect_status(context: RunContext, executor: Executor) -> dict:
    commands = []
    for command in build_ray_status_commands(context):
        commands.append(executor.container(command.node, command.label, command.command, allow_failure=True).record.to_dict())
    for node in context.selected_nodes:
        commands.append(
            executor.container(
                node,
                "training-process-status",
                "ps -ef | grep -E 'verl.trainer.main_ppo|verl_plugins.trainers.sft_trainer' | grep -v grep || true",
                allow_failure=True,
            ).record.to_dict()
        )
    latest = _read_optional_json(executor.state.root / "deployment-manifest.json")
    launch = _read_optional_json(executor.state.root / "launch-metadata.json")
    logs = [str(path) for path in list_logs(executor.state.root)[-20:]]
    return {
        "run_id": context.run_id,
        "release_id": context.release_id,
        "latest_deployment": latest,
        "launch": launch,
        "recent_logs": logs,
        "commands": commands,
        "warnings": ["process checks are informational"],
    }


def list_logs(root: str | Path, *, node: str | None = None, label: str | None = None) -> list[Path]:
    root = Path(root)
    logs_dir = root / "logs"
    metadata_logs = [Path(path) for path in (_read_optional_json(root / "launch-metadata.json") or {}).get("background_log_paths", [])]
    if not logs_dir.exists():
        return metadata_logs
    logs = sorted(logs_dir.glob("*.log"))
    if node:
        logs = [path for path in logs if f"-{node}." in path.name]
    if label:
        logs = [path for path in logs if path.name.startswith(label)]
    return [*logs, *metadata_logs]


def _read_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())
