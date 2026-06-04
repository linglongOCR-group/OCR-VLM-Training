from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tools.training_ops.config import RunContext, dumps_json


@dataclass(frozen=True)
class CommandRecord:
    label: str
    node: str
    host: str
    run_rank: int
    rank: int
    container: str | None
    command: str
    argv: list[str]
    started_at: str
    ended_at: str
    exit_code: int
    stdout_path: str
    stderr_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpsState:
    def __init__(self, context: RunContext, root: Path) -> None:
        self.context = context
        self.root = root
        self.logs_dir = root / "logs"
        self.packages_dir = root / "packages"
        self.manifests_dir = root / "manifests"

    @classmethod
    def create(cls, context: RunContext, *, root: str | Path | None = None) -> "OpsState":
        if root is None:
            ops_root = context.run.get("ops_state_root") or context.inventory.shared_paths.get("ops_state_root")
            root_path = Path(ops_root) / context.run_id if ops_root else context.repo_root / "runs" / "training-ops" / context.run_id
        else:
            root_path = Path(root)
            if root_path.name != context.run_id:
                root_path = root_path / context.run_id
        state = cls(context, root_path)
        for path in (state.root, state.logs_dir, state.packages_dir, state.manifests_dir):
            path.mkdir(parents=True, exist_ok=True)
        return state

    def snapshot_configs(self) -> None:
        _write_yaml(self.root / "run.yaml", self.context.run_payload)
        _write_yaml(self.root / "inventory.yaml", self.context.inventory_payload)
        self.write_json("effective-env.json", self.context.effective_env())
        self.write_json("run-metadata.json", self.context.to_snapshot())

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_json(payload) + "\n")
        return path

    def append_command(self, record: CommandRecord) -> None:
        path = self.root / "commands.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
