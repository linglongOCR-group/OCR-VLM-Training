from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.training_ops.errors import ConfigError
from tools.training_ops.inventory import Inventory, SelectedNode, parse_inventory, select_nodes


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    run_config_path: Path
    inventory_path: Path
    run_payload: dict[str, Any]
    inventory_payload: dict[str, Any]
    inventory: Inventory
    selected_nodes: tuple[SelectedNode, ...]

    @property
    def run(self) -> dict[str, Any]:
        return self.run_payload["run"]

    @property
    def run_id(self) -> str:
        return str(self.run.get("id"))

    @property
    def mode(self) -> str:
        return str(self.run.get("mode", "")).lower()

    @property
    def deployment(self) -> dict[str, Any]:
        return dict(self.run.get("deployment") or {})

    @property
    def paths(self) -> dict[str, Any]:
        return dict(self.run.get("paths") or {})

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.run.get("training") or {})

    @property
    def head_node(self) -> SelectedNode:
        head_name = self.run.get("head_node")
        if head_name:
            for node in self.selected_nodes:
                if node.name == head_name:
                    return node
        return self.selected_nodes[0]

    @property
    def release_id(self) -> str:
        return str(self.deployment.get("release_id") or self.run.get("release_id") or self.run_id)

    def project_root(self) -> str:
        deployment = self.deployment
        method = str(deployment.get("method", "shared"))
        if method == "tmp":
            root = deployment.get("container_release_root", "/tmp/ocr-vlm-training/releases")
            return f"{root.rstrip('/')}/{self.release_id}"
        return str(deployment.get("current_link") or self.inventory.shared_paths.get("code_root") or self.repo_root)

    def effective_env(self, *, node: SelectedNode | None = None, project_root: str | None = None) -> dict[str, str]:
        paths = self.paths
        training_env = dict((self.training.get("env") or {}))
        env: dict[str, str] = {str(key): str(value) for key, value in training_env.items()}
        env.pop("NODE_RANK", None)
        root = project_root or self.project_root()
        env.update(
            {
                "PROJECT_ROOT": root,
                "PYTHONPATH": f"{root}:${{PYTHONPATH:-}}",
                "TRAINOPS_RUN_ID": self.run_id,
                "MODEL_PATH": str(paths.get("model_path", "")),
                "CKPTS_DIR": str(paths.get("ckpts_dir", "")),
                "OCR_DATA_ROOT": str(paths.get("ocr_data_root", "")),
                "NNODES": str(len(self.selected_nodes)),
                "NPUS_PER_NODE": str(env.get("NPUS_PER_NODE", self.run.get("npus_per_node", 8))),
            }
        )
        if self.mode == "grpo":
            env["TRAIN_FILE"] = str(paths.get("train_file", ""))
            if paths.get("val_file") is not None:
                env["VAL_FILE"] = str(paths.get("val_file"))
            else:
                env.setdefault("USE_VALIDATION", "False")
            env.setdefault("RAY_ADDRESS", str(self.training.get("ray_address", "auto")))
        elif self.mode == "sft":
            env["TRAIN_FILES"] = str(paths.get("train_files") or paths.get("train_file", ""))
            if paths.get("val_files") is not None:
                env["VAL_FILES"] = str(paths.get("val_files"))
            elif paths.get("val_file") is not None:
                env["VAL_FILES"] = str(paths.get("val_file"))
            env["MASTER_ADDR"] = str(self.training.get("master_addr", self.head_node.host_ip))
            env["MASTER_PORT"] = str(self.training.get("master_port", 29500))
        if node is not None and self.mode == "sft":
            env["NODE_RANK"] = str(node.run_rank)
            env["TRAIN_IFACE"] = node.train_iface
        return env

    def extra_args(self) -> list[str]:
        return [str(arg) for arg in self.training.get("args") or []]

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "release_id": self.release_id,
            "run_config_path": str(self.run_config_path),
            "inventory_path": str(self.inventory_path),
            "selected_nodes": [node.to_dict() for node in self.selected_nodes],
            "head_node": self.head_node.name,
            "effective_env": self.effective_env(),
        }


def load_run_context(run_config_path: str | Path, *, repo_root: str | Path | None = None) -> RunContext:
    run_config_path = Path(run_config_path).expanduser().resolve()
    repo_root_path = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd().resolve()
    run_payload = _read_yaml(run_config_path)
    run = run_payload.get("run")
    if not isinstance(run, dict):
        raise ConfigError("run config requires top-level run mapping")
    if not run.get("id"):
        raise ConfigError("run.id is required")
    mode = str(run.get("mode", "")).lower()
    if mode not in {"grpo", "sft"}:
        raise ConfigError("run.mode must be grpo or sft")
    inventory_ref = run.get("inventory")
    if not inventory_ref:
        raise ConfigError("run.inventory is required")
    inventory_path = Path(inventory_ref).expanduser()
    if not inventory_path.is_absolute():
        inventory_path = (run_config_path.parent / inventory_path).resolve()
        if not inventory_path.exists():
            inventory_path = (repo_root_path / inventory_ref).resolve()
    inventory_payload = _read_yaml(inventory_path)
    inventory = parse_inventory(inventory_payload)
    selected = select_nodes(inventory, list(run.get("nodes") or []))
    if not selected:
        raise ConfigError("run must select at least one node")
    head_node = run.get("head_node")
    if head_node is not None and str(head_node) not in {node.name for node in selected}:
        raise ConfigError(f"run.head_node must be one of selected nodes: {head_node}")
    context = RunContext(
        repo_root=repo_root_path,
        run_config_path=run_config_path,
        inventory_path=inventory_path,
        run_payload=run_payload,
        inventory_payload=inventory_payload,
        inventory=inventory,
        selected_nodes=selected,
    )
    _validate_context(context)
    return context


def format_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={_quote_env_value(value)}" for key, value in sorted(env.items()))


def command_with_env(env: dict[str, str], command: str, args: list[str] | None = None) -> str:
    suffix = ""
    if args:
        suffix = " " + " ".join(shlex.quote(arg) for arg in args)
    return f"{format_env(env)} {command}{suffix}"


def _quote_env_value(value: str) -> str:
    text = str(value)
    if text == "":
        return "''"
    if all(char.isalnum() or char in "/._:-${}" for char in text):
        return text
    return shlex.quote(text)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return payload


def _validate_context(context: RunContext) -> None:
    deployment = context.deployment
    method = str(deployment.get("method", "shared"))
    if method not in {"shared", "tmp"}:
        raise ConfigError("deployment.method must be shared or tmp")
    if method == "shared":
        for key in ("release_root", "current_link"):
            if not deployment.get(key):
                raise ConfigError(f"deployment.{key} is required for shared deployment")
    if method == "tmp" and not deployment.get("container_release_root") and not deployment.get("host_release_root"):
        raise ConfigError("tmp deployment requires container_release_root or host_release_root")
    paths = context.paths
    missing = []
    if not paths.get("model_path"):
        missing.append("MODEL_PATH")
    if not paths.get("train_file") and not paths.get("train_files"):
        missing.append("TRAIN_FILE")
    if not paths.get("ckpts_dir"):
        missing.append("CKPTS_DIR")
    if context.mode == "grpo" and not paths.get("val_file") and str(context.training.get("use_validation", "True")) == "True":
        missing.append("VAL_FILE")
    if missing:
        raise ConfigError("missing required fields for launch: " + ", ".join(missing))


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
