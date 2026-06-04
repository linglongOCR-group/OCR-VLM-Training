from __future__ import annotations

import shlex

from tools.training_ops.config import RunContext, command_with_env
from tools.training_ops.ray import NodeCommand


def grpo_background_log_path(project_root: str) -> str:
    return f"{project_root.rstrip('/')}/training-grpo.log"


def sft_background_log_path(project_root: str, rank: int) -> str:
    return f"{project_root.rstrip('/')}/training-sft-rank{rank}.log"


def kd_sft_background_log_path(project_root: str, rank: int) -> str:
    return f"{project_root.rstrip('/')}/training-kd-sft-rank{rank}.log"


def background_command(command: str, log_path: str) -> str:
    return f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(log_path)} 2>&1 &"


def build_grpo_launches(context: RunContext, *, project_root: str | None = None, background: bool = False) -> list[NodeCommand]:
    root = project_root or context.project_root()
    env = context.effective_env(project_root=root)
    command = command_with_env(env, "cd $PROJECT_ROOT && bash scripts/train/run_grpo_fsdp.sh", context.extra_args())
    if background:
        command = background_command(command, grpo_background_log_path(root))
    return [NodeCommand(context.head_node, "launch-grpo", command)]


def build_sft_launches(context: RunContext, *, project_root: str | None = None, background: bool = False) -> list[NodeCommand]:
    root = project_root or context.project_root()
    commands: list[NodeCommand] = []
    for node in context.selected_nodes:
        env = context.effective_env(node=node, project_root=root)
        command = command_with_env(env, "cd $PROJECT_ROOT && bash scripts/train/run_multinode_sft_new.sh", context.extra_args())
        if background:
            command = background_command(command, sft_background_log_path(root, node.run_rank))
        commands.append(NodeCommand(node, "launch-sft", command))
    return commands


def build_kd_sft_launches(context: RunContext, *, project_root: str | None = None, background: bool = False) -> list[NodeCommand]:
    root = project_root or context.project_root()
    commands: list[NodeCommand] = []
    for node in context.selected_nodes:
        env = context.effective_env(node=node, project_root=root)
        command = command_with_env(env, "cd $PROJECT_ROOT && bash scripts/train/run_multinode_kd_sft.sh", context.extra_args())
        if background:
            command = background_command(command, kd_sft_background_log_path(root, node.run_rank))
        commands.append(NodeCommand(node, "launch-kd-sft", command))
    return commands


def launch_metadata(
    context: RunContext,
    *,
    project_root: str | None = None,
    mode: str | None = None,
    background: bool = False,
) -> dict:
    mode = mode or context.mode
    root = project_root or context.project_root()
    if mode == "grpo":
        commands = build_grpo_launches(context, project_root=root, background=background)
    elif mode == "kd_sft":
        commands = build_kd_sft_launches(context, project_root=root, background=background)
    else:
        commands = build_sft_launches(context, project_root=root, background=background)
    background_log_paths: list[str] = []
    if background and mode == "grpo":
        background_log_paths = [grpo_background_log_path(root)]
    elif background and mode == "kd_sft":
        background_log_paths = [kd_sft_background_log_path(root, command.node.run_rank) for command in commands]
    elif background:
        background_log_paths = [sft_background_log_path(root, command.node.run_rank) for command in commands]
    return {
        "run_id": context.run_id,
        "release_id": context.release_id,
        "mode": mode,
        "project_root": root,
        "background": background,
        "background_log_paths": background_log_paths,
        "effective_env": context.effective_env(project_root=root),
        "extra_args": context.extra_args(),
        "target_nodes": [command.node.name for command in commands],
        "run_ranks": {node.name: node.run_rank for node in context.selected_nodes},
        "commands": [{"node": command.node.name, "label": command.label, "command": command.command} for command in commands],
    }
