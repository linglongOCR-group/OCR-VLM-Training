from __future__ import annotations

from dataclasses import dataclass

from tools.training_ops.config import RunContext, command_with_env
from tools.training_ops.inventory import SelectedNode


@dataclass(frozen=True)
class NodeCommand:
    node: SelectedNode
    label: str
    command: str


def ray_head_address(context: RunContext) -> str:
    port = context.training.get("ray_port", context.deployment.get("ray_port", 6379))
    return f"{context.head_node.host_ip}:{port}"


def build_ray_head_command(context: RunContext, *, project_root: str | None = None) -> NodeCommand:
    env = {
        "PROJECT_ROOT": project_root or context.project_root(),
        "NPUS_PER_NODE": context.effective_env()["NPUS_PER_NODE"],
        "RAY_PORT": str(context.training.get("ray_port", 6379)),
        "RAY_DASHBOARD_PORT": str(context.training.get("ray_dashboard_port", 8265)),
        "NODE_IP_ADDRESS": context.head_node.host_ip,
    }
    return NodeCommand(
        context.head_node,
        "ray-start-head",
        command_with_env(env, "cd $PROJECT_ROOT && bash scripts/cluster/start_ray_head.sh"),
    )


def build_ray_worker_commands(context: RunContext, *, project_root: str | None = None) -> list[NodeCommand]:
    commands: list[NodeCommand] = []
    for node in context.selected_nodes:
        if node.name == context.head_node.name:
            continue
        env = {
            "PROJECT_ROOT": project_root or context.project_root(),
            "NPUS_PER_NODE": context.effective_env()["NPUS_PER_NODE"],
            "RAY_HEAD_ADDRESS": ray_head_address(context),
        }
        commands.append(
            NodeCommand(
                node,
                "ray-start-worker",
                command_with_env(env, "cd $PROJECT_ROOT && bash scripts/cluster/start_ray_worker.sh"),
            )
        )
    return commands


def build_ray_status_commands(context: RunContext) -> list[NodeCommand]:
    return [NodeCommand(node, "ray-status", "ray status || ps -ef | grep -E 'ray|gcs' | grep -v grep || true") for node in context.selected_nodes]


def build_ray_stop_commands(context: RunContext) -> list[NodeCommand]:
    return [NodeCommand(node, "ray-stop", "ray stop || true") for node in context.selected_nodes]
