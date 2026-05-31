from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.training_ops.errors import ConfigError


@dataclass(frozen=True)
class Node:
    name: str
    host: str
    rank: int
    host_ip: str
    train_iface: str
    ssh_user: str
    container: str

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}" if self.ssh_user else self.host


@dataclass(frozen=True)
class Inventory:
    name: str
    ssh_user: str
    default_container: str
    nodes: tuple[Node, ...]
    shared_paths: dict[str, Any]

    def by_name(self) -> dict[str, Node]:
        return {node.name: node for node in self.nodes}


def parse_inventory(payload: dict[str, Any]) -> Inventory:
    cluster = payload.get("cluster") or {}
    ssh_user = str(cluster.get("ssh_user", "root"))
    default_container = str(cluster.get("default_container", "verl-vlm-grpo"))
    raw_nodes = cluster.get("nodes") or []
    if not raw_nodes:
        raise ConfigError("inventory requires cluster.nodes")
    nodes: list[Node] = []
    for raw in raw_nodes:
        missing = [field for field in ("name", "host", "rank", "host_ip", "train_iface") if field not in raw]
        if missing:
            raise ConfigError(f"inventory node missing required fields: {', '.join(missing)}")
        nodes.append(
            Node(
                name=str(raw["name"]),
                host=str(raw["host"]),
                rank=int(raw["rank"]),
                host_ip=str(raw["host_ip"]),
                train_iface=str(raw["train_iface"]),
                ssh_user=str(raw.get("ssh_user", ssh_user)),
                container=str(raw.get("container", default_container)),
            )
        )
    return Inventory(
        name=str(cluster.get("name", "cluster")),
        ssh_user=ssh_user,
        default_container=default_container,
        nodes=tuple(sorted(nodes, key=lambda node: node.rank)),
        shared_paths=dict(cluster.get("shared_paths") or {}),
    )


def select_nodes(inventory: Inventory, names: list[str] | None) -> tuple[Node, ...]:
    if not names:
        return inventory.nodes
    by_name = inventory.by_name()
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ConfigError(f"run selected unknown inventory nodes: {', '.join(missing)}")
    return tuple(sorted((by_name[name] for name in names), key=lambda node: node.rank))
