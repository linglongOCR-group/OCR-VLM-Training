from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.training_ops.errors import ConfigError


@dataclass(frozen=True)
class Node:
    name: str
    host: str
    host_ip: str
    train_iface: str
    ssh_user: str
    container: str

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}" if self.ssh_user else self.host


@dataclass(frozen=True)
class SelectedNode:
    node: Node
    run_rank: int

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def host(self) -> str:
        return self.node.host

    @property
    def host_ip(self) -> str:
        return self.node.host_ip

    @property
    def train_iface(self) -> str:
        return self.node.train_iface

    @property
    def ssh_user(self) -> str:
        return self.node.ssh_user

    @property
    def container(self) -> str:
        return self.node.container

    @property
    def ssh_target(self) -> str:
        return self.node.ssh_target

    @property
    def rank(self) -> int:
        """Compatibility alias for callers that still read command rank."""
        return self.run_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host,
            "host_ip": self.host_ip,
            "train_iface": self.train_iface,
            "ssh_user": self.ssh_user,
            "container": self.container,
            "run_rank": self.run_rank,
        }


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
    names: set[str] = set()
    for raw in raw_nodes:
        missing = [field for field in ("name", "host", "host_ip", "train_iface") if field not in raw]
        if missing:
            raise ConfigError(f"inventory node missing required fields: {', '.join(missing)}")
        name = str(raw["name"])
        if name in names:
            raise ConfigError(f"inventory contains duplicate node name: {name}")
        names.add(name)
        nodes.append(
            Node(
                name=name,
                host=str(raw["host"]),
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
        nodes=tuple(nodes),
        shared_paths=dict(cluster.get("shared_paths") or {}),
    )


def select_nodes(inventory: Inventory, names: list[str] | None) -> tuple[SelectedNode, ...]:
    if not names:
        return tuple(SelectedNode(node, run_rank=index) for index, node in enumerate(inventory.nodes))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"run selected duplicate inventory nodes: {', '.join(duplicates)}")
    by_name = inventory.by_name()
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ConfigError(f"run selected unknown inventory nodes: {', '.join(missing)}")
    return tuple(SelectedNode(by_name[name], run_rank=index) for index, name in enumerate(names))
