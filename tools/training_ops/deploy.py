from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from tools.training_ops.config import RunContext
from tools.training_ops.executor import Executor
from tools.training_ops.package import PackageResult
from tools.training_ops.state import OpsState, utc_now


def deploy_shared(context: RunContext, package: PackageResult, state: OpsState, executor: Executor) -> dict[str, Any]:
    release_root = str(context.deployment["release_root"]).rstrip("/")
    release_id = str(package.manifest["release_id"])
    release_path = f"{release_root}/{release_id}"
    current_link = str(context.deployment["current_link"])
    marker = _release_marker(package, release_path)
    marker_text = shlex.quote(_compact_json(marker))
    checksum = shlex.quote(str(package.manifest["artifact_sha256"]))
    for node in context.selected_nodes:
        extract_command = (
            "set -euo pipefail; "
            f"release={shlex.quote(release_path)}; "
            f"tmp={shlex.quote(release_path + '.tmp.$$')}; "
            'marker=\"$release/.training-ops-release.json\"; '
            f"checksum={checksum}; "
            'if [ -e \"$release\" ]; then '
            'test -f \"$marker\" && grep -F \"artifact_sha256\" \"$marker\" | grep -F \"$checksum\" >/dev/null; '
            "else "
            'rm -rf \"$tmp\"; mkdir -p \"$tmp\"; '
            f"tar -xzf {shlex.quote(str(package.artifact_path))} -C \"$tmp\"; "
            f"printf '%s\\n' {marker_text} > \"$tmp/.training-ops-release.json\"; "
            'mv -T \"$tmp\" \"$release\"; '
            "fi"
        )
        executor.host(node, "deploy-shared-extract", extract_command)
        executor.host(node, "deploy-shared-host-verify", f"test -d {shlex.quote(release_path)}")
        executor.container(
            node,
            "deploy-shared-container-verify",
            f"test -d {shlex.quote(release_path)} && grep -F {checksum} {shlex.quote(release_path + '/.training-ops-release.json')} >/dev/null",
        )
    link_tmp = f"{current_link}.tmp.$$"
    executor.host(
        context.head_node,
        "deploy-shared-current-link",
        f"ln -sfn {shlex.quote(release_path)} {shlex.quote(link_tmp)} && mv -Tf {shlex.quote(link_tmp)} {shlex.quote(current_link)}",
    )
    manifest = _deployment_manifest(context, package, release_path, "shared")
    state.write_json("deployment-manifest.json", manifest)
    state.write_json("manifests/deployment-manifest.json", manifest)
    return manifest


def deploy_tmp(context: RunContext, package: PackageResult, state: OpsState, executor: Executor) -> dict[str, Any]:
    deployment = context.deployment
    mode = str(deployment.get("tmp_target", "container"))
    container_root = str(deployment.get("container_release_root", "/tmp/ocr-vlm-training/releases")).rstrip("/")
    host_root = str(deployment.get("host_release_root", "/tmp/ocr-vlm-training/releases")).rstrip("/")
    target_root = host_root if mode == "host" else container_root
    release_id = str(package.manifest["release_id"])
    release_path = f"{target_root}/{release_id}"
    marker = _release_marker(package, release_path)
    marker_text = shlex.quote(_compact_json(marker))
    checksum = shlex.quote(str(package.manifest["artifact_sha256"]))
    for node in context.selected_nodes:
        stream_command = (
            "set -euo pipefail; "
            f"release={shlex.quote(release_path)}; "
            f"tmp={shlex.quote(release_path + '.tmp.$$')}; "
            'marker=\"$release/.training-ops-release.json\"; '
            f"checksum={checksum}; "
            'if [ -e \"$release\" ]; then '
            'test -f \"$marker\" && grep -F \"artifact_sha256\" \"$marker\" | grep -F \"$checksum\" >/dev/null; '
            "cat >/dev/null; "
            "else "
            'rm -rf \"$tmp\"; mkdir -p \"$tmp\"; '
            'tar -xzf - -C \"$tmp\"; '
            f"printf '%s\\n' {marker_text} > \"$tmp/.training-ops-release.json\"; "
            'mv -T \"$tmp\" \"$release\"; '
            "fi; "
            'test -d \"$release\" && grep -F \"$checksum\" \"$release/.training-ops-release.json\" >/dev/null'
        )
        if mode == "host":
            executor.host_stream(node, "deploy-tmp-host-stream", stream_command, package.artifact_path)
            executor.container(node, "deploy-tmp-container-verify-host-path", f"test -d {shlex.quote(release_path)}")
        else:
            executor.container_stream(node, "deploy-tmp-container-stream", stream_command, package.artifact_path)
        executor.container(node, "deploy-tmp-container-space", f"df -Pk {shlex.quote(release_path)} || df -Pk /tmp")
    manifest = _deployment_manifest(context, package, release_path, "tmp")
    state.write_json("deployment-manifest.json", manifest)
    return manifest


def _deployment_manifest(context: RunContext, package: PackageResult, target_path: str, method: str) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "release_id": package.manifest["release_id"],
        "method": method,
        "timestamp": utc_now(),
        "source_path": package.manifest["source_path"],
        "target_path": target_path,
        "target_nodes": [node.name for node in context.selected_nodes],
        "artifact_checksum": package.manifest["artifact_sha256"],
        "verification": {node.name: {"host": node.host, "container": node.container, "status": "recorded"} for node in context.selected_nodes},
    }


def _release_marker(package: PackageResult, target_path: str) -> dict[str, Any]:
    return {
        "release_id": package.manifest["release_id"],
        "artifact_sha256": package.manifest["artifact_sha256"],
        "source_path": package.manifest["source_path"],
        "target_path": target_path,
        "timestamp": utc_now(),
    }


def _compact_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
