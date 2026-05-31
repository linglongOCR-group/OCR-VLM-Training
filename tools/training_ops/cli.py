from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from tools.training_ops.cleanup import execute_cleanup, plan_cleanup
from tools.training_ops.config import load_run_context
from tools.training_ops.deploy import deploy_shared, deploy_tmp
from tools.training_ops.executor import Executor
from tools.training_ops.launch import build_grpo_launches, build_sft_launches, launch_metadata
from tools.training_ops.package import create_package
from tools.training_ops.preflight import run_preflight
from tools.training_ops.ray import build_ray_head_command, build_ray_status_commands, build_ray_stop_commands, build_ray_worker_commands
from tools.training_ops.state import OpsState
from tools.training_ops.status import collect_status, list_logs


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inventory":
        context = _context(args)
        print(json.dumps({"cluster": context.inventory.name, "nodes": [node.__dict__ for node in context.selected_nodes]}, indent=2))
        return
    if args.command == "logs":
        context = _context(args)
        state = OpsState.create(context, root=args.state_root)
        for path in list_logs(state.root, node=args.node, label=args.label):
            print(path)
        return

    context = _context(args)
    state = OpsState.create(context, root=getattr(args, "state_root", None))
    state.snapshot_configs()
    executor = Executor(state, dry_run=getattr(args, "dry_run", False))

    if args.command == "preflight":
        state.write_json("status.json", run_preflight(context, executor))
    elif args.command == "package":
        result = _package(context, state)
        state.write_json("package-manifest.json", result.manifest)
        print(result.artifact_path)
    elif args.command == "deploy":
        package = _package(context, state)
        manifest = deploy_shared(context, package, state, executor) if args.deploy_method == "shared" else deploy_tmp(context, package, state, executor)
        print(json.dumps(manifest, indent=2))
    elif args.command == "exec":
        for node in context.selected_nodes:
            if args.scope == "host":
                executor.host(node, "exec", args.remote_command)
            else:
                executor.container(node, "exec", args.remote_command)
    elif args.command == "ray":
        _run_node_commands(executor, _ray_commands(args.ray_action, context))
    elif args.command == "launch":
        project_root = args.project_root or context.project_root()
        state.write_json(
            "launch-metadata.json",
            launch_metadata(context, project_root=project_root, mode=args.launch_mode, background=args.background),
        )
        commands = build_grpo_launches(context, project_root=project_root, background=args.background) if args.launch_mode == "grpo" else build_sft_launches(context, project_root=project_root, background=args.background)
        _run_node_commands(executor, commands)
    elif args.command == "status":
        state.write_json("status.json", collect_status(context, executor))
    elif args.command == "stop":
        if not args.target:
            raise SystemExit("stop requires --target")
        if args.target == "ray":
            _run_node_commands(executor, build_ray_stop_commands(context))
        else:
            marker = f"TRAINOPS_RUN_ID={context.run_id}"
            experiment = context.effective_env().get("EXPERIMENT_NAME", context.run_id)
            if args.target == "run":
                pattern = f"{marker}|EXPERIMENT_NAME={experiment}"
            else:
                trainer = "verl.trainer.main_ppo" if args.target == "grpo" else "verl_plugins.trainers.sft_trainer"
                pattern = f"({marker}|EXPERIMENT_NAME={experiment}).*{trainer}"
            stop_command = (
                f"pattern={shlex.quote(pattern)}; "
                'if pgrep -f \"$pattern\" >/dev/null; then pkill -f \"$pattern\"; else true; fi'
            )
            for node in context.selected_nodes:
                executor.container(node, f"stop-{args.target}", stop_command)
    elif args.command == "cleanup":
        plan = plan_cleanup(release_root=args.release_root, current_link=args.current_link, dry_run=not args.force)
        state.write_json("cleanup-plan.json", {"dry_run": plan.dry_run, "candidates": [str(path) for path in plan.candidates]})
        execute_cleanup(plan, force=args.force)
    else:
        parser.error(f"unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainops", description="OCR-VLM training operations CLI")
    parser.add_argument("--run", required=True, help="Run YAML config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--state-root")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    sub.add_parser("preflight")
    sub.add_parser("package")
    deploy = sub.add_parser("deploy")
    deploy_sub = deploy.add_subparsers(dest="deploy_method", required=True)
    deploy_sub.add_parser("shared")
    deploy_sub.add_parser("tmp")
    exec_cmd = sub.add_parser("exec")
    exec_cmd.add_argument("--scope", choices=["host", "container"], default="container")
    exec_cmd.add_argument("remote_command")
    ray = sub.add_parser("ray")
    ray_sub = ray.add_subparsers(dest="ray_action", required=True)
    ray_sub.add_parser("start-head")
    ray_sub.add_parser("start-worker")
    ray_sub.add_parser("status")
    ray_sub.add_parser("stop")
    launch = sub.add_parser("launch")
    launch_sub = launch.add_subparsers(dest="launch_mode", required=True)
    for name in ("grpo", "sft"):
        child = launch_sub.add_parser(name)
        child.add_argument("--project-root")
        child.add_argument("--background", action="store_true")
    sub.add_parser("status")
    logs = sub.add_parser("logs")
    logs.add_argument("--node")
    logs.add_argument("--label")
    stop = sub.add_parser("stop")
    stop.add_argument("--target", choices=["ray", "grpo", "sft", "run"], required=True)
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--release-root", required=True)
    cleanup.add_argument("--current-link")
    cleanup.add_argument("--force", action="store_true")
    return parser


def _context(args: argparse.Namespace):
    return load_run_context(args.run, repo_root=args.repo_root)


def _package(context, state):
    return create_package(
        source_root=context.repo_root,
        output_dir=state.packages_dir,
        release_id=context.release_id,
        target_nodes=[node.name for node in context.selected_nodes],
        extra_excludes=context.deployment.get("extra_excludes") or (),
        extra_includes=context.deployment.get("extra_includes") or (),
    )


def _ray_commands(action: str, context):
    if action == "start-head":
        return [build_ray_head_command(context)]
    if action == "start-worker":
        return build_ray_worker_commands(context)
    if action == "status":
        return build_ray_status_commands(context)
    if action == "stop":
        return build_ray_stop_commands(context)
    raise SystemExit(f"unsupported ray action: {action}")


def _run_node_commands(executor: Executor, commands) -> None:
    for command in commands:
        executor.container(command.node, command.label, command.command)


if __name__ == "__main__":
    main(sys.argv[1:])
