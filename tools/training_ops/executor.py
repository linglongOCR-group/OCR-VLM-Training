from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.training_ops.errors import CommandExecutionError
from tools.training_ops.inventory import Node
from tools.training_ops.state import CommandRecord, OpsState, utc_now


@dataclass(frozen=True)
class ExecutionResult:
    record: CommandRecord
    stdout: str = ""
    stderr: str = ""


class Executor:
    def __init__(self, state: OpsState, *, dry_run: bool = False) -> None:
        self.state = state
        self.dry_run = dry_run

    def host(self, node: Node, label: str, command: str, *, allow_failure: bool = False) -> ExecutionResult:
        argv = ["ssh", node.ssh_target, command]
        return self._run(node, label, command, argv, container=None, allow_failure=allow_failure)

    def container(self, node: Node, label: str, command: str, *, allow_failure: bool = False) -> ExecutionResult:
        remote_command = shlex.join(["docker", "exec", node.container, "bash", "-lc", command])
        argv = ["ssh", node.ssh_target, remote_command]
        return self._run(node, label, remote_command, argv, container=node.container, allow_failure=allow_failure)

    def container_stream(
        self,
        node: Node,
        label: str,
        command: str,
        input_path: Path,
        *,
        allow_failure: bool = False,
    ) -> ExecutionResult:
        remote_command = shlex.join(["docker", "exec", "-i", node.container, "bash", "-lc", command])
        argv = ["ssh", node.ssh_target, remote_command]
        display = f"cat {shlex.quote(str(input_path))} | {remote_command}"
        return self._run(
            node,
            label,
            display,
            argv,
            container=node.container,
            stdin_path=input_path,
            allow_failure=allow_failure,
        )

    def host_stream(
        self,
        node: Node,
        label: str,
        command: str,
        input_path: Path,
        *,
        allow_failure: bool = False,
    ) -> ExecutionResult:
        argv = ["ssh", node.ssh_target, command]
        display = f"cat {shlex.quote(str(input_path))} | ssh {shlex.quote(node.ssh_target)} {shlex.quote(command)}"
        return self._run(
            node,
            label,
            display,
            argv,
            container=None,
            stdin_path=input_path,
            allow_failure=allow_failure,
        )

    def _run(
        self,
        node: Node,
        label: str,
        command: str,
        argv: list[str],
        *,
        container: str | None,
        stdin_path: Path | None = None,
        allow_failure: bool = False,
    ) -> ExecutionResult:
        started = utc_now()
        stdout_path = self.state.logs_dir / f"{label}-{node.name}.stdout.log"
        stderr_path = self.state.logs_dir / f"{label}-{node.name}.stderr.log"
        if self.dry_run:
            exit_code = 0
            stdout = ""
            stderr = ""
        else:
            stdin = stdin_path.open("rb") if stdin_path else None
            try:
                completed = subprocess.run(argv, stdin=stdin, capture_output=True, text=True, check=False)
            finally:
                if stdin is not None:
                    stdin.close()
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        record = CommandRecord(
            label=label,
            node=node.name,
            host=node.host,
            rank=node.rank,
            container=container,
            command=command,
            argv=argv,
            started_at=started,
            ended_at=utc_now(),
            exit_code=exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        self.state.append_command(record)
        if exit_code != 0 and not allow_failure:
            raise CommandExecutionError(
                f"required command failed: label={label} node={node.name} exit_code={exit_code} stderr={stderr_path}"
            )
        return ExecutionResult(record=record, stdout=stdout, stderr=stderr)


class FakeExecutor(Executor):
    def __init__(self, state: OpsState) -> None:
        super().__init__(state, dry_run=True)
        self.commands: list[CommandRecord] = []

    def _run(self, *args, **kwargs) -> ExecutionResult:
        result = super()._run(*args, **kwargs)
        self.commands.append(result.record)
        return result
