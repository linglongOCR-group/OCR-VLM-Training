from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.training_ops.state import utc_now


DEFAULT_EXCLUDES = (
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    "venv",
    "venv/**",
    "__pycache__",
    "**/__pycache__/**",
    ".pytest_cache",
    ".pytest_cache/**",
    ".mypy_cache",
    ".mypy_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
    "checkpoints",
    "checkpoints/**",
    "datasets",
    "datasets/**",
    "wandb",
    "wandb/**",
    "runs",
    "runs/**",
    "outputs",
    "outputs/**",
    "artifacts",
    "artifacts/**",
    "*.log",
    "*.tmp",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.zip",
)


@dataclass(frozen=True)
class PackageResult:
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def create_package(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    release_id: str,
    target_nodes: Iterable[str] = (),
    extra_excludes: Iterable[str] = (),
    extra_includes: Iterable[str] = (),
) -> PackageResult:
    source_root = Path(source_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{release_id}.tar.gz"
    manifest_path = output_dir / f"{release_id}.package-manifest.json"
    excludes = tuple(DEFAULT_EXCLUDES) + tuple(extra_excludes)
    output_rel = _relative_or_none(output_dir, source_root)
    output_patterns = (str(output_rel), f"{output_rel}/**") if output_rel else ()

    with tarfile.open(artifact_path, "w:gz") as archive:
        for path in _iter_files(source_root):
            rel = path.relative_to(source_root).as_posix()
            if _matches(rel, excludes + output_patterns) and not _matches(rel, tuple(extra_includes)):
                continue
            archive.add(path, arcname=rel, recursive=False)

    checksum = sha256_file(artifact_path)
    manifest: dict[str, Any] = {
        "branch": _git_value(source_root, ["rev-parse", "--abbrev-ref", "HEAD"], default="unknown"),
        "commit_sha": _git_value(source_root, ["rev-parse", "HEAD"], default="unknown"),
        "dirty": _git_dirty(source_root),
        "release_id": release_id,
        "timestamp": utc_now(),
        "source_path": str(source_root),
        "artifact_path": str(artifact_path),
        "target_nodes": list(target_nodes),
        "exclude_rules": list(excludes),
        "artifact_sha256": checksum,
    }
    manifest_path.write_text(_json(manifest))
    return PackageResult(artifact_path=artifact_path, manifest_path=manifest_path, manifest=manifest)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _matches(rel: str, patterns: tuple[str, ...]) -> bool:
    parts = rel.split("/")
    return any(fnmatch.fnmatch(rel, pattern) or any(fnmatch.fnmatch(part, pattern) for part in parts) for pattern in patterns)


def _relative_or_none(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _git_value(root: Path, args: list[str], *, default: str) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return default
    return completed.stdout.strip() if completed.returncode == 0 else default


def _git_dirty(root: Path) -> bool:
    try:
        completed = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return True
    return bool(completed.stdout.strip()) if completed.returncode == 0 else True


def _json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
