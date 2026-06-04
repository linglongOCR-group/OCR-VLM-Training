from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupPlan:
    candidates: list[Path]
    dry_run: bool


def plan_cleanup(*, release_root: str | Path, current_link: str | Path | None = None, dry_run: bool = True) -> CleanupPlan:
    release_root = Path(release_root)
    active: Path | None = None
    if current_link:
        link = Path(current_link)
        if link.is_symlink():
            active = link.resolve()
    candidates = []
    if release_root.exists():
        for path in sorted(p for p in release_root.iterdir() if p.is_dir()):
            if active is not None and path.resolve() == active:
                continue
            candidates.append(path)
    return CleanupPlan(candidates=candidates, dry_run=dry_run)


def execute_cleanup(plan: CleanupPlan, *, force: bool = False) -> None:
    if plan.dry_run and not force:
        return
    for path in plan.candidates:
        shutil.rmtree(path)
