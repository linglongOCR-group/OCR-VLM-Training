from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TextIO


class ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        log_every: int = 1000,
        stream: TextIO | None = None,
        root: str | Path | None = None,
        force_tty: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.log_every = max(int(log_every), 1)
        self.stream = stream or sys.stderr
        self.root = Path(root).resolve() if root is not None else None
        self.started_at = time.monotonic()
        self._tty = bool(force_tty) if force_tty is not None else bool(getattr(self.stream, "isatty", lambda: False)())
        self._bar_active = False

    @classmethod
    def disabled(cls) -> "ProgressReporter":
        return cls(enabled=False)

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._clear_bar()
        payload = " ".join(f"{key}={self._format_value(value)}" for key, value in fields.items() if value is not None)
        suffix = f" {payload}" if payload else ""
        self.stream.write(f"[docds] {event}{suffix}\n")
        self.stream.flush()

    def update(self, event: str, current: int, *, total: int | None = None, force: bool = False, **fields: Any) -> None:
        if not self.enabled:
            return
        if self._tty:
            self._draw_bar(event, current, total=total, **fields)
            return
        if force or current == 1 or current % self.log_every == 0 or (total is not None and current >= total):
            self.log(event, current=current, total=total, **self._monitoring_fields(current, total), **fields)

    def finish(self, event: str, *, total: int | None = None, **fields: Any) -> None:
        if not self.enabled:
            return
        self._clear_bar()
        self.log(event, phase="done", total=total, **fields)

    def path(self, value: str | Path | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if self.root is not None:
            try:
                return path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                pass
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def _draw_bar(self, event: str, current: int, *, total: int | None, **fields: Any) -> None:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        rate = current / elapsed
        if total:
            fraction = min(max(current / total, 0.0), 1.0)
            width = 24
            filled = int(width * fraction)
            bar = "#" * filled + "-" * (width - filled)
            count = f"{current}/{total}"
        else:
            bar = "#" * 24
            count = str(current)
        payload = " ".join(f"{key}={self._format_value(value)}" for key, value in fields.items() if value is not None)
        suffix = f" {payload}" if payload else ""
        self.stream.write(f"\r[docds] {event} [{bar}] {count} {rate:.1f}/s{suffix}")
        self.stream.flush()
        self._bar_active = True

    def _monitoring_fields(self, current: int, total: int | None) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        rate = current / elapsed
        fields: dict[str, Any] = {
            "elapsed_s": round(elapsed, 1),
            "rate_per_s": round(rate, 2),
        }
        if total:
            remaining = max(total - current, 0)
            fields["pct"] = round(min(max(current / total, 0.0), 1.0) * 100, 1)
            fields["eta_s"] = round(remaining / rate, 1) if rate > 0 else None
        return fields

    def _clear_bar(self) -> None:
        if self._bar_active:
            self.stream.write("\n")
            self.stream.flush()
            self._bar_active = False

    def _format_value(self, value: Any) -> str:
        if isinstance(value, Path):
            return self.path(value) or ""
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(item) for item in value)
        return str(value)
