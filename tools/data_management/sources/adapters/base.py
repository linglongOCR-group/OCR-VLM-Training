from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from tools.data_management.canonical.writer import CanonicalWriteReport
from tools.data_management.progress import ProgressReporter


class SourceAdapter(ABC):
    name: str
    version: str

    @abstractmethod
    def scan_documents(self) -> Iterable[dict]:
        """Return source-level document descriptors."""

    @abstractmethod
    def export(
        self,
        canonical_root: str | Path,
        *,
        tasks: list[str] | None = None,
        overwrite_partitions: bool = True,
        progress: ProgressReporter | None = None,
    ) -> CanonicalWriteReport:
        """Export the source into canonical parquet partitions."""
