from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class SourceAdapter(ABC):
    """Converts source-specific raw data into canonical entities, records, and assets."""

    name: str
    version: str

    @abstractmethod
    def scan_documents(self) -> Iterable[dict]:
        """Return source-level document descriptors."""

    @abstractmethod
    def export_documents(self) -> Iterable[dict]:
        """Export canonical document entities."""

    @abstractmethod
    def export_pages(self) -> Iterable[dict]:
        """Export canonical page entities and page image assets."""

    @abstractmethod
    def export_regions(self) -> Iterable[dict]:
        """Export canonical region entities and crop assets."""

    @abstractmethod
    def export_task_records(self, task: str) -> Iterable[dict]:
        """Export canonical task records for a specific task."""
