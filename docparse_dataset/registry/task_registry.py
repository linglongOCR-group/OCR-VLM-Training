from __future__ import annotations

TASKS = ["layout", "text", "table", "formula", "diagram", "seal"]


class TaskRegistry:
    _tasks: dict[str, dict] = {}

    @classmethod
    def register(cls, name: str, schema: dict | None = None) -> None:
        cls._tasks[name] = schema or {}

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._tasks)

    @classmethod
    def get_schema(cls, task: str) -> dict:
        if task not in cls._tasks:
            raise KeyError(f"task not registered: {task!r}")
        return cls._tasks[task]


for _task in TASKS:
    TaskRegistry.register(_task)
