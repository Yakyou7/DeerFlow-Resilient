"""DualLedger: Unified orchestration state combining TaskLedger and ProgressLedger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deerflow.agents.ledgers.progress_ledger import ProgressLedger
from deerflow.agents.ledgers.task_ledger import TaskLedger


@dataclass
class DualLedger:
    task_ledger: TaskLedger = field(default_factory=TaskLedger)
    progress_ledger: ProgressLedger = field(default_factory=ProgressLedger)

    def render_prompt(self) -> str:
        parts: list[str] = []
        task_md = self.task_ledger.render_markdown()
        if task_md:
            parts.append(task_md)
        progress_md = self.progress_ledger.render_markdown()
        if progress_md:
            parts.append(progress_md)
        return "\n\n".join(parts).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_ledger": self.task_ledger.to_dict(),
            "progress_ledger": self.progress_ledger.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DualLedger:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            task_ledger=TaskLedger.from_dict(data.get("task_ledger")),
            progress_ledger=ProgressLedger.from_dict(data.get("progress_ledger")),
        )
