"""ProgressLedger: Micro-loop self-reflection ledger inspired by Microsoft Magentic-One.

Answers the five core metacognitive self-reflection questions per iteration:
1. Is the task completed?
2. Is the agent stuck or looping?
3. Has substantive progress been made in this step?
4. Who is the next worker or what is the next core action?
5. What is the specific instruction or remediation?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class ProgressLedger:
    is_task_completed: bool = False
    is_stuck_or_looping: bool = False
    has_substantive_progress: bool = False
    next_worker_or_action: str = ""
    instruction_or_remediation: str = ""
    step_index: int = 0
    updated_at: str = ""
    evidence: str = ""

    def update(
        self,
        *,
        is_task_completed: bool | None = None,
        is_stuck_or_looping: bool | None = None,
        has_substantive_progress: bool | None = None,
        next_worker_or_action: str | None = None,
        instruction_or_remediation: str | None = None,
        step_index: int | None = None,
        evidence: str | None = None,
    ) -> None:
        if is_task_completed is not None:
            self.is_task_completed = is_task_completed
        if is_stuck_or_looping is not None:
            self.is_stuck_or_looping = is_stuck_or_looping
        if has_substantive_progress is not None:
            self.has_substantive_progress = has_substantive_progress
        if next_worker_or_action is not None:
            self.next_worker_or_action = next_worker_or_action.strip()
        if instruction_or_remediation is not None:
            self.instruction_or_remediation = instruction_or_remediation.strip()
        if step_index is not None:
            self.step_index = step_index
        else:
            self.step_index += 1
        if evidence is not None:
            self.evidence = evidence.strip()
        self.updated_at = _utc_now_iso()

    def render_markdown(self) -> str:
        completed_str = "YES" if self.is_task_completed else "NO"
        stuck_str = "YES (Intervention Needed)" if self.is_stuck_or_looping else "NO (Healthy)"
        progress_str = "YES" if self.has_substantive_progress else "NO"

        lines: list[str] = [
            "<progress_ledger>",
            f"### Progress Assessment (Step {self.step_index}" + (f" @ {self.updated_at}" if self.updated_at else "") + ")",
            f"- **Task Completed**: {completed_str}",
            f"- **Stuck or Looping**: {stuck_str}",
            f"- **Substantive Progress**: {progress_str}",
            f"- **Next Worker / Action**: {self.next_worker_or_action or 'N/A'}",
            f"- **Instruction / Remediation**: {self.instruction_or_remediation or 'N/A'}",
        ]
        if self.evidence:
            lines.append(f"- **Evidence / Reasoning**: {self.evidence}")
        lines.append("</progress_ledger>")
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProgressLedger:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            is_task_completed=bool(data.get("is_task_completed", False)),
            is_stuck_or_looping=bool(data.get("is_stuck_or_looping", False)),
            has_substantive_progress=bool(data.get("has_substantive_progress", False)),
            next_worker_or_action=str(data.get("next_worker_or_action", "")),
            instruction_or_remediation=str(data.get("instruction_or_remediation", "")),
            step_index=int(data.get("step_index", 0)),
            updated_at=str(data.get("updated_at", "")),
            evidence=str(data.get("evidence", "")),
        )
