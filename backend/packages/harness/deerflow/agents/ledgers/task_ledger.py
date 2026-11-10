"""TaskLedger: Macro-level task and facts ledger inspired by Microsoft Magentic-One.

Maintains structured epistemic state for long-horizon agent tasks:
- Given facts: Known immutable truths provided by user/system
- Facts to look up: External information needing tool/search retrieval
- Facts to derive: Analytical conclusions needing computation or deduction
- Guesses: Working hypotheses with explicit confirmation/refutation
- Milestones: Linear or branching execution steps with verified statuses
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MilestoneStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]
GuessStatus = Literal["unverified", "confirmed", "refuted"]


@dataclass
class TaskMilestone:
    name: str
    status: MilestoneStatus = "pending"
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMilestone:
        return cls(
            name=str(data.get("name", "")),
            status=data.get("status", "pending"),
            details=str(data.get("details", "")),
        )


@dataclass
class TaskLedger:
    task: str = ""
    given_facts: list[str] = field(default_factory=list)
    facts_to_lookup: list[dict[str, Any]] = field(default_factory=list)
    facts_to_derive: list[dict[str, Any]] = field(default_factory=list)
    guesses: list[dict[str, Any]] = field(default_factory=list)
    milestones: list[TaskMilestone] = field(default_factory=list)

    def add_given_fact(self, fact: str) -> None:
        cleaned = fact.strip()
        if cleaned and cleaned not in self.given_facts:
            self.given_facts.append(cleaned)

    def add_fact_to_lookup(self, fact: str) -> None:
        cleaned = fact.strip()
        if cleaned and not any(item["fact"] == cleaned for item in self.facts_to_lookup):
            self.facts_to_lookup.append({"fact": cleaned, "resolved": False, "answer": ""})

    def resolve_fact_lookup(self, fact_or_index: str | int, answer: str) -> bool:
        if isinstance(fact_or_index, int):
            if 0 <= fact_or_index < len(self.facts_to_lookup):
                self.facts_to_lookup[fact_or_index]["resolved"] = True
                self.facts_to_lookup[fact_or_index]["answer"] = answer.strip()
                return True
            return False
        for item in self.facts_to_lookup:
            if item["fact"] == fact_or_index:
                item["resolved"] = True
                item["answer"] = answer.strip()
                return True
        return False

    def add_fact_to_derive(self, fact: str) -> None:
        cleaned = fact.strip()
        if cleaned and not any(item["fact"] == cleaned for item in self.facts_to_derive):
            self.facts_to_derive.append({"fact": cleaned, "derived": False, "result": ""})

    def resolve_fact_derive(self, fact_or_index: str | int, result: str) -> bool:
        if isinstance(fact_or_index, int):
            if 0 <= fact_or_index < len(self.facts_to_derive):
                self.facts_to_derive[fact_or_index]["derived"] = True
                self.facts_to_derive[fact_or_index]["result"] = result.strip()
                return True
            return False
        for item in self.facts_to_derive:
            if item["fact"] == fact_or_index:
                item["derived"] = True
                item["result"] = result.strip()
                return True
        return False

    def add_guess(self, guess: str) -> None:
        cleaned = guess.strip()
        if cleaned and not any(item["guess"] == cleaned for item in self.guesses):
            self.guesses.append({"guess": cleaned, "status": "unverified", "evidence": ""})

    def evaluate_guess(self, guess_or_index: str | int, confirmed: bool, evidence: str = "") -> bool:
        status: GuessStatus = "confirmed" if confirmed else "refuted"
        if isinstance(guess_or_index, int):
            if 0 <= guess_or_index < len(self.guesses):
                self.guesses[guess_or_index]["status"] = status
                self.guesses[guess_or_index]["evidence"] = evidence.strip()
                return True
            return False
        for item in self.guesses:
            if item["guess"] == guess_or_index:
                item["status"] = status
                item["evidence"] = evidence.strip()
                return True
        return False

    def add_milestone(self, name: str, status: MilestoneStatus = "pending", details: str = "") -> None:
        cleaned = name.strip()
        if cleaned and not any(m.name == cleaned for m in self.milestones):
            self.milestones.append(TaskMilestone(name=cleaned, status=status, details=details.strip()))

    def update_milestone(self, name_or_index: str | int, status: MilestoneStatus, details: str = "") -> bool:
        if isinstance(name_or_index, int):
            if 0 <= name_or_index < len(self.milestones):
                self.milestones[name_or_index].status = status
                if details:
                    self.milestones[name_or_index].details = details.strip()
                return True
            return False
        for m in self.milestones:
            if m.name == name_or_index:
                m.status = status
                if details:
                    m.details = details.strip()
                return True
        return False

    def render_markdown(self) -> str:
        lines: list[str] = ["<task_ledger>"]
        if self.task:
            lines.append(f"### Current Task\n{self.task}\n")

        if self.given_facts:
            lines.append("### Given Facts")
            for fact in self.given_facts:
                lines.append(f"- {fact}")
            lines.append("")

        if self.facts_to_lookup:
            lines.append("### Facts to Look Up")
            for item in self.facts_to_lookup:
                box = "[x]" if item["resolved"] else "[ ]"
                ans = f" -> {item['answer']}" if item["resolved"] and item["answer"] else ""
                lines.append(f"- {box} {item['fact']}{ans}")
            lines.append("")

        if self.facts_to_derive:
            lines.append("### Facts to Derive")
            for item in self.facts_to_derive:
                box = "[x]" if item["derived"] else "[ ]"
                res = f" -> {item['result']}" if item["derived"] and item["result"] else ""
                lines.append(f"- {box} {item['fact']}{res}")
            lines.append("")

        if self.guesses:
            lines.append("### Working Guesses & Hypotheses")
            for item in self.guesses:
                st = item["status"]
                mark = "[?]" if st == "unverified" else ("[✓]" if st == "confirmed" else "[✗]")
                ev = f" (evidence: {item['evidence']})" if item["evidence"] else ""
                lines.append(f"- {mark} {item['guess']} [{st}]{ev}")
            lines.append("")

        if self.milestones:
            lines.append("### Milestones")
            for idx, m in enumerate(self.milestones, start=1):
                box = "[x]" if m.status == "completed" else ("[/]" if m.status == "in_progress" else "[ ]")
                det = f" ({m.details})" if m.details else ""
                lines.append(f"- {box} {idx}. {m.name} [{m.status}]{det}")
            lines.append("")

        lines.append("</task_ledger>")
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "given_facts": list(self.given_facts),
            "facts_to_lookup": [dict(x) for x in self.facts_to_lookup],
            "facts_to_derive": [dict(x) for x in self.facts_to_derive],
            "guesses": [dict(x) for x in self.guesses],
            "milestones": [m.to_dict() for m in self.milestones],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaskLedger:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            task=str(data.get("task", "")),
            given_facts=list(data.get("given_facts", [])),
            facts_to_lookup=[dict(x) for x in data.get("facts_to_lookup", []) if isinstance(x, dict)],
            facts_to_derive=[dict(x) for x in data.get("facts_to_derive", []) if isinstance(x, dict)],
            guesses=[dict(x) for x in data.get("guesses", []) if isinstance(x, dict)],
            milestones=[TaskMilestone.from_dict(m) for m in data.get("milestones", []) if isinstance(m, dict)],
        )
