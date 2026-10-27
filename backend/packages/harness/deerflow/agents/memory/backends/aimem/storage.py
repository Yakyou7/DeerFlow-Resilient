"""Markdown storage engine for Aimem.

Treats human-readable Markdown files as the Single Source of Truth.
Supports loading, formatting, updating, contradiction resolution, and parsing human edits.
"""

from __future__ import annotations

import datetime
import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AimemConfig

# Supports:
# - [fact_id] content <!-- updated: ... | accessed: ... | count: N -->
# - [fact_id] content <!-- updated: ... -->
# - [fact_id] content
_FACT_LINE_REGEX = re.compile(r"^\s*-\s*(?:\[(?P<id>[a-zA-Z0-9_-]+)\])?\s*(?P<content>.+?)(?:\s*<!--\s*(?P<meta>[^>]+)\s*-->)?\s*$")

# Common topic prefixes used for detecting superseding facts
_SUPERSEDING_PREFIXES = [
    re.compile(r"^(?:my\s+)?favorite\s+(?P<topic>\w+)\s+is\s+", re.IGNORECASE),
    re.compile(r"^(?:user\s+)?prefers?\s+(?P<topic>\w+)\s+(?:over|instead\s+of)?", re.IGNORECASE),
    re.compile(r"^(?:my\s+)?preferred\s+(?P<topic>\w+)\s+is\s+", re.IGNORECASE),
    re.compile(r"^(?:user\s+)?timezone\s+is\s+", re.IGNORECASE),
    re.compile(r"^项目(?:采用|使用|依赖)\s*(?P<topic>\w+)", re.IGNORECASE),
]


def _extract_topic_key(content: str) -> str | None:
    """Extract a canonical topic key to identify contradictory / superseding facts."""
    norm = content.strip().lower()
    for pattern in _SUPERSEDING_PREFIXES:
        m = pattern.search(norm)
        if m:
            topic = m.groupdict().get("topic")
            return f"{pattern.pattern[:15]}:{topic or 'general'}"
    return None


@dataclass
class MemoryFact:
    """A discrete unit of stored memory."""

    id: str
    content: str
    category: str = "profile"
    confidence: float = 1.0
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0
    last_accessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
        }


class MarkdownStorage:
    """Manages reading and writing structured facts to/from Markdown files."""

    def __init__(self, config: AimemConfig) -> None:
        self.config = config
        base = Path(config.storage_path) if config.storage_path else Path(".deer-flow")
        self.root_dir = base / "aimem" / "users"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: str) -> Path:
        safe_user = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", user_id or "default")
        path = self.root_dir / safe_user
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _file_for_category(self, user_dir: Path, category: str) -> Path:
        cat = (category or "").lower()
        if "entity" in cat or "knowledge" in cat:
            return user_dir / self.config.entities_filename
        if "archive" in cat or "history" in cat or "decision" in cat:
            return user_dir / self.config.archive_filename
        return user_dir / self.config.profile_filename

    def _header_for_category(self, category: str) -> str:
        cat = (category or "").lower()
        if "entity" in cat or "knowledge" in cat:
            return "# Project Entities & Knowledge\n\n"
        if "archive" in cat or "history" in cat or "decision" in cat:
            return "# Project Decisions & Archived Context\n\n"
        return "# User Profile & Preferences\n\n"

    def parse_markdown(self, text: str, default_category: str) -> list[MemoryFact]:
        """Parse facts from markdown content, preserving IDs, timestamps, and access counts."""
        facts: list[MemoryFact] = []
        now = datetime.datetime.now(datetime.UTC).isoformat()

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            match = _FACT_LINE_REGEX.match(line)
            if not match:
                continue
            content = match.group("content").strip()
            if not content:
                continue

            fact_id = match.group("id")
            if not fact_id:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
                fact_id = f"fact_{digest}"

            updated = now
            accessed = ""
            count = 0

            meta_str = match.group("meta") or ""
            if meta_str:
                for part in meta_str.split("|"):
                    part = part.strip()
                    if part.startswith("updated:"):
                        updated = part.split(":", 1)[1].strip()
                    elif part.startswith("accessed:"):
                        accessed = part.split(":", 1)[1].strip()
                    elif part.startswith("count:"):
                        try:
                            count = int(part.split(":", 1)[1].strip())
                        except ValueError:
                            count = 0

            facts.append(
                MemoryFact(
                    id=fact_id,
                    content=content,
                    category=default_category,
                    confidence=1.0,
                    created_at=updated,
                    updated_at=updated,
                    access_count=count,
                    last_accessed_at=accessed or updated,
                )
            )
        return facts

    def format_markdown(self, facts: list[MemoryFact], category: str) -> str:
        """Render facts list into clean Markdown with comment metadata."""
        lines = [self._header_for_category(category)]
        for f in facts:
            meta_parts = []
            if f.updated_at:
                meta_parts.append(f"updated: {f.updated_at}")
            if f.last_accessed_at:
                meta_parts.append(f"accessed: {f.last_accessed_at}")
            if f.access_count > 0:
                meta_parts.append(f"count: {f.access_count}")

            meta_tag = f" <!-- {' | '.join(meta_parts)} -->" if meta_parts else ""
            lines.append(f"- [{f.id}] {f.content}{meta_tag}")
        lines.append("")
        return "\n".join(lines)

    def load_user_facts(self, user_id: str, category: str | None = None) -> list[MemoryFact]:
        """Load all facts for a user, optionally filtered by category."""
        user_dir = self._user_dir(user_id)
        categories = [category] if category else ["profile", "entity", "archive"]
        all_facts: list[MemoryFact] = []
        for cat in categories:
            file_path = self._file_for_category(user_dir, cat)
            if file_path.is_file():
                try:
                    text = file_path.read_text(encoding="utf-8")
                    all_facts.extend(self.parse_markdown(text, default_category=cat))
                except Exception:
                    pass
        return all_facts

    def save_facts_for_category(self, user_id: str, category: str, facts: list[MemoryFact]) -> None:
        """Write facts of a specific category to its corresponding Markdown file."""
        user_dir = self._user_dir(user_id)
        file_path = self._file_for_category(user_dir, category)
        md = self.format_markdown(facts, category)
        file_path.write_text(md, encoding="utf-8")

    def upsert_fact(self, user_id: str, fact: MemoryFact) -> tuple[MemoryFact, str | None]:
        """Insert or update a fact with contradiction / superseding resolution.

        Returns (saved_fact, superseded_fact_id).
        """
        user_dir = self._user_dir(user_id)
        category = fact.category or "profile"
        file_path = self._file_for_category(user_dir, category)
        existing: list[MemoryFact] = []
        if file_path.is_file():
            try:
                existing = self.parse_markdown(file_path.read_text(encoding="utf-8"), category)
            except Exception:
                existing = []

        now = datetime.datetime.now(datetime.UTC).isoformat()
        if not fact.id:
            fact.id = f"fact_{uuid.uuid4().hex[:8]}"
        if not fact.created_at:
            fact.created_at = now
        if not fact.updated_at:
            fact.updated_at = now
        if not fact.last_accessed_at:
            fact.last_accessed_at = fact.updated_at

        new_topic = _extract_topic_key(fact.content)
        superseded_id: str | None = None
        replaced = False
        new_list: list[MemoryFact] = []

        for item in existing:
            # 1. Exact ID or byte match
            if item.id == fact.id or item.content.strip().lower() == fact.content.strip().lower():
                fact.access_count = max(fact.access_count, item.access_count)
                new_list.append(fact)
                replaced = True
            # 2. Contradiction / Superseding match on topic prefix in the same category
            elif new_topic is not None and _extract_topic_key(item.content) == new_topic:
                superseded_id = item.id
                fact.access_count = max(fact.access_count, item.access_count + 1)
                new_list.append(fact)
                replaced = True
            else:
                new_list.append(item)

        if not replaced:
            new_list.append(fact)

        # Automatic staleness pruning if category facts exceed limit
        max_facts = self.config.max_facts_per_category
        if len(new_list) > max_facts:
            # Keep highest access count and most recently updated
            new_list.sort(key=lambda x: (x.access_count, x.updated_at), reverse=True)
            new_list = new_list[:max_facts]

        self.save_facts_for_category(user_id, category, new_list)
        return fact, superseded_id

    def touch_facts(self, user_id: str, fact_ids: list[str]) -> None:
        """Increment access count and touch timestamp for retrieved facts."""
        if not fact_ids:
            return
        id_set = set(fact_ids)
        now = datetime.datetime.now(datetime.UTC).isoformat()
        for cat in ["profile", "entity", "archive"]:
            facts = self.load_user_facts(user_id, category=cat)
            changed = False
            for f in facts:
                if f.id in id_set:
                    f.access_count += 1
                    f.last_accessed_at = now
                    changed = True
            if changed:
                self.save_facts_for_category(user_id, cat, facts)

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        """Delete a fact by ID across all categories for the user."""
        user_dir = self._user_dir(user_id)
        deleted = False
        for cat in ["profile", "entity", "archive"]:
            file_path = self._file_for_category(user_dir, cat)
            if not file_path.is_file():
                continue
            existing = self.parse_markdown(file_path.read_text(encoding="utf-8"), cat)
            remaining = [f for f in existing if f.id != fact_id]
            if len(remaining) < len(existing):
                self.save_facts_for_category(user_id, cat, remaining)
                deleted = True
        return deleted

    def clear_user(self, user_id: str) -> None:
        """Clear all memory files for a user."""
        user_dir = self._user_dir(user_id)
        for cat in ["profile", "entity", "archive"]:
            f = self._file_for_category(user_dir, cat)
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
