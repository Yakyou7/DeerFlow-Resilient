"""Aimem hierarchical memory manager implementation.

Implements MemoryManager with human-readable Markdown storage as Source of Truth
and local SQLite FTS5 with BM25 + Recency Decay hybrid retrieval.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

# The ONLY permitted import from deerflow in this backend
from deerflow.agents.memory.manager import MemoryManager

from .config import AimemConfig
from .indexer import FtsIndexer
from .storage import MarkdownStorage, MemoryFact

logger = logging.getLogger(__name__)

# Heuristic patterns for rule-based preference/fact detection
_PREFERENCE_PATTERNS = [
    re.compile(r"(?:remember that|note that|please note)\s+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"i (?:prefer|like|love|dislike|hate)\s+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"my (?:favorite|preferred|current)\s+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"(?:always|never)\s+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"(?:请记住|记住|记录一下|我的偏好是|我喜欢|我不喜欢|我的习惯是)\s*[:：]?\s*(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"项目(?:使用|采用|配置|规范|依赖)\s*[:：]?\s*(?P<fact>.+)", re.IGNORECASE),
]

_TRIVIAL_WORDS = {"ok", "okay", "thanks", "thank you", "thx", "got it", "sure", "好的", "收到", "知道了", "谢谢", "嗯", "对", "行", "明白"}


class AimemMemoryManager(MemoryManager):
    """Hierarchical memory backend with Markdown persistence and SQLite FTS5 index."""

    supports_search: ClassVar[bool] = True
    supports_agent_scoped_management: ClassVar[bool] = True

    _config: AimemConfig = PrivateAttr(default=None)
    _storage: MarkdownStorage = PrivateAttr(default=None)
    _indexer: FtsIndexer = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._config = AimemConfig.from_backend_config(self.backend_config)
        self._storage = MarkdownStorage(self._config)
        self._indexer = FtsIndexer(self._config)

    @classmethod
    def from_config(
        cls,
        backend_config: dict[str, Any] | None = None,
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> AimemMemoryManager:
        cfg = backend_config or {}
        merged = {**cfg, **host_hooks}
        return cls(backend_config=merged, mode=mode)

    # ── Write Operations (Tier 1) ──────────────────────────────────────────

    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Extract and persist facts from conversation messages with superseding support."""
        uid = user_id or "default"
        if not messages:
            return

        for msg in messages:
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue

            stripped = content.strip()
            # Gating filter 1: Skip short or trivial messages
            if len(stripped) < self._config.min_message_chars:
                continue
            if stripped.lower() in _TRIVIAL_WORDS:
                continue

            # Gating filter 2: Match high-confidence preference / fact patterns
            matched_fact = None
            category = "profile"
            for pattern in _PREFERENCE_PATTERNS:
                m = pattern.search(stripped)
                if m:
                    matched_fact = m.group("fact").strip()
                    if "项目" in pattern.pattern or "project" in matched_fact.lower():
                        category = "entity"
                    break

            if matched_fact:
                fact = MemoryFact(
                    id="",
                    content=matched_fact,
                    category=category,
                    confidence=1.0,
                )
                saved_fact, superseded_id = self._storage.upsert_fact(uid, fact)
                if superseded_id:
                    self._indexer.delete_fact(superseded_id)
                    logger.info("Aimem superseded outdated fact: [%s]", superseded_id)

                self._indexer.upsert_fact(uid, saved_fact)
                logger.info("Aimem stored fact: [%s] (%s) %s", saved_fact.id, saved_fact.category, saved_fact.content)

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.add(thread_id, messages, agent_name=agent_name, user_id=user_id)

    # ── Read Operations (Tier 1) ───────────────────────────────────────────

    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
        query: str | None = None,
    ) -> str:
        """Generate tiered context memory injection with budget allocation and recency decay."""
        uid = user_id or "default"
        max_facts = self._config.max_context_facts
        if max_facts <= 0:
            return ""

        # Tiered budget allocation
        profile_budget = max(1, int(round(max_facts * self._config.profile_budget_ratio)))
        entity_budget = max(1, int(round(max_facts * self._config.entity_budget_ratio)))
        archive_budget = max(0, max_facts - profile_budget - entity_budget)

        # 1. Profile facts (User habits & hard preferences)
        profiles: list[dict[str, Any]] = []
        if query and self._config.enable_fts:
            profiles = self._indexer.search(query, top_k=profile_budget, user_id=uid, category="profile")
        if len(profiles) < profile_budget:
            all_profiles = self._storage.load_user_facts(uid, category="profile")
            seen = {p["id"] for p in profiles}
            for p in all_profiles:
                if len(profiles) >= profile_budget:
                    break
                if p.id not in seen:
                    profiles.append(p.to_dict())
                    seen.add(p.id)

        # 2. Entity & Project Knowledge facts
        entities: list[dict[str, Any]] = []
        if query and self._config.enable_fts:
            entities = self._indexer.search(query, top_k=entity_budget, user_id=uid, category="entity")
        if len(entities) < entity_budget:
            all_entities = self._storage.load_user_facts(uid, category="entity")
            seen = {e["id"] for e in entities}
            for e in all_entities:
                if len(entities) >= entity_budget:
                    break
                if e.id not in seen:
                    entities.append(e.to_dict())
                    seen.add(e.id)

        # 3. Archive & Historical Decisions
        archives: list[dict[str, Any]] = []
        if archive_budget > 0:
            if query and self._config.enable_fts:
                archives = self._indexer.search(query, top_k=archive_budget, user_id=uid, category="archive")
            if len(archives) < archive_budget:
                all_archives = self._storage.load_user_facts(uid, category="archive")
                seen = {a["id"] for a in archives}
                for a in all_archives:
                    if len(archives) >= archive_budget:
                        break
                    if a.id not in seen:
                        archives.append(a.to_dict())
                        seen.add(a.id)

        # Touch accessed facts so their access count and last_accessed_at update
        accessed_ids = [f["id"] for f in [*profiles, *entities, *archives]]
        if accessed_ids:
            self._storage.touch_facts(uid, accessed_ids)
            self._indexer.touch_facts(accessed_ids)

        if not profiles and not entities and not archives:
            return ""

        # Format context injection block (outer <memory> tag is added by prompt builder)
        sections: list[str] = ["User Profile & Long-Term Context:"]
        for p in profiles:
            sections.append(f"- (profile) {p.get('content', '')}")
        for e in entities:
            sections.append(f"- (entity) {e.get('content', '')}")
        for a in archives:
            sections.append(f"- (archive) {a.get('content', '')}")

        return "\n".join(sections)

    # ── Search & Retrieval (Tier 2) ────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memory facts using hybrid ranking (BM25 + recency decay + access frequency)."""
        uid = user_id or "default"
        if not self._config.enable_fts:
            facts = self._storage.load_user_facts(uid, category=category)
            q = query.lower()
            matched = [f.to_dict() for f in facts if q in f.content.lower()]
            return matched[:top_k]

        results = self._indexer.search(query, top_k=top_k, user_id=uid, category=category)
        if results:
            retrieved_ids = [r["id"] for r in results]
            self._storage.touch_facts(uid, retrieved_ids)
            self._indexer.touch_facts(retrieved_ids)

        return results

    # ── Management Operations (Tier 2) ─────────────────────────────────────

    def get_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Return memory in DeerMem-compatible schema."""
        uid = user_id or "default"
        facts = self._storage.load_user_facts(uid)
        now = datetime.datetime.now(datetime.UTC).isoformat()
        return {
            "version": "1.0",
            "lastUpdated": now,
            "user": uid,
            "history": [],
            "facts": [f.to_dict() for f in facts],
        }

    def clear_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Clear memory for the user."""
        uid = user_id or "default"
        self._storage.clear_user(uid)
        self._indexer.sync_user_facts(uid, [])
        return self.get_memory(user_id=uid, agent_name=agent_name)

    def import_memory(
        self,
        memory_data: dict[str, Any],
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Import facts from a dict into Markdown and FTS5."""
        uid = user_id or "default"
        raw_facts = memory_data.get("facts", [])
        facts_to_save: list[MemoryFact] = []
        for rf in raw_facts:
            if isinstance(rf, dict) and "content" in rf:
                facts_to_save.append(
                    MemoryFact(
                        id=str(rf.get("id", "")),
                        content=str(rf.get("content", "")),
                        category=str(rf.get("category", "profile")),
                        confidence=float(rf.get("confidence", 1.0)),
                    )
                )
        for f in facts_to_save:
            saved, superseded_id = self._storage.upsert_fact(uid, f)
            if superseded_id:
                self._indexer.delete_fact(superseded_id)
            self._indexer.upsert_fact(uid, saved)
        return self.get_memory(user_id=uid, agent_name=agent_name)

    # ── Fact CRUD & Hooks (Tier 3) ─────────────────────────────────────────

    def create_fact(
        self,
        content: str,
        category: str = "profile",
        confidence: float = 1.0,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Create a fact directly (supports frontend UI Add Fact button)."""
        uid = user_id or "default"
        fact = MemoryFact(
            id="",
            content=content.strip(),
            category=category,
            confidence=confidence,
        )
        saved, superseded_id = self._storage.upsert_fact(uid, fact)
        if superseded_id:
            self._indexer.delete_fact(superseded_id)
        self._indexer.upsert_fact(uid, saved)
        return self.get_memory(user_id=uid, agent_name=agent_name), saved.id

    def delete_fact(
        self,
        fact_id: str,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete a fact by ID (supports frontend UI Delete Fact button)."""
        uid = user_id or "default"
        self._storage.delete_fact(uid, fact_id)
        self._indexer.delete_fact(fact_id)
        return self.get_memory(user_id=uid, agent_name=agent_name)

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing fact by ID."""
        uid = user_id or "default"
        facts = self._storage.load_user_facts(uid)
        target = next((f for f in facts if f.id == fact_id), None)
        if target:
            if content is not None:
                target.content = content.strip()
            if category is not None:
                target.category = category
            if confidence is not None:
                target.confidence = confidence
            saved, superseded_id = self._storage.upsert_fact(uid, target)
            if superseded_id and superseded_id != target.id:
                self._indexer.delete_fact(superseded_id)
            self._indexer.upsert_fact(uid, saved)
        return self.get_memory(user_id=uid, agent_name=agent_name)

    def reload_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Resync FTS5 index from on-disk Markdown files (e.g. after manual human edits)."""
        uid = user_id or "default"
        facts = self._storage.load_user_facts(uid)
        self._indexer.sync_user_facts(uid, facts)
        logger.info("Aimem reloaded and re-indexed %d facts for user %s", len(facts), uid)
        return self.get_memory(user_id=uid, agent_name=agent_name)

    def shutdown_flush(self, timeout: float) -> bool:
        return True
