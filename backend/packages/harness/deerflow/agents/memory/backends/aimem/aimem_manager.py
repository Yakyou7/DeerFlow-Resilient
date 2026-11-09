"""Aimem hierarchical memory manager implementation.

Implements MemoryManager with human-readable Markdown storage as Source of Truth
and local SQLite FTS5 with BM25 + Recency Decay hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import html
import logging
import re
import threading
import time
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
    re.compile(r"(?:please remember that|remember that|please note that|note that|please note|remember)\s+(?:that\s+)?(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"(?P<fact>(?:user|i)\s+(?:prefers?|likes?|loves?|dislikes?|hates?)\s+.+)", re.IGNORECASE),
    re.compile(r"(?P<fact>(?:my|user's)\s+(?:favorite|preferred|current)\s+.+)", re.IGNORECASE),
    re.compile(r"(?:always|never)\s+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"(?:请记住|记住|记录一下|我的偏好是|我喜欢|我不喜欢|我的习惯是)\s*[:：]?\s*(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"项目(?:使用|采用|配置|规范|依赖)\s*[:：]?\s*(?P<fact>.+)", re.IGNORECASE),
]

_TRIVIAL_WORDS = {"ok", "okay", "thanks", "thank you", "thx", "got it", "sure", "好的", "收到", "知道了", "谢谢", "嗯", "对", "行", "明白"}


@dataclass
class _RawTrace:
    """Buffered raw conversation trace awaiting sleep-time crystallization."""

    user_id: str
    thread_id: str
    messages: list[Any]
    agent_name: str | None = None
    trace_id: str | None = None
    timestamp: float = 0.0


class AimemMemoryManager(MemoryManager):
    """Hierarchical memory backend with Markdown persistence and SQLite FTS5 index."""

    supports_search: ClassVar[bool] = True
    supports_agent_scoped_management: ClassVar[bool] = True

    _config: AimemConfig = PrivateAttr(default=None)
    _storage: MarkdownStorage = PrivateAttr(default=None)
    _indexer: FtsIndexer = PrivateAttr(default=None)
    _pending_traces: list[_RawTrace] = PrivateAttr(default_factory=list)
    _sleep_timer: threading.Timer | None = PrivateAttr(default=None)
    _sleep_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context: Any) -> None:
        self._config = AimemConfig.from_backend_config(self.backend_config)
        self._storage = MarkdownStorage(self._config)
        self._indexer = FtsIndexer(self._config)
        self._pending_traces = []
        self._sleep_timer = None
        self._sleep_lock = threading.Lock()

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

    @property
    def pending_traces_count(self) -> int:
        """Return the number of buffered raw conversation traces pending crystallization."""
        with self._sleep_lock:
            return len(self._pending_traces)

    # ── Write Operations (Tier 1) ──────────────────────────────────────────

    def _extract_and_persist(self, uid: str, messages: list[Any]) -> tuple[int, int]:
        """Synchronously extract preference/entity facts from messages and persist to Markdown & FTS.

        Returns (crystallized_count, superseded_count).
        """
        if not messages:
            return 0, 0

        crystallized_count = 0
        superseded_count = 0

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
                    superseded_count += 1
                    logger.info("Aimem superseded outdated fact: [%s]", superseded_id)

                self._indexer.upsert_fact(uid, saved_fact)
                crystallized_count += 1
                logger.info("Aimem stored fact: [%s] (%s) %s", saved_fact.id, saved_fact.category, saved_fact.content)

        return crystallized_count, superseded_count

    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Extract and persist facts from conversation messages with sleep-time compute support."""
        uid = user_id or "default"
        if not messages:
            return

        if not self._config.sleep_time_compute_enabled:
            self._extract_and_persist(uid, messages)
            return

        trace = _RawTrace(
            user_id=uid,
            thread_id=thread_id,
            messages=list(messages),
            agent_name=agent_name,
            trace_id=trace_id,
            timestamp=time.time(),
        )

        with self._sleep_lock:
            self._pending_traces.append(trace)
            while len(self._pending_traces) > self._config.max_pending_traces:
                self._pending_traces.pop(0)

            if self._sleep_timer is not None:
                self._sleep_timer.cancel()
                self._sleep_timer = None

            debounce = max(0.01, self._config.sleep_idle_debounce_seconds)
            self._sleep_timer = threading.Timer(debounce, self._on_sleep_timer_trigger)
            self._sleep_timer.daemon = True
            self._sleep_timer.start()

    def _on_sleep_timer_trigger(self) -> None:
        """Triggered when agent conversation experiences idle period."""
        try:
            self.sleep_time_compute()
        except Exception:
            logger.exception("Error during Aimem sleep_time_compute background crystallization")

    def sleep_time_compute(self, user_id: str | None = None) -> dict[str, Any]:
        """Drain buffered conversation traces and crystallize into long-term memory.

        Inspired by Letta / Sleep-Time Compute: separates raw collection from
        cognitive extraction and deduplication/superseding.
        """
        with self._sleep_lock:
            if user_id is not None:
                traces = [t for t in self._pending_traces if t.user_id == user_id]
                self._pending_traces = [t for t in self._pending_traces if t.user_id != user_id]
                if not self._pending_traces and self._sleep_timer is not None:
                    self._sleep_timer.cancel()
                    self._sleep_timer = None
            else:
                traces = list(self._pending_traces)
                self._pending_traces.clear()
                if self._sleep_timer is not None:
                    self._sleep_timer.cancel()
                    self._sleep_timer = None

        if not traces:
            return {"crystallized_facts": 0, "superseded_facts": 0, "processed_traces": 0}

        total_crystallized = 0
        total_superseded = 0
        for trace in traces:
            c, s = self._extract_and_persist(trace.user_id, trace.messages)
            total_crystallized += c
            total_superseded += s

        logger.info(
            "Aimem sleep_time_compute completed: processed %d traces, crystallized %d facts, superseded %d facts",
            len(traces),
            total_crystallized,
            total_superseded,
        )
        return {
            "crystallized_facts": total_crystallized,
            "superseded_facts": total_superseded,
            "processed_traces": len(traces),
        }

    def _ensure_crystallized(self, user_id: str) -> None:
        """Guarantee Read-Your-Own-Writes consistency by draining pending traces for user_id."""
        with self._sleep_lock:
            has_pending = any(t.user_id == user_id for t in self._pending_traces)
        if has_pending:
            self.sleep_time_compute(user_id=user_id)

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
        self._ensure_crystallized(uid)
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

        def _escape_context(val: str) -> str:
            return html.escape(str(val), quote=False)

        # Format context injection block (outer <memory> tag is added by prompt builder)
        sections: list[str] = ["User Profile & Long-Term Context:"]
        for p in profiles:
            sections.append(f"- (profile) {_escape_context(p.get('content', ''))}")
        for e in entities:
            sections.append(f"- (entity) {_escape_context(e.get('content', ''))}")
        for a in archives:
            sections.append(f"- (archive) {_escape_context(a.get('content', ''))}")

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
        self._ensure_crystallized(uid)
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
        self._ensure_crystallized(uid)
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
        with self._sleep_lock:
            self._pending_traces = [t for t in self._pending_traces if t.user_id != uid]
            if not self._pending_traces and self._sleep_timer is not None:
                self._sleep_timer.cancel()
                self._sleep_timer = None
        self._storage.clear_user(uid)
        self._indexer.sync_user_facts(uid, [])
        return self.get_memory(user_id=uid, agent_name=agent_name)

    def delete_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Delete memory for the user (alias for clear_memory)."""
        return self.clear_memory(user_id=user_id, agent_name=agent_name)

    def export_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Export memory in DeerMem-compatible schema."""
        return self.get_memory(user_id=user_id, agent_name=agent_name)

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
        category: str = "context",
        confidence: float = 1.0,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Create a fact directly (supports frontend UI Add Fact button)."""
        uid = user_id or "default"
        cat = "profile" if category == "context" else (category or "profile")
        fact = MemoryFact(
            id="",
            content=content.strip(),
            category=cat,
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

    def shutdown_flush(self, timeout: float = 5.0) -> bool:
        """Best-effort bounded drain of pending updates on graceful shutdown."""
        with self._sleep_lock:
            if self._sleep_timer is not None:
                self._sleep_timer.cancel()
                self._sleep_timer = None
        self.sleep_time_compute()
        return True

    def close(self) -> None:
        """Release backend resources and flush pending sleep compute traces."""
        self.shutdown_flush(5.0)
        if self._indexer is not None:
            self._indexer.close()
        super().close()
