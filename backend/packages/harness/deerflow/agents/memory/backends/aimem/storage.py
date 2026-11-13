"""Markdown storage engine for Aimem.

Treats human-readable Markdown files as the Single Source of Truth.
Supports loading, formatting, updating, contradiction resolution, and parsing human edits.
"""

from __future__ import annotations

import datetime
import hashlib
import re
import threading
import uuid
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import AimemConfig

# Supports:
# - [fact_id] content <!-- updated: ... | accessed: ... | count: N -->
# - [fact_id] content <!-- updated: ... -->
# - [fact_id] content
_FACT_LINE_REGEX = re.compile(r"^\s*-\s*(?:\[(?P<id>[a-zA-Z0-9_-]+)\])?\s*(?P<content>.+?)(?:\s*<!--\s*(?P<meta>[^>]+)\s*-->)?\s*$")

# Canonical topic domain mappings for detecting superseding facts
_SUPERSEDING_DOMAINS = [
    ("preference", re.compile(r"^(?:that\s+)?(?:my\s+)?favorite\s+(?P<topic>\w+)\s+is\s+", re.IGNORECASE)),
    ("preference", re.compile(r"^(?:that\s+)?(?:user\s+)?prefers?\s+(?P<topic>\w+)\s+(?:over|instead\s+of)?", re.IGNORECASE)),
    ("preference", re.compile(r"^(?:that\s+)?(?:my\s+)?preferred\s+(?P<topic>\w+)\s+is\s+", re.IGNORECASE)),
    ("timezone", re.compile(r"^(?:that\s+)?(?:user\s+)?timezone\s+is\s+", re.IGNORECASE)),
    ("config_kv", re.compile(r"^(?:the\s+)?(?:primary|default|current)\s+(?P<topic>\w+)\s+is\s+", re.IGNORECASE)),
    ("project_tech", re.compile(r"^项目(?:采用|使用|依赖)\s*(?P<topic>\w+)", re.IGNORECASE)),
    ("preference", re.compile(r"^(?:我的偏好是|我喜欢|我不喜欢|我的习惯是)\s*[:：]?\s*(?P<topic>\w+)", re.IGNORECASE)),
]


def _extract_topic_key(content: str) -> str | None:
    """Extract a canonical topic key to identify contradictory / superseding facts."""
    norm = content.strip().lower()
    for domain, pattern in _SUPERSEDING_DOMAINS:
        m = pattern.search(norm)
        if m:
            topic = m.groupdict().get("topic") or "general"
            return f"{domain}:{topic.strip().lower()}"
    return None


def _normalize_agent_name(agent_name: str | None) -> str:
    """Canonicalize an agent identifier for storage and scope checks."""
    return (agent_name or "").strip().lower()


def resolve_default_scope_role(agent_name: str | None, scope_role: str | None = None) -> str:
    """Resolve effective scope_role for a fact or bucket.

    Named agents default to ``private`` (owner-isolated); unscoped calls
    default to ``default``; explicit roles (``shared`` or a named RBAC role)
    are preserved lowercased.
    """
    explicit = (scope_role or "").strip().lower()
    if explicit:
        return explicit
    if _normalize_agent_name(agent_name):
        return "private"
    return "default"


def is_fact_authorized(
    fact_user_id: str,
    fact_agent_name: str,
    fact_scope_role: str,
    *,
    caller_user_id: str,
    caller_agent_name: str | None = None,
    caller_scope_role: str | None = None,
    caller_roles: Collection[str] | None = None,
) -> bool:
    """Symmetric scope authorization predicate shared by point-lookup and search paths.

    Enforces tenant isolation (``user_id``) plus role/agent ACL so ``get_by_id``
    and ``search`` never diverge on visibility boundaries.
    """
    if (fact_user_id or "default") != (caller_user_id or "default"):
        return False

    f_agent = _normalize_agent_name(fact_agent_name)
    f_role = resolve_default_scope_role(f_agent, fact_scope_role)
    c_agent = _normalize_agent_name(caller_agent_name)

    roles = {r.strip().lower() for r in (caller_roles or ()) if r and r.strip()}
    if caller_scope_role and caller_scope_role.strip():
        roles.add(caller_scope_role.strip().lower())

    if f_role == "shared":
        return True
    if f_role == "default":
        return c_agent == ""
    if f_role == "private":
        if f_agent:
            return c_agent == f_agent
        return c_agent == ""
    return f_role in roles


def _sanitize_meta_token(value: str) -> str:
    """Strip Markdown comment delimiters and pipe separators from metadata tokens."""
    cleaned = re.sub(r"[\r\n|]+", "_", str(value or "").strip())
    return cleaned.replace("<!--", "_").replace("-->", "_").strip()


def _normalize_for_dedup(content: str) -> str:
    """Normalize whitespace and trailing punctuation for Stage-2 deduplication."""
    collapsed = " ".join(content.strip().lower().split())
    return collapsed.rstrip(".,!?;:。，！？；：")


def _is_near_duplicate(existing_content: str, new_content: str) -> bool:
    """Detect non-contradictory exact or high-overlap re-statements (Stage-2 only)."""
    norm_a = _normalize_for_dedup(existing_content)
    norm_b = _normalize_for_dedup(new_content)
    if norm_a == norm_b:
        return True
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if len(tokens_a) >= 4 and len(tokens_b) >= 4:
        union_size = len(tokens_a | tokens_b)
        if union_size > 0 and (len(tokens_a & tokens_b) / union_size) >= 0.85:
            return True
    return False


@dataclass
class MemoryFact:
    """A discrete unit of stored memory with Bi-Temporal edge and provenance metadata."""

    id: str
    content: str
    category: str = "profile"
    confidence: float = 1.0
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0
    last_accessed_at: str = ""
    valid_from: str = ""
    invalidated_at: str = ""
    agent_name: str = ""
    scope_role: str = ""
    writer_id: str = ""
    parent_provenance_ids: list[str] = field(default_factory=list)
    supersedes_id: str = ""

    @property
    def is_active(self) -> bool:
        """Check if fact is currently active/valid (not invalidated)."""
        return not bool(self.invalidated_at)

    def is_valid_at(self, timestamp: str | None = None) -> bool:
        """Point-in-time temporal evaluation (Zep Graphiti style)."""
        if not timestamp:
            return self.is_active
        if self.valid_from and self.valid_from > timestamp:
            return False
        if self.invalidated_at and self.invalidated_at <= timestamp:
            return False
        return True

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
            "valid_from": self.valid_from,
            "invalidated_at": self.invalidated_at,
            "agent_name": self.agent_name,
            "scope_role": self.scope_role,
            "writer_id": self.writer_id,
            "parent_provenance_ids": list(self.parent_provenance_ids),
            "supersedes_id": self.supersedes_id,
        }


class MarkdownStorage:
    """Manages reading and writing structured facts to/from Markdown files."""

    def __init__(self, config: AimemConfig) -> None:
        self.config = config
        base = Path(config.storage_path) if config.storage_path else Path(".deer-flow")
        self.root_dir = (base / "aimem" / "users").resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _sanitize_segment(raw: str, default: str = "default") -> str:
        val = (raw or default).strip()
        if not val:
            val = default
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", val).strip("_")
        if not sanitized:
            sanitized = default
        if sanitized != val:
            digest = hashlib.sha256(val.encode("utf-8")).hexdigest()[:12]
            return f"{sanitized}_{digest}"
        return sanitized

    def _user_dir(self, user_id: str) -> Path:
        safe_user = self._sanitize_segment(user_id, "default")
        path = (self.root_dir / safe_user).resolve()
        if not path.is_relative_to(self.root_dir):
            path = self.root_dir / "default"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _agent_dir(self, user_id: str, agent_name: str, *, create: bool = True) -> Path:
        user_dir = self._user_dir(user_id)
        safe_agent = self._sanitize_segment(agent_name, "agent")
        agents_root = (user_dir / "agents").resolve()
        path = (agents_root / safe_agent).resolve()
        if not path.is_relative_to(user_dir):
            path = user_dir / "agents" / "agent"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _shared_dir(self, user_id: str, *, create: bool = True) -> Path:
        user_dir = self._user_dir(user_id)
        path = (user_dir / "shared").resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _bucket_dir(
        self,
        user_id: str,
        *,
        agent_name: str | None = None,
        scope_role: str | None = None,
        create: bool = True,
    ) -> Path:
        norm_agent = _normalize_agent_name(agent_name)
        norm_role = resolve_default_scope_role(norm_agent, scope_role)
        if norm_role not in ("private", "default"):
            return self._shared_dir(user_id, create=create)
        if norm_agent:
            return self._agent_dir(user_id, norm_agent, create=create)
        return self._user_dir(user_id)

    def _candidate_read_dirs(
        self,
        user_id: str,
        *,
        agent_name: str | None = None,
    ) -> list[tuple[Path, str, str]]:
        """Return (dir_path, default_agent_name, default_scope_role) tuples for scoped reads."""
        user_dir = self._user_dir(user_id)
        norm_agent = _normalize_agent_name(agent_name)
        candidates: list[tuple[Path, str, str]] = []
        if norm_agent:
            agent_dir = self._agent_dir(user_id, norm_agent, create=False)
            if agent_dir.is_dir():
                candidates.append((agent_dir, norm_agent, "private"))
        else:
            candidates.append((user_dir, "", "default"))

        shared_dir = self._shared_dir(user_id, create=False)
        if shared_dir.is_dir():
            candidates.append((shared_dir, "", "shared"))
        return candidates

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

    def parse_markdown(
        self,
        text: str,
        default_category: str,
        *,
        default_agent_name: str = "",
        default_scope_role: str = "",
    ) -> list[MemoryFact]:
        """Parse facts from markdown content, preserving IDs, timestamps, access counts, and governance metadata."""
        facts: list[MemoryFact] = []
        now = datetime.datetime.now(datetime.UTC).isoformat()

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            match = _FACT_LINE_REGEX.match(line)
            if not match:
                continue
            content = match.group("content").strip().replace(r"\n", "\n").replace("&lt;!--", "<!--")
            if not content:
                continue

            fact_id = match.group("id")
            if not fact_id:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
                fact_id = f"fact_{digest}"

            updated = now
            accessed = ""
            count = 0
            valid_from = ""
            invalidated_at = ""
            agent_name = default_agent_name
            scope_role = default_scope_role
            writer_id = ""
            parent_provenance_ids: list[str] = []
            supersedes_id = ""

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
                    elif part.startswith("valid_from:"):
                        valid_from = part.split(":", 1)[1].strip()
                    elif part.startswith("invalidated_at:"):
                        invalidated_at = part.split(":", 1)[1].strip()
                    elif part.startswith("agent:"):
                        agent_name = _normalize_agent_name(part.split(":", 1)[1])
                    elif part.startswith("scope_role:"):
                        scope_role = part.split(":", 1)[1].strip().lower()
                    elif part.startswith("writer:"):
                        writer_id = part.split(":", 1)[1].strip()
                    elif part.startswith("parents:"):
                        raw_parents = part.split(":", 1)[1].strip()
                        parent_provenance_ids = [p.strip() for p in raw_parents.split(",") if p.strip()]
                    elif part.startswith("supersedes:"):
                        supersedes_id = part.split(":", 1)[1].strip()

            resolved_role = resolve_default_scope_role(agent_name, scope_role)
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
                    valid_from=valid_from or updated,
                    invalidated_at=invalidated_at,
                    agent_name=agent_name,
                    scope_role=resolved_role,
                    writer_id=writer_id or agent_name,
                    parent_provenance_ids=parent_provenance_ids,
                    supersedes_id=supersedes_id,
                )
            )
        return facts

    def format_markdown(self, facts: list[MemoryFact], category: str) -> str:
        """Render facts list into clean Markdown with comment metadata."""
        lines = [self._header_for_category(category)]
        for f in facts:
            meta_parts = []
            if f.updated_at:
                meta_parts.append(f"updated: {_sanitize_meta_token(f.updated_at)}")
            if f.valid_from:
                meta_parts.append(f"valid_from: {_sanitize_meta_token(f.valid_from)}")
            if f.invalidated_at:
                meta_parts.append(f"invalidated_at: {_sanitize_meta_token(f.invalidated_at)}")
            if f.last_accessed_at:
                meta_parts.append(f"accessed: {_sanitize_meta_token(f.last_accessed_at)}")
            if f.access_count > 0:
                meta_parts.append(f"count: {f.access_count}")
            if f.agent_name:
                meta_parts.append(f"agent: {_sanitize_meta_token(f.agent_name)}")
            if f.scope_role and f.scope_role != "default":
                meta_parts.append(f"scope_role: {_sanitize_meta_token(f.scope_role)}")
            if f.writer_id:
                meta_parts.append(f"writer: {_sanitize_meta_token(f.writer_id)}")
            if f.parent_provenance_ids:
                safe_parents = ",".join(_sanitize_meta_token(p).replace(",", "_") for p in f.parent_provenance_ids if p)
                if safe_parents:
                    meta_parts.append(f"parents: {safe_parents}")
            if f.supersedes_id:
                meta_parts.append(f"supersedes: {_sanitize_meta_token(f.supersedes_id)}")

            meta_tag = f" <!-- {' | '.join(meta_parts)} -->" if meta_parts else ""
            escaped_content = f.content.replace("\r\n", r"\n").replace("\n", r"\n").replace("<!--", "&lt;!--")
            lines.append(f"- [{f.id}] {escaped_content}{meta_tag}")
        lines.append("")
        return "\n".join(lines)

    def load_user_facts(
        self,
        user_id: str,
        category: str | None = None,
        *,
        agent_name: str | None = None,
        scope_role: str | None = None,
        caller_roles: Collection[str] | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
    ) -> list[MemoryFact]:
        """Load authorized facts for a user/scope, optionally filtered by category or point-in-time."""
        with self._lock:
            uid = user_id or "default"
            categories = [category] if category else ["profile", "entity", "archive"]
            all_facts: list[MemoryFact] = []
            for bucket_dir, default_agent, default_role in self._candidate_read_dirs(uid, agent_name=agent_name):
                for cat in categories:
                    file_path = self._file_for_category(bucket_dir, cat)
                    if not file_path.is_file():
                        continue
                    try:
                        text = file_path.read_text(encoding="utf-8")
                        parsed = self.parse_markdown(
                            text,
                            default_category=cat,
                            default_agent_name=default_agent,
                            default_scope_role=default_role,
                        )
                    except Exception:
                        continue
                    for f in parsed:
                        if not is_fact_authorized(
                            uid,
                            f.agent_name,
                            f.scope_role,
                            caller_user_id=uid,
                            caller_agent_name=agent_name,
                            caller_scope_role=scope_role,
                            caller_roles=caller_roles,
                        ):
                            continue
                        if as_of:
                            if f.is_valid_at(as_of):
                                all_facts.append(f)
                        elif include_invalidated or f.is_active:
                            all_facts.append(f)
            return all_facts

    def load_all_facts_for_user(
        self,
        user_id: str,
        *,
        agent_name: str | None = None,
        include_invalidated: bool = True,
    ) -> list[MemoryFact]:
        """Load raw facts from disk across buckets for index resync (reload_memory)."""
        with self._lock:
            uid = user_id or "default"
            user_dir = self._user_dir(uid)
            norm_agent = _normalize_agent_name(agent_name)
            buckets: list[tuple[Path, str, str]] = []
            if norm_agent:
                agent_dir = self._agent_dir(uid, norm_agent, create=False)
                if agent_dir.is_dir():
                    buckets.append((agent_dir, norm_agent, "private"))
            else:
                buckets.append((user_dir, "", "default"))
                shared_dir = self._shared_dir(uid, create=False)
                if shared_dir.is_dir():
                    buckets.append((shared_dir, "", "shared"))
                agents_root = user_dir / "agents"
                if agents_root.is_dir():
                    for child in sorted(agents_root.iterdir()):
                        if child.is_dir():
                            buckets.append((child, child.name, "private"))

            collected: list[MemoryFact] = []
            for bucket_dir, default_agent, default_role in buckets:
                for cat in ["profile", "entity", "archive"]:
                    file_path = self._file_for_category(bucket_dir, cat)
                    if not file_path.is_file():
                        continue
                    try:
                        text = file_path.read_text(encoding="utf-8")
                        for f in self.parse_markdown(
                            text,
                            default_category=cat,
                            default_agent_name=default_agent,
                            default_scope_role=default_role,
                        ):
                            if include_invalidated or f.is_active:
                                collected.append(f)
                    except Exception:
                        continue
            return collected

    def get_fact_by_id(
        self,
        user_id: str,
        fact_id: str,
        *,
        agent_name: str | None = None,
        scope_role: str | None = None,
        caller_roles: Collection[str] | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
    ) -> MemoryFact | None:
        """Direct ID lookup gated by the exact same scope authorization as search."""
        if not fact_id:
            return None
        facts = self.load_user_facts(
            user_id,
            agent_name=agent_name,
            scope_role=scope_role,
            caller_roles=caller_roles,
            include_invalidated=include_invalidated,
            as_of=as_of,
        )
        return next((f for f in facts if f.id == fact_id), None)

    def save_facts_for_category(
        self,
        user_id: str,
        category: str,
        facts: list[MemoryFact],
        *,
        agent_name: str | None = None,
        scope_role: str | None = None,
    ) -> None:
        """Write facts of a specific category to its corresponding scoped Markdown file."""
        with self._lock:
            bucket_dir = self._bucket_dir(user_id, agent_name=agent_name, scope_role=scope_role, create=True)
            file_path = self._file_for_category(bucket_dir, category)
            md = self.format_markdown(facts, category)
            file_path.write_text(md, encoding="utf-8")

    @staticmethod
    def _resolve_temporal_supersession(
        existing: list[MemoryFact],
        fact: MemoryFact,
        now: str,
    ) -> tuple[list[MemoryFact], list[str], bool]:
        """Stage 1: Resolve contradiction & temporal supersession BEFORE deduplication."""
        new_topic = _extract_topic_key(fact.content)
        target_agent = _normalize_agent_name(fact.agent_name)
        target_role = resolve_default_scope_role(target_agent, fact.scope_role)

        superseded_ids: list[str] = []
        id_replaced_with_history = False
        post_supersession: list[MemoryFact] = []

        for item in existing:
            item_agent = _normalize_agent_name(item.agent_name)
            item_role = resolve_default_scope_role(item_agent, item.scope_role)
            same_write_scope = (item_agent, item_role) == (target_agent, target_role)

            if item.invalidated_at or not same_write_scope:
                post_supersession.append(item)
                continue

            # Explicit ID update with changed content -> preserve invalidated history snapshot
            if item.id == fact.id and item.content.strip() != fact.content.strip():
                archived = replace(
                    item,
                    id=f"{item.id}__v_{uuid.uuid4().hex[:6]}",
                    invalidated_at=now,
                )
                post_supersession.append(archived)
                superseded_ids.append(item.id)
                if not fact.supersedes_id:
                    fact.supersedes_id = archived.id
                fact.access_count = max(fact.access_count, item.access_count + 1)
                fact.valid_from = now
                post_supersession.append(fact)
                id_replaced_with_history = True
                continue

            # Explicit supersedes_id OR topic-level contradiction
            explicit_supersede = bool(fact.supersedes_id and item.id == fact.supersedes_id)
            topic_contradiction = (
                new_topic is not None
                and item.id != fact.id
                and _extract_topic_key(item.content) == new_topic
                and item.content.strip().lower() != fact.content.strip().lower()
            )
            if explicit_supersede or topic_contradiction:
                superseded_ids.append(item.id)
                item.invalidated_at = now
                post_supersession.append(item)
                if not fact.supersedes_id:
                    fact.supersedes_id = item.id
                fact.access_count = max(fact.access_count, item.access_count + 1)
                fact.valid_from = now
                continue

            post_supersession.append(item)

        return post_supersession, superseded_ids, id_replaced_with_history

    def _deduplicate_and_compact(
        self,
        post_supersession: list[MemoryFact],
        fact: MemoryFact,
        *,
        already_inserted: bool,
    ) -> list[MemoryFact]:
        """Stage 2: Deduplicate non-contradictory active facts and prune stale entries."""
        target_agent = _normalize_agent_name(fact.agent_name)
        target_role = resolve_default_scope_role(target_agent, fact.scope_role)

        new_list: list[MemoryFact] = []
        merged = already_inserted

        for item in post_supersession:
            item_agent = _normalize_agent_name(item.agent_name)
            item_role = resolve_default_scope_role(item_agent, item.scope_role)
            same_write_scope = (item_agent, item_role) == (target_agent, target_role)

            if not merged and not item.invalidated_at and same_write_scope:
                if item.id == fact.id or _is_near_duplicate(item.content, fact.content):
                    fact.access_count = max(fact.access_count, item.access_count)
                    fact.valid_from = item.valid_from or fact.valid_from
                    if item.parent_provenance_ids or fact.parent_provenance_ids:
                        merged_parents: list[str] = []
                        for pid in [*item.parent_provenance_ids, *fact.parent_provenance_ids]:
                            if pid and pid not in merged_parents:
                                merged_parents.append(pid)
                        fact.parent_provenance_ids = merged_parents
                    new_list.append(fact)
                    merged = True
                    continue
            new_list.append(item)

        if not merged:
            new_list.append(fact)

        max_facts = self.config.max_facts_per_category
        active_facts = [f for f in new_list if not f.invalidated_at]
        invalid_facts = [f for f in new_list if f.invalidated_at]
        if len(active_facts) > max_facts:
            active_facts.sort(key=lambda x: (x.access_count, x.updated_at), reverse=True)
            active_facts = active_facts[:max_facts]
        if len(invalid_facts) > max_facts:
            invalid_facts.sort(key=lambda x: x.invalidated_at, reverse=True)
            invalid_facts = invalid_facts[:max_facts]
        return active_facts + invalid_facts

    def upsert_fact_governed(self, user_id: str, fact: MemoryFact) -> tuple[MemoryFact, list[str]]:
        """Insert or update a fact using the decoupled 2-stage pipeline:
        Stage 1 (Temporal Supersession) -> Stage 2 (Deduplication & Compaction).

        Returns (saved_fact, superseded_fact_ids).
        """
        with self._lock:
            norm_agent = _normalize_agent_name(fact.agent_name)
            norm_role = resolve_default_scope_role(norm_agent, fact.scope_role)
            fact.agent_name = norm_agent
            fact.scope_role = norm_role
            if not fact.writer_id:
                fact.writer_id = norm_agent

            category = fact.category or "profile"
            bucket_dir = self._bucket_dir(user_id, agent_name=norm_agent, scope_role=norm_role, create=True)
            file_path = self._file_for_category(bucket_dir, category)
            existing: list[MemoryFact] = []
            if file_path.is_file():
                try:
                    existing = self.parse_markdown(
                        file_path.read_text(encoding="utf-8"),
                        category,
                        default_agent_name=norm_agent,
                        default_scope_role=norm_role,
                    )
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
            if not fact.valid_from:
                fact.valid_from = fact.created_at or now

            # Stage 1: Temporal supersession & contradiction resolution FIRST
            post_supersession, superseded_ids, already_inserted = self._resolve_temporal_supersession(existing, fact, now)

            # Stage 2: Near-duplicate compaction & staleness pruning SECOND
            final_list = self._deduplicate_and_compact(post_supersession, fact, already_inserted=already_inserted)

            self.save_facts_for_category(
                user_id,
                category,
                final_list,
                agent_name=norm_agent,
                scope_role=norm_role,
            )
            return fact, superseded_ids

    def upsert_fact(self, user_id: str, fact: MemoryFact) -> tuple[MemoryFact, str | None]:
        """Insert or update a fact with contradiction / superseding resolution.

        Returns (saved_fact, last_superseded_fact_id) for backward compatibility.
        """
        saved, superseded_ids = self.upsert_fact_governed(user_id, fact)
        return saved, (superseded_ids[-1] if superseded_ids else None)

    def touch_facts(
        self,
        user_id: str,
        fact_ids: list[str],
        *,
        agent_name: str | None = None,
        scope_role: str | None = None,
        caller_roles: Collection[str] | None = None,
    ) -> None:
        """Increment access count and touch timestamp for authorized retrieved facts."""
        if not fact_ids:
            return
        with self._lock:
            uid = user_id or "default"
            id_set = set(fact_ids)
            now = datetime.datetime.now(datetime.UTC).isoformat()
            for bucket_dir, default_agent, default_role in self._candidate_read_dirs(uid, agent_name=agent_name):
                for cat in ["profile", "entity", "archive"]:
                    file_path = self._file_for_category(bucket_dir, cat)
                    if not file_path.is_file():
                        continue
                    try:
                        facts = self.parse_markdown(
                            file_path.read_text(encoding="utf-8"),
                            default_category=cat,
                            default_agent_name=default_agent,
                            default_scope_role=default_role,
                        )
                    except Exception:
                        continue
                    changed = False
                    for f in facts:
                        if f.id in id_set and is_fact_authorized(
                            uid,
                            f.agent_name,
                            f.scope_role,
                            caller_user_id=uid,
                            caller_agent_name=agent_name,
                            caller_scope_role=scope_role,
                            caller_roles=caller_roles,
                        ):
                            f.access_count += 1
                            f.last_accessed_at = now
                            changed = True
                    if changed:
                        file_path.write_text(self.format_markdown(facts, cat), encoding="utf-8")

    def delete_fact(
        self,
        user_id: str,
        fact_id: str,
        *,
        agent_name: str | None = None,
        scope_role: str | None = None,
        caller_roles: Collection[str] | None = None,
    ) -> bool:
        """Delete an authorized fact by ID across accessible categories for the user/scope."""
        with self._lock:
            uid = user_id or "default"
            deleted = False
            for bucket_dir, default_agent, default_role in self._candidate_read_dirs(uid, agent_name=agent_name):
                for cat in ["profile", "entity", "archive"]:
                    file_path = self._file_for_category(bucket_dir, cat)
                    if not file_path.is_file():
                        continue
                    existing = self.parse_markdown(
                        file_path.read_text(encoding="utf-8"),
                        cat,
                        default_agent_name=default_agent,
                        default_scope_role=default_role,
                    )
                    remaining: list[MemoryFact] = []
                    for f in existing:
                        if f.id == fact_id and is_fact_authorized(
                            uid,
                            f.agent_name,
                            f.scope_role,
                            caller_user_id=uid,
                            caller_agent_name=agent_name,
                            caller_scope_role=scope_role,
                            caller_roles=caller_roles,
                        ):
                            deleted = True
                            continue
                        remaining.append(f)
                    if len(remaining) < len(existing):
                        file_path.write_text(self.format_markdown(remaining, cat), encoding="utf-8")
            return deleted

    def clear_user(self, user_id: str, *, agent_name: str | None = None) -> None:
        """Clear memory files for a user (or a specific agent bucket when agent_name is set)."""
        with self._lock:
            uid = user_id or "default"
            norm_agent = _normalize_agent_name(agent_name)
            if norm_agent:
                target_dirs = [self._agent_dir(uid, norm_agent, create=False)]
            else:
                user_dir = self._user_dir(uid)
                target_dirs = [user_dir, self._shared_dir(uid, create=False)]
                agents_root = user_dir / "agents"
                if agents_root.is_dir():
                    target_dirs.extend(child for child in agents_root.iterdir() if child.is_dir())

            for target_dir in target_dirs:
                if not target_dir.is_dir():
                    continue
                for cat in ["profile", "entity", "archive"]:
                    f = self._file_for_category(target_dir, cat)
                    if f.is_file():
                        try:
                            f.unlink()
                        except Exception:
                            pass
