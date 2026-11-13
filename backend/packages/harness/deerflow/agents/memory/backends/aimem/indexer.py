"""SQLite FTS5 derived indexer for Aimem.

Maintains a fast local full-text search index synced from Markdown files.
Supports BM25 + Ebbinghaus recency decay + access frequency hybrid ranking.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from collections.abc import Collection
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import AimemConfig
from .storage import (
    MemoryFact,
    _normalize_agent_name,
    is_fact_authorized,
    resolve_default_scope_role,
)

logger = logging.getLogger(__name__)


def _make_scope_key(user_id: str | None, agent_name: str | None, scope_role: str | None, fact_id: str) -> str:
    """Build collision-free composite primary key for (user_id, agent_name, scope_role, fact_id)."""
    uid = user_id or "default"
    agent = _normalize_agent_name(agent_name)
    role = resolve_default_scope_role(agent, scope_role)
    scope_hash = hashlib.sha256(f"{uid}\0{agent}\0{role}".encode()).hexdigest()[:16]
    return f"{scope_hash}:{fact_id}"


def _parse_provenance_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x]
    except Exception:
        pass
    return []


def _build_sql_scope_clause(
    alias: str,
    *,
    user_id: str | None,
    agent_name: str | None,
    scope_role: str | None,
    caller_roles: Collection[str] | None,
) -> tuple[str, list[Any]]:
    """Build SQL WHERE predicates matching ``is_fact_authorized`` so LIMIT never starves."""
    clauses: list[str] = []
    params: list[Any] = []

    if user_id:
        clauses.append(f"{alias}.user_id = ?")
        params.append(user_id)

    c_agent = _normalize_agent_name(agent_name)
    roles = sorted(
        {
            *(r.strip().lower() for r in (caller_roles or ()) if r and r.strip()),
            *([scope_role.strip().lower()] if scope_role and scope_role.strip() else []),
        }
    )

    role_or = [
        f"{alias}.scope_role = 'shared'",
        f"({alias}.scope_role = 'default' AND ? = '')",
        f"({alias}.scope_role = 'private' AND {alias}.agent_name = ?)",
    ]
    params.extend([c_agent, c_agent])

    if roles:
        placeholders = ",".join("?" for _ in roles)
        role_or.append(f"{alias}.scope_role IN ({placeholders})")
        params.extend(roles)

    clauses.append(f"({' OR '.join(role_or)})")
    return " AND ".join(clauses), params


def _sanitize_fts_query(query: str) -> str:
    """Sanitize user query into safe FTS5 query tokens."""
    cleaned = re.sub(r'["*^:()\[\]\{\}]', " ", query)
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"*' for token in tokens)


def _compute_recency_score(updated_at_str: str, half_life_days: float) -> float:
    """Compute exponential decay score in [0.0, 1.0] based on elapsed days."""
    if not updated_at_str or half_life_days <= 0:
        return 0.5
    try:
        t = datetime.datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.UTC)
        delta_days = max(0.0, (now - t).total_seconds() / 86400.0)
        # Half-life exponential decay: 0.5 ** (days / half_life)
        return float(math.pow(0.5, delta_days / half_life_days))
    except Exception:
        return 0.5


class FtsIndexer:
    """SQLite FTS5 derived index for Aimem facts with hybrid ranking."""

    def __init__(self, config: AimemConfig) -> None:
        self.config = config
        base = Path(config.storage_path) if config.storage_path else Path(".deer-flow")
        db_dir = base / "aimem" / "derived"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / config.fts_db_name
        self._lock = threading.Lock()
        self._fts_available = False
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA busy_timeout = 5000;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            except Exception as e:
                logger.debug("SQLite PRAGMA tuning notice: %s", e)

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts_meta (
                    scope_key TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_name TEXT DEFAULT '',
                    scope_role TEXT DEFAULT 'default',
                    writer_id TEXT DEFAULT '',
                    parent_provenance_ids TEXT DEFAULT '[]',
                    supersedes_id TEXT DEFAULT '',
                    category TEXT,
                    content TEXT,
                    confidence REAL,
                    created_at TEXT,
                    updated_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT,
                    valid_from TEXT,
                    invalidated_at TEXT
                );
                """
            )
            cursor = conn.execute("PRAGMA table_info(facts_meta);")
            columns = {row["name"] for row in cursor.fetchall()}

            if "scope_key" not in columns:
                # Migrate legacy table where bare fact_id was PRIMARY KEY
                conn.execute(
                    """
                    CREATE TABLE facts_meta_v2 (
                        scope_key TEXT PRIMARY KEY,
                        fact_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        agent_name TEXT DEFAULT '',
                        scope_role TEXT DEFAULT 'default',
                        writer_id TEXT DEFAULT '',
                        parent_provenance_ids TEXT DEFAULT '[]',
                        supersedes_id TEXT DEFAULT '',
                        category TEXT,
                        content TEXT,
                        confidence REAL,
                        created_at TEXT,
                        updated_at TEXT,
                        access_count INTEGER DEFAULT 0,
                        last_accessed_at TEXT,
                        valid_from TEXT,
                        invalidated_at TEXT
                    );
                    """
                )
                legacy_rows = conn.execute("SELECT * FROM facts_meta;").fetchall()
                for row in legacy_rows:
                    r = dict(row)
                    uid = r.get("user_id") or "default"
                    fid = r.get("fact_id") or ""
                    sk = _make_scope_key(uid, "", "default", fid)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO facts_meta_v2 (
                            scope_key, fact_id, user_id, agent_name, scope_role, writer_id,
                            parent_provenance_ids, supersedes_id, category, content, confidence,
                            created_at, updated_at, access_count, last_accessed_at, valid_from, invalidated_at
                        ) VALUES (?, ?, ?, '', 'default', '', '[]', '', ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            sk,
                            fid,
                            uid,
                            r.get("category"),
                            r.get("content"),
                            r.get("confidence", 1.0),
                            r.get("created_at", ""),
                            r.get("updated_at", ""),
                            r.get("access_count", 0),
                            r.get("last_accessed_at", ""),
                            r.get("valid_from", ""),
                            r.get("invalidated_at", ""),
                        ),
                    )
                conn.execute("DROP TABLE facts_meta;")
                conn.execute("ALTER TABLE facts_meta_v2 RENAME TO facts_meta;")
            else:
                for col_name, col_def in (
                    ("agent_name", "TEXT DEFAULT ''"),
                    ("scope_role", "TEXT DEFAULT 'default'"),
                    ("writer_id", "TEXT DEFAULT ''"),
                    ("parent_provenance_ids", "TEXT DEFAULT '[]'"),
                    ("supersedes_id", "TEXT DEFAULT ''"),
                    ("content", "TEXT"),
                    ("access_count", "INTEGER DEFAULT 0"),
                    ("last_accessed_at", "TEXT"),
                    ("valid_from", "TEXT"),
                    ("invalidated_at", "TEXT"),
                ):
                    if col_name not in columns:
                        conn.execute(f"ALTER TABLE facts_meta ADD COLUMN {col_name} {col_def};")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_meta_user_fact ON facts_meta(user_id, fact_id);")

            try:
                fts_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='aimem_fts';"
                ).fetchone()
                recreate_fts = fts_exists is None
                if not recreate_fts:
                    fts_cols = {row["name"] for row in conn.execute("PRAGMA table_info(aimem_fts);").fetchall()}
                    if "scope_key" not in fts_cols:
                        conn.execute("DROP TABLE aimem_fts;")
                        recreate_fts = True

                if recreate_fts:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE aimem_fts USING fts5(
                            scope_key UNINDEXED,
                            fact_id UNINDEXED,
                            user_id UNINDEXED,
                            agent_name UNINDEXED,
                            scope_role UNINDEXED,
                            category,
                            content,
                            tokenize='unicode61'
                        );
                        """
                    )
                    active_rows = conn.execute(
                        """
                        SELECT scope_key, fact_id, user_id, agent_name, scope_role, category, content
                        FROM facts_meta
                        WHERE (invalidated_at IS NULL OR invalidated_at = '');
                        """
                    ).fetchall()
                    for row in active_rows:
                        conn.execute(
                            """
                            INSERT INTO aimem_fts (scope_key, fact_id, user_id, agent_name, scope_role, category, content)
                            VALUES (?, ?, ?, ?, ?, ?, ?);
                            """,
                            (
                                row["scope_key"],
                                row["fact_id"],
                                row["user_id"],
                                row["agent_name"] or "",
                                row["scope_role"] or "default",
                                row["category"] or "profile",
                                row["content"] or "",
                            ),
                        )
                conn.commit()
                self._fts_available = True
            except Exception as e:
                logger.warning("FTS5 table initialization notice: %s; falling back to LIKE search", e)
                self._fts_available = False

    def _insert_fact_rows(self, conn: sqlite3.Connection, user_id: str, f: MemoryFact) -> None:
        uid = user_id or "default"
        agent = _normalize_agent_name(f.agent_name)
        role = resolve_default_scope_role(agent, f.scope_role)
        writer = f.writer_id or agent
        parents_json = json.dumps(list(f.parent_provenance_ids or []))
        scope_key = _make_scope_key(uid, agent, role, f.id)

        conn.execute(
            """
            INSERT INTO facts_meta (
                scope_key, fact_id, user_id, agent_name, scope_role, writer_id,
                parent_provenance_ids, supersedes_id, category, content, confidence,
                created_at, updated_at, access_count, last_accessed_at, valid_from, invalidated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                scope_key,
                f.id,
                uid,
                agent,
                role,
                writer,
                parents_json,
                f.supersedes_id or "",
                f.category,
                f.content,
                f.confidence,
                f.created_at,
                f.updated_at,
                f.access_count,
                f.last_accessed_at,
                f.valid_from,
                f.invalidated_at,
            ),
        )
        if self._fts_available and not f.invalidated_at:
            try:
                conn.execute(
                    """
                    INSERT INTO aimem_fts (scope_key, fact_id, user_id, agent_name, scope_role, category, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (scope_key, f.id, uid, agent, role, f.category, f.content),
                )
            except Exception:
                pass

    def sync_user_facts(
        self,
        user_id: str,
        facts: list[MemoryFact],
        *,
        agent_name: str | None = None,
    ) -> None:
        """Replace index entries for a user (or specific agent bucket) with the provided facts."""
        uid = user_id or "default"
        norm_agent = _normalize_agent_name(agent_name)
        with self._lock, self._connection() as conn:
            if agent_name is not None and norm_agent:
                conn.execute(
                    "DELETE FROM facts_meta WHERE user_id = ? AND agent_name = ? AND scope_role = 'private'",
                    (uid, norm_agent),
                )
                if self._fts_available:
                    try:
                        conn.execute(
                            "DELETE FROM aimem_fts WHERE user_id = ? AND agent_name = ? AND scope_role = 'private'",
                            (uid, norm_agent),
                        )
                    except Exception:
                        pass
            else:
                conn.execute("DELETE FROM facts_meta WHERE user_id = ?", (uid,))
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE user_id = ?", (uid,))
                    except Exception:
                        pass

            for f in facts:
                self._insert_fact_rows(conn, uid, f)
            conn.commit()

    def upsert_fact(self, user_id: str, fact: MemoryFact) -> None:
        """Update or insert a single scoped fact in the index without cross-scope collision."""
        uid = user_id or "default"
        scope_key = _make_scope_key(uid, fact.agent_name, fact.scope_role, fact.id)
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM facts_meta WHERE scope_key = ?", (scope_key,))
            if self._fts_available:
                try:
                    conn.execute("DELETE FROM aimem_fts WHERE scope_key = ?", (scope_key,))
                except Exception:
                    pass

            self._insert_fact_rows(conn, uid, fact)
            conn.commit()

    def invalidate_fact(
        self,
        fact_id: str,
        invalidated_at: str,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        scope_role: str | None = None,
    ) -> None:
        """Mark a scoped fact as invalidated in SQLite metadata and remove from active FTS index."""
        with self._lock, self._connection() as conn:
            if user_id is not None and (agent_name is not None or scope_role is not None):
                scope_key = _make_scope_key(user_id, agent_name, scope_role, fact_id)
                conn.execute(
                    "UPDATE facts_meta SET invalidated_at = ? WHERE scope_key = ?;",
                    (invalidated_at, scope_key),
                )
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE scope_key = ?", (scope_key,))
                    except Exception:
                        pass
            elif user_id is not None:
                conn.execute(
                    "UPDATE facts_meta SET invalidated_at = ? WHERE fact_id = ? AND user_id = ?;",
                    (invalidated_at, fact_id, user_id),
                )
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE fact_id = ? AND user_id = ?", (fact_id, user_id))
                    except Exception:
                        pass
            else:
                conn.execute(
                    "UPDATE facts_meta SET invalidated_at = ? WHERE fact_id = ?;",
                    (invalidated_at, fact_id),
                )
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE fact_id = ?", (fact_id,))
                    except Exception:
                        pass
            conn.commit()

    def touch_facts(
        self,
        fact_ids: list[str],
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Update access count and timestamp in SQLite index for the given scope."""
        if not fact_ids:
            return
        now = datetime.datetime.now(datetime.UTC).isoformat()
        with self._lock, self._connection() as conn:
            placeholders = ",".join("?" for _ in fact_ids)
            sql = f"UPDATE facts_meta SET access_count = access_count + 1, last_accessed_at = ? WHERE fact_id IN ({placeholders})"
            params: list[Any] = [now, *fact_ids]
            if user_id is not None:
                sql += " AND user_id = ?"
                params.append(user_id)
            if agent_name is not None:
                norm_agent = _normalize_agent_name(agent_name)
                sql += " AND (agent_name = ? OR scope_role NOT IN ('private', 'default'))"
                params.append(norm_agent)
            conn.execute(sql + ";", params)
            conn.commit()

    def delete_fact(
        self,
        fact_id: str,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        scope_role: str | None = None,
    ) -> None:
        with self._lock, self._connection() as conn:
            if user_id is not None and (agent_name is not None or scope_role is not None):
                scope_key = _make_scope_key(user_id, agent_name, scope_role, fact_id)
                conn.execute("DELETE FROM facts_meta WHERE scope_key = ?", (scope_key,))
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE scope_key = ?", (scope_key,))
                    except Exception:
                        pass
            elif user_id is not None:
                conn.execute("DELETE FROM facts_meta WHERE fact_id = ? AND user_id = ?", (fact_id, user_id))
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE fact_id = ? AND user_id = ?", (fact_id, user_id))
                    except Exception:
                        pass
            else:
                conn.execute("DELETE FROM facts_meta WHERE fact_id = ?", (fact_id,))
                if self._fts_available:
                    try:
                        conn.execute("DELETE FROM aimem_fts WHERE fact_id = ?", (fact_id,))
                    except Exception:
                        pass
            conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        scope_role: str | None = None,
        caller_roles: Collection[str] | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search facts using hybrid ranking: BM25 + Recency Decay + Access Frequency,
        enforcing symmetric scope authorization.
        """
        clean_query = _sanitize_fts_query(query)
        if not clean_query:
            return []

        if not self._fts_available:
            return self._fallback_like_search(
                query,
                top_k=top_k,
                user_id=user_id,
                agent_name=agent_name,
                scope_role=scope_role,
                caller_roles=caller_roles,
                category=category,
            )

        uid = user_id or "default"
        scope_sql, scope_params = _build_sql_scope_clause(
            "m",
            user_id=uid,
            agent_name=agent_name,
            scope_role=scope_role,
            caller_roles=caller_roles,
        )

        raw_candidates: list[dict[str, Any]] = []
        sql = f"""
            SELECT f.fact_id, f.user_id, f.category, f.content, bm25(aimem_fts) as raw_bm25,
                   m.agent_name, m.scope_role, m.writer_id, m.parent_provenance_ids, m.supersedes_id,
                   m.confidence, m.created_at, m.updated_at, m.access_count, m.last_accessed_at,
                   m.valid_from, m.invalidated_at
            FROM aimem_fts f
            INNER JOIN facts_meta m ON f.scope_key = m.scope_key
            WHERE aimem_fts MATCH ?
              AND (m.invalidated_at IS NULL OR m.invalidated_at = '')
              AND {scope_sql}
        """
        params: list[Any] = [clean_query, *scope_params]

        if category:
            sql += " AND f.category = ?"
            params.append(category)

        candidate_limit = max(top_k * 3, 20)
        sql += " ORDER BY raw_bm25 LIMIT ?"
        params.append(candidate_limit)

        try:
            with self._connection() as conn:
                cursor = conn.execute(sql, params)
                for row in cursor.fetchall():
                    if not is_fact_authorized(
                        row["user_id"],
                        row["agent_name"] or "",
                        row["scope_role"] or "",
                        caller_user_id=uid,
                        caller_agent_name=agent_name,
                        caller_scope_role=scope_role,
                        caller_roles=caller_roles,
                    ):
                        continue
                    raw_candidates.append(
                        {
                            "id": row["fact_id"],
                            "content": row["content"],
                            "category": row["category"],
                            "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                            "created_at": row["created_at"] or "",
                            "updated_at": row["updated_at"] or "",
                            "access_count": int(row["access_count"] or 0),
                            "last_accessed_at": row["last_accessed_at"] or "",
                            "valid_from": row["valid_from"] or "",
                            "invalidated_at": row["invalidated_at"] or "",
                            "agent_name": row["agent_name"] or "",
                            "scope_role": row["scope_role"] or "default",
                            "writer_id": row["writer_id"] or "",
                            "parent_provenance_ids": _parse_provenance_json(row["parent_provenance_ids"]),
                            "supersedes_id": row["supersedes_id"] or "",
                            "raw_bm25": float(row["raw_bm25"]),
                        }
                    )
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 match failed (%s); trying fallback LIKE query", e)
            return self._fallback_like_search(
                query,
                top_k=top_k,
                user_id=user_id,
                agent_name=agent_name,
                scope_role=scope_role,
                caller_roles=caller_roles,
                category=category,
            )

        half_life = self.config.decay_half_life_days
        for c in raw_candidates:
            r_bm25 = max(0.1, -1.0 * c["raw_bm25"])
            r_recency = _compute_recency_score(c["updated_at"] or c["last_accessed_at"], half_life)
            r_freq = math.log1p(c["access_count"])
            c["score"] = round(r_bm25 * (1.0 + 0.6 * r_recency + 0.4 * r_freq), 4)

        raw_candidates.sort(key=lambda x: x["score"], reverse=True)
        return raw_candidates[:top_k]

    def _fallback_like_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        scope_role: str | None = None,
        caller_roles: Collection[str] | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        uid = user_id or "default"
        scope_sql, scope_params = _build_sql_scope_clause(
            "facts_meta",
            user_id=uid,
            agent_name=agent_name,
            scope_role=scope_role,
            caller_roles=caller_roles,
        )
        sql = f"""
            SELECT fact_id, user_id, agent_name, scope_role, writer_id, parent_provenance_ids, supersedes_id,
                   category, content, confidence, created_at, updated_at, access_count, last_accessed_at,
                   valid_from, invalidated_at
            FROM facts_meta
            WHERE content LIKE ?
              AND (invalidated_at IS NULL OR invalidated_at = '')
              AND {scope_sql}
        """
        params: list[Any] = [f"%{query.strip()}%", *scope_params]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " LIMIT ?"
        params.append(max(1, top_k))

        results: list[dict[str, Any]] = []
        try:
            with self._connection() as conn:
                for row in conn.execute(sql, params).fetchall():
                    if not is_fact_authorized(
                        row["user_id"],
                        row["agent_name"] or "",
                        row["scope_role"] or "",
                        caller_user_id=uid,
                        caller_agent_name=agent_name,
                        caller_scope_role=scope_role,
                        caller_roles=caller_roles,
                    ):
                        continue
                    results.append(
                        {
                            "id": row["fact_id"],
                            "content": row["content"],
                            "category": row["category"],
                            "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                            "created_at": row["created_at"] or "",
                            "updated_at": row["updated_at"] or "",
                            "access_count": int(row["access_count"] or 0),
                            "last_accessed_at": row["last_accessed_at"] or "",
                            "valid_from": row["valid_from"] or "",
                            "invalidated_at": row["invalidated_at"] or "",
                            "agent_name": row["agent_name"] or "",
                            "scope_role": row["scope_role"] or "default",
                            "writer_id": row["writer_id"] or "",
                            "parent_provenance_ids": _parse_provenance_json(row["parent_provenance_ids"]),
                            "supersedes_id": row["supersedes_id"] or "",
                            "score": 1.0,
                        }
                    )
        except Exception:
            pass
        return results

    def close(self) -> None:
        pass
