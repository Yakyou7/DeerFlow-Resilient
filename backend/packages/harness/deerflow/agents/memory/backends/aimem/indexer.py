"""SQLite FTS5 derived indexer for Aimem.

Maintains a fast local full-text search index synced from Markdown files.
Supports BM25 + Ebbinghaus recency decay + access frequency hybrid ranking.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import AimemConfig
from .storage import MemoryFact

logger = logging.getLogger(__name__)


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts_meta (
                    fact_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    category TEXT,
                    confidence REAL,
                    created_at TEXT,
                    updated_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT
                );
                """
            )
            # Add columns if migrating from earlier schema
            cursor = conn.execute("PRAGMA table_info(facts_meta);")
            columns = {row["name"] for row in cursor.fetchall()}
            if "access_count" not in columns:
                conn.execute("ALTER TABLE facts_meta ADD COLUMN access_count INTEGER DEFAULT 0;")
            if "last_accessed_at" not in columns:
                conn.execute("ALTER TABLE facts_meta ADD COLUMN last_accessed_at TEXT;")

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS aimem_fts USING fts5(
                        fact_id UNINDEXED,
                        user_id UNINDEXED,
                        category,
                        content,
                        tokenize='unicode61'
                    );
                    """
                )
                conn.commit()
            except Exception as e:
                logger.warning("FTS5 table initialization notice: %s", e)

    def sync_user_facts(self, user_id: str, facts: list[MemoryFact]) -> None:
        """Replace all index entries for a user with the provided facts."""
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM facts_meta WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM aimem_fts WHERE user_id = ?", (user_id,))

            for f in facts:
                conn.execute(
                    """
                    INSERT INTO facts_meta (fact_id, user_id, category, confidence, created_at, updated_at, access_count, last_accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (f.id, user_id, f.category, f.confidence, f.created_at, f.updated_at, f.access_count, f.last_accessed_at),
                )
                conn.execute(
                    """
                    INSERT INTO aimem_fts (fact_id, user_id, category, content)
                    VALUES (?, ?, ?, ?);
                    """,
                    (f.id, user_id, f.category, f.content),
                )
            conn.commit()

    def upsert_fact(self, user_id: str, fact: MemoryFact) -> None:
        """Update or insert a single fact in the index."""
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM facts_meta WHERE fact_id = ?", (fact.id,))
            conn.execute("DELETE FROM aimem_fts WHERE fact_id = ?", (fact.id,))

            conn.execute(
                """
                INSERT INTO facts_meta (fact_id, user_id, category, confidence, created_at, updated_at, access_count, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (fact.id, user_id, fact.category, fact.confidence, fact.created_at, fact.updated_at, fact.access_count, fact.last_accessed_at),
            )
            conn.execute(
                """
                INSERT INTO aimem_fts (fact_id, user_id, category, content)
                VALUES (?, ?, ?, ?);
                """,
                (fact.id, user_id, fact.category, fact.content),
            )
            conn.commit()

    def touch_facts(self, fact_ids: list[str]) -> None:
        """Update access count and timestamp in SQLite index."""
        if not fact_ids:
            return
        now = datetime.datetime.now(datetime.UTC).isoformat()
        with self._lock, self._connection() as conn:
            placeholders = ",".join("?" for _ in fact_ids)
            conn.execute(
                f"""
                UPDATE facts_meta
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE fact_id IN ({placeholders});
                """,
                [now, *fact_ids],
            )
            conn.commit()

    def delete_fact(self, fact_id: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM facts_meta WHERE fact_id = ?", (fact_id,))
            conn.execute("DELETE FROM aimem_fts WHERE fact_id = ?", (fact_id,))
            conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search facts using hybrid ranking: BM25 + Recency Decay + Access Frequency."""
        clean_query = _sanitize_fts_query(query)
        if not clean_query:
            return []

        raw_candidates: list[dict[str, Any]] = []
        sql = """
            SELECT f.fact_id, f.user_id, f.category, f.content, bm25(aimem_fts) as raw_bm25,
                   m.confidence, m.created_at, m.updated_at, m.access_count, m.last_accessed_at
            FROM aimem_fts f
            LEFT JOIN facts_meta m ON f.fact_id = m.fact_id
            WHERE aimem_fts MATCH ?
        """
        params: list[Any] = [clean_query]

        if user_id:
            sql += " AND f.user_id = ?"
            params.append(user_id)
        if category:
            sql += " AND f.category = ?"
            params.append(category)

        # Retrieve more candidates to re-rank with recency and frequency
        candidate_limit = max(top_k * 3, 20)
        sql += " ORDER BY raw_bm25 LIMIT ?"
        params.append(candidate_limit)

        try:
            with self._connection() as conn:
                cursor = conn.execute(sql, params)
                for row in cursor.fetchall():
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
                            "raw_bm25": float(row["raw_bm25"]),
                        }
                    )
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 match failed (%s); trying fallback LIKE query", e)
            return self._fallback_like_search(query, top_k=top_k, user_id=user_id, category=category)

        # Compute hybrid score
        half_life = self.config.decay_half_life_days
        for c in raw_candidates:
            # BM25 is negative; lower is better match -> positive relevance
            r_bm25 = max(0.1, -1.0 * c["raw_bm25"])
            r_recency = _compute_recency_score(c["updated_at"] or c["last_accessed_at"], half_life)
            r_freq = math.log1p(c["access_count"])
            # Hybrid formula
            c["score"] = round(r_bm25 * (1.0 + 0.6 * r_recency + 0.4 * r_freq), 4)

        # Sort descending by hybrid composite score
        raw_candidates.sort(key=lambda x: x["score"], reverse=True)
        return raw_candidates[:top_k]

    def _fallback_like_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT f.fact_id, f.user_id, f.category, f.content,
                   m.confidence, m.created_at, m.updated_at, m.access_count, m.last_accessed_at
            FROM aimem_fts f
            LEFT JOIN facts_meta m ON f.fact_id = m.fact_id
            WHERE f.content LIKE ?
        """
        params: list[Any] = [f"%{query.strip()}%"]
        if user_id:
            sql += " AND f.user_id = ?"
            params.append(user_id)
        if category:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " LIMIT ?"
        params.append(max(1, top_k))

        results: list[dict[str, Any]] = []
        try:
            with self._connection() as conn:
                for row in conn.execute(sql, params).fetchall():
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
                            "score": 1.0,
                        }
                    )
        except Exception:
            pass
        return results

    def close(self) -> None:
        pass
