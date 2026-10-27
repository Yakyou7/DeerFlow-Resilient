"""Aimem backend configuration.

Portability golden rule:
A backend receives host information through MemoryManager arguments and backend_config.
It must not import deer-flow internal path helpers or singletons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class AimemConfig:
    """Configuration for the Aimem hierarchical memory backend."""

    #: Writable state dir injected by the host factory (e.g. {runtime_home}).
    storage_path: str = ""

    #: Maximum number of relevant facts to inject into agent context.
    max_context_facts: int = 10

    #: Minimum message length (in characters) required to evaluate for memory extraction.
    min_message_chars: int = 10

    #: Whether to build and query the SQLite FTS5 derived index.
    enable_fts: bool = True

    #: Name of the SQLite FTS5 database file under the aimem storage root.
    fts_db_name: str = "aimem_index.db"

    #: Markdown filename for user profiles & long-term preferences.
    profile_filename: str = "profile.md"

    #: Markdown filename for project entities & knowledge.
    entities_filename: str = "entities.md"

    #: Markdown filename for historical decisions & epics.
    archive_filename: str = "archive.md"

    #: Half-life for Ebbinghaus recency decay scoring (in days).
    decay_half_life_days: float = 30.0

    #: Maximum facts retained per category before automatic staleness pruning.
    max_facts_per_category: int = 50

    #: Context budget ratio allocated to user profile facts (default: 40%).
    profile_budget_ratio: float = 0.4

    #: Context budget ratio allocated to domain entities & project knowledge (default: 40%).
    entity_budget_ratio: float = 0.4

    #: Context budget ratio allocated to historical decisions/epics (default: 20%).
    archive_budget_ratio: float = 0.2

    #: Optional host-injected hidden message filter hook.
    should_keep_hidden_message: Callable[[dict[str, Any]], bool] | None = None

    #: Optional host-injected LLM factory hook for advanced extraction.
    host_llm_factory: Callable[..., Any] | None = None

    @classmethod
    def from_backend_config(
        cls,
        backend_config: dict[str, Any] | None,
        **host_hooks: Any,
    ) -> AimemConfig:
        """Parse host backend_config dict layered with host-provided hooks."""
        cfg = backend_config or {}
        return cls(
            storage_path=str(cfg.get("storage_path", "")),
            max_context_facts=int(cfg.get("max_context_facts", 10)),
            min_message_chars=int(cfg.get("min_message_chars", 10)),
            enable_fts=bool(cfg.get("enable_fts", True)),
            fts_db_name=str(cfg.get("fts_db_name", "aimem_index.db")),
            profile_filename=str(cfg.get("profile_filename", "profile.md")),
            entities_filename=str(cfg.get("entities_filename", "entities.md")),
            archive_filename=str(cfg.get("archive_filename", "archive.md")),
            decay_half_life_days=float(cfg.get("decay_half_life_days", 30.0)),
            max_facts_per_category=int(cfg.get("max_facts_per_category", 50)),
            profile_budget_ratio=float(cfg.get("profile_budget_ratio", 0.4)),
            entity_budget_ratio=float(cfg.get("entity_budget_ratio", 0.4)),
            archive_budget_ratio=float(cfg.get("archive_budget_ratio", 0.2)),
            should_keep_hidden_message=host_hooks.get("should_keep_hidden_message"),
            host_llm_factory=host_hooks.get("host_llm_factory"),
        )
