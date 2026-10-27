# Plan: Aimem Pluggable Memory Backend

> **Source / Context**: Refactoring DeerFlow's memory subsystem to solve high LLM API costs, lack of human auditability, and context rot during long-horizon multi-agent tasks. Built upon principles from `ai-memory` (Git-backed Markdown Wiki + Zero-LLM heuristic gating + Access-weighted decay) and `ai-agent-book` (Chapter 3: Memory & RAG).

---

## Architectural decisions

1. **Storage Topology (Markdown as Source of Truth, SQLite FTS5 as Derived Index)**:
   - Ground truth records are stored as readable, editable, git-trackable Markdown files under `.deer-flow/memory/wiki/<user_id>/` with YAML frontmatter.
   - SQLite FTS5 (`memory-fts5.sqlite3`) acts strictly as a **derived read index (Derived Cache)**. If the database file is corrupted or deleted, it can be 100% rebuilt by scanning the Markdown files.
   - Solves the opacity of monolithic JSON / binary DB stores and allows humans to directly audit or curate memories in Obsidian, VS Code, or Git.

2. **Compliance with DeerFlow's Pluggable Memory Contract**:
   - Resides cleanly in `backend/packages/harness/deerflow/agents/memory/backends/aimem/`.
   - The **only** import allowed from the host framework is `from deerflow.agents.memory.manager import MemoryManager`. All other configuration and storage paths are passed via `backend_config`.
   - Implements the 3-Tier contract:
     - **Tier 1 (Required)**: `add` (ingest conversation messages), `get_context` (inject relevant memory into LLM prompts).
     - **Tier 2 (Management & Search)**: `search` (BM25 search via SQLite FTS5), `get_memory`, `clear_memory` (conforming to DeerMem shape: `version`, `lastUpdated`, `user`, `history`, `facts[]`).
     - **Tier 3 (Lifecycle hooks)**: `warm`, `shutdown_flush`, `on_pre_compress`.

3. **Zero-LLM Prescreening by Default (Decoupled Ingest)**:
   - Normal conversation turns do **not** invoke expensive, slow LLM extraction prompts (`memory_update.chat.yaml`).
   - Fast deterministic heuristic rules (inspired by `ai-memory-core/src/sanitize.rs` and `entropy_filter.rs`) capture key signals: tool error codes, user architectural preferences, configuration directives, and persistent decisions.
   - LLM extraction is reserved only as an opt-in or offline consolidation pass.

4. **Access-Weighted Retention & Compaction (Mitigating Context Rot)**:
   - Memories have a half-life decay curve. Every time a memory page is hit by `get_context` or `search`, its access count increases and its expiration is extended.
   - When episodic memories grow cold, they are compacted (summarizing transient steps into permanent facts) rather than hard-deleted.

5. **Sub-Agent Typed Handoff State Protocol**:
   - For SuperAgent ↔ Sub-agent delegations, memories bridge session bounds via typed handoff records (`handoff_id`, `task_id`, `status`, `attempted_approaches`, `blockers`, `unresolved_questions`).
   - Incorporates single-claim semantics to prevent multiple sub-agents from colliding or repeating failed exploration paths.

---

## Phase 1: Minimal Drop-in Backend Skeleton (Contract & Factory)

**Goal**: Establish `backends/aimem` with clean discovery, config validation, and factory integration without breaking any existing tests.

### What to build
1. `backends/aimem/config.py`: Pydantic model `AimemConfig` with fields `storage_path`, `max_context_tokens`, `fts_enabled`, `decay_enabled`.
2. `backends/aimem/aimem_manager.py`: `AimemMemoryManager(MemoryManager)` inheriting Pydantic base, implementing minimal `add` and `get_context`.
3. `backends/aimem/__init__.py`: Export `MANAGER_CLASS = AimemMemoryManager`.
4. Register `aimem` in `backend/packages/harness/deerflow/config/memory_config.py` allowed managers list (if constrained).

### Acceptance criteria
- [ ] `AimemMemoryManager` instantiates cleanly via `from_backend_config`.
- [ ] Setting `memory.manager_class: aimem` in `config.yaml` starts DeerFlow without `TypeError` or invariant violation.
- [ ] Calling `get_context` returns safe default text without throwing exceptions.
- [ ] Passes new unit test `tests/test_aimem_memory_backend.py` covering instantiation and discovery.

---

## Phase 2: Markdown Wiki Storage & Derived SQLite FTS5 Indexing

**Goal**: Implement local file persistence as Markdown with YAML frontmatter, paired with an SQLite FTS5 full-text search index.

### What to build
1. **Wiki Store (`wiki_store.py`)**:
   - File layout: `<storage_path>/wiki/<user_id>/<fact_id>.md`.
   - YAML frontmatter: `id`, `title`, `created_at`, `updated_at`, `access_count`, `category` (architecture, tool_error, user_pref, rule).
   - Atomic file write via temporary file + rename.
2. **SQLite FTS5 Derived Index (`fts_index.py`)**:
   - Tables: `memory_fts` (virtual table FTS5 `title`, `content`, `tags`, `user_id`).
   - Auto-initialization and `rebuild_from_markdown()` routine.
3. **`AimemMemoryManager.get_context()` & `search()`**:
   - `search()` queries FTS5 with BM25 ranking, returns list of matching fact dicts.
   - `get_context()` selects top-ranked active facts formatted as clean system injection blocks.
   - `get_memory()` maps Markdown records to standard `MemoryResponse` shape (`{"version": "1.0", "facts": [...]}`).

### Acceptance criteria
- [ ] Writing a fact creates a valid `.md` file on disk with YAML frontmatter.
- [ ] Deleting the `.sqlite3` file and invoking `rebuild_index()` reconstructs all search records from `.md` files.
- [ ] `search("keyword")` accurately retrieves matching Markdown documents.
- [ ] `get_memory()` output matches the DeerMem contract shape required by Gateway API.

---

## Phase 3: Zero-LLM Ingest & Access-Weighted Retention

**Goal**: Filter raw conversation messages into durable facts without LLM calls, and implement biological memory decay.

### What to build
1. **Heuristic Filter (`prescreen_filter.py`)**:
   - Extracts explicit user preferences ("I prefer...", "Always use...", "Never...").
   - Captures tool failure signatures (non-zero exits, stack traces) to prevent repeating mistakes.
   - Replaces per-turn LLM extraction calls with deterministic parsing.
2. **Retention & Decay Policy (`retention.py`)**:
   - Access weight: reading or retrieving a memory increments `access_count` and refreshes `last_accessed_at`.
   - Cold memory sweep: facts with zero accesses over `cold_threshold_days` are marked for compaction or demoted from `get_context`.

### Acceptance criteria
- [ ] Ingesting 10 conversation turns incurs **0 LLM API calls** while successfully capturing explicit user rules and tool errors.
- [ ] Accessing a memory increases its retention priority in subsequent `get_context` calls.
- [ ] Benchmarked latency of `add()` is under 15ms.

---

## Phase 4: Sub-Agent Typed Handoff Protocol

**Goal**: Standardize task state transfer between SuperAgent and spawned Sub-agents to prevent duplicate exploration.

### What to build
1. **Handoff Model (`handoff.py`)**:
   - Schema: `handoff_id`, `task_id`, `from_agent`, `target_agent`, `status` (`pending`, `claimed`, `completed`), `summary`, `blockers`.
2. **Manager Methods**:
   - `create_handoff(...)`: Called upon sub-agent completion or timeout.
   - `claim_handoff(...)`: Single-claim consumption by the next assigned agent.

### Acceptance criteria
- [ ] Sub-agent failure creates a pending handoff containing the error trace and attempted steps.
- [ ] Claiming a handoff transitions its status from `pending` to `claimed`; second claim is rejected.
- [ ] Injected context for resuming agents includes the handoff summary.

---

## Verification & Testing Strategy

- **Unit tests**: `tests/test_aimem_memory_backend.py` (instantiation, config, wiki CRUD, FTS5 rebuild, return shapes).
- **Format & Lint**: `cd backend && make format` (enforce Ruff format and lint rules).
- **Regression test suite**: `cd backend && make test TEST_JOBS=1` (ensure zero breakage across existing DeerFlow features).
