from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.memory.backends.aimem.aimem_manager import AimemMemoryManager
from deerflow.agents.memory.backends.aimem.config import AimemConfig
from deerflow.agents.memory.backends.aimem.storage import MarkdownStorage, MemoryFact


@pytest.fixture
def temp_aimem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = AimemMemoryManager.from_config(
            backend_config={"storage_path": tmp_dir, "add_only_mode": True, "lazy_conflict_resolution": True},
            mode="middleware",
        )
        try:
            yield manager, tmp_dir
        finally:
            manager.close()


def test_aimem_contract_and_invariants(temp_aimem):
    manager, tmp_dir = temp_aimem
    assert manager.supports_search is True
    assert manager.mode == "middleware"
    assert manager._config.storage_path == tmp_dir
    assert manager._config.add_only_mode is True
    assert manager._config.lazy_conflict_resolution is True


def test_markdown_storage_and_human_edits(temp_aimem):
    _, tmp_dir = temp_aimem
    config = AimemConfig(storage_path=tmp_dir)
    storage = MarkdownStorage(config)

    # 1. Upsert a fact
    fact = MemoryFact(id="pref_01", content="User prefers Python 3.12", category="profile")
    saved, _ = storage.upsert_fact("user_1", fact)
    assert saved.id == "pref_01"

    # Verify file content
    profile_path = Path(tmp_dir) / "aimem" / "users" / "user_1" / "profile.md"
    assert profile_path.is_file()
    content = profile_path.read_text(encoding="utf-8")
    assert "- [pref_01] User prefers Python 3.12" in content

    # 2. Simulate human manual edit directly to the Markdown file
    modified_content = (
        "# User Profile & Preferences\n\n- [pref_01] User prefers Python 3.12 on Linux\n- User prefers dark mode UI\n"
    )
    profile_path.write_text(modified_content, encoding="utf-8")

    # 3. Reload facts from storage
    reloaded = storage.load_user_facts("user_1", category="profile")
    assert len(reloaded) == 2
    f1 = next(f for f in reloaded if f.id == "pref_01")
    assert f1.content == "User prefers Python 3.12 on Linux"

    f2 = next(f for f in reloaded if f.id != "pref_01")
    assert f2.content == "User prefers dark mode UI"
    assert f2.id.startswith("fact_")


def test_fts5_bm25_search_and_escaping(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Add facts
    manager.create_fact("User prefers dark roast pour-over coffee", category="profile", user_id="u1")
    manager.create_fact("Project uses FastAPI and SQLAlchemy for backend", category="entity", user_id="u1")
    manager.create_fact("Database migration completed for SQLite FTS5", category="archive", user_id="u1")

    # 1. Search keyword "coffee"
    results = manager.search("coffee", top_k=5, user_id="u1")
    assert len(results) >= 1
    assert "coffee" in results[0]["content"].lower()
    assert results[0]["category"] == "profile"

    # 2. Search keyword "FastAPI"
    results = manager.search("FastAPI", top_k=5, user_id="u1")
    assert len(results) >= 1
    assert "fastapi" in results[0]["content"].lower()
    assert results[0]["category"] == "entity"

    # 3. Query with special characters (must not crash FTS5 parser)
    safe_results = manager.search('coffee AND OR * "foo : bar ()', top_k=5, user_id="u1")
    assert isinstance(safe_results, list)


def test_gated_message_extraction(temp_aimem):
    manager, tmp_dir = temp_aimem

    # 1. Trivial messages should NOT trigger memory write
    msg_trivial = [
        HumanMessage(content="ok"),
        HumanMessage(content="thanks!"),
        HumanMessage(content="好的"),
        HumanMessage(content="123"),
    ]
    manager.add("thread_1", msg_trivial, user_id="u2")
    mem = manager.get_memory(user_id="u2")
    assert len(mem["facts"]) == 0

    # 2. Genuine preference statement SHOULD trigger memory write
    msg_pref = [
        HumanMessage(content="Please remember that my favorite beverage is iced latte"),
    ]
    manager.add("thread_1", msg_pref, user_id="u2")
    mem = manager.get_memory(user_id="u2")
    assert len(mem["facts"]) == 1
    assert "favorite beverage is iced latte" in mem["facts"][0]["content"]

    # 3. Project fact statement
    msg_project = [
        HumanMessage(content="项目采用 Qoder 本地代理端点进行快速测试"),
    ]
    manager.add("thread_1", msg_project, user_id="u2")
    mem = manager.get_memory(user_id="u2")
    assert len(mem["facts"]) == 2


def test_agent_generated_facts_as_first_class(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Assistant confirms milestone/decision
    msg_agent = [
        AIMessage(content="Successfully verified: SQLite FTS5 migration passed with 0 errors"),
    ]
    manager.add("thread_agent", msg_agent, user_id="u_agent")
    mem = manager.get_memory(user_id="u_agent")
    assert len(mem["facts"]) == 1
    assert mem["facts"][0]["source"] == "agent"
    assert "SQLite FTS5 migration passed" in mem["facts"][0]["content"]
    assert mem["facts"][0]["category"] == "archive"


def test_single_pass_add_only_and_lazy_conflict_resolution(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Turn 1: User specifies favorite framework is PyTorch
    msg1 = [HumanMessage(content="Please remember that my favorite framework is PyTorch")]
    manager.add("t1", msg1, user_id="u_lazy")

    mem1 = manager.get_memory(user_id="u_lazy")
    assert len(mem1["facts"]) == 1
    assert "PyTorch" in mem1["facts"][0]["content"]

    # Turn 2: User changes preference to JAX
    msg2 = [HumanMessage(content="Please remember that my favorite framework is JAX")]
    manager.add("t2", msg2, user_id="u_lazy")

    # In Single-Pass ADD-Only mode:
    # 1. When querying active memory (include_superseded=False), Lazy Conflict Resolution
    #    ensures only the newer active fact (JAX) is surfaced
    mem2 = manager.get_memory(user_id="u_lazy")
    assert len(mem2["facts"]) == 1
    assert "JAX" in mem2["facts"][0]["content"]

    # 2. When querying audit history (include_superseded=True), BOTH facts exist on disk!
    #    Zero destructive deletions!
    mem_audit = manager.get_memory(user_id="u_lazy", include_superseded=True)
    assert len(mem_audit["facts"]) == 2
    contents = [f["content"] for f in mem_audit["facts"]]
    assert any("PyTorch" in c for c in contents)
    assert any("JAX" in c for c in contents)

    # 3. Search also applies Lazy Conflict Resolution: only JAX is returned
    search_res = manager.search("framework", user_id="u_lazy")
    assert len(search_res) == 1
    assert "JAX" in search_res[0]["content"]


def test_entity_boosting_ranking(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Two facts with identical content words but different entity keys
    f_general = MemoryFact(
        id="f_gen",
        content="General documentation note regarding Python configuration",
        category="entity",
        entity_key="documentation",
    )
    f_theme = MemoryFact(
        id="f_boosted",
        content="Theme mode is Dark by default for UI components",
        category="profile",
        entity_key="theme",
    )
    manager._storage.upsert_fact("u_boost", f_general)
    manager._indexer.upsert_fact("u_boost", f_general)
    manager._storage.upsert_fact("u_boost", f_theme)
    manager._indexer.upsert_fact("u_boost", f_theme)

    # Query targeting theme specifically
    results = manager.search("theme mode", user_id="u_boost")
    assert len(results) >= 1
    assert results[0]["id"] == "f_boosted"
    assert results[0]["score"] > 0


def test_get_context_generation(temp_aimem):
    manager, tmp_dir = temp_aimem

    ctx_empty = manager.get_context("u3", query="coffee")
    assert ctx_empty == ""

    manager.create_fact("User timezone is UTC+8", category="profile", user_id="u3")
    manager.create_fact("User works on DeerFlow architecture", category="profile", user_id="u3")

    ctx = manager.get_context("u3", query="timezone")
    assert "User Profile & Long-Term Context:" in ctx
    assert "UTC+8" in ctx


def test_fact_crud_lifecycle(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Create
    mem, fact_id = manager.create_fact("User loves vim keybindings", user_id="u4")
    assert fact_id is not None
    assert len(mem["facts"]) == 1

    # Update
    mem = manager.update_fact(fact_id, content="User loves neovim keybindings", user_id="u4")
    assert mem["facts"][0]["content"] == "User loves neovim keybindings"

    # Search updated content
    found = manager.search("neovim", user_id="u4")
    assert len(found) == 1
    assert found[0]["id"] == fact_id

    # Delete
    mem = manager.delete_fact(fact_id, user_id="u4")
    assert len(mem["facts"]) == 0
    assert len(manager.search("neovim", user_id="u4")) == 0


def test_reload_memory_syncs_human_disk_edits(temp_aimem):
    manager, tmp_dir = temp_aimem

    manager.create_fact("Initial fact A", user_id="u5")

    profile_path = Path(tmp_dir) / "aimem" / "users" / "u5" / "profile.md"
    profile_path.write_text("# Profiles\n\n- [f_manual] Manually injected knowledge about Quantum Computing\n", encoding="utf-8")

    mem = manager.reload_memory(user_id="u5")
    assert len(mem["facts"]) == 1
    assert mem["facts"][0]["id"] == "f_manual"

    search_res = manager.search("Quantum", user_id="u5")
    assert len(search_res) == 1
    assert search_res[0]["id"] == "f_manual"
