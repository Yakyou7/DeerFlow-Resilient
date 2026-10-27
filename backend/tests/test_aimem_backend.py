from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from deerflow.agents.memory.backends.aimem.aimem_manager import AimemMemoryManager
from deerflow.agents.memory.backends.aimem.config import AimemConfig
from deerflow.agents.memory.backends.aimem.storage import MarkdownStorage, MemoryFact


@pytest.fixture
def temp_aimem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = AimemMemoryManager.from_config(
            backend_config={"storage_path": tmp_dir},
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
    # Human adds a new line without ID and edits pref_01
    modified_content = "# User Profile & Preferences\n\n- [pref_01] User prefers Python 3.12 on Linux\n- User prefers dark mode UI\n"
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


def test_get_context_generation(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Empty context when no facts
    ctx_empty = manager.get_context("u3", query="coffee")
    assert ctx_empty == ""

    # Populate facts
    manager.create_fact("User timezone is UTC+8", category="profile", user_id="u3")
    manager.create_fact("User works on DeerFlow architecture", category="profile", user_id="u3")

    # Query context
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

    # Add initial fact
    manager.create_fact("Initial fact A", user_id="u5")

    # Manually edit the markdown file on disk
    profile_path = Path(tmp_dir) / "aimem" / "users" / "u5" / "profile.md"
    profile_path.write_text("# Profiles\n\n- [f_manual] Manually injected knowledge about Quantum Computing\n", encoding="utf-8")

    # Call reload_memory
    mem = manager.reload_memory(user_id="u5")
    assert len(mem["facts"]) == 1
    assert mem["facts"][0]["id"] == "f_manual"

    # Verify FTS5 index was resynced
    search_res = manager.search("Quantum", user_id="u5")
    assert len(search_res) == 1
    assert search_res[0]["id"] == "f_manual"
