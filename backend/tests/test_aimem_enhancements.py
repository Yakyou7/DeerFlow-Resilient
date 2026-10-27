from __future__ import annotations

import datetime
import tempfile

import pytest
from langchain_core.messages import HumanMessage

from deerflow.agents.memory.backends.aimem.aimem_manager import AimemMemoryManager
from deerflow.agents.memory.backends.aimem.storage import MemoryFact


@pytest.fixture
def temp_aimem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = AimemMemoryManager.from_config(
            backend_config={
                "storage_path": tmp_dir,
                "decay_half_life_days": 30.0,
                "max_facts_per_category": 5,
                "profile_budget_ratio": 0.4,
                "entity_budget_ratio": 0.4,
                "archive_budget_ratio": 0.2,
            },
            mode="middleware",
        )
        try:
            yield manager, tmp_dir
        finally:
            manager.close()


def test_ebbinghaus_recency_decay_scoring(temp_aimem):
    manager, tmp_dir = temp_aimem
    now = datetime.datetime.now(datetime.UTC)
    old_date = (now - datetime.timedelta(days=90)).isoformat()
    fresh_date = now.isoformat()

    # Create an old fact (90 days old -> 3 half-lives -> 0.125 recency multiplier)
    fact_old = MemoryFact(
        id="old_doc",
        content="Deployment target is Kubernetes on AWS EKS",
        category="entity",
        created_at=old_date,
        updated_at=old_date,
    )
    manager._storage.upsert_fact("u_decay", fact_old)
    manager._indexer.upsert_fact("u_decay", fact_old)

    # Create a fresh fact (today -> 1.0 recency multiplier)
    fact_fresh = MemoryFact(
        id="fresh_doc",
        content="Deployment target is bare-metal Kubernetes cluster",
        category="entity",
        created_at=fresh_date,
        updated_at=fresh_date,
    )
    manager._storage.upsert_fact("u_decay", fact_fresh)
    manager._indexer.upsert_fact("u_decay", fact_fresh)

    # Search for "Kubernetes"
    results = manager.search("Kubernetes", top_k=5, user_id="u_decay")
    assert len(results) == 2
    # The fresh fact should rank FIRST due to recency decay weighting
    assert results[0]["id"] == "fresh_doc"
    assert results[1]["id"] == "old_doc"
    assert results[0]["score"] > results[1]["score"]


def test_access_frequency_boost(temp_aimem):
    manager, tmp_dir = temp_aimem
    now = datetime.datetime.now(datetime.UTC).isoformat()

    # Two facts with identical timestamps
    f1 = MemoryFact(id="f1", content="Compiler flag optimization is O2", category="entity", created_at=now, updated_at=now)
    f2 = MemoryFact(id="f2", content="Compiler flag optimization is O3", category="entity", created_at=now, updated_at=now)

    manager._storage.upsert_fact("u_freq", f1)
    manager._indexer.upsert_fact("u_freq", f1)
    manager._storage.upsert_fact("u_freq", f2)
    manager._indexer.upsert_fact("u_freq", f2)

    # Simulate f2 being frequently accessed (retrieved multiple times)
    for _ in range(5):
        manager._storage.touch_facts("u_freq", ["f2"])
        manager._indexer.touch_facts(["f2"])

    # Search: f2 should now rank higher than f1 due to frequency boost
    results = manager.search("Compiler flag", top_k=5, user_id="u_freq")
    assert len(results) == 2
    assert results[0]["id"] == "f2"
    assert results[0]["access_count"] >= 5
    assert results[0]["score"] > results[1]["score"]


def test_contradiction_superseding(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Turn 1: Store favorite framework as PyTorch
    msg1 = [HumanMessage(content="Please remember that my favorite framework is PyTorch")]
    manager.add("t1", msg1, user_id="u_contra")

    mem1 = manager.get_memory(user_id="u_contra")
    assert len(mem1["facts"]) == 1
    assert "PyTorch" in mem1["facts"][0]["content"]

    # Turn 2: User explicitly changes preference to JAX
    msg2 = [HumanMessage(content="Please remember that my favorite framework is JAX")]
    manager.add("t2", msg2, user_id="u_contra")

    mem2 = manager.get_memory(user_id="u_contra")
    # There should NOT be 2 conflicting facts; the older PyTorch preference should be superseded!
    assert len(mem2["facts"]) == 1
    assert "JAX" in mem2["facts"][0]["content"]

    # Verify search only returns the new superseding fact
    results = manager.search("framework", user_id="u_contra")
    assert len(results) == 1
    assert "JAX" in results[0]["content"]


def test_category_budget_allocation(temp_aimem):
    manager, tmp_dir = temp_aimem

    # Insert 6 profile facts, 6 entity facts, 6 archive facts
    for i in range(6):
        manager.create_fact(f"User preference #{i}", category="profile", user_id="u_budget")
        manager.create_fact(f"Project entity system #{i}", category="entity", user_id="u_budget")
        manager.create_fact(f"Historical decision log #{i}", category="archive", user_id="u_budget")

    # Request context with max 10 facts (budget: 4 profile, 4 entity, 2 archive)
    manager._config.max_context_facts = 10
    ctx = manager.get_context("u_budget", query="system preference")

    lines = ctx.splitlines()
    profile_lines = [line for line in lines if "(profile)" in line]
    entity_lines = [line for line in lines if "(entity)" in line]
    archive_lines = [line for line in lines if "(archive)" in line]

    # Check budget quotas: neither category starves the others
    assert len(profile_lines) == 4
    assert len(entity_lines) == 4
    assert len(archive_lines) == 2


def test_staleness_eviction(temp_aimem):
    manager, tmp_dir = temp_aimem
    # Config max_facts_per_category is 5

    # Insert 7 facts into entity category
    for i in range(7):
        manager.create_fact(f"Temporary note #{i}", category="entity", user_id="u_evict")

    facts = manager._storage.load_user_facts("u_evict", category="entity")
    # Must be capped at max_facts_per_category (5)
    assert len(facts) == 5
