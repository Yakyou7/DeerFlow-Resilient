from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import pytest

from deerflow.sandbox.worktree import GitWorktreeManager, WorktreeContext
from deerflow.subagents.typed_handoff import (
    TypedHandoff,
    negotiate_peer_handoffs,
    parse_typed_handoff,
    render_handoff_section,
)


@pytest.fixture
def temp_git_repo():
    """Create a temporary real Git repository with an initial commit on main."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir).resolve()

        def _git(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *args],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"},
            )

        _git("init", "-b", "main")
        _git("config", "user.name", "Test Agent")
        _git("config", "user.email", "agent@deerflow.ai")

        initial_file = repo_path / "README.md"
        initial_file.write_text("# Test Repo\n", encoding="utf-8")
        _git("add", "README.md")
        _git("commit", "-m", "Initial commit")

        manager = GitWorktreeManager(worktrees_subdir=".deerflow/worktrees")
        yield repo_path, manager


def test_worktree_creation_and_branch_isolation(temp_git_repo):
    repo_path, manager = temp_git_repo

    # 1. Create two parallel worker worktrees
    wt_a = manager.create_worktree(repo_path, "worker_alpha", base_branch="main")
    wt_b = manager.create_worktree(repo_path, "worker_beta", base_branch="main")

    assert wt_a.worktree_path.exists()
    assert wt_b.worktree_path.exists()
    assert wt_a.branch == "worktree/worker_alpha"
    assert wt_b.branch == "worktree/worker_beta"

    # 2. Worker A writes a file
    file_a = wt_a.worktree_path / "alpha_output.txt"
    file_a.write_text("Worker Alpha private content", encoding="utf-8")

    # 3. Worker B writes a file
    file_b = wt_b.worktree_path / "beta_output.txt"
    file_b.write_text("Worker Beta private content", encoding="utf-8")

    # 4. Strict isolation: Worker A cannot see Worker B's file, and vice versa
    assert (wt_a.worktree_path / "alpha_output.txt").exists()
    assert not (wt_a.worktree_path / "beta_output.txt").exists()

    assert (wt_b.worktree_path / "beta_output.txt").exists()
    assert not (wt_b.worktree_path / "alpha_output.txt").exists()

    # 5. Primary repo working directory remains completely untouched and clean
    assert not (repo_path / "alpha_output.txt").exists()
    assert not (repo_path / "beta_output.txt").exists()


def test_worktree_commit_and_clean_merge(temp_git_repo):
    repo_path, manager = temp_git_repo

    wt = manager.create_worktree(repo_path, "worker_feature", base_branch="main")
    feat_file = wt.worktree_path / "feature.py"
    feat_file.write_text("def test_feature(): return True\n", encoding="utf-8")

    # Commit changes in worktree
    commit_sha = manager.commit_worktree(wt, "Add new test_feature function")
    assert commit_sha is not None
    assert len(commit_sha) == 40

    # Clean merge into main
    success, msg = manager.merge_worktree(wt, target_branch="main", auto_commit=False)
    assert success is True
    assert "Successfully merged" in msg

    # Merged file now exists on main repository
    assert (repo_path / "feature.py").exists()
    assert "def test_feature" in (repo_path / "feature.py").read_text(encoding="utf-8")


def test_worktree_conflict_detection_and_safe_abort(temp_git_repo):
    repo_path, manager = temp_git_repo

    # Create a shared file on main first
    shared = repo_path / "shared.txt"
    shared.write_text("line 1\nline 2: original\nline 3\n", encoding="utf-8")
    subprocess.run(["git", "add", "shared.txt"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "commit", "-m", "Add shared.txt"], cwd=str(repo_path), check=True)

    # Worker 1 and Worker 2 branch from current main
    wt1 = manager.create_worktree(repo_path, "worker_1", base_branch="main")
    wt2 = manager.create_worktree(repo_path, "worker_2", base_branch="main")

    # Worker 1 edits line 2
    (wt1.worktree_path / "shared.txt").write_text("line 1\nline 2: worker_1 version\nline 3\n", encoding="utf-8")
    manager.commit_worktree(wt1, "Worker 1 update")

    # Worker 2 edits line 2 differently
    (wt2.worktree_path / "shared.txt").write_text("line 1\nline 2: worker_2 version\nline 3\n", encoding="utf-8")
    manager.commit_worktree(wt2, "Worker 2 update")

    # Worker 1 merges first -> clean
    success1, _ = manager.merge_worktree(wt1, target_branch="main", auto_commit=False)
    assert success1 is True

    # Worker 2 merges second -> conflict!
    success2, msg2 = manager.merge_worktree(wt2, target_branch="main", auto_commit=False)
    assert success2 is False
    assert "conflict" in msg2.lower()

    # Verify main repo is NOT left in dirty conflicted state (merge was safely aborted)
    status_proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_path), capture_output=True, text=True)
    dirty_lines = [line for line in status_proc.stdout.splitlines() if not line.strip().endswith(".deerflow/")]
    assert len(dirty_lines) == 0


def test_peer_to_peer_collision_detection(temp_git_repo):
    repo_path, manager = temp_git_repo

    wt1 = manager.create_worktree(repo_path, "w1", base_branch="main")
    wt2 = manager.create_worktree(repo_path, "w2", base_branch="main")

    # Worker 1 touches file_shared.py and file_w1.py
    (wt1.worktree_path / "file_shared.py").write_text("v1", encoding="utf-8")
    (wt1.worktree_path / "file_w1.py").write_text("w1", encoding="utf-8")

    # Worker 2 touches file_shared.py and file_w2.py
    (wt2.worktree_path / "file_shared.py").write_text("v2", encoding="utf-8")
    (wt2.worktree_path / "file_w2.py").write_text("w2", encoding="utf-8")

    collisions = manager.detect_collisions(wt1, wt2, base_branch="main")
    assert collisions == ["file_shared.py"]


def test_typed_handoff_worktree_metadata_roundtrip():
    raw_text = """
Execution finished.

<handoff>
status: completed
summary: Refactored authentication middleware.
worktree_branch: worktree/worker_auth_42
merge_status: clean
deliverables:
- deerflow/auth/middleware.py
- tests/test_auth.py
next:
- Run integration tests
</handoff>
"""
    handoff = parse_typed_handoff(raw_text)
    assert handoff.status == "completed"
    assert handoff.worktree_branch == "worktree/worker_auth_42"
    assert handoff.merge_status == "clean"
    assert len(handoff.deliverables) == 2

    rendered = render_handoff_section(handoff)
    assert "- **Worktree Branch**: `worktree/worker_auth_42` (merge: clean)" in rendered
    assert "- **Deliverables**:" in rendered
    assert "`deerflow/auth/middleware.py`" in rendered


def test_typed_handoff_peer_negotiation():
    h1 = TypedHandoff(
        status="completed",
        summary="Updated database queries",
        deliverables=["src/db/user.py", "src/models/user.py"],
        worktree_branch="worktree/db_worker",
    )
    h2 = TypedHandoff(
        status="completed",
        summary="Added user validation",
        deliverables=["src/models/user.py", "src/validators/user.py"],
        worktree_branch="worktree/val_worker",
    )

    negotiation = negotiate_peer_handoffs(h1, h2)
    assert negotiation["has_collision"] is True
    assert negotiation["colliding_files"] == ["src/models/user.py"]
    assert negotiation["suggested_action"] == "serialize_or_rebase"

    assert h1.peer_negotiation == negotiation
    assert h2.peer_negotiation == negotiation

    rendered1 = render_handoff_section(h1)
    assert "- **Peer Negotiation**: serialize_or_rebase" in rendered1


def test_worktree_cleanup(temp_git_repo):
    repo_path, manager = temp_git_repo

    wt = manager.create_worktree(repo_path, "worker_cleanup", base_branch="main")
    wt_path = wt.worktree_path
    assert wt_path.exists()

    # Clean up worktree and delete its branch
    removed = manager.remove_worktree(wt, force=True, delete_branch=True)
    assert removed is True
    assert not wt_path.exists()

    # Branch should also be deleted
    branch_check = subprocess.run(
        ["git", "rev-parse", "--verify", wt.branch],
        cwd=str(repo_path),
        capture_output=True,
    )
    assert branch_check.returncode != 0
