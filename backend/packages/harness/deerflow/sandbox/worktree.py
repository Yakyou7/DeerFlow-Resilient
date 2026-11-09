"""Git Worktree Branch Isolation for Concurrent Subagents (Route 4 / RFC-004).

Inspired by:
- Microsoft Research Agensh (Scaling Organizational Intelligence to 1,024 Agents)
- Open Multi-Agent (OMA) Worktree Isolation Protocol
- Google Research harness2 workspace virtualization

Provides lightweight, copy-on-write workspace isolation for concurrent subagents
via native `git worktree` commands, preventing uncommitted dirty state collisions
and git lock contention while enabling peer-to-peer collision negotiation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorktreeContext:
    """Context metadata for an isolated worker worktree."""

    worker_id: str
    branch: str
    worktree_path: Path
    repo_root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "branch": self.branch,
            "worktree_path": str(self.worktree_path),
            "repo_root": str(self.repo_root),
        }


class GitWorktreeManager:
    """Manages lightweight Git worktrees for concurrent subagents."""

    def __init__(self, worktrees_subdir: str = ".deerflow/worktrees") -> None:
        self.worktrees_subdir = worktrees_subdir

    def _run_git(
        self,
        args: list[str],
        cwd: Path | str,
        *,
        timeout: float = 30.0,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a git command safely with timeouts and env isolation."""
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_PAGER"] = "cat"

        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
            check=check,
        )
        return proc

    def is_git_repo(self, path: Path | str) -> bool:
        """Check whether the given path is inside a valid Git repository."""
        path_obj = Path(path)
        if not path_obj.exists():
            return False
        res = self._run_git(["rev-parse", "--is-inside-work-tree"], cwd=path_obj)
        return res.returncode == 0 and res.stdout.strip() == "true"

    def get_current_branch(self, repo_root: Path | str) -> str:
        """Get the current checked-out branch name for the repository."""
        res = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        if res.returncode == 0:
            return res.stdout.strip()
        return "main"

    def create_worktree(
        self,
        repo_root: Path | str,
        worker_id: str,
        base_branch: str | None = None,
    ) -> WorktreeContext:
        """Create a lightweight isolated Git worktree for a worker subagent.

        Args:
            repo_root: Root path of the primary repository.
            worker_id: Unique subagent or worker identifier.
            base_branch: Base branch or commit ref to branch from (default: HEAD).

        Returns:
            WorktreeContext containing the worktree directory and branch.
        """
        root = Path(repo_root).resolve()
        if not self.is_git_repo(root):
            raise ValueError(f"Path is not a valid Git repository: {root}")

        # Ensure worktrees directory is excluded in .git/info/exclude to avoid untracked noise
        exclude_path = root / ".git" / "info" / "exclude"
        if exclude_path.exists():
            try:
                content = exclude_path.read_text(encoding="utf-8")
                if ".deerflow" not in content:
                    exclude_path.write_text(content.rstrip() + "\n.deerflow/\n", encoding="utf-8")
            except OSError:
                pass

        # Sanitize worker_id for branch and directory name
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", worker_id)
        branch_name = f"worktree/{safe_id}"
        worktree_dir = (root / self.worktrees_subdir / safe_id).resolve()

        # If worktree dir already exists, clean it up first
        if worktree_dir.exists():
            self._run_git(["worktree", "remove", "--force", str(worktree_dir)], cwd=root)
            if worktree_dir.exists():
                shutil.rmtree(worktree_dir, ignore_errors=True)

        # Ensure parent directory exists
        worktree_dir.parent.mkdir(parents=True, exist_ok=True)

        base_ref = base_branch or self.get_current_branch(root) or "HEAD"

        # Check if the branch already exists; if so, delete it or detach
        branch_check = self._run_git(["rev-parse", "--verify", branch_name], cwd=root)
        if branch_check.returncode == 0:
            self._run_git(["branch", "-D", branch_name], cwd=root)

        # Run git worktree add -b <branch_name> <worktree_dir> <base_ref>
        add_res = self._run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_dir), base_ref],
            cwd=root,
        )
        if add_res.returncode != 0:
            # Fallback if branch creation failed (e.g. detached HEAD)
            add_res2 = self._run_git(
                ["worktree", "add", "--detach", str(worktree_dir), base_ref],
                cwd=root,
            )
            if add_res2.returncode != 0:
                raise RuntimeError(
                    f"Failed to create git worktree for worker {worker_id}: {add_res.stderr or add_res2.stderr}"
                )

        logger.info(
            "Created Git worktree for worker %s at %s on branch %s",
            worker_id,
            worktree_dir,
            branch_name,
        )
        return WorktreeContext(
            worker_id=worker_id,
            branch=branch_name,
            worktree_path=worktree_dir,
            repo_root=root,
        )

    def get_changed_files(self, ctx: WorktreeContext) -> list[str]:
        """Return list of modified/untracked files in the worktree."""
        res = self._run_git(["status", "--porcelain"], cwd=ctx.worktree_path)
        if res.returncode != 0 or not res.stdout.strip():
            return []
        files: list[str] = []
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if len(line_str) > 3:
                files.append(line_str[3:].strip())
        return files

    def commit_worktree(self, ctx: WorktreeContext, message: str) -> str | None:
        """Stage all changes in the worktree and commit.

        Returns commit hash or None if no changes exist.
        """
        changed = self.get_changed_files(ctx)
        if not changed:
            logger.debug("No changes to commit in worktree %s", ctx.worktree_path)
            return None

        self._run_git(["add", "-A"], cwd=ctx.worktree_path)
        commit_res = self._run_git(["commit", "-m", message], cwd=ctx.worktree_path)
        if commit_res.returncode != 0:
            logger.warning("Worktree commit failed: %s", commit_res.stderr)
            return None

        rev_res = self._run_git(["rev-parse", "HEAD"], cwd=ctx.worktree_path)
        return rev_res.stdout.strip() if rev_res.returncode == 0 else None

    def detect_collisions(
        self,
        ctx1: WorktreeContext,
        ctx2: WorktreeContext,
        base_branch: str = "main",
    ) -> list[str]:
        """Detect file-level collisions (overlapping modified files) between two workers."""
        files1 = set(self.get_changed_files(ctx1))
        files2 = set(self.get_changed_files(ctx2))

        # Also inspect committed diffs relative to base if branches exist
        diff1 = self._run_git(["diff", "--name-only", f"{base_branch}...{ctx1.branch}"], cwd=ctx1.repo_root)
        if diff1.returncode == 0 and diff1.stdout.strip():
            files1.update(diff1.stdout.splitlines())

        diff2 = self._run_git(["diff", "--name-only", f"{base_branch}...{ctx2.branch}"], cwd=ctx2.repo_root)
        if diff2.returncode == 0 and diff2.stdout.strip():
            files2.update(diff2.stdout.splitlines())

        collisions = sorted(files1.intersection(files2))
        return collisions

    def merge_worktree(
        self,
        ctx: WorktreeContext,
        target_branch: str = "main",
        auto_commit: bool = True,
    ) -> tuple[bool, str]:
        """Merge worktree branch into the target branch.

        If conflicts occur, safely aborts the merge to keep the repo clean.

        Returns (success: bool, message: str).
        """
        if auto_commit:
            self.commit_worktree(ctx, f"Auto-commit subagent {ctx.worker_id} work")

        # Checkout target branch in main repository
        root = ctx.repo_root
        current_branch = self.get_current_branch(root)

        if current_branch != target_branch:
            co = self._run_git(["checkout", target_branch], cwd=root)
            if co.returncode != 0:
                return False, f"Failed to switch to target branch {target_branch}: {co.stderr}"

        merge_res = self._run_git(
            ["merge", "--no-ff", "-m", f"Merge worktree branch {ctx.branch}", ctx.branch],
            cwd=root,
        )
        if merge_res.returncode == 0:
            logger.info("Successfully merged %s into %s", ctx.branch, target_branch)
            return True, f"Successfully merged {ctx.branch} into {target_branch}"

        # Conflict detected! Safely abort merge to leave repo in clean state
        abort_res = self._run_git(["merge", "--abort"], cwd=root)
        conflict_msg = merge_res.stderr or merge_res.stdout
        logger.warning("Conflict merging worktree branch %s: %s (aborted)", ctx.branch, conflict_msg)
        return False, f"Merge conflict in branch {ctx.branch}: {conflict_msg.strip()}"

    def remove_worktree(
        self,
        ctx: WorktreeContext,
        force: bool = True,
        delete_branch: bool = False,
    ) -> bool:
        """Remove worktree and prune Git metadata."""
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(ctx.worktree_path))

        res = self._run_git(args, cwd=ctx.repo_root)
        self._run_git(["worktree", "prune"], cwd=ctx.repo_root)

        # Clean directory if leftovers remain
        if ctx.worktree_path.exists():
            shutil.rmtree(ctx.worktree_path, ignore_errors=True)

        if delete_branch:
            self._run_git(["branch", "-D", ctx.branch], cwd=ctx.repo_root)

        logger.info("Removed worktree at %s", ctx.worktree_path)
        return res.returncode == 0
