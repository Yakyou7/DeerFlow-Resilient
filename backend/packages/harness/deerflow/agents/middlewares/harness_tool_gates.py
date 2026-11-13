"""Deterministic pre-tool Constraint Gates and executable SOP FSM evaluators (Route C #6)."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from deerflow.skills.types import SOPStateMachineSpec

if TYPE_CHECKING:
    from deerflow.config.harness_guard_config import HarnessGuardConfig

_WRITE_PATH_TOOLS = frozenset({"write_file", "str_replace", "edit_file"})
_SHELL_TOOLS = frozenset({"bash", "execute", "shell"})

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_SHELL_SPLIT_RE = re.compile(r"(?:\r?\n|&&|\|\||[;|])")
_ENV_ASSIGN_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|\S+)\s+)+")


def _normalize_posix_path(raw: str) -> str:
    """Normalize slash direction and collapse dot segments without hitting disk."""
    replaced = str(raw).replace("\\", "/").strip()
    if not replaced:
        return ""
    norm = posixpath.normpath(replaced)
    if norm == ".":
        return "."
    return norm.rstrip("/") or "/"


def _path_matches_prefix(target: str, prefix: str) -> bool:
    if not prefix:
        return False
    if prefix == "/":
        return target.startswith("/")
    return target == prefix or target.startswith(prefix + "/")


def _iter_arg_strings(value: Any, *, depth: int = 0) -> list[str]:
    """Flatten nested tool-call arguments up to bounded depth for domain inspection."""
    if depth > 4 or value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for v in value.values():
            out.extend(_iter_arg_strings(v, depth=depth + 1))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_iter_arg_strings(item, depth=depth + 1))
        return out
    return []


def _host_allowed(host: str, allowed_domains: list[str]) -> bool:
    norm_host = host.strip().lower().rstrip(".")
    if not norm_host:
        return False
    for domain in allowed_domains:
        norm_domain = domain.strip().lower().lstrip(".").rstrip(".")
        if not norm_domain:
            continue
        if norm_host == norm_domain or norm_host.endswith("." + norm_domain):
            return True
    return False


def _extract_command_segments(command: str) -> list[str]:
    segments: list[str] = []
    for raw_seg in _SHELL_SPLIT_RE.split(command):
        seg = raw_seg.strip()
        if not seg:
            continue
        seg = _ENV_ASSIGN_RE.sub("", seg).strip()
        normalized = " ".join(seg.split()).lower()
        if normalized:
            segments.append(normalized)
    return segments


def evaluate_constraint_tool_gate(
    tool_name: str,
    args: Mapping[str, Any] | dict[str, Any] | None,
    config: HarnessGuardConfig,
) -> str | None:
    """Evaluate machine-checkable constraints before executing a tool call.

    Returns a denial reason string if blocked, or ``None`` if permitted.
    """
    safe_args = args if isinstance(args, Mapping) else {}

    # 1. Write path allowlist
    if config.allowed_write_paths and tool_name in _WRITE_PATH_TOOLS:
        raw_path = safe_args.get("path") or safe_args.get("file_path") or ""
        if not isinstance(raw_path, str) or "\x00" in raw_path or not raw_path.strip():
            return f"Blocked '{tool_name}': missing or invalid target path under active write allowlist {config.allowed_write_paths}."
        norm_target = _normalize_posix_path(raw_path)
        prefixes = [_normalize_posix_path(p) for p in config.allowed_write_paths if p and p.strip()]
        if not any(_path_matches_prefix(norm_target, p) for p in prefixes):
            return f"Blocked '{tool_name}': path '{norm_target}' is outside allowed write paths {prefixes}."

    # 2. Domain / email host allowlist
    if config.allowed_domains:
        for text in _iter_arg_strings(safe_args):
            for match in _URL_RE.findall(text):
                parsed = urlparse(match)
                host = parsed.hostname or ""
                if host and not _host_allowed(host, config.allowed_domains):
                    return f"Blocked '{tool_name}': URL host '{host}' is not in allowed domains {config.allowed_domains}."
            for domain in _EMAIL_RE.findall(text):
                if domain and not _host_allowed(domain, config.allowed_domains):
                    return f"Blocked '{tool_name}': email domain '{domain}' is not in allowed domains {config.allowed_domains}."

    # 3. Forbidden command prefixes
    if config.blocked_command_prefixes and tool_name in _SHELL_TOOLS:
        raw_cmd = safe_args.get("command") or safe_args.get("cmd") or ""
        if isinstance(raw_cmd, str) and raw_cmd.strip():
            blocked = [" ".join(p.split()).lower() for p in config.blocked_command_prefixes if p and p.strip()]
            for seg in _extract_command_segments(raw_cmd):
                for prefix in blocked:
                    if seg == prefix or seg.startswith(prefix + " ") or seg.startswith(prefix + "/"):
                        return f"Blocked '{tool_name}': command segment '{seg}' matches forbidden prefix '{prefix}'."

    return None


def known_sop_states(spec: SOPStateMachineSpec) -> set[str]:
    """Return every declared state name in an SOP state machine specification."""
    states: set[str] = {spec.initial_state}
    states.update(spec.exit_states)
    states.update(spec.allowed_tools_by_state.keys())
    for src, edges in spec.transitions.items():
        if src != "*":
            states.add(src)
        states.update(edges.values())
    return states


def evaluate_sop_tool_gate(
    spec: SOPStateMachineSpec,
    current_state: str,
    tool_name: str,
) -> str | None:
    """Enforce SOP per-state tool permission table before tool execution."""
    allowed = spec.allowed_tools_by_state.get(current_state)
    if allowed is None:
        return None
    if tool_name in allowed:
        return None
    permitted_str = ", ".join(sorted(allowed)) if allowed else "none"
    return f"Blocked '{tool_name}': SOP '{spec.name}' is in state '{current_state}', which only permits tools: [{permitted_str}]."


def next_sop_state(
    spec: SOPStateMachineSpec,
    current_state: str,
    tool_name: str,
    *,
    succeeded: bool,
) -> str:
    """Advance SOP state deterministically on successful tool execution only."""
    if not succeeded:
        return current_state
    candidate = spec.transitions.get(current_state, {}).get(tool_name)
    if candidate is None:
        candidate = spec.transitions.get("*", {}).get(tool_name)
    if candidate and candidate in known_sop_states(spec):
        return candidate
    return current_state


def format_sop_stop_gate_message(spec: SOPStateMachineSpec, current_state: str) -> str:
    """Format an actionable continuation prompt when Stop Gate rejects premature exit."""
    allowed = spec.allowed_tools_by_state.get(current_state)
    permitted_str = ", ".join(sorted(allowed)) if allowed else "any permitted tool"
    exit_str = ", ".join(sorted(spec.exit_states)) or "terminal state"
    return (
        f"[HARNESS GUARD: SOP STOP GATE] Cannot conclude while SOP '{spec.name}' is in non-terminal "
        f"state '{current_state}' (required exit state: {exit_str}). Permitted tools in "
        f"'{current_state}': [{permitted_str}]. Execute the next required step to advance the workflow."
    )
