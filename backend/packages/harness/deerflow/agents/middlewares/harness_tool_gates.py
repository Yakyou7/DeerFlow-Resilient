"""Deterministic pre-tool Constraint Gates and executable SOP FSM evaluators (Route C #6)."""

from __future__ import annotations

import ast
import posixpath
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from deerflow.skills.types import SOPStateMachineSpec

if TYPE_CHECKING:
    from deerflow.config.harness_guard_config import HarnessGuardConfig

_WRITE_PATH_TOOLS = frozenset({"write_file", "str_replace", "edit_file"})
_SHELL_TOOLS = frozenset({"bash", "execute", "shell"})
_CODE_ACTION_TOOLS = frozenset({"execute_python", "python_exec", "run_python", "code_action", "local_python"})

_DEFAULT_ALLOWED_LOCAL_IMPORTS = frozenset(
    {"collections", "datetime", "functools", "itertools", "json", "math", "re", "statistics", "typing"}
)
_FORBIDDEN_AST_CALLS = frozenset(
    {"eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "breakpoint", "input", "memoryview"}
)
_SANDBOX_REQUIRED_CALLS = frozenset({"open"})
_SANDBOX_REQUIRED_MODULES = frozenset(
    {
        "asyncio",
        "builtins",
        "ctypes",
        "fcntl",
        "glob",
        "http",
        "importlib",
        "io",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "pty",
        "requests",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
        "httpx",
    }
)

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


@dataclass(frozen=True)
class CodeExecutionDecision:
    """Two-tier AST security classification for Python code actions."""

    allowed_local: bool
    tier: Literal["blocked", "local_ast", "sandbox"]
    reason: str | None = None
    findings: tuple[str, ...] = ()


def classify_code_execution_tier(
    code: str,
    *,
    allowed_imports: Collection[str] | None = None,
) -> CodeExecutionDecision:
    """Inspect Python source via ``ast.parse`` and assign an execution tier.

    - ``blocked``: syntax errors, dynamic code-eval/import primitives
      (``eval``, ``exec``, ``compile``, ``__import__``, ``getattr``/``setattr``),
      or dunder attribute escapes (``__class__``, ``__subclasses__``, etc.).
    - ``sandbox``: IO/system primitives (``open``, ``os``, ``subprocess``,
      ``pathlib``, ``socket``, etc.) or modules outside ``allowed_imports``,
      which MUST downgrade to an isolated sandbox and never run locally.
    - ``local_ast``: pure-memory operations eligible for bounded in-process AST
      evaluation.
    """
    if not isinstance(code, str) or not code.strip():
        return CodeExecutionDecision(
            allowed_local=False,
            tier="blocked",
            reason="Empty or non-string Python code payload.",
            findings=("empty_code",),
        )

    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        return CodeExecutionDecision(
            allowed_local=False,
            tier="blocked",
            reason=f"AST syntax error: {exc}",
            findings=("syntax_error",),
        )

    whitelist = frozenset(
        str(m).strip().split(".")[0]
        for m in (allowed_imports if allowed_imports is not None else _DEFAULT_ALLOWED_LOCAL_IMPORTS)
        if str(m).strip()
    )

    blocked_findings: list[str] = []
    sandbox_findings: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_mod = alias.name.split(".")[0]
                if top_mod == "builtins":
                    blocked_findings.append("import:builtins")
                elif top_mod in _SANDBOX_REQUIRED_MODULES or top_mod not in whitelist:
                    sandbox_findings.append(f"import:{top_mod}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                sandbox_findings.append("relative_import")
            elif node.module:
                top_mod = node.module.split(".")[0]
                if top_mod == "builtins":
                    blocked_findings.append("import:builtins")
                elif top_mod in _SANDBOX_REQUIRED_MODULES or top_mod not in whitelist:
                    sandbox_findings.append(f"import:{top_mod}")
            else:
                sandbox_findings.append("import:unknown")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                blocked_findings.append(f"dunder_name:{node.id}")
            elif node.id in _FORBIDDEN_AST_CALLS:
                blocked_findings.append(f"forbidden_symbol:{node.id}")
            elif node.id in _SANDBOX_REQUIRED_CALLS:
                sandbox_findings.append(f"io_call:{node.id}")
            elif node.id in _SANDBOX_REQUIRED_MODULES:
                sandbox_findings.append(f"module_ref:{node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                blocked_findings.append(f"dunder_attr:{node.attr}")
            elif node.attr in _FORBIDDEN_AST_CALLS:
                blocked_findings.append(f"forbidden_attr:{node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn = node.func.id
                if fn in _FORBIDDEN_AST_CALLS:
                    blocked_findings.append(f"call:{fn}")
                elif fn in _SANDBOX_REQUIRED_CALLS:
                    sandbox_findings.append(f"call:{fn}")
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                base = node.func.value.id
                if base in _SANDBOX_REQUIRED_MODULES:
                    sandbox_findings.append(f"call:{base}.{node.func.attr}")

    if blocked_findings:
        unique_blocked = tuple(dict.fromkeys(blocked_findings))
        return CodeExecutionDecision(
            allowed_local=False,
            tier="blocked",
            reason=f"AST security gate rejected forbidden construct(s): {', '.join(unique_blocked)}.",
            findings=unique_blocked,
        )

    if sandbox_findings:
        unique_sandbox = tuple(dict.fromkeys(sandbox_findings))
        return CodeExecutionDecision(
            allowed_local=False,
            tier="sandbox",
            reason=f"Code requires isolated sandbox execution due to IO/external module(s): {', '.join(unique_sandbox)}.",
            findings=unique_sandbox,
        )

    return CodeExecutionDecision(allowed_local=True, tier="local_ast", reason=None, findings=())


def evaluate_code_action_gate(
    tool_name: str,
    args: Mapping[str, Any] | dict[str, Any] | None,
    config: HarnessGuardConfig,
) -> str | None:
    """Enforce pre-execution AST gate and two-tier routing on code-action tools."""
    if not config.code_ast_gate_enabled or tool_name not in _CODE_ACTION_TOOLS:
        return None
    safe_args = args if isinstance(args, Mapping) else {}
    code = safe_args.get("code") or safe_args.get("script") or safe_args.get("source") or ""
    if not isinstance(code, str):
        return f"Blocked '{tool_name}': code argument must be a string."

    decision = classify_code_execution_tier(code, allowed_imports=config.allowed_local_python_imports)
    if decision.tier == "blocked":
        return f"Blocked '{tool_name}': {decision.reason}"
    if decision.tier == "sandbox":
        requested_tier = str(safe_args.get("execution_tier") or safe_args.get("target") or "").strip().lower()
        if requested_tier != "sandbox":
            return f"Blocked '{tool_name}': {decision.reason} Route to isolated sandbox instead of local execution."
    return None


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

    # 4. Code-as-Action AST gate & two-tier routing
    code_denial = evaluate_code_action_gate(tool_name, safe_args, config)
    if code_denial is not None:
        return code_denial

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
