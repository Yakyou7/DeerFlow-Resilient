"""Bounded in-process AST Python evaluator for pure-memory data aggregation (Route C).

Implements a minimal smolagents-style LocalPythonInterpreter backed by Python's
standard library ``ast`` module without calling ``exec()`` or ``eval()``.
Not a security boundary for untrusted IO: any code requiring IO or non-whitelist
modules is deterministically routed to ``tier="sandbox"`` and refused locally.
"""

from __future__ import annotations

import ast
import importlib
import operator
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from deerflow.agents.middlewares.harness_tool_gates import (
    _DEFAULT_ALLOWED_LOCAL_IMPORTS,
    classify_code_execution_tier,
)

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _FinalAnswerSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class LocalAstSecurityError(RuntimeError):
    """Raised when AST evaluation encounters a disallowed operation or budget cap."""


@dataclass(frozen=True)
class LocalExecutionResult:
    """Structured result of two-tier code-action routing and execution."""

    ok: bool
    tier: str
    output: Any = None
    stdout: str = ""
    error: str | None = None


class LocalAstPythonExecutor:
    """Bounded pure-memory AST interpreter with mandatory two-tier sandbox routing."""

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        allowed_imports: Collection[str] | None = None,
        max_operations: int = 10000,
        max_output_chars: int = 16384,
        max_call_depth: int = 32,
    ) -> None:
        self.tools = dict(tools or {})
        self.allowed_imports = frozenset(
            str(m).strip().split(".")[0]
            for m in (allowed_imports if allowed_imports is not None else _DEFAULT_ALLOWED_LOCAL_IMPORTS)
            if str(m).strip()
        )
        self.max_operations = max(100, int(max_operations))
        self.max_output_chars = max(256, int(max_output_chars))
        self.max_call_depth = max(4, int(max_call_depth))

    def execute(
        self,
        code: str,
        *,
        variables: Mapping[str, Any] | None = None,
        sandbox_runner: Callable[[str], Any] | None = None,
    ) -> LocalExecutionResult:
        """Route code through AST security gate; run locally only if ``tier == 'local_ast'``."""
        decision = classify_code_execution_tier(code, allowed_imports=self.allowed_imports)
        if decision.tier == "blocked":
            return LocalExecutionResult(ok=False, tier="blocked", error=decision.reason)

        if decision.tier == "sandbox":
            if sandbox_runner is not None:
                try:
                    out = sandbox_runner(code)
                    return LocalExecutionResult(ok=True, tier="sandbox", output=out)
                except Exception as exc:
                    return LocalExecutionResult(ok=False, tier="sandbox", error=str(exc))
            return LocalExecutionResult(
                ok=False,
                tier="sandbox",
                error=decision.reason or "Code requires isolated sandbox execution.",
            )

        tree = ast.parse(code, mode="exec")
        stdout_parts: list[str] = []
        stdout_len = 0

        def _capture_print(*args: Any, sep: str = " ", end: str = "\n") -> None:
            nonlocal stdout_len
            chunk = sep.join(str(a) for a in args) + end
            if stdout_len < self.max_output_chars:
                remaining = self.max_output_chars - stdout_len
                stdout_parts.append(chunk[:remaining])
                stdout_len += min(len(chunk), remaining)

        def _final_answer(val: Any = None) -> None:
            raise _FinalAnswerSignal(val)

        env: dict[str, Any] = {**_SAFE_BUILTINS, **self.tools, "print": _capture_print, "final_answer": _final_answer}
        if variables:
            env.update(variables)

        ops = 0

        def _tick() -> None:
            nonlocal ops
            ops += 1
            if ops > self.max_operations:
                raise LocalAstSecurityError(f"Local AST operation budget exceeded ({self.max_operations}).")

        def _assign_target(target: ast.AST, value: Any, scope: dict[str, Any]) -> None:
            if isinstance(target, ast.Name):
                if target.id.startswith("_"):
                    raise LocalAstSecurityError(f"Assignment to underscore symbol '{target.id}' is not permitted.")
                scope[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)):
                seq = list(value)
                if len(seq) != len(target.elts):
                    raise ValueError(f"Cannot unpack {len(seq)} values into {len(target.elts)} targets.")
                for elt, item in zip(target.elts, seq, strict=True):
                    _assign_target(elt, item, scope)
            elif isinstance(target, ast.Subscript):
                container = _eval_expr(target.value, scope, depth=0)
                idx = _eval_expr(target.slice, scope, depth=0)
                container[idx] = value
            else:
                raise LocalAstSecurityError(f"Unsupported assignment target: {type(target).__name__}")

        def _eval_expr(node: ast.AST, scope: dict[str, Any], depth: int) -> Any:
            _tick()
            if depth > self.max_call_depth:
                raise LocalAstSecurityError("Maximum AST evaluation depth exceeded.")

            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id in scope:
                    return scope[node.id]
                if node.id in env:
                    return env[node.id]
                raise NameError(f"name '{node.id}' is not defined")
            if isinstance(node, ast.List):
                return [_eval_expr(e, scope, depth + 1) for e in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple(_eval_expr(e, scope, depth + 1) for e in node.elts)
            if isinstance(node, ast.Set):
                return {_eval_expr(e, scope, depth + 1) for e in node.elts}
            if isinstance(node, ast.Dict):
                return {
                    _eval_expr(k, scope, depth + 1): _eval_expr(v, scope, depth + 1)
                    for k, v in zip(node.keys, node.values, strict=True)
                    if k is not None
                }
            if isinstance(node, ast.JoinedStr):
                return "".join(str(_eval_expr(v, scope, depth + 1)) for v in node.values)
            if isinstance(node, ast.FormattedValue):
                val = _eval_expr(node.value, scope, depth + 1)
                if node.format_spec is not None:
                    spec = str(_eval_expr(node.format_spec, scope, depth + 1))
                    return format(val, spec)
                return str(val)
            if isinstance(node, ast.UnaryOp):
                op_fn = _UNARY_OPS.get(type(node.op))
                if op_fn is None:
                    raise LocalAstSecurityError(f"Unsupported unary op: {type(node.op).__name__}")
                return op_fn(_eval_expr(node.operand, scope, depth + 1))
            if isinstance(node, ast.BinOp):
                op_fn = _BIN_OPS.get(type(node.op))
                if op_fn is None:
                    raise LocalAstSecurityError(f"Unsupported binary op: {type(node.op).__name__}")
                left = _eval_expr(node.left, scope, depth + 1)
                right = _eval_expr(node.right, scope, depth + 1)
                if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
                    raise LocalAstSecurityError("Exponent exceeds safe local AST bound.")
                return op_fn(left, right)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    res: Any = True
                    for v in node.values:
                        res = _eval_expr(v, scope, depth + 1)
                        if not res:
                            return res
                    return res
                if isinstance(node.op, ast.Or):
                    res = False
                    for v in node.values:
                        res = _eval_expr(v, scope, depth + 1)
                        if res:
                            return res
                    return res
                raise LocalAstSecurityError(f"Unsupported bool op: {type(node.op).__name__}")
            if isinstance(node, ast.Compare):
                left = _eval_expr(node.left, scope, depth + 1)
                for cmp_op, comp_node in zip(node.ops, node.comparators, strict=True):
                    cmp_fn = _CMP_OPS.get(type(cmp_op))
                    if cmp_fn is None:
                        raise LocalAstSecurityError(f"Unsupported comparison: {type(cmp_op).__name__}")
                    right = _eval_expr(comp_node, scope, depth + 1)
                    if not cmp_fn(left, right):
                        return False
                    left = right
                return True
            if isinstance(node, ast.IfExp):
                cond = _eval_expr(node.test, scope, depth + 1)
                return _eval_expr(node.body if cond else node.orelse, scope, depth + 1)
            if isinstance(node, ast.Slice):
                lower = _eval_expr(node.lower, scope, depth + 1) if node.lower else None
                upper = _eval_expr(node.upper, scope, depth + 1) if node.upper else None
                step = _eval_expr(node.step, scope, depth + 1) if node.step else None
                return slice(lower, upper, step)
            if isinstance(node, ast.Subscript):
                obj = _eval_expr(node.value, scope, depth + 1)
                slc = _eval_expr(node.slice, scope, depth + 1)
                return obj[slc]
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    raise LocalAstSecurityError(f"Attribute access '{node.attr}' is forbidden in local AST tier.")
                obj = _eval_expr(node.value, scope, depth + 1)
                return getattr(obj, node.attr)
            if isinstance(node, ast.Lambda):
                arg_names = [a.arg for a in node.args.args]

                def _lambda_fn(*call_args: Any, _closure: dict[str, Any] = scope) -> Any:
                    local_scope = dict(_closure)
                    for name, val in zip(arg_names, call_args, strict=True):
                        local_scope[name] = val
                    return _eval_expr(node.body, local_scope, depth + 1)

                return _lambda_fn
            if isinstance(node, ast.Call):
                if node.keywords and any(kw.arg is None for kw in node.keywords):
                    raise LocalAstSecurityError("Keyword unpacking (**kwargs) is not permitted in local AST tier.")
                fn = _eval_expr(node.func, scope, depth + 1)
                pos_args: list[Any] = []
                for arg in node.args:
                    if isinstance(arg, ast.Starred):
                        pos_args.extend(_eval_expr(arg.value, scope, depth + 1))
                    else:
                        pos_args.append(_eval_expr(arg, scope, depth + 1))
                kw_args = {kw.arg: _eval_expr(kw.value, scope, depth + 1) for kw in node.keywords if kw.arg is not None}
                return fn(*pos_args, **kw_args)
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                items: list[Any] = []
                _eval_comprehension(node.generators, 0, scope, depth + 1, lambda s: items.append(_eval_expr(node.elt, s, depth + 1)))
                if isinstance(node, ast.SetComp):
                    return set(items)
                return items
            if isinstance(node, ast.DictComp):
                out_dict: dict[Any, Any] = {}
                def _add_kv(s: dict[str, Any]) -> None:
                    k = _eval_expr(node.key, s, depth + 1)
                    v = _eval_expr(node.value, s, depth + 1)
                    out_dict[k] = v

                _eval_comprehension(node.generators, 0, scope, depth + 1, _add_kv)
                return out_dict

            raise LocalAstSecurityError(f"AST node '{type(node).__name__}' is not permitted in local AST tier.")

        def _eval_comprehension(
            generators: list[ast.comprehension],
            idx: int,
            scope: dict[str, Any],
            depth: int,
            on_leaf: Callable[[dict[str, Any]], None],
        ) -> None:
            if idx >= len(generators):
                on_leaf(scope)
                return
            gen = generators[idx]
            iterable = _eval_expr(gen.iter, scope, depth + 1)
            for item in iterable:
                _tick()
                child = dict(scope)
                _assign_target(gen.target, item, child)
                if all(_eval_expr(cond, child, depth + 1) for cond in gen.ifs):
                    _eval_comprehension(generators, idx + 1, child, depth + 1, on_leaf)

        last_expr_value: Any = None

        def _exec_stmts(stmts: list[ast.stmt], scope: dict[str, Any], depth: int) -> None:
            nonlocal last_expr_value
            for stmt in stmts:
                _tick()
                if isinstance(stmt, ast.Pass):
                    continue
                if isinstance(stmt, ast.Break):
                    raise _BreakSignal()
                if isinstance(stmt, ast.Continue):
                    raise _ContinueSignal()
                if isinstance(stmt, ast.Return):
                    val = _eval_expr(stmt.value, scope, depth + 1) if stmt.value is not None else None
                    raise _ReturnSignal(val)
                if isinstance(stmt, ast.Expr):
                    last_expr_value = _eval_expr(stmt.value, scope, depth + 1)
                    continue
                if isinstance(stmt, ast.Assign):
                    val = _eval_expr(stmt.value, scope, depth + 1)
                    for target in stmt.targets:
                        _assign_target(target, val, scope)
                    continue
                if isinstance(stmt, ast.AugAssign):
                    cur = _eval_expr(stmt.target, scope, depth + 1)
                    rhs = _eval_expr(stmt.value, scope, depth + 1)
                    op_fn = _BIN_OPS.get(type(stmt.op))
                    if op_fn is None:
                        raise LocalAstSecurityError(f"Unsupported aug-assign op: {type(stmt.op).__name__}")
                    _assign_target(stmt.target, op_fn(cur, rhs), scope)
                    continue
                if isinstance(stmt, ast.AnnAssign):
                    if stmt.value is not None:
                        val = _eval_expr(stmt.value, scope, depth + 1)
                        _assign_target(stmt.target, val, scope)
                    continue
                if isinstance(stmt, ast.If):
                    branch = stmt.body if _eval_expr(stmt.test, scope, depth + 1) else stmt.orelse
                    _exec_stmts(branch, scope, depth + 1)
                    continue
                if isinstance(stmt, ast.For):
                    iterable = _eval_expr(stmt.iter, scope, depth + 1)
                    for item in iterable:
                        _tick()
                        _assign_target(stmt.target, item, scope)
                        try:
                            _exec_stmts(stmt.body, scope, depth + 1)
                        except _ContinueSignal:
                            continue
                        except _BreakSignal:
                            break
                    else:
                        _exec_stmts(stmt.orelse, scope, depth + 1)
                    continue
                if isinstance(stmt, ast.While):
                    while _eval_expr(stmt.test, scope, depth + 1):
                        _tick()
                        try:
                            _exec_stmts(stmt.body, scope, depth + 1)
                        except _ContinueSignal:
                            continue
                        except _BreakSignal:
                            break
                    else:
                        _exec_stmts(stmt.orelse, scope, depth + 1)
                    continue
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        top_mod = alias.name.split(".")[0]
                        if top_mod not in self.allowed_imports:
                            raise LocalAstSecurityError(f"Import '{alias.name}' is not in allowed local imports.")
                        mod = importlib.import_module(alias.name)
                        bind_name = alias.asname or top_mod
                        scope[bind_name] = mod if alias.asname else importlib.import_module(top_mod)
                    continue
                if isinstance(stmt, ast.ImportFrom):
                    if not stmt.module or stmt.module.split(".")[0] not in self.allowed_imports:
                        raise LocalAstSecurityError(f"ImportFrom '{stmt.module}' is not in allowed local imports.")
                    mod = importlib.import_module(stmt.module)
                    for alias in stmt.names:
                        if alias.name == "*" or alias.name.startswith("_"):
                            raise LocalAstSecurityError(f"Forbidden import member '{alias.name}'.")
                        scope[alias.asname or alias.name] = getattr(mod, alias.name)
                    continue
                if isinstance(stmt, ast.FunctionDef):
                    param_names = [a.arg for a in stmt.args.args]
                    defaults = [_eval_expr(d, scope, depth + 1) for d in stmt.args.defaults]

                    def _user_fn(*call_args: Any, _fn_node: ast.FunctionDef = stmt, _params: list[str] = param_names, _defs: list[Any] = defaults, **call_kwargs: Any) -> Any:
                        if depth + 1 > self.max_call_depth:
                            raise LocalAstSecurityError("Maximum function call depth exceeded.")
                        fn_scope = dict(scope)
                        offset = len(_params) - len(_defs)
                        for idx_p, pname in enumerate(_params):
                            if idx_p < len(call_args):
                                fn_scope[pname] = call_args[idx_p]
                            elif pname in call_kwargs:
                                fn_scope[pname] = call_kwargs[pname]
                            elif idx_p >= offset:
                                fn_scope[pname] = _defs[idx_p - offset]
                            else:
                                raise TypeError(f"Missing required argument '{pname}'")
                        try:
                            _exec_stmts(_fn_node.body, fn_scope, depth + 1)
                        except _ReturnSignal as ret:
                            return ret.value
                        return None

                    scope[stmt.name] = _user_fn
                    continue

                raise LocalAstSecurityError(f"AST statement '{type(stmt).__name__}' is not permitted in local AST tier.")

        try:
            _exec_stmts(tree.body, env, 0)
            output = env.get("result", last_expr_value)
            return LocalExecutionResult(ok=True, tier="local_ast", output=output, stdout="".join(stdout_parts))
        except _FinalAnswerSignal as fin:
            return LocalExecutionResult(ok=True, tier="local_ast", output=fin.value, stdout="".join(stdout_parts))
        except Exception as exc:
            return LocalExecutionResult(ok=False, tier="local_ast", stdout="".join(stdout_parts), error=str(exc))
