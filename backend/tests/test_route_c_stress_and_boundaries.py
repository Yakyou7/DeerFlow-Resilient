"""Stress, boundary, and adversarial tests for Route C (RFC-003 Stage 3).

Validates:
1. Adversarial XML and tag injection resilience in Typed Handoff.
2. Malformed, truncated, and massive binary/text outputs.
3. Multi-threaded high-concurrency race condition and deadlock freedom in HarnessGuard.
4. Deep traceback regex backtracking safety (ReDoS protection).
5. Edge cases in Write-Verify syntax validation.
"""

from __future__ import annotations

import concurrent.futures
import time
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from deerflow.agents.middlewares.harness_guard_middleware import (
    HarnessGuardMiddleware,
    classify_tool_error,
    verify_content_syntax,
)
from deerflow.config.harness_guard_config import HarnessGuardConfig
from deerflow.subagents.typed_handoff import (
    TypedHandoff,
    parse_typed_handoff,
)


class TestAdversarialAndMalformedHandoff:
    def test_unclosed_and_nested_handoff_tags(self):
        # Adversarial unclosed tag
        text = "Process log...\n<handoff>\nstatus: completed\ndeliverables:\n- /workspace/data.csv"
        handoff = parse_typed_handoff(text, status="completed")
        # Should gracefully fall back without raising or hanging
        assert isinstance(handoff, TypedHandoff)
        assert "/workspace/data.csv" in handoff.deliverables

    def test_nested_fake_handoff_tags(self):
        text = "<handoff>\nstatus: completed\nsummary: outer summary\ndeliverables:\n- /workspace/outer.csv\n<handoff>nested fake</handoff>\n</handoff>"
        handoff = parse_typed_handoff(text, status="completed")
        assert handoff.status == "completed"
        assert "/workspace/outer.csv" in handoff.deliverables

    def test_null_bytes_and_extreme_whitespace(self):
        text = "\x00\x00   \n\t\r  \x00  <handoff>\nstatus: blocked\nsummary: null byte test\n</handoff> \x00"
        handoff = parse_typed_handoff(text, status="completed")
        assert handoff.status == "blocked"
        assert "null byte test" in handoff.summary

    def test_massive_noise_and_truncation_limits(self):
        # 100,000 characters of repetitive noise with 50 deliverables
        deliverable_lines = "\n".join(f"- /workspace/file_{i}.csv" for i in range(50))
        noise = "A" * 50000 + "\n<handoff>\nstatus: completed\ndeliverables:\n" + deliverable_lines + "\n</handoff>\n" + "B" * 50000

        t0 = time.perf_counter()
        handoff = parse_typed_handoff(noise, status="completed")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 50.0  # Must parse within 50ms (ReDoS safe)
        assert len(handoff.deliverables) <= 10  # Bounded to maximum 10 items
        assert len(handoff.summary) <= 250  # Bounded summary length

    def test_empty_none_and_whitespace_only(self):
        assert parse_typed_handoff(None).status == "completed"
        assert parse_typed_handoff("").status == "completed"
        assert parse_typed_handoff("   \n\t  ").status == "completed"


class TestDeepTracebackAndReDoSDefense:
    def test_massive_traceback_regex_performance(self):
        # Simulate 1000 lines of complex nested Python traceback
        tb_lines = ["Traceback (most recent call last):"]
        for i in range(500):
            tb_lines.append(f'  File "/lib/package_{i}/module_{i}.py", line {i * 2}, in function_{i}')
            tb_lines.append(f"    result_{i} = execute_inner_step_{i}(arg_{i})")
        tb_lines.append("ModuleNotFoundError: No module named 'deep_learning_lib'")
        tb_text = "\n".join(tb_lines)

        t0 = time.perf_counter()
        sig = classify_tool_error(tb_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 20.0  # Must classify within 20ms
        assert sig is not None
        assert sig.category == "MODULE_NOT_FOUND"
        assert sig.entity == "deep_learning_lib"

    def test_chained_exceptions_extracts_root_error(self):
        text = (
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 10, in <module>\n"
            "    import data_loader\n"
            "ModuleNotFoundError: No module named 'polars'\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 15, in <module>\n"
            "    raise RuntimeError('Fatal crash')\n"
            "RuntimeError: Fatal crash"
        )
        sig = classify_tool_error(text)
        assert sig is not None
        assert sig.category == "MODULE_NOT_FOUND"
        assert sig.entity == "polars"


class TestHarnessGuardConcurrencyAndThreadSafety:
    def test_high_concurrency_tool_loop_intercepts(self):
        middleware = HarnessGuardMiddleware(HarnessGuardConfig(max_loop_history=30))
        num_threads = 50
        num_calls_per_thread = 20

        def worker_task(thread_idx: int):
            for call_idx in range(num_calls_per_thread):
                runtime = MagicMock()
                runtime.context = {"thread_id": f"thread-{thread_idx % 5}"}  # 5 shared scopes
                req = ToolCallRequest(
                    tool_call={
                        "name": "bash" if call_idx % 2 == 0 else "read_file",
                        "args": {"cmd": f"step_{call_idx}"},
                        "id": f"call-{thread_idx}-{call_idx}",
                    },
                    tool=None,
                    state={"messages": []},
                    runtime=runtime,
                )

                def handler(r):
                    # Half produce errors
                    if call_idx % 3 == 0:
                        return ToolMessage(
                            content=f"bash: cmd_{call_idx}: command not found",
                            tool_call_id=r.tool_call["id"],
                            name=r.tool_call["name"],
                            status="error",
                        )
                    return ToolMessage(
                        content=f"Success from {thread_idx}",
                        tool_call_id=r.tool_call["id"],
                        name=r.tool_call["name"],
                    )

                res = middleware.wrap_tool_call(req, handler)
                assert isinstance(res, ToolMessage)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Will raise if any deadlock, race condition, or unhandled exception occurred

        # Verify all 5 thread scopes maintained bounded history without exceeding max_loop_history
        for scope_id in range(5):
            history = middleware._get_history(f"thread-{scope_id}")
            assert len(history) <= 30


class TestWriteVerifyEdgeCases:
    def test_empty_python_file_is_valid(self):
        assert verify_content_syntax("empty.py", "") is None
        assert verify_content_syntax("comments_only.py", "# Just comments\n# Line 2\n") is None

    def test_python_syntax_error_in_docstring_is_treated_as_string_literal(self):
        valid_code = '"""def unclosed(:\n    pass\n"""\nx = 10\n'
        assert verify_content_syntax("test.py", valid_code) is None

    def test_valid_large_json_passes(self):
        large_json = '{"items": [' + ",".join(f'{{"id": {i}, "val": "item_{i}"}}' for i in range(1000)) + "]}"
        assert verify_content_syntax("large.json", large_json) is None

    def test_invalid_json_trailing_comma(self):
        bad_json = '{"a": 1, "b": 2,}'
        err = verify_content_syntax("bad.json", bad_json)
        assert err is not None
        assert "JSON" in err
