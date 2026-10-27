"""End-to-end integration and live task test for Route C (RFC-003 Stage 4).

Validates all 4 pillars functioning together in the runtime harness:
1. Write-Verify gate blocks syntax corruption.
2. Error classification & targeted hints resolve failures without doom loops.
3. Subagent task delegation returns structured Typed Handoff to parent agent.
4. Dynamic skill router provides zero-LLM JIT context activation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.harness_guard_middleware import (
    HARNESS_GUARD_KEY,
    HarnessGuardMiddleware,
)
from deerflow.config.harness_guard_config import HarnessGuardConfig
from deerflow.skills.router import DynamicSkillRouter
from deerflow.skills.types import Skill, SkillCategory
from deerflow.tools.builtins.task_tool import _task_result_command


class TestRouteCEndToEndLive:
    def test_e2e_write_verify_blocks_bad_syntax_live(self):
        """Pillar 1a: Writing broken code is blocked before file creation."""
        middleware = HarnessGuardMiddleware(HarnessGuardConfig(block_on_syntax_error=True))
        runtime = SimpleNamespace(context={"thread_id": "live-session-1"})
        bad_code = "def calculate_sales(total, tax:\n    return total * tax\n"
        req = ToolCallRequest(
            tool_call={"name": "write_file", "args": {"path": "sales_calculator.py", "content": bad_code}, "id": "tc-write-1"},
            tool=None,
            state={"messages": []},
            runtime=runtime,
        )

        file_system_write_called = False

        def dummy_writer(r):
            nonlocal file_system_write_called
            file_system_write_called = True
            return ToolMessage(content="File saved", tool_call_id=r.tool_call["id"], name="write_file")

        result = middleware.wrap_tool_call(req, dummy_writer)
        assert file_system_write_called is False  # Write to disk was prevented
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "[HARNESS GUARD: WRITE BLOCKED]" in result.content
        assert "syntax error" in result.content.lower()
        assert result.additional_kwargs[HARNESS_GUARD_KEY]["blocked"] is True

    def test_e2e_failing_bash_triggers_targeted_hint_and_anti_loop_live(self):
        """Pillar 1b: Tool failure produces targeted hint and blocks doom loop repetition."""
        middleware = HarnessGuardMiddleware(HarnessGuardConfig(max_repeated_error_signatures=2))
        runtime = SimpleNamespace(context={"thread_id": "live-session-1"})
        req = ToolCallRequest(
            tool_call={"name": "bash", "args": {"command": "python train.py"}, "id": "tc-bash-1"},
            tool=None,
            state={"messages": []},
            runtime=runtime,
        )

        def failing_bash(r):
            return ToolMessage(
                content="Traceback (most recent call last):\n  File 'train.py', line 2\nModuleNotFoundError: No module named 'onnxruntime'",
                tool_call_id=r.tool_call["id"],
                name="bash",
                status="error",
            )

        # First run: Targeted Hint injected
        res1 = middleware.wrap_tool_call(req, failing_bash)
        assert "[HARNESS GUARD]" in res1.content
        assert "pip install onnxruntime" in res1.content
        assert "CRITICAL REPETITION" not in res1.content

        # Second consecutive run: Doom loop intercepted with CRITICAL REPETITION warning
        res2 = middleware.wrap_tool_call(req, failing_bash)
        assert "[HARNESS GUARD]" in res2.content
        assert "[HARNESS GUARD: CRITICAL REPETITION]" in res2.content
        assert "has occurred 2 times consecutively" in res2.content

    def test_e2e_subagent_task_delegation_produces_typed_handoff_live(self):
        """Pillar 3: Subagent task tool delegation generates typed handoff."""
        subagent_output = (
            "Successfully trained model and computed cross-validation accuracy.\n\n"
            "<handoff>\n"
            "status: completed\n"
            "summary: High-accuracy model training completed across 10 folds.\n"
            "deliverables:\n"
            "- /workspace/models/model_v1.onnx\n"
            "- /workspace/reports/training_report.md\n"
            "attempted:\n"
            "- Data preprocessing: SUCCESS\n"
            "- 10-fold cross validation: SUCCESS (accuracy 94.2%)\n"
            "fatal_blocks:\n"
            "- None\n"
            "next:\n"
            "- Deploy model weights or export inference graphs\n"
            "</handoff>"
        )

        cmd = _task_result_command(
            tool_call_id="task-call-99",
            status="completed",
            result=subagent_output,
            model_name="qoder-qwen3.8-flash",
            usage={"input_tokens": 1200, "output_tokens": 450, "total_tokens": 1650},
        )

        assert isinstance(cmd, Command)
        msg = cmd.update["messages"][0]
        assert isinstance(msg, ToolMessage)

        # Model-visible text contains rendered handoff section
        assert '<subagent_handoff status="completed">' in msg.content
        assert "High-accuracy model training completed" in msg.content
        assert "`/workspace/models/model_v1.onnx`" in msg.content
        assert "</subagent_handoff>" in msg.content

        # additional_kwargs contains typed handoff metadata
        assert "subagent_handoff" in msg.additional_kwargs
        handoff_dict = msg.additional_kwargs["subagent_handoff"]
        assert handoff_dict["status"] == "completed"
        assert "/workspace/models/model_v1.onnx" in handoff_dict["deliverables"]
        assert len(handoff_dict["attempted_paths"]) == 2

    def test_e2e_dynamic_skill_routing_activates_jit_context(self):
        """Pillar 2: Zero-LLM dynamic skill intent router generates JIT context."""
        skills = [
            Skill(
                name="onnx-converter",
                description="Convert and optimize deep learning models into ONNX runtime format",
                license="Apache-2.0",
                skill_dir=Path("/mnt/skills/onnx"),
                skill_file=Path("/mnt/skills/onnx/SKILL.md"),
                relative_path=Path("onnx"),
                category=SkillCategory.PUBLIC,
                enabled=True,
                allowed_tools=("bash", "write_file"),
            ),
            Skill(
                name="sql-analyst",
                description="Query relational databases, run SQL statements, and inspect table schemas",
                license="MIT",
                skill_dir=Path("/mnt/skills/sql"),
                skill_file=Path("/mnt/skills/sql/SKILL.md"),
                relative_path=Path("sql"),
                category=SkillCategory.PUBLIC,
                enabled=True,
                allowed_tools=("read_file",),
            ),
        ]

        router = DynamicSkillRouter(skills)
        # Query 1: ONNX task
        matches1 = router.route_intent("Export our PyTorch weights into ONNX runtime format")
        assert len(matches1) == 1
        assert matches1[0].skill.name == "onnx-converter"
        jit_context = router.render_jit_skill_context(matches1)
        assert "<activated_skills>" in jit_context
        assert "Skill: onnx-converter" in jit_context

        # Query 2: General query (no skills triggered, keeps prompt prefix clean)
        matches2 = router.route_intent("Explain the difference between supervised and unsupervised learning")
        assert len(matches2) == 0
        assert router.render_jit_skill_context(matches2) == ""
