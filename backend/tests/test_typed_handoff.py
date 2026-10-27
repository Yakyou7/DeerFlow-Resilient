"""Unit tests for Typed Subagent Handoff Protocol (Route C / RFC-003 Phase 2)."""

from deerflow.subagents.report_contract import build_report_contract_section
from deerflow.subagents.typed_handoff import (
    TypedHandoff,
    parse_typed_handoff,
    render_handoff_section,
)
from deerflow.tools.builtins.task_tool import _task_result_command


class TestTypedHandoffParsing:
    def test_parse_explicit_handoff_tag(self):
        text = (
            "Here is my analysis of the dataset.\n\n"
            "<handoff>\n"
            "status: completed\n"
            "summary: Successfully generated 20000 sales records and aggregated revenue.\n"
            "deliverables:\n"
            "- workspace/sales.csv\n"
            "- workspace/report.md\n"
            "attempted:\n"
            "- Generated CSV via Python pandas: SUCCESS\n"
            "- Regional aggregation: SUCCESS\n"
            "fatal_blocks:\n"
            "- None\n"
            "next:\n"
            "- Synthesize report highlights for user\n"
            "</handoff>"
        )
        handoff = parse_typed_handoff(text, status="completed")
        assert handoff.status == "completed"
        assert "Successfully generated 20000" in handoff.summary
        assert "workspace/sales.csv" in handoff.deliverables
        assert "workspace/report.md" in handoff.deliverables
        assert len(handoff.attempted_paths) == 2
        assert len(handoff.fatal_blocks) == 0  # 'None' is filtered out
        assert "Synthesize report highlights" in handoff.remaining_goals[0]

    def test_parse_explicit_blocked_handoff_tag(self):
        text = "<handoff>\nstatus: blocked\nsummary: Database connection timeout.\nfatal_blocks:\n- Connection refused on port 5432: Postgres service unreachable\nnext:\n- Ensure database daemon is active before retry\n</handoff>"
        handoff = parse_typed_handoff(text, status="completed")
        assert handoff.status == "blocked"
        assert len(handoff.fatal_blocks) == 1
        assert "Postgres service unreachable" in handoff.fatal_blocks[0]

    def test_parse_heuristic_fallback_from_plain_text(self):
        text = "I processed the data and saved output to workspace/analysis.csv. However, the script failed due to: missing required environment variable DB_SECRET."
        handoff = parse_typed_handoff(text, status="failed")
        assert handoff.status in ("failed", "blocked")
        assert "workspace/analysis.csv" in handoff.deliverables
        assert len(handoff.fatal_blocks) >= 1
        assert "missing required environment variable" in handoff.fatal_blocks[0]

    def test_parse_with_tool_receipts(self):
        receipts = [
            {"tool_name": "write_file", "args": {"path": "/workspace/output.json"}},
            {"tool_name": "bash", "args": {"command": "python test.py"}},
        ]
        text = "Executed scripts and generated output."
        handoff = parse_typed_handoff(text, status="completed", tool_receipts=receipts)
        assert "/workspace/output.json" in handoff.deliverables
        assert any("write_file" in a for a in handoff.attempted_paths)
        assert any("bash" in a for a in handoff.attempted_paths)


class TestTypedHandoffRendering:
    def test_render_handoff_section(self):
        handoff = TypedHandoff(
            status="completed",
            summary="All unit tests passed cleanly.",
            deliverables=["workspace/test_report.xml"],
            attempted_paths=["pytest -v: 12 passed"],
            fatal_blocks=[],
            remaining_goals=["Proceed to deployment"],
        )
        rendered = render_handoff_section(handoff)
        assert '<subagent_handoff status="completed">' in rendered
        assert "**Summary**: All unit tests passed cleanly." in rendered
        assert "`workspace/test_report.xml`" in rendered
        assert "**Fatal Blocks**: None" in rendered
        assert "pytest -v: 12 passed" in rendered
        assert "</subagent_handoff>" in rendered


class TestTaskToolIntegration:
    def test_task_tool_build_command_stamps_handoff(self):
        result_text = "Done processing.\n\n<handoff>\nstatus: completed\nsummary: Exported final results.\ndeliverables:\n- final_metrics.csv\n</handoff>"
        cmd = _task_result_command(
            tool_call_id="call-42",
            status="completed",
            result=result_text,
            model_name="test-model",
        )
        msg = cmd.update["messages"][0]
        assert "<subagent_handoff" in msg.content
        assert "final_metrics.csv" in msg.content

        # Verify additional_kwargs
        kwargs = msg.additional_kwargs
        assert "subagent_handoff" in kwargs
        handoff_meta = kwargs["subagent_handoff"]
        assert handoff_meta["status"] == "completed"
        assert "final_metrics.csv" in handoff_meta["deliverables"]

    def test_report_contract_includes_handoff_instruction(self):
        contract_text = build_report_contract_section(receipts_enabled=True)
        assert "<handoff>" in contract_text
        assert "RFC-003" in contract_text
