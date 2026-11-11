"""Unit tests for Typed Subagent Handoff Protocol, input_filter, and nest_handoff_history (Route C / RFC-003)."""

import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.subagents.context_snapshot import ParentContextSnapshot
from deerflow.subagents.report_contract import build_report_contract_section
from deerflow.subagents.status_contract import SUBAGENT_HANDOFF_KEY
from deerflow.subagents.typed_handoff import (
    HandoffInputData,
    TypedHandoff,
    nest_handoff_history,
    parse_typed_handoff,
    remove_all_tools,
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
        text = (
            "<handoff>\n"
            "status: blocked\n"
            "summary: Database connection timeout.\n"
            "fatal_blocks:\n"
            "- Connection refused on port 5432: Postgres service unreachable\n"
            "next:\n"
            "- Ensure database daemon is active before retry\n"
            "</handoff>"
        )
        handoff = parse_typed_handoff(text, status="completed")
        assert handoff.status == "blocked"
        assert len(handoff.fatal_blocks) == 1
        assert "Postgres service unreachable" in handoff.fatal_blocks[0]

    def test_parse_unclosed_handoff_tag_tolerance(self):
        """Simulate token cutoff where </handoff> closing tag was truncated."""
        text = (
            "Partial output...\n"
            "<handoff>\n"
            "status: completed\n"
            "summary: Step 1 completed before token limit\n"
            "deliverables:\n"
            "- partial_state.json"
        )
        handoff = parse_typed_handoff(text, status="completed")
        assert handoff.status == "completed"
        assert "Step 1 completed" in handoff.summary
        assert "partial_state.json" in handoff.deliverables

    def test_parse_heuristic_fallback_from_plain_text(self):
        text = (
            "I processed the data and saved output to workspace/analysis.csv. "
            "However, the script failed due to: missing required environment variable DB_SECRET."
        )
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


class TestHandoffInputFilterAndHistoryFolding:
    def test_remove_all_tools_strips_tool_messages_and_calls(self):
        messages = [
            HumanMessage(content="Please investigate why user login fails."),
            AIMessage(
                content="I will check the auth logs.",
                tool_calls=[{"name": "bash", "args": {"command": "cat /var/log/auth.log"}, "id": "call-1"}],
            ),
            ToolMessage(content="Dec 10 12:00:01 pam_unix(sshd:auth): authentication failure", tool_call_id="call-1", name="bash"),
            AIMessage(
                content="Now reading config.",
                tool_calls=[{"name": "read_file", "args": {"path": "/etc/pam.d/sshd"}, "id": "call-2"}],
            ),
            ToolMessage(content="# PAM configuration for sshd\nauth required pam_deny.so", tool_call_id="call-2", name="read_file"),
            AIMessage(content="Root cause identified: pam_deny.so is blocking auth."),
        ]

        filtered = remove_all_tools(messages)
        # All 2 ToolMessages are removed
        assert not any(isinstance(m, ToolMessage) for m in filtered)
        # AIMessages with tool_calls have tool_calls stripped to []
        for m in filtered:
            if isinstance(m, AIMessage):
                assert m.tool_calls == []
        # Total messages: 1 Human + 3 AIMessage = 4 messages (from 6)
        assert len(filtered) == 4
        assert filtered[0].content == "Please investigate why user login fails."
        assert filtered[1].content == "I will check the auth logs."
        assert filtered[-1].content == "Root cause identified: pam_deny.so is blocking auth."

    def test_nest_handoff_history_compresses_into_single_block(self):
        messages = [
            HumanMessage(content="Fix bug in auth middleware."),
            AIMessage(
                content="Investigating logs.",
                tool_calls=[{"name": "bash", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="Err: 500", tool_call_id="c1", name="bash"),
            AIMessage(content="The issue is a missing null check on user ID."),
        ]

        folded_text = nest_handoff_history(messages, filter_tools=True)
        assert "<CONVERSATION HISTORY>" in folded_text
        assert "</CONVERSATION HISTORY>" in folded_text
        assert "User: Fix bug in auth middleware." in folded_text
        assert "Assistant: Investigating logs." in folded_text
        assert "Assistant: The issue is a missing null check on user ID." in folded_text
        # Transient tool result must not leak into folded history
        assert "Err: 500" not in folded_text

    def test_handoff_input_data_validation(self):
        input_data = HandoffInputData(
            target_agent="db_specialist",
            reason="Query optimization required for slow query",
            priority="urgent",
            payload={"table": "orders", "latency_ms": 1250},
        )
        assert input_data.target_agent == "db_specialist"
        assert input_data.priority == "urgent"
        assert input_data.payload["table"] == "orders"

    def test_parent_context_snapshot_folded_mode(self):
        state = {
            "summary_text": "Prior session summary",
            "messages": [
                HumanMessage(content="Deploy the new auth microservice."),
                AIMessage(
                    content="Running deployment script.",
                    tool_calls=[{"name": "bash", "args": {}, "id": "c9"}],
                ),
                ToolMessage(content="Transient docker output: pulling layers 1..100", tool_call_id="c9", name="bash"),
                AIMessage(content="Service container deployed successfully on port 8080."),
            ],
        }

        # Ordinary snapshot dumps verbose tool result blocks
        raw_snapshot = ParentContextSnapshot.from_state(state)
        assert raw_snapshot is not None
        raw_msg = raw_snapshot.to_message()
        raw_text = json.dumps(raw_msg.content)
        assert "Transient docker output" in raw_text

        # Folded snapshot with input_filter and nest_history strips tool noise and nests
        folded_snapshot = ParentContextSnapshot.from_state(
            state,
            input_filter=remove_all_tools,
            nest_history=True,
        )
        assert folded_snapshot is not None
        folded_msg = folded_snapshot.to_message()
        folded_text = json.dumps(folded_msg.content)
        assert "<CONVERSATION HISTORY>" in folded_text
        assert "Deploy the new auth microservice." in folded_text
        assert "Service container deployed successfully" in folded_text
        # Transient docker output is completely stripped!
        assert "Transient docker output" not in folded_text


class TestTaskToolIntegration:
    def test_task_tool_build_command_stamps_handoff(self):
        result_text = (
            "Done processing.\n\n"
            "<handoff>\n"
            "status: completed\n"
            "summary: Exported final results.\n"
            "deliverables:\n"
            "- final_metrics.csv\n"
            "</handoff>"
        )
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
        assert SUBAGENT_HANDOFF_KEY in kwargs
        handoff_meta = kwargs[SUBAGENT_HANDOFF_KEY]
        assert handoff_meta["status"] == "completed"
        assert "final_metrics.csv" in handoff_meta["deliverables"]

    def test_report_contract_includes_handoff_instruction(self):
        contract_text = build_report_contract_section(receipts_enabled=True)
        assert "<handoff>" in contract_text
        assert "RFC-003" in contract_text
