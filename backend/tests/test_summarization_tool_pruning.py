from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.summarization_middleware import (
    DeerFlowSummarizationMiddleware,
)


def _mock_middleware(*, prune_tool_outputs=True, max_tool_output_chars=200, keep_recent_tool_turns=2):
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(text="summary text")
    model.ainvoke = MagicMock()
    model.with_config.return_value = model
    return DeerFlowSummarizationMiddleware(
        model=model,
        trigger=("messages", 4),
        keep=("messages", 4),
        token_counter=len,
        prune_tool_outputs=prune_tool_outputs,
        max_tool_output_chars=max_tool_output_chars,
        keep_recent_tool_turns=keep_recent_tool_turns,
    )


def test_tool_pruning_protects_recent_and_truncates_older():
    middleware = _mock_middleware(
        prune_tool_outputs=True,
        max_tool_output_chars=100,
        keep_recent_tool_turns=1,
    )

    t1 = ToolMessage(content="A" * 500, tool_call_id="call_1", name="bash", id="msg_1")
    t2 = ToolMessage(content="B" * 500, tool_call_id="call_2", name="read_file", id="msg_2")
    ai = AIMessage(content="done", id="msg_3")

    preserved = [t1, t2, ai]
    pruned = middleware._prune_preserved_tool_outputs(preserved)

    # t1 (older) should be truncated
    assert len(pruned[0].content) < 500
    assert "[Output truncated: 500 chars total, tool=bash]" in pruned[0].content
    assert pruned[0].tool_call_id == "call_1"
    assert pruned[0].id == "msg_1"
    assert pruned[0].name == "bash"

    # t2 (most recent) should NOT be truncated (protected by keep_recent_tool_turns=1)
    assert pruned[1].content == "B" * 500
    assert pruned[1].tool_call_id == "call_2"

    # ai message untouched
    assert pruned[2] == ai


def test_tool_pruning_disabled():
    middleware = _mock_middleware(prune_tool_outputs=False)

    t1 = ToolMessage(content="A" * 500, tool_call_id="call_1", name="bash", id="msg_1")
    preserved = [t1]
    pruned = middleware._prune_preserved_tool_outputs(preserved)

    # untouched when disabled
    assert pruned[0].content == "A" * 500


def test_end_to_end_compaction_with_tool_pruning():
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(text="compacted summary")
    model.with_config.return_value = model

    middleware = DeerFlowSummarizationMiddleware(
        model=model,
        trigger=("messages", 4),
        keep=("messages", 2),
        token_counter=lambda msgs: 100,  # always triggers
        prune_tool_outputs=True,
        max_tool_output_chars=80,
        keep_recent_tool_turns=1,
    )

    messages = [
        HumanMessage(content="user request", id="u1"),
        AIMessage(content="call tool 1", tool_calls=[{"id": "c1", "name": "bash", "args": {}}], id="a1"),
        ToolMessage(content="Output 1: " + "X" * 300, tool_call_id="c1", name="bash", id="t1"),
        AIMessage(content="call tool 2", tool_calls=[{"id": "c2", "name": "read_file", "args": {}}], id="a2"),
        ToolMessage(content="Output 2: " + "Y" * 300, tool_call_id="c2", name="read_file", id="t2"),
        AIMessage(content="call tool 3", tool_calls=[{"id": "c3", "name": "read_file", "args": {}}], id="a3"),
        ToolMessage(content="Output 3: " + "Z" * 300, tool_call_id="c3", name="read_file", id="t3"),
    ]

    state = {"messages": messages}
    runtime = SimpleNamespace(context={"thread_id": "t-123"})
    result = middleware.compact_state(state, runtime)

    assert result is not None
    assert result.summary_text == "compacted summary"
    # Preserved messages should have user request rescued + last messages
    preserved = result.preserved_messages
    tool_msgs = [m for m in preserved if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    # Check that tool message ids and pairings remain intact
    for tm in tool_msgs:
        assert tm.tool_call_id in ["c1", "c2", "c3"]


def test_reported_tokens_trigger_without_model_provider():
    middleware = _mock_middleware()
    # AIMessage without model_provider in response_metadata (typical OpenAI-compatible endpoint)
    ai_msg = AIMessage(
        content="calling tool",
        usage_metadata={"input_tokens": 33000, "output_tokens": 200, "total_tokens": 33200},
    )
    # Threshold 32000: reported 33200 >= 32000, should trigger!
    assert middleware._should_summarize_based_on_reported_tokens([ai_msg], 32000.0) is True

    # Threshold 35000: reported 33200 < 35000, should not trigger!
    assert middleware._should_summarize_based_on_reported_tokens([ai_msg], 35000.0) is False


def test_reported_tokens_with_matching_and_mismatching_provider():
    middleware = _mock_middleware()
    middleware.model._get_ls_params = lambda: {"ls_provider": "openai"}

    # Matching provider
    ai_matching = AIMessage(
        content="calling tool",
        response_metadata={"model_provider": "openai"},
        usage_metadata={"input_tokens": 33000, "output_tokens": 200, "total_tokens": 33200},
    )
    assert middleware._should_summarize_based_on_reported_tokens([ai_matching], 32000.0) is True

    # Mismatching provider
    ai_mismatching = AIMessage(
        content="calling tool",
        response_metadata={"model_provider": "anthropic"},
        usage_metadata={"input_tokens": 33000, "output_tokens": 200, "total_tokens": 33200},
    )
    assert middleware._should_summarize_based_on_reported_tokens([ai_mismatching], 32000.0) is False
