"""Tests for compaction-time externalized tool preview folding in DeerFlowSummarizationMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware
from deerflow.agents.middlewares.tool_output_budget_middleware import TOOL_OUTPUT_BLOB_KEY
from deerflow.agents.middlewares.tool_transform_meta import append_tool_transform, read_tool_transforms
from deerflow.config.summarization_config import SummarizationConfig


def _runtime(thread_id: str = "test-thread-pruning") -> SimpleNamespace:
    return SimpleNamespace(context={"thread_id": thread_id, "agent_name": "lead_agent"})


def _mock_model():
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(text="summary of conversation")
    model.ainvoke = AsyncMock(return_value=SimpleNamespace(text="summary of conversation"))
    model.with_config.return_value = model
    return model


def _make_externalized_preview(
    *,
    tool_name: str = "bash",
    virtual_path: str = "/mnt/user-data/outputs/.tool-results/bash_call_1.txt",
    total_chars: int = 24000,
    preview_body_chars: int = 3600,
    exit_marker: str | None = None,
) -> str:
    body = ("line_payload_sample_0123456789abcdef\n" * ((preview_body_chars // 36) + 1))[:preview_body_chars]
    parts = [
        f"[Full {tool_name} output saved to {virtual_path} ({total_chars} chars, ~{total_chars // 4} tokens).]",
        body,
        "Access:",
        f"- Use read_file on {virtual_path} with start_line and end_line to inspect the raw output.",
    ]
    if exit_marker is not None:
        parts.append(exit_marker)
    return "\n".join(parts)


def _ext_kwargs(
    *,
    virtual_path: str | None = "/mnt/user-data/outputs/.tool-results/bash_call_1.txt",
    with_blob: bool = False,
) -> dict:
    kwargs: dict = {}
    append_tool_transform(kwargs, "externalized", by="ToolOutputBudgetMiddleware")
    if with_blob and virtual_path is not None:
        kwargs[TOOL_OUTPUT_BLOB_KEY] = {
            "version": 1,
            "virtual_path": virtual_path,
            "storage_subdir": ".tool-results",
            "encoding": "utf-8",
        }
    return kwargs


def test_summarization_config_defaults_and_validation() -> None:
    config = SummarizationConfig()
    assert config.prune_tool_outputs is False
    assert config.max_tool_output_chars == 300
    assert config.keep_recent_tool_turns == 2

    with pytest.raises(ValueError, match="must be an integer, not a boolean"):
        SummarizationConfig(max_tool_output_chars=True)

    with pytest.raises(ValueError, match="must be an integer, not a boolean"):
        SummarizationConfig(keep_recent_tool_turns=False)


def test_release_policy_parameters_reflects_pruning_config() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 4),
        keep=("messages", 2),
        token_counter=len,
        prune_tool_outputs=True,
        max_tool_output_chars=250,
        keep_recent_tool_turns=3,
    )
    policy = middleware.release_policy_parameters()
    assert policy["prune_tool_outputs"] is True
    assert policy["max_tool_output_chars"] == 250
    assert policy["keep_recent_tool_turns"] == 3


def test_pruning_disabled_by_default() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 4),
        keep=("messages", 3),
        token_counter=len,
        prune_tool_outputs=False,
    )

    preview = _make_externalized_preview()
    messages = [
        HumanMessage(id="h0", content="historical query"),
        AIMessage(id="a0", content="historical answer"),
        HumanMessage(id="h1", content="current request"),
        AIMessage(id="a1", content="calling tool", tool_calls=[{"id": "c1", "name": "bash", "args": {}}]),
        ToolMessage(id="t1", tool_call_id="c1", name="bash", content=preview, additional_kwargs=_ext_kwargs()),
        HumanMessage(id="h2", content="next"),
    ]

    state = {"messages": list(messages), "summary_text": None}
    res = middleware.compact_state(state, _runtime(), force=True)
    assert res is not None

    kept_tool_msgs = [m for m in res.preserved_messages if isinstance(m, ToolMessage)]
    assert len(kept_tool_msgs) == 1
    assert kept_tool_msgs[0].content == preview
    assert all(tr.get("kind") != "pruned" for tr in read_tool_transforms(kept_tool_msgs[0]))


def test_folds_externalized_previews_preserving_reference_and_exit_code() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 6),
        keep=("messages", 7),
        token_counter=len,
        prune_tool_outputs=True,
        max_tool_output_chars=300,
        keep_recent_tool_turns=1,
    )

    vpath_old = "/mnt/user-data/outputs/.tool-results/bash_call_old.txt"
    vpath_err = "/mnt/user-data/outputs/.tool-results/bash_call_err.txt"
    vpath_recent = "/mnt/user-data/outputs/.tool-results/bash_call_recent.txt"

    old_preview = _make_externalized_preview(
        tool_name="bash",
        virtual_path=vpath_old,
        total_chars=24000,
        exit_marker="Exit Code: 0",
    )
    err_preview = _make_externalized_preview(
        tool_name="bash",
        virtual_path=vpath_err,
        total_chars=18000,
        exit_marker="Exit Code: 1",
    )
    recent_preview = _make_externalized_preview(
        tool_name="bash",
        virtual_path=vpath_recent,
        total_chars=19000,
        exit_marker="Exit Code: 0",
    )

    messages = [
        HumanMessage(id="h_hist", content="historical initial goal"),
        AIMessage(id="a_hist", content="historical initial response"),
        HumanMessage(id="h0", content="current task start"),
        AIMessage(
            id="a1",
            content="call 1",
            tool_calls=[
                {"id": "c1", "name": "bash", "args": {}},
                {"id": "c2", "name": "bash", "args": {}},
            ],
        ),
        ToolMessage(
            id="t1",
            tool_call_id="c1",
            name="bash",
            content=old_preview,
            artifact={"meta": 1},
            additional_kwargs=_ext_kwargs(virtual_path=vpath_old, with_blob=True),
        ),
        ToolMessage(
            id="t2",
            tool_call_id="c2",
            name="bash",
            content=err_preview,
            status="error",
            additional_kwargs=_ext_kwargs(virtual_path=vpath_err),
        ),
        AIMessage(id="a2", content="call 2", tool_calls=[{"id": "c3", "name": "bash", "args": {}}]),
        ToolMessage(
            id="t3",
            tool_call_id="c3",
            name="bash",
            content=recent_preview,
            additional_kwargs=_ext_kwargs(virtual_path=vpath_recent),
        ),
        AIMessage(id="a3", content="finished"),
    ]

    state = {"messages": list(messages), "summary_text": None}
    res = middleware.compact_state(state, _runtime(), force=True)
    assert res is not None

    preserved_by_id = {m.id: m for m in res.preserved_messages if isinstance(m, ToolMessage)}

    # t1 (older externalized output): MUST be folded while retaining path, Access instruction, and Exit Code
    t1 = preserved_by_id["t1"]
    assert len(t1.content) < 350
    assert len(t1.content) < len(old_preview) // 5
    assert f"[Full bash output saved to {vpath_old} (24000 chars, ~6000 tokens).]" in t1.content
    assert f"- Use read_file on {vpath_old} with start_line and end_line to inspect the raw output." in t1.content
    assert t1.content.rstrip().endswith("Exit Code: 0")
    assert t1.artifact == {"meta": 1}
    assert t1.tool_call_id == "c1"
    assert t1.additional_kwargs[TOOL_OUTPUT_BLOB_KEY]["virtual_path"] == vpath_old
    transforms = read_tool_transforms(t1)
    assert [tr.get("kind") for tr in transforms] == ["externalized", "pruned"]
    assert transforms[-1].get("by") == "DeerFlowSummarizationMiddleware"

    # t2 (error tool): MUST remain intact
    assert preserved_by_id["t2"].content == err_preview
    assert preserved_by_id["t2"].status == "error"

    # t3 (recent tool turn): MUST remain intact
    assert preserved_by_id["t3"].content == recent_preview


def test_unexternalized_and_spoofed_messages_are_never_destructively_truncated() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 4),
        keep=("messages", 5),
        token_counter=len,
        prune_tool_outputs=True,
        max_tool_output_chars=200,
        keep_recent_tool_turns=0,
    )

    raw_inline = "IMPORTANT_INLINE_DATA_" * 40
    # Even if raw output happens to print text matching externalized formatting,
    # lacking control-plane metadata ("externalized" transform or blob ref) must keep it untouched.
    spoofed_text = _make_externalized_preview(virtual_path="/mnt/user-data/outputs/.tool-results/fake.txt")

    messages = [
        HumanMessage(id="h0", content="hist"),
        AIMessage(id="a0", content="hist_ack"),
        HumanMessage(id="h1", content="task"),
        AIMessage(
            id="a1",
            content="calls",
            tool_calls=[
                {"id": "c_raw", "name": "read_file", "args": {}},
                {"id": "c_spoof", "name": "bash", "args": {}},
            ],
        ),
        ToolMessage(id="t_raw", tool_call_id="c_raw", name="read_file", content=raw_inline),
        ToolMessage(id="t_spoof", tool_call_id="c_spoof", name="bash", content=spoofed_text),
        AIMessage(id="a2", content="done"),
    ]

    state = {"messages": list(messages), "summary_text": None}
    res = middleware.compact_state(state, _runtime(), force=True)
    assert res is not None

    by_id = {m.id: m for m in res.preserved_messages if isinstance(m, ToolMessage)}
    assert by_id["t_raw"].content == raw_inline
    assert by_id["t_spoof"].content == spoofed_text
    assert read_tool_transforms(by_id["t_raw"]) == ()
    assert read_tool_transforms(by_id["t_spoof"]) == ()


def test_idempotent_on_already_folded_messages_and_preserves_multimodal_blocks() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 4),
        keep=("messages", 5),
        token_counter=len,
        prune_tool_outputs=True,
        max_tool_output_chars=100,
        keep_recent_tool_turns=0,
    )

    vpath = "/mnt/user-data/outputs/.tool-results/view_call_1.txt"
    preview = _make_externalized_preview(tool_name="view", virtual_path=vpath, total_chars=15000)
    list_content = [
        {"type": "text", "text": preview},
        {"type": "image", "data": "binary-blob"},
    ]

    messages = [
        HumanMessage(id="h0", content="hist"),
        AIMessage(id="a0", content="hist_ack"),
        HumanMessage(id="h1", content="hi"),
        AIMessage(id="a1", content="c", tool_calls=[{"id": "c1", "name": "view", "args": {}}]),
        ToolMessage(id="t1", tool_call_id="c1", name="view", content=list_content, additional_kwargs=_ext_kwargs(virtual_path=vpath)),
        AIMessage(id="a2", content="done"),
    ]

    res1 = middleware.compact_state({"messages": list(messages), "summary_text": None}, _runtime(), force=True)
    assert res1 is not None
    kept_t1 = [m for m in res1.preserved_messages if m.id == "t1"][0]
    assert isinstance(kept_t1.content, list)
    assert len(kept_t1.content) == 2
    assert kept_t1.content[1] == {"type": "image", "data": "binary-blob"}
    folded_text = kept_t1.content[0]["text"]
    assert f"Use read_file on {vpath}" in folded_text
    assert "[Preview folded during compaction]" in folded_text

    # Re-compacting must be idempotent (never duplicate transforms or alter pointer)
    res2 = middleware.compact_state(
        {"messages": [HumanMessage(id="h_x", content="x"), *res1.preserved_messages], "summary_text": res1.summary_text},
        _runtime(),
        force=True,
    )
    assert res2 is not None
    kept_t1_again = [m for m in res2.preserved_messages if m.id == "t1"][0]
    assert kept_t1_again.content == kept_t1.content
    assert [tr.get("kind") for tr in read_tool_transforms(kept_t1_again)] == ["externalized", "pruned"]


def test_auto_compaction_keep_tokens_stay_below_trigger_threshold() -> None:
    """Folding older externalized previews prevents the post-compaction Keep zone from immediately re-triggering."""

    def char_token_counter(msgs) -> int:
        return sum(len(str(getattr(m, "content", ""))) for m in msgs)

    def make_mw(prune: bool) -> DeerFlowSummarizationMiddleware:
        return DeerFlowSummarizationMiddleware(
            model=_mock_model(),
            trigger=("tokens", 8500),
            keep=("messages", 7),
            token_counter=char_token_counter,
            prune_tool_outputs=prune,
            max_tool_output_chars=300,
            keep_recent_tool_turns=1,
        )

    p1 = _make_externalized_preview(virtual_path="/mnt/user-data/outputs/.tool-results/call_1.txt", preview_body_chars=3500)
    p2 = _make_externalized_preview(virtual_path="/mnt/user-data/outputs/.tool-results/call_2.txt", preview_body_chars=3500)
    p3 = _make_externalized_preview(virtual_path="/mnt/user-data/outputs/.tool-results/call_3.txt", preview_body_chars=3500)

    messages = [
        HumanMessage(id="h0", content="initial goal"),
        AIMessage(id="a0", content="ack"),
        HumanMessage(id="h1", content="step 1"),
        AIMessage(id="a1", content="run 1", tool_calls=[{"id": "c1", "name": "bash", "args": {}}]),
        ToolMessage(id="t1", tool_call_id="c1", name="bash", content=p1, additional_kwargs=_ext_kwargs(virtual_path="/mnt/user-data/outputs/.tool-results/call_1.txt")),
        AIMessage(id="a2", content="run 2", tool_calls=[{"id": "c2", "name": "bash", "args": {}}]),
        ToolMessage(id="t2", tool_call_id="c2", name="bash", content=p2, additional_kwargs=_ext_kwargs(virtual_path="/mnt/user-data/outputs/.tool-results/call_2.txt")),
        AIMessage(id="a3", content="run 3", tool_calls=[{"id": "c3", "name": "bash", "args": {}}]),
        ToolMessage(id="t3", tool_call_id="c3", name="bash", content=p3, additional_kwargs=_ext_kwargs(virtual_path="/mnt/user-data/outputs/.tool-results/call_3.txt")),
    ]

    mw_off = make_mw(prune=False)
    mw_on = make_mw(prune=True)

    res_off = mw_off.compact_state({"messages": list(messages), "summary_text": None}, _runtime(), force=False)
    res_on = mw_on.compact_state({"messages": list(messages), "summary_text": None}, _runtime(), force=False)
    assert res_off is not None
    assert res_on is not None

    tokens_after_off = char_token_counter(res_off.preserved_messages)
    tokens_after_on = char_token_counter(res_on.preserved_messages)
    assert tokens_after_off > 8500  # Still above trigger without folding
    assert tokens_after_on < 5000  # Well below trigger with externalized previews folded


def test_normal_turn_without_compaction_never_alters_messages() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 20),
        keep=("messages", 5),
        token_counter=len,
        prune_tool_outputs=True,
    )

    preview = _make_externalized_preview()
    messages = [
        HumanMessage(id="h1", content="read"),
        ToolMessage(id="t1", tool_call_id="c1", name="bash", content=preview, additional_kwargs=_ext_kwargs()),
    ]

    state = {"messages": list(messages), "summary_text": None}
    res = middleware.before_model(state, _runtime())
    assert res is None
    assert state["messages"][1].content == preview


@pytest.mark.asyncio
async def test_async_acompact_state_pruning() -> None:
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 4),
        keep=("messages", 4),
        token_counter=len,
        prune_tool_outputs=True,
        max_tool_output_chars=300,
        keep_recent_tool_turns=0,
    )

    vpath = "/mnt/user-data/outputs/.tool-results/search_call_1.txt"
    preview = _make_externalized_preview(tool_name="grep", virtual_path=vpath, total_chars=16000)

    messages = [
        HumanMessage(id="h0", content="hist"),
        AIMessage(id="a0", content="hist_ack"),
        HumanMessage(id="h1", content="q"),
        AIMessage(id="a1", content="c", tool_calls=[{"id": "c1", "name": "grep", "args": {}}]),
        ToolMessage(id="t1", tool_call_id="c1", name="grep", content=preview, additional_kwargs=_ext_kwargs(virtual_path=vpath)),
        AIMessage(id="a2", content="r"),
    ]

    state = {"messages": list(messages), "summary_text": None}
    res = await middleware.acompact_state(state, _runtime(), force=True)
    assert res is not None

    kept_t1 = [m for m in res.preserved_messages if m.id == "t1"][0]
    assert f"[Full grep output saved to {vpath} (16000 chars, ~4000 tokens).]" in kept_t1.content
    assert f"- Use read_file on {vpath} with start_line and end_line to inspect the raw output." in kept_t1.content


def test_repeated_tool_call_ids_across_turns_keep_distinct_turn_boundaries() -> None:
    """Providers that number every turn from 'call_0' must not merge older turns into the protected recent tail."""
    middleware = DeerFlowSummarizationMiddleware(
        model=_mock_model(),
        trigger=("messages", 6),
        keep=("messages", 6),
        token_counter=len,
        prune_tool_outputs=True,
        max_tool_output_chars=300,
        keep_recent_tool_turns=1,
    )

    vpath_old = "/mnt/user-data/outputs/.tool-results/bash_turn1_call0.txt"
    vpath_new = "/mnt/user-data/outputs/.tool-results/bash_turn2_call0.txt"
    preview_old = _make_externalized_preview(tool_name="bash", virtual_path=vpath_old, total_chars=18000)
    preview_new = _make_externalized_preview(tool_name="bash", virtual_path=vpath_new, total_chars=18000)

    messages = [
        HumanMessage(id="h0", content="hist"),
        AIMessage(id="a0", content="hist_ack"),
        HumanMessage(id="h1", content="task"),
        AIMessage(id="a1", content="turn 1", tool_calls=[{"id": "call_0", "name": "bash", "args": {}}]),
        ToolMessage(id="t1", tool_call_id="call_0", name="bash", content=preview_old, additional_kwargs=_ext_kwargs(virtual_path=vpath_old)),
        AIMessage(id="a2", content="turn 2", tool_calls=[{"id": "call_0", "name": "bash", "args": {}}]),
        ToolMessage(id="t2", tool_call_id="call_0", name="bash", content=preview_new, additional_kwargs=_ext_kwargs(virtual_path=vpath_new)),
        AIMessage(id="a3", content="done"),
    ]

    res = middleware.compact_state({"messages": list(messages), "summary_text": None}, _runtime(), force=True)
    assert res is not None

    by_id = {m.id: m for m in res.preserved_messages if isinstance(m, ToolMessage)}
    assert "[Preview folded during compaction]" in by_id["t1"].content
    assert f"Use read_file on {vpath_old}" in by_id["t1"].content
    assert by_id["t2"].content == preview_new
    assert [tr.get("kind") for tr in read_tool_transforms(by_id["t1"])] == ["externalized", "pruned"]
    assert [tr.get("kind") for tr in read_tool_transforms(by_id["t2"])] == ["externalized"]
