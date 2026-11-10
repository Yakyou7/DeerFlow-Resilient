"""DualLedgerMiddleware: Injects and maintains TaskLedger and ProgressLedger in model prompts.

Inspired by Microsoft Magentic-One's dual ledger architecture:
- TaskLedger maintains macro-level task facts, lookup requirements, derivations, hypotheses and milestones.
- ProgressLedger maintains micro-loop metacognitive reflection on the 5 core questions:
  completion, stagnation/looping, progress, next actor, remediation.

This middleware guarantees that dual-ledger state is deterministically injected
into the prompt context, remaining decoupled from unstructured conversational history
and immune to context loss during long runs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.ledgers.dual_ledger import DualLedger
from deerflow.agents.ledgers.progress_ledger import ProgressLedger
from deerflow.agents.ledgers.task_ledger import TaskLedger
from deerflow.agents.middlewares.message_utils import insert_after_leading_system_messages

DUAL_LEDGER_MESSAGE_NAME = "dual_ledger_context"


class DualLedgerMiddleware(AgentMiddleware):
    def __init__(self, dual_ledger: DualLedger | None = None) -> None:
        super().__init__()
        self._ledger = dual_ledger or DualLedger()

    @property
    def ledger(self) -> DualLedger:
        return self._ledger

    def resolve_ledger(self, state: dict[str, Any] | None) -> DualLedger:
        if not state:
            return self._ledger
        # State may contain "dual_ledger" dict, or individual "task_ledger" / "progress_ledger" dicts
        if "dual_ledger" in state and isinstance(state["dual_ledger"], dict):
            return DualLedger.from_dict(state["dual_ledger"])
        if "task_ledger" in state or "progress_ledger" in state:
            t = TaskLedger.from_dict(state.get("task_ledger")) if isinstance(state.get("task_ledger"), dict) else self._ledger.task_ledger
            p = ProgressLedger.from_dict(state.get("progress_ledger")) if isinstance(state.get("progress_ledger"), dict) else self._ledger.progress_ledger
            return DualLedger(task_ledger=t, progress_ledger=p)
        return self._ledger

    def _inject(self, request: ModelRequest) -> ModelRequest:
        state = request.state or {}
        active_ledger = self.resolve_ledger(state)
        prompt_segment = active_ledger.render_prompt()
        if not prompt_segment.strip():
            return request

        # Check if already present to avoid duplicate injections
        for msg in request.messages:
            if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == DUAL_LEDGER_MESSAGE_NAME:
                return request

        ledger_msg = HumanMessage(
            content=f"## Explicit Task & Progress Ledgers (Magentic-One State Orchestration)\n\n{prompt_segment}",
            name=DUAL_LEDGER_MESSAGE_NAME,
        )
        updated_messages = insert_after_leading_system_messages(list(request.messages), [ledger_msg])
        return request.override(messages=updated_messages)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._inject(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._inject(request))
