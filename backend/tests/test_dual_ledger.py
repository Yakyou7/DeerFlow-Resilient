"""Unit tests for Magentic-One style dual ledger state orchestration."""

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deerflow.agents.ledgers import (
    DualLedger,
    ProgressLedger,
    TaskLedger,
    TaskMilestone,
)
from deerflow.agents.middlewares.dual_ledger_middleware import (
    DUAL_LEDGER_MESSAGE_NAME,
    DualLedgerMiddleware,
)
from deerflow.agents.middlewares.summarization_middleware import (
    extract_task_ledger_blocks,
    merge_task_ledger_blocks,
)


class TestTaskLedger:
    def test_facts_lifecycle(self):
        ledger = TaskLedger(task="Refactor authentication system")
        ledger.add_given_fact("User database runs on PostgreSQL 16")
        ledger.add_given_fact("User database runs on PostgreSQL 16")  # Deduplication
        assert len(ledger.given_facts) == 1

        ledger.add_fact_to_lookup("Where are session tokens generated?")
        assert len(ledger.facts_to_lookup) == 1
        assert not ledger.facts_to_lookup[0]["resolved"]

        # Resolve fact to lookup
        ok = ledger.resolve_fact_lookup("Where are session tokens generated?", "In auth/tokens.py")
        assert ok
        assert ledger.facts_to_lookup[0]["resolved"]
        assert ledger.facts_to_lookup[0]["answer"] == "In auth/tokens.py"

        # Fact to derive
        ledger.add_fact_to_derive("Does token revocation require Redis flush?")
        ok = ledger.resolve_fact_derive(0, "Yes, Redis key session:<id> must be deleted")
        assert ok
        assert ledger.facts_to_derive[0]["derived"]
        assert "Redis key" in ledger.facts_to_derive[0]["result"]

    def test_guesses_and_milestones(self):
        ledger = TaskLedger(task="Fix deadlock")
        ledger.add_guess("Deadlock caused by lock acquisition order in worker.py")
        assert ledger.guesses[0]["status"] == "unverified"

        ok = ledger.evaluate_guess(0, confirmed=True, evidence="Confirmed in log trace #42")
        assert ok
        assert ledger.guesses[0]["status"] == "confirmed"
        assert ledger.guesses[0]["evidence"] == "Confirmed in log trace #42"

        ledger.add_milestone("Reproduce deadlock in test", status="completed", details="Reproduced in test_worker.py")
        ledger.add_milestone("Invert lock acquisition order", status="in_progress")
        ledger.add_milestone("Verify no regression", status="pending")

        assert len(ledger.milestones) == 3
        ledger.update_milestone("Invert lock acquisition order", status="completed", details="Fixed in PR #10")
        assert ledger.milestones[1].status == "completed"

        md = ledger.render_markdown()
        assert "<task_ledger>" in md
        assert "</task_ledger>" in md
        assert "### Current Task" in md
        assert "Fix deadlock" in md
        assert "[✓] Deadlock caused by lock acquisition order in worker.py [confirmed]" in md
        assert "1. Reproduce deadlock in test [completed]" in md

    def test_dict_serialization_roundtrip(self):
        ledger = TaskLedger(task="Build parser")
        ledger.add_given_fact("Grammar is LL(1)")
        ledger.add_milestone("Lexer", "completed")
        data = ledger.to_dict()

        restored = TaskLedger.from_dict(data)
        assert restored.task == "Build parser"
        assert restored.given_facts == ["Grammar is LL(1)"]
        assert len(restored.milestones) == 1
        assert restored.milestones[0].name == "Lexer"
        assert restored.milestones[0].status == "completed"


class TestProgressLedger:
    def test_metacognitive_questions_and_rendering(self):
        pl = ProgressLedger()
        pl.update(
            is_task_completed=False,
            is_stuck_or_looping=False,
            has_substantive_progress=True,
            next_worker_or_action="CodeReviewer",
            instruction_or_remediation="Review git diff for concurrency safety",
            evidence="All unit tests pass",
        )
        assert pl.step_index == 1
        assert not pl.is_task_completed
        assert not pl.is_stuck_or_looping
        assert pl.has_substantive_progress
        assert pl.next_worker_or_action == "CodeReviewer"

        md = pl.render_markdown()
        assert "<progress_ledger>" in md
        assert "</progress_ledger>" in md
        assert "- **Task Completed**: NO" in md
        assert "- **Stuck or Looping**: NO (Healthy)" in md
        assert "- **Substantive Progress**: YES" in md
        assert "- **Next Worker / Action**: CodeReviewer" in md
        assert "Review git diff for concurrency safety" in md

    def test_stuck_reflection(self):
        pl = ProgressLedger()
        pl.update(
            is_task_completed=False,
            is_stuck_or_looping=True,
            has_substantive_progress=False,
            next_worker_or_action="HarnessGuard",
            instruction_or_remediation="Trigger git rollback and adopt alternative design",
            evidence="Repeated identical compiler error 3 times",
        )
        assert pl.is_stuck_or_looping
        md = pl.render_markdown()
        assert "- **Stuck or Looping**: YES (Intervention Needed)" in md
        assert "- **Substantive Progress**: NO" in md


class TestDualLedger:
    def test_dual_ledger_prompt_and_dict(self):
        task_ledger = TaskLedger(task="Deploy cluster")
        task_ledger.add_given_fact("3 nodes available")
        progress_ledger = ProgressLedger()
        progress_ledger.update(
            is_task_completed=False,
            is_stuck_or_looping=False,
            has_substantive_progress=True,
            next_worker_or_action="Deployer",
        )

        dual = DualLedger(task_ledger=task_ledger, progress_ledger=progress_ledger)
        prompt = dual.render_prompt()
        assert "<task_ledger>" in prompt
        assert "<progress_ledger>" in prompt
        assert "3 nodes available" in prompt

        data = dual.to_dict()
        restored = DualLedger.from_dict(data)
        assert restored.task_ledger.task == "Deploy cluster"
        assert restored.progress_ledger.next_worker_or_action == "Deployer"


class TestDualLedgerMiddleware:
    def test_inject_into_model_request(self):
        dual = DualLedger()
        dual.task_ledger.task = "Analyze performance"
        dual.task_ledger.add_given_fact("P99 latency is 1200ms")
        dual.progress_ledger.update(has_substantive_progress=True, next_worker_or_action="Profiler")

        middleware = DualLedgerMiddleware(dual_ledger=dual)

        system_msg = SystemMessage(content="You are an expert engineer.")
        user_msg = HumanMessage(content="Investigate latency.")
        req = ModelRequest(messages=[system_msg, user_msg], model="test-model", state={})

        def mock_handler(injected_req: ModelRequest) -> ModelResponse:
            messages = list(injected_req.messages)
            assert len(messages) == 3
            # Injected right after leading system message
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage)
            assert getattr(messages[1], "name", None) == DUAL_LEDGER_MESSAGE_NAME
            assert "<task_ledger>" in messages[1].content
            assert "<progress_ledger>" in messages[1].content
            assert "P99 latency is 1200ms" in messages[1].content
            assert messages[2] == user_msg
            return ModelResponse(result=AIMessage(content="Analysis complete"))

        res = middleware.wrap_model_call(req, mock_handler)
        assert res.result.content == "Analysis complete"

    def test_no_duplicate_injection(self):
        dual = DualLedger()
        dual.task_ledger.task = "Test task"
        middleware = DualLedgerMiddleware(dual_ledger=dual)

        existing_injected = HumanMessage(content="Existing ledger", name=DUAL_LEDGER_MESSAGE_NAME)
        req = ModelRequest(messages=[existing_injected], model="test-model", state={})

        def mock_handler(injected_req: ModelRequest) -> ModelResponse:
            assert len(injected_req.messages) == 1
            return ModelResponse(result=AIMessage(content="ok"))

        middleware.wrap_model_call(req, mock_handler)


class TestSummarizationLedgerPreservation:
    def test_extract_and_merge_task_ledger_blocks(self):
        text = "Some summary\n<task_ledger>\n### Current Task\nMigrate schema\n</task_ledger>\nOther text"
        extracted = extract_task_ledger_blocks(text)
        assert len(extracted) == 1
        assert "Migrate schema" in extracted[0]

        shortened_text = "Some truncated summary..."
        merged = merge_task_ledger_blocks(extracted, shortened_text)
        assert "Some truncated summary..." in merged
        assert "<task_ledger>" in merged
        assert "Migrate schema" in merged
