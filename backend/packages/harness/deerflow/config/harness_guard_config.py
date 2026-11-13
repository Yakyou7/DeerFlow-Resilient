"""Configuration for HarnessGuard tool-loop interceptor middleware."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class HarnessGuardConfig(BaseModel):
    """Configuration for system harness tool-loop interception, write-verification,
    and anti-doom-loop mechanisms (RFC-003, Route C)."""

    enabled: bool = Field(
        default=True,
        description="Whether the HarnessGuard tool-loop interceptor middleware is enabled",
    )
    write_verify_syntax: bool = Field(
        default=True,
        description="Whether to perform pre-execution syntax validation on .py (ast.parse) and .json (json.loads) writes",
    )
    block_on_syntax_error: bool = Field(
        default=True,
        description="Whether to reject the write immediately if a syntax error is detected",
    )
    targeted_hints_enabled: bool = Field(
        default=True,
        description="Whether to analyze tool error outputs and append actionable targeted hints",
    )
    max_repeated_error_signatures: int = Field(
        default=2,
        ge=1,
        description="Threshold of consecutive identical error signatures before triggering a critical repetition warning",
    )
    detect_alternating_loops: bool = Field(
        default=True,
        description="Whether to detect A -> B -> A -> B alternating failure patterns between different tools",
    )
    max_loop_history: int = Field(
        default=20,
        ge=5,
        description="Sliding window size of recent tool calls tracked for loop and pattern detection",
    )
    max_tracked_scopes: int = Field(
        default=1000,
        ge=10,
        description="Maximum number of tracked thread/run scopes retained in the bounded LRU history cache",
    )
    dpo_data_collection_enabled: bool = Field(
        default=True,
        description="Whether to capture repetition/syntax failures and subsequent recoveries as DPO preference pairs (arXiv:2610.19376)",
    )
    dpo_dataset_path: str | None = Field(
        default=None,
        description="Optional local filesystem path for append-only JSONL persistence of DPO training pairs",
    )
    git_snapshot_enabled: bool = Field(
        default=False,
        description="Whether to track workspace Git snapshots on successful actions (SWE-Search-style)",
    )
    git_rollback_on_loop: bool = Field(
        default=False,
        description="Whether to trigger Git hard rollback to the last healthy snapshot when a loop is detected",
    )
    workspace_root: str | None = Field(
        default=None,
        description="Optional explicit root directory for the Git workspace",
    )
    stagnant_readonly_warn_threshold: int = Field(
        default=12,
        ge=0,
        description="Consecutive read-only exploration tool calls without an action step before injecting a convergence warning (0 disables)",
    )
    stagnant_readonly_hard_limit: int = Field(
        default=24,
        ge=0,
        description="Consecutive read-only exploration tool calls before stripping read-only tool_calls in after_model to force convergence (0 disables hard stop)",
    )
    allowed_write_paths: list[str] = Field(
        default_factory=list,
        description="Optional path prefixes allowed for file write/edit tools (empty disables write-path allowlist gate)",
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Optional host/domain allowlist enforced on URL/email tool arguments (empty disables domain allowlist gate)",
    )
    blocked_command_prefixes: list[str] = Field(
        default_factory=list,
        description="Optional forbidden command prefixes blocked before shell tool execution (empty disables command gate)",
    )
    sop_max_stop_blocks: int = Field(
        default=3,
        ge=0,
        description="Maximum consecutive premature stop attempts blocked by an active SOP state machine before surfacing loop_capped (0 disables Stop Gate)",
    )
    code_ast_gate_enabled: bool = Field(
        default=True,
        description="Whether to enforce AST static safety checks and two-tier execution routing on code-action tool calls",
    )
    allowed_local_python_imports: list[str] = Field(
        default_factory=lambda: ["collections", "datetime", "functools", "itertools", "json", "math", "re", "statistics", "typing"],
        description="Pure-memory standard library modules permitted in the local_ast execution tier",
    )
    local_python_max_operations: int = Field(
        default=10000,
        ge=100,
        description="Maximum AST evaluation operations permitted per local_ast execution call",
    )

    @model_validator(mode="after")
    def validate_stagnant_readonly_limits(self) -> HarnessGuardConfig:
        """Ensure hard limit does not trigger before the warning threshold when both are active."""
        if (
            self.stagnant_readonly_hard_limit > 0
            and self.stagnant_readonly_warn_threshold > 0
            and self.stagnant_readonly_hard_limit < self.stagnant_readonly_warn_threshold
        ):
            raise ValueError("stagnant_readonly_hard_limit must be >= stagnant_readonly_warn_threshold")
        return self
