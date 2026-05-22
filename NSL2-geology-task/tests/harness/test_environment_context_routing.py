"""OrchestratorModeHarness must wire ``TaskPromptSpec.environment_context``
into its outgoing system prompt. Earlier it silently ignored the field —
tasks that split episode-specific context out of ``system_instruction``
would lose it entirely.
"""

from __future__ import annotations

from src.harness.orchestrator_modes import _compose_system_prompt
from src.task.types import Capability, TaskPromptSpec


def test_compose_folds_environment_context_into_system():
    spec = TaskPromptSpec(
        system_instruction="You are X.",
        environment_context="Target: 0xABC\nFork: 18000000",
        capabilities=[Capability(name="analyzer", description="look")],
    )
    composed = _compose_system_prompt(spec)
    assert "You are X." in composed
    assert "Target: 0xABC" in composed
    assert "Fork: 18000000" in composed


def test_compose_passes_through_when_environment_context_empty():
    spec = TaskPromptSpec(system_instruction="only system")
    assert _compose_system_prompt(spec) == "only system"


def test_compose_handles_environment_only():
    spec = TaskPromptSpec(system_instruction="", environment_context="env only")
    assert _compose_system_prompt(spec).strip() == "env only"
