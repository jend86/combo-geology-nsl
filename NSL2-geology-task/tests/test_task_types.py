from __future__ import annotations

from src.task.types import WorkflowStep


def test_workflow_step_terminator_capabilities_defaults_empty() -> None:
    step = WorkflowStep(name="x", prompt="do something")
    assert step.terminator_capabilities == ()


def test_workflow_step_accepts_terminator_capabilities() -> None:
    step = WorkflowStep(
        name="submit_seed",
        prompt="submit",
        terminator_capabilities=("report_metric",),
    )
    assert step.terminator_capabilities == ("report_metric",)


def test_workflow_step_terminator_capabilities_multi() -> None:
    step = WorkflowStep(
        name="explore_data",
        prompt="explore",
        terminator_capabilities=("record_phase", "analysis_shell"),
    )
    assert step.terminator_capabilities == ("record_phase", "analysis_shell")
