from __future__ import annotations

import pytest

from src.task.types import Workflow, WorkflowStep


def test_workflow_entry_step_explicit_and_implicit() -> None:
    implicit = Workflow(
        steps=(
            WorkflowStep(name="plan", prompt="plan", next_steps=("act",)),
            WorkflowStep(name="act", prompt="act"),
        )
    )
    assert implicit.entry_step.name == "plan"

    explicit = Workflow(
        steps=(
            WorkflowStep(name="recover", prompt="recover"),
            WorkflowStep(name="plan", prompt="plan", is_entry=True),
        )
    )
    assert explicit.entry_step.name == "plan"


def test_workflow_step_accepts_terminator_capabilities() -> None:
    assert WorkflowStep(name="plan", prompt="plan").terminator_capabilities == ()

    step = WorkflowStep(
        name="submit",
        prompt="submit",
        terminator_capabilities=("report_metric",),
    )

    assert step.terminator_capabilities == ("report_metric",)


def test_workflow_topological_order_detects_isolated_cycle() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(name="a", prompt="a", next_steps=("b",)),
            WorkflowStep(name="b", prompt="b", next_steps=("a",)),
        )
    )

    with pytest.raises(ValueError, match="cycle"):
        workflow.topological_order()


def test_workflow_validate_accepts_empty_explicit_allowlist() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="summarize",
                prompt="summarize",
                inherit_all_capabilities=False,
                capabilities=(),
            ),
        )
    )

    workflow.validate({"run_python"})


def test_workflow_validate_rejects_inherit_all_with_named_capabilities() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(name="bad", prompt="bad", capabilities=("run_python",)),
        )
    )

    with pytest.raises(ValueError, match="inherit_all_capabilities"):
        workflow.validate({"run_python"})


def test_workflow_validate_rejects_fan_in_and_fan_out() -> None:
    fan_in = Workflow(
        steps=(
            WorkflowStep(name="a", prompt="a", next_steps=("c",), is_entry=True),
            WorkflowStep(name="b", prompt="b", next_steps=("c",)),
            WorkflowStep(name="c", prompt="c"),
        )
    )
    with pytest.raises(ValueError, match="fan-in"):
        fan_in.validate(set())

    fan_out = Workflow(
        steps=(
            WorkflowStep(name="a", prompt="a", next_steps=("b", "c")),
            WorkflowStep(name="b", prompt="b"),
            WorkflowStep(name="c", prompt="c"),
        )
    )
    with pytest.raises(ValueError, match="fan-out"):
        fan_out.validate(set())
