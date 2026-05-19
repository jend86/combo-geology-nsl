from __future__ import annotations

from src.harness.profiles.aiq import AiqProfile, AiqProfileConfig
from src.task.types import Capability, WorkflowStep


def _profile(group: str = "capabilities") -> AiqProfile:
    return AiqProfile(AiqProfileConfig(model="test-model", function_group_name=group))


def _agent_yaml(profile: AiqProfile, step: WorkflowStep, caps: list[Capability] | None = None) -> dict:
    return profile._agent_yaml(
        inference_url="http://localhost:8000/v1",
        mcp_url="http://localhost:9000/mcp",
        token="test-token",
        prompt_spec=None,
        capabilities=caps or [],
        step=step,
        transcript_filename="trace.jsonl",
    )


def test_return_direct_set_when_terminator_capabilities_present() -> None:
    profile = _profile()
    step = WorkflowStep(
        name="submit_seed",
        prompt="submit the seed graph",
        inherit_all_capabilities=False,
        capabilities=("seed_submit", "report_metric"),
        terminator_capabilities=("report_metric",),
    )
    caps = [
        Capability(name="seed_submit", description="submit seed"),
        Capability(name="report_metric", description="report metric"),
    ]
    result = _agent_yaml(profile, step, caps)
    assert result["workflow"]["return_direct"] == ["capabilities__report_metric"]


def test_return_direct_supports_atomic_candidate_submit() -> None:
    profile = _profile()
    step = WorkflowStep(
        name="submit",
        prompt="submit candidate",
        inherit_all_capabilities=False,
        capabilities=("candidate_submit_and_report",),
        terminator_capabilities=("candidate_submit_and_report",),
    )
    caps = [Capability(name="candidate_submit_and_report", description="submit candidate")]

    result = _agent_yaml(profile, step, caps)

    assert result["workflow"]["return_direct"] == ["capabilities__candidate_submit_and_report"]


def test_return_direct_absent_when_no_terminator_capabilities() -> None:
    profile = _profile()
    step = WorkflowStep(name="explore", prompt="explore the data")
    result = _agent_yaml(profile, step)
    assert "return_direct" not in result["workflow"]


def test_return_direct_absent_when_terminator_capabilities_empty() -> None:
    profile = _profile()
    step = WorkflowStep(
        name="hypothesise",
        prompt="form a hypothesis",
        terminator_capabilities=(),
    )
    result = _agent_yaml(profile, step)
    assert "return_direct" not in result["workflow"]


def test_return_direct_uses_function_group_name() -> None:
    profile = _profile(group="mytools")
    step = WorkflowStep(
        name="explore_data",
        prompt="explore",
        terminator_capabilities=("record_phase",),
    )
    result = _agent_yaml(profile, step)
    assert result["workflow"]["return_direct"] == ["mytools__record_phase"]


def test_return_direct_multiple_terminators() -> None:
    profile = _profile()
    step = WorkflowStep(
        name="submit_seed",
        prompt="submit",
        terminator_capabilities=("seed_submit", "report_metric"),
    )
    result = _agent_yaml(profile, step)
    assert result["workflow"]["return_direct"] == [
        "capabilities__seed_submit",
        "capabilities__report_metric",
    ]
