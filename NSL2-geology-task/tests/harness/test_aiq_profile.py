from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.harness.profiles import resolve_profile
from src.harness.profiles.aiq import AiqProfile, AiqProfileConfig
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeConstraints,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    Workflow,
    WorkflowStep,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _profile(**overrides) -> AiqProfile:
    cfg = AiqProfileConfig(
        model=overrides.pop("model", "nsl-memory-cleanup"),
        **overrides,
    )
    return AiqProfile(cfg)


def _prompt_spec() -> TaskPromptSpec:
    return TaskPromptSpec(
        system_instruction="Free non-volatile storage space.",
        environment_context="Container: special-learn-compose_service-a_1",
        capabilities=[Capability(name="run_python", description="Execute Python.")],
    )


def test_render_config_no_workflow_writes_single_agent_yaml(tmp_path: Path) -> None:
    profile = _profile()
    spec = _prompt_spec()

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://127.0.0.1:9001/v1",
        mcp_url="http://127.0.0.1:9002/mcp",
        token="tok",
        prompt_spec=spec,
    )

    cfg = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    assert cfg["llms"]["shim_llm"]["base_url"] == "http://127.0.0.1:9001/v1"
    assert cfg["llms"]["shim_llm"]["api_key"] == "tok"
    assert cfg["llms"]["shim_llm"]["model_name"] == "nsl-memory-cleanup"
    # NSL OpenAiShim rejects `stream=true`; NAT's tool_calling_agent calls
    # `astream` unconditionally, so `disable_streaming=True` must reach the
    # underlying LangChain ChatOpenAI client to keep the wire request
    # non-streaming.
    assert cfg["llms"]["shim_llm"]["disable_streaming"] is True
    assert cfg["workflow"]["system_prompt"].startswith(
        "Free non-volatile storage space."
    )
    assert (tmp_path / "query.txt").read_text() == profile.render_query(spec)
    assert not (tmp_path / "workflow.json").exists()


def test_render_config_workflow_writes_per_step_yaml_and_manifest(
    tmp_path: Path,
) -> None:
    profile = _profile()
    spec = _prompt_spec()
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="explore_container",
                prompt="Explore.",
                is_entry=True,
                next_steps=("plan_cleanup",),
            ),
            WorkflowStep(
                name="plan_cleanup",
                prompt="Plan.",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("execute_cleanup",),
            ),
            WorkflowStep(
                name="execute_cleanup",
                prompt="Execute.",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
            ),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    manifest = json.loads((tmp_path / "workflow.json").read_text())
    steps = {step["name"]: step for step in manifest["steps"]}
    assert list(steps) == ["explore_container", "plan_cleanup", "execute_cleanup"]
    assert steps["explore_container"]["config"] == "explore_container.yaml"
    assert steps["explore_container"]["inherit_context"] is True
    assert steps["explore_container"]["prompt"].endswith("Explore.")
    assert "do not describe what you would do in plain text" in steps[
        "explore_container"
    ]["prompt"]
    assert "ends the episode without making progress" not in steps[
        "explore_container"
    ]["prompt"]
    assert steps["plan_cleanup"] == {
        "name": "plan_cleanup",
        "config": "plan_cleanup.yaml",
        "prompt": "Plan.",
        "inherit_context": True,
    }
    assert steps["execute_cleanup"]["prompt"].endswith("Execute.")
    assert "do not describe what you would do in plain text" in steps[
        "execute_cleanup"
    ]["prompt"]
    assert "tool_choice" not in steps["explore_container"]
    assert "tool_choice" not in steps["plan_cleanup"]
    assert "markdown code fences" not in json.dumps(manifest)
    assert not (tmp_path / "agent.yaml").exists()
    assert not (tmp_path / "query.txt").exists()
    for filename in ["explore_container.yaml", "plan_cleanup.yaml", "execute_cleanup.yaml"]:
        assert (tmp_path / filename).exists()


def test_render_config_workflow_renders_effective_step_constraints(
    tmp_path: Path,
) -> None:
    profile = _profile()
    spec = _prompt_spec()
    workflow = Workflow(
        steps=(
            WorkflowStep(name="explore", prompt="Explore", next_steps=("plan",)),
            WorkflowStep(
                name="plan",
                prompt="Plan",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("execute",),
            ),
            WorkflowStep(
                name="execute",
                prompt="Execute",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
            ),
        )
    )
    constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=15, max_llm_turns=18),
        success=SuccessConstraints(min_task_tool_calls_for_success=1),
        step_overrides={
            "plan": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=0, max_llm_turns=4),
                success=SuccessConstraints(min_task_tool_calls_for_success=0),
            )
        },
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec, constraints=constraints),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
        constraints=constraints,
    )

    manifest = json.loads((tmp_path / "workflow.json").read_text())
    steps = {step["name"]: step for step in manifest["steps"]}
    assert "task tool calls: at most 15" in steps["explore"]["prompt"]
    assert "task tool calls: at most 0" in steps["plan"]["prompt"]
    assert "success requires at least 0 task tool calls" in steps["plan"]["prompt"]
    assert "llm turns: advisory limit 4" in steps["plan"]["prompt"]
    assert "task tool calls: at most 15" in steps["execute"]["prompt"]


def test_render_query_adds_tool_preamble_only_when_capabilities_present() -> None:
    profile = _profile()
    query = profile.render_query(_prompt_spec())

    assert query.startswith("To take any action, CALL the appropriate tool")
    assert "do not describe what you would do in plain text" in query
    assert "ends the episode without making progress" not in query

    no_tool_query = profile.render_query(
        TaskPromptSpec(system_instruction="Think only.", environment_context="ctx")
    )
    assert no_tool_query == "ctx"


def test_render_query_includes_static_constraints_block() -> None:
    profile = _profile()
    constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=15, max_llm_turns=18),
        success=SuccessConstraints(min_task_tool_calls_for_success=1),
    )

    query = profile.render_query(_prompt_spec(), constraints=constraints)

    assert "Task constraints:" in query
    assert "task tool calls: at most 15" in query
    assert "llm turns: advisory limit 18" in query
    assert "success requires at least 1 task tool call" in query


def test_render_config_threads_system_prompt_and_bearer_token(
    tmp_path: Path,
) -> None:
    profile = _profile(tool_call_timeout_s=45)
    spec = _prompt_spec()

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
    )

    cfg = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    assert "Free non-volatile storage space." in cfg["workflow"]["system_prompt"]
    assert "capabilities__run_python" in cfg["workflow"]["system_prompt"]
    assert cfg["function_groups"]["capabilities"]["server"]["custom_headers"] == {
        "Authorization": "Bearer tok"
    }
    assert cfg["function_groups"]["capabilities"]["tool_call_timeout"] == 45


def test_render_config_step_tool_names_respect_capabilities(tmp_path: Path) -> None:
    profile = _profile()
    spec = _prompt_spec()
    workflow = Workflow(
        steps=(
            WorkflowStep(name="all", prompt="All", next_steps=("none",)),
            WorkflowStep(
                name="none",
                prompt="None",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("one",),
            ),
            WorkflowStep(
                name="one",
                prompt="One",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
            ),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    all_cfg = yaml.safe_load((tmp_path / "all.yaml").read_text())
    none_cfg = yaml.safe_load((tmp_path / "none.yaml").read_text())
    one_cfg = yaml.safe_load((tmp_path / "one.yaml").read_text())

    assert all_cfg["workflow"]["tool_names"] == ["capabilities"]
    assert none_cfg["workflow"]["tool_names"] == []
    assert one_cfg["workflow"]["tool_names"] == ["capabilities__run_python"]
    assert all_cfg["function_groups"]["capabilities"]["server"]["custom_headers"][
        "X-NSL-Workflow-Step"
    ] == "all"
    assert none_cfg["function_groups"]["capabilities"]["server"]["custom_headers"][
        "X-NSL-Workflow-Step"
    ] == "none"
    assert one_cfg["function_groups"]["capabilities"]["server"]["custom_headers"][
        "X-NSL-Workflow-Step"
    ] == "one"

    # Inherit-all and reasoning-only steps must NOT pin `include`; that field
    # restricts what NAT registers under the prefixed name.
    assert "include" not in all_cfg["function_groups"]["capabilities"]
    assert "include" not in none_cfg["function_groups"]["capabilities"]
    # Explicit capability list MUST set `include` so NAT registers
    # `capabilities__run_python` in its global function registry — otherwise
    # the agent fails to build with "Function ... not found in list of functions".
    assert one_cfg["function_groups"]["capabilities"]["include"] == ["run_python"]
    assert all_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Workflow-Step": "all"
    }
    assert none_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Workflow-Step": "none"
    }
    assert one_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Workflow-Step": "one"
    }


def test_render_config_step_terminators_render_return_direct(tmp_path: Path) -> None:
    profile = _profile()
    spec = TaskPromptSpec(
        system_instruction="Submit a metric.",
        capabilities=[Capability(name="report_metric", description="Report metric.")],
    )
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="think",
                prompt="Think",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("submit",),
            ),
            WorkflowStep(
                name="submit",
                prompt="Submit",
                inherit_all_capabilities=False,
                capabilities=("report_metric",),
                terminator_capabilities=("report_metric",),
            ),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    think_cfg = yaml.safe_load((tmp_path / "think.yaml").read_text())
    submit_cfg = yaml.safe_load((tmp_path / "submit.yaml").read_text())

    assert "return_direct" not in think_cfg["workflow"]
    assert submit_cfg["workflow"]["return_direct"] == [
        "capabilities__report_metric"
    ]


def test_force_tool_choice_plumbs_to_tool_capable_step_yaml_and_manifest(
    tmp_path: Path,
) -> None:
    profile = _profile(force_tool_choice=True)
    spec = _prompt_spec()
    workflow = Workflow(
        steps=(
            WorkflowStep(name="all", prompt="All", next_steps=("none",)),
            WorkflowStep(
                name="none",
                prompt="None",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("one",),
            ),
            WorkflowStep(
                name="one",
                prompt="One",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
            ),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    all_cfg = yaml.safe_load((tmp_path / "all.yaml").read_text())
    none_cfg = yaml.safe_load((tmp_path / "none.yaml").read_text())
    one_cfg = yaml.safe_load((tmp_path / "one.yaml").read_text())
    assert all_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Workflow-Step": "all",
        "X-NSL-Tool-Choice": "required"
    }
    assert none_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Workflow-Step": "none"
    }
    assert one_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Workflow-Step": "one",
        "X-NSL-Tool-Choice": "required"
    }

    manifest = json.loads((tmp_path / "workflow.json").read_text())
    steps = {step["name"]: step for step in manifest["steps"]}
    assert steps["all"]["tool_choice"] == "required"
    assert "tool_choice" not in steps["none"]
    assert steps["one"]["tool_choice"] == "required"


def test_force_tool_choice_plumbs_to_single_agent_yaml(tmp_path: Path) -> None:
    profile = _profile(force_tool_choice=True)
    spec = _prompt_spec()

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
    )

    cfg = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    assert cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Tool-Choice": "required"
    }


def test_render_config_without_prompt_spec_uses_capabilities_argument(
    tmp_path: Path,
) -> None:
    profile = _profile(force_tool_choice=True)

    with_tools = tmp_path / "with-tools"
    with_tools.mkdir()
    profile.render_config(
        scratch=with_tools,
        query="q",
        capabilities=[Capability(name="run_python", description="Execute Python.")],
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
    )
    with_tools_cfg = yaml.safe_load((with_tools / "agent.yaml").read_text())
    assert with_tools_cfg["workflow"]["tool_names"] == ["capabilities"]
    assert with_tools_cfg["llms"]["shim_llm"]["default_headers"] == {
        "X-NSL-Tool-Choice": "required"
    }

    no_tools = tmp_path / "no-tools"
    no_tools.mkdir()
    profile.render_config(
        scratch=no_tools,
        query="q",
        capabilities=[],
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
    )
    no_tools_cfg = yaml.safe_load((no_tools / "agent.yaml").read_text())
    assert no_tools_cfg["workflow"]["tool_names"] == []
    assert "default_headers" not in no_tools_cfg["llms"]["shim_llm"]


def test_render_query_does_not_emit_nsl_server_phrase() -> None:
    profile = _profile()
    query = profile.render_query(_prompt_spec())

    assert "'nsl' server" not in query
    assert "function_groups.capabilities" in query
    assert "capabilities__run_python" in query
    assert "Free non-volatile storage space." not in query
    assert "markdown code fences" not in query


def test_tool_capable_step_names_excludes_reasoning_only_steps() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(name="all", prompt="All", next_steps=("none",)),
            WorkflowStep(
                name="none",
                prompt="None",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("one",),
            ),
            WorkflowStep(
                name="one",
                prompt="One",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
            ),
        )
    )

    assert _profile().tool_capable_step_names(workflow) == {"all", "one"}
    assert _profile().tool_capable_step_names(None) == set()


def test_read_transcript_parses_payload_event_type(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "trace.jsonl").write_text((FIXTURES / "aiq_trace.jsonl").read_text())

    transcript = _profile().read_transcript(tmp_path)

    assert transcript is not None
    assert transcript["messages"] == [
        {"role": "assistant", "content": "First answer."},
        {"role": "tool", "name": "capabilities__run_python", "content": "ok"},
        {"role": "assistant", "content": "Second answer."},
    ]


def test_read_transcript_uses_final_answer_txt_for_final_response(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "trace.jsonl").write_text((FIXTURES / "aiq_trace.jsonl").read_text())
    (tmp_path / "final_answer.txt").write_text("Final from run.py")

    transcript = _profile().read_transcript(tmp_path)

    assert transcript is not None
    assert transcript["final_response"] == "Final from run.py"
    assert _profile().count_llm_turns(transcript) == 2


def test_read_transcript_returns_none_when_no_jsonl_and_no_final(
    tmp_path: Path,
) -> None:
    assert _profile().read_transcript(tmp_path) is None


def test_read_transcript_tolerates_missing_final_answer_after_crash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "trace.jsonl").write_text((FIXTURES / "aiq_trace.jsonl").read_text())

    transcript = _profile().read_transcript(tmp_path)

    assert transcript is not None
    assert transcript["final_response"] is None
    assert _profile().count_llm_turns(transcript) == 2


def test_read_transcript_preserves_workflow_manifest_order_and_step_summaries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "workflow.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"name": "z_step", "config": "z_step.yaml", "prompt": "z"},
                    {"name": "a_step", "config": "a_step.yaml", "prompt": "a"},
                ]
            }
        )
    )
    (output / "a_step.jsonl").write_text(
        '{"payload":{"event_type":"LLM_END","name":"shim_llm",'
        '"data":{"output":"A answer"}}}\n'
        '{"payload":{"event_type":"TOOL_END","name":"capabilities__run_python",'
        '"data":{"output":"ok"}}}\n'
    )
    (output / "z_step.jsonl").write_text(
        '{"payload":{"event_type":"LLM_END","name":"shim_llm",'
        '"data":{"output":"Z answer"}}}\n'
    )

    transcript = _profile().read_transcript(tmp_path)

    assert transcript is not None
    assert [msg["content"] for msg in transcript["messages"]] == [
        "Z answer",
        "A answer",
        "ok",
    ]
    assert transcript["workflow"]["z_step"]["tool_end_count"] == 0
    assert transcript["workflow"]["z_step"]["first_assistant_content"] == "Z answer"
    assert transcript["workflow"]["a_step"]["tool_end_count"] == 1
    assert transcript["last_workflow_step"] == "a_step"


def test_to_artifacts_pulls_final_response_from_transcript() -> None:
    pairs = [
        (
            CapabilityInvocation(name="run_python", input={"code": "print(1)"}),
            CapabilityResult(name="run_python", output={"stdout": "1"}),
        )
    ]

    artifacts = _profile().to_artifacts(
        transcript={"messages": [], "final_response": "final"},
        capability_pairs=pairs,
    )

    assert artifacts.capability_invocations == [pairs[0][0]]
    assert artifacts.capability_results == [pairs[0][1]]
    assert artifacts.final_response == "final"


def test_default_args_env_and_native_workflow_support(tmp_path: Path) -> None:
    profile = _profile(tool_call_timeout_s=45)

    assert profile.default_args(tmp_path) == ["python", "/opt/nsl/run.py"]
    assert profile.env(tmp_path) == {"TOOL_CALL_TIMEOUT": "45"}
    assert profile.supports_native_workflow(
        Workflow(steps=(WorkflowStep(name="s", prompt="p"),))
    ) is True


def test_resolve_profile_aiq() -> None:
    profile = resolve_profile("aiq", {"model": "nsl-memory-cleanup"})

    assert isinstance(profile, AiqProfile)
