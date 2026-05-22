from __future__ import annotations

import math
from pathlib import Path

from src.task.types import (
    CapabilityExecutionContext,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
)
from tasks.geology_graph import GeologyGraphState, GeologyGraphTask, GeologyGraphVariation


class _FakeG2VShim:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.calls: list[tuple[str, dict]] = []

    def dispatch(self, tool: str, args: dict) -> dict:
        self.calls.append((tool, dict(args)))
        return dict(self.output)


class _ScriptedG2VShim:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.expected = list(calls)
        self.calls: list[tuple[str, dict]] = []

    def dispatch(self, tool: str, args: dict) -> dict:
        self.calls.append((tool, dict(args)))
        expected_tool, output = self.expected.pop(0)
        assert tool == expected_tool
        return dict(output)


def _task(tmp_path: Path) -> GeologyGraphTask:
    dataset = Path("data/geology/36572_smolianova_1984").resolve()
    return GeologyGraphTask(
        {
            "dataset_dir": str(dataset),
            "pool_root": str(tmp_path / "pools"),
        }
    )


def test_variations_and_validation(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variations = task.list_variations()

    assert len(variations) >= 4
    assert all(isinstance(v, GeologyGraphVariation) for v in variations)
    assert all(Path(v.dataset_dir).exists() for v in variations if isinstance(v, GeologyGraphVariation))
    assert not hasattr(variations[0], "seed_graphs")
    task.validate()


def test_variation_names_filter_supports_short_verification_runs(tmp_path: Path) -> None:
    dataset = Path("data/geology/36572_smolianova_1984").resolve()
    task = GeologyGraphTask(
        {
            "dataset_dir": str(dataset),
            "pool_root": str(tmp_path / "pools"),
            "variation_names": ["smolianova_basic"],
        }
    )

    variations = task.list_variations()

    assert [variation.name for variation in variations] == ["smolianova_basic"]


def test_bootstrap_and_regular_workflow_shapes(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)

    bootstrap = task.workflow(variation, {"pool_snapshot": {"graph_ids": []}})
    assert bootstrap is not None
    assert [step.name for step in bootstrap.steps] == ["explore_data", "hypothesise", "execute", "submit_seed"]
    bootstrap_terminators = {
        step.name: step.terminator_capabilities for step in bootstrap.steps
    }
    assert bootstrap_terminators == {
        "explore_data": ("record_phase",),
        "hypothesise": ("record_phase",),
        "execute": ("record_phase",),
        "submit_seed": ("seed_graph_submit",),
    }
    submit_seed_caps = next(step.capabilities for step in bootstrap.steps if step.name == "submit_seed")
    assert submit_seed_caps == ("seed_graph_submit", "phase_get", "analysis_shell")
    modes = {step.name: step.context_mode for step in bootstrap.steps}
    assert modes["execute"] == "isolated"
    assert modes["submit_seed"] == "isolated"

    regular = task.workflow(
        variation,
        {"pool_snapshot": {"graph_ids": ["g2v://graph/a", "g2v://graph/b"]}},
    )
    assert regular is not None
    assert [step.name for step in regular.steps] == ["explore", "hypothesise", "execute", "refine", "submit"]
    regular_terminators = {step.name: step.terminator_capabilities for step in regular.steps}
    assert regular_terminators == {
        "explore": ("record_phase",),
        "hypothesise": ("record_phase",),
        "execute": ("record_phase",),
        "refine": ("record_phase",),
        "submit": ("candidate_submit_and_report",),
    }
    modes = {step.name: step.context_mode for step in regular.steps}
    assert modes["execute"] == "isolated"
    assert modes["refine"] == "isolated"
    assert modes["submit"] == "inherit"
    execute_caps = next(step.capabilities for step in regular.steps if step.name == "execute")
    assert "mcp_execute_call" in execute_caps
    assert "mcp_submit_call" not in execute_caps
    refine_caps = next(step.capabilities for step in regular.steps if step.name == "refine")
    assert refine_caps == (
        "refine_commit",
        "phase_get",
        "analysis_shell",
        "promote_analysis_artifact",
        "record_phase",
    )
    submit_caps = next(step.capabilities for step in regular.steps if step.name == "submit")
    assert submit_caps == ("candidate_submit_and_report", "phase_get")


def test_capability_surface_excludes_scoring_tools(tmp_path: Path) -> None:
    task = _task(tmp_path)
    caps = {cap.name: cap for cap in task.prompt_spec(task.list_variations()[0], {}).capabilities}
    for name, cap in caps.items():
        schema_text = str(cap.schema)
        assert "ic_score" not in schema_text, name
        assert "ic_score_from_graphs" not in schema_text, name
    assert caps["mcp_submit_call"].schema["properties"]["tool"]["enum"] == ["candidate_submit"]
    assert "candidate_submit" not in caps["mcp_submit_call_seed"].schema["properties"]["tool"]["enum"]
    assert caps["seed_graph_submit"].schema["required"] == [
        "filename",
        "content_text",
        "predicted_passed_gates",
    ]
    assert caps["refine_commit"].schema["required"] == [
        "reference_graph_uri",
        "operations",
        "message",
    ]
    assert caps["candidate_submit_and_report"].schema["required"] == [
        "candidate_graph_uri",
        "reference_pair",
        "predicted_score_bits",
    ]


def test_regular_episode_constraints_use_atomic_submit_success(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)

    constraints = task.episode_constraints(variation, {"workflow_kind": "regular"})

    assert constraints.success.terminal_capability_for_success == "candidate_submit_and_report"
    assert constraints.step_overrides["submit"].success is not None
    assert constraints.step_overrides["submit"].success.terminal_capability_for_success == "candidate_submit_and_report"


def test_record_phase_and_phase_get_use_trusted_step(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    episode_context = {"phase_records": {}}
    ctx = CapabilityExecutionContext("ep", "execute", episode_context)

    bad = task.execute_capability(
        CapabilityInvocation("record_phase", {"phase": "hypothesise"}),
        [],
        variation,
        ctx,
    )
    assert bad.success is False

    good = task.execute_capability(
        CapabilityInvocation("record_phase", {"phase": "execute", "status": "ok"}),
        [],
        variation,
        ctx,
    )
    assert good.success is True
    assert episode_context["phase_records"]["execute"]["status"] == "ok"

    got = task.execute_capability(
        CapabilityInvocation("phase_get", {"phase": "execute"}),
        [],
        variation,
        CapabilityExecutionContext("ep", "refine", episode_context),
    )
    assert got.output["found"] is True
    assert got.output["payload"]["status"] == "ok"


def test_seed_submit_rejects_non_g2v_uris(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    ctx = CapabilityExecutionContext("ep", "submit_seed", {})

    placeholder = task.execute_capability(
        CapabilityInvocation("seed_submit", {"seed_graph_uri": "<path-to-seed-graph>"}),
        [],
        variation,
        ctx,
    )
    assert placeholder.success is False
    assert "g2v://graph" in (placeholder.error or "")

    bad_field = task.execute_capability(
        CapabilityInvocation(
            "seed_submit",
            {
                "seed_graph_uri": "g2v://graph/" + "a" * 16,
                "seed_field_uri": "not-a-uri",
            },
        ),
        [],
        variation,
        ctx,
    )
    assert bad_field.success is False
    assert "g2v://field" in (bad_field.error or "")

    good = task.execute_capability(
        CapabilityInvocation(
            "seed_submit",
            {"seed_graph_uri": "g2v://graph/" + "a" * 16, "seed_field_uri": None},
        ),
        [],
        variation,
        ctx,
    )
    assert good.success is True


def test_seed_graph_submit_records_atomic_terminal_payloads(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    graph_uri = "g2v://graph/" + "a" * 16
    shim = _FakeG2VShim({"graph_uri": graph_uri, "node_count": 2, "edge_count": 1})
    episode_context = {"_g2v_shim": shim}

    result = task.execute_capability(
        CapabilityInvocation(
            "seed_graph_submit",
            {
                "filename": "seed.json",
                "content_text": '{"nodes": [], "edges": []}',
                "predicted_passed_gates": True,
                "gate_failures": [],
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "submit_seed", episode_context),
    )

    assert result.success is True
    assert shim.calls[0][0] == "seed_graph_submit"
    assert shim.calls[0][1]["filename"] == "seed.json"
    assert shim.calls[0][1]["content_text"] == '{"nodes": [], "edges": []}'
    assert result.output["seed_submit"] == {"seed_graph_uri": graph_uri, "seed_field_uri": None}
    assert result.output["report_metric"] == {"predicted_passed_gates": True, "gate_failures": []}
    assert episode_context["terminal_records"]["seed_submit"]["seed_graph_uri"] == graph_uri
    assert episode_context["terminal_records"]["report_metric"]["predicted_passed_gates"] is True


def test_seed_graph_submit_rejects_invalid_graph_uri(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    shim = _FakeG2VShim({"graph_uri": "<path-to-seed-graph>"})

    result = task.execute_capability(
        CapabilityInvocation(
            "seed_graph_submit",
            {
                "filename": "seed.json",
                "content_text": '{"nodes": [], "edges": []}',
                "predicted_passed_gates": True,
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "submit_seed", {"_g2v_shim": shim}),
    )

    assert result.success is False
    assert "g2v://graph" in (result.error or "")


def test_refine_commit_rejects_placeholder_reference_uri(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)

    result = task.execute_capability(
        CapabilityInvocation(
            "refine_commit",
            {
                "reference_graph_uri": "<reference_a_uri>",
                "operations": [],
                "message": "try refine",
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "refine", {}),
    )

    assert result.success is False
    assert "g2v://graph" in (result.error or "")


def test_refine_commit_dispatches_atomic_g2v_tool_and_preview(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    ref_uri = "g2v://graph/" + "a" * 16
    candidate_uri = "g2v://graph/" + "b" * 16
    field_uri = "g2v://field/" + "c" * 16
    shim = _ScriptedG2VShim(
        [
            (
                "refine_commit",
                {
                    "graph_uri": candidate_uri,
                    "scratch_uri": "g2v://scratch/abc",
                    "head_rev_uri": "g2v://scratch/abc@rev/1",
                    "validation_report": {"count": 1},
                },
            ),
            ("engine_run_preview", {"graph_uri": candidate_uri, "field_uri": field_uri}),
        ]
    )
    episode_context = {"_g2v_shim": shim, "field_spec": {"grid_shape": [2, 2, 2]}}

    result = task.execute_capability(
        CapabilityInvocation(
            "refine_commit",
            {
                "reference_graph_uri": ref_uri,
                "operations": [{"op": "update_node", "node_id": "u1", "patch": {"metadata": {"tag": "v2"}}}],
                "message": "regular refine",
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "refine", episode_context),
    )

    assert result.success is True
    assert shim.calls[0] == (
        "refine_commit",
        {
            "graph_uri": ref_uri,
            "operations": [{"op": "update_node", "node_id": "u1", "patch": {"metadata": {"tag": "v2"}}}],
            "message": "regular refine",
        },
    )
    assert shim.calls[1] == (
        "engine_run_preview",
        {"graph_ref": candidate_uri, "field_spec": {"grid_shape": [2, 2, 2]}},
    )
    assert result.output["candidate_graph_uri"] == candidate_uri
    assert result.output["candidate_field_uri"] == field_uri
    assert episode_context["phase_records"]["refine"]["candidate_graph_uri"] == candidate_uri


def test_seed_graph_submit_projects_terminal_artifacts() -> None:
    graph_uri = "g2v://graph/" + "b" * 16
    artifacts = EpisodeArtifacts(
        capability_invocations=[
            CapabilityInvocation(
                "seed_graph_submit",
                {
                    "filename": "seed.json",
                    "content_text": "{}",
                    "predicted_passed_gates": True,
                    "gate_failures": [],
                },
            )
        ],
        capability_results=[
            CapabilityResult(
                "seed_graph_submit",
                output={
                    "seed_submit": {"seed_graph_uri": graph_uri, "seed_field_uri": None},
                    "report_metric": {"predicted_passed_gates": True, "gate_failures": []},
                },
                success=True,
            )
        ],
    )

    terminals = GeologyGraphTask._terminal_artifacts(artifacts)

    assert terminals["seed_submit"]["input"] == {"seed_graph_uri": graph_uri, "seed_field_uri": None}
    assert terminals["report_metric"]["input"] == {"predicted_passed_gates": True, "gate_failures": []}


def test_candidate_submit_and_report_records_atomic_terminal_payloads(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    graph_uri = "g2v://graph/" + "d" * 16
    ref_a = "g2v://graph/" + "a" * 16
    ref_b = "g2v://graph/" + "b" * 16
    shim = _FakeG2VShim({"candidate_uri": "g2v://candidate/1", "graph_uri": graph_uri})
    episode_context = {"_g2v_shim": shim}

    result = task.execute_capability(
        CapabilityInvocation(
            "candidate_submit_and_report",
            {
                "candidate_graph_uri": graph_uri,
                "reference_pair": [ref_a, ref_b],
                "predicted_score_bits": 42.5,
                "gate_failures": [],
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "submit", episode_context),
    )

    assert result.success is True
    assert shim.calls[0] == ("candidate_submit", {"graph_uri": graph_uri, "reference_pair": [ref_a, ref_b]})
    assert result.output["candidate_submit"]["candidate_graph_uri"] == graph_uri
    assert result.output["report_metric"] == {"predicted_score_bits": 42.5, "gate_failures": []}
    assert episode_context["terminal_records"]["candidate_submit"]["candidate_graph_uri"] == graph_uri
    assert episode_context["terminal_records"]["report_metric"]["predicted_score_bits"] == 42.5


def test_candidate_submit_and_report_rejects_invalid_graph_uri_without_refine_fallback(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)

    result = task.execute_capability(
        CapabilityInvocation(
            "candidate_submit_and_report",
            {
                "candidate_graph_uri": "<candidate_graph_uri>",
                "reference_pair": ["g2v://graph/" + "a" * 16, "g2v://graph/" + "b" * 16],
                "predicted_score_bits": 10.0,
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "submit", {}),
    )

    assert result.success is False
    assert "g2v://graph" in (result.error or "")


def test_candidate_submit_and_report_uses_recorded_refine_uri_for_placeholder(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    graph_uri = "g2v://graph/" + "e" * 16
    shim = _FakeG2VShim({"candidate_uri": "g2v://candidate/2", "graph_uri": graph_uri})
    episode_context = {
        "_g2v_shim": shim,
        "phase_records": {"refine": {"phase": "refine", "candidate_graph_uri": graph_uri}},
    }

    result = task.execute_capability(
        CapabilityInvocation(
            "candidate_submit_and_report",
            {
                "candidate_graph_uri": "<from phase_get>",
                "reference_pair": ["g2v://graph/" + "a" * 16, "g2v://graph/" + "b" * 16],
                "predicted_score_bits": None,
            },
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "submit", episode_context),
    )

    assert result.success is True
    assert shim.calls[0][1]["graph_uri"] == graph_uri


def test_candidate_submit_and_report_projects_terminal_artifacts() -> None:
    graph_uri = "g2v://graph/" + "f" * 16
    artifacts = EpisodeArtifacts(
        capability_invocations=[
            CapabilityInvocation(
                "candidate_submit_and_report",
                {
                    "candidate_graph_uri": graph_uri,
                    "reference_pair": ["g2v://graph/" + "a" * 16, "g2v://graph/" + "b" * 16],
                    "predicted_score_bits": 12.0,
                    "gate_failures": [],
                },
            )
        ],
        capability_results=[
            CapabilityResult(
                "candidate_submit_and_report",
                output={
                    "candidate_submit": {"candidate_graph_uri": graph_uri, "graph_uri": graph_uri},
                    "report_metric": {"predicted_score_bits": 12.0, "gate_failures": []},
                },
                success=True,
            )
        ],
    )

    terminals = GeologyGraphTask._terminal_artifacts(artifacts)

    assert terminals["candidate_submit"]["input"]["candidate_graph_uri"] == graph_uri
    assert terminals["report_metric"]["input"] == {"predicted_score_bits": 12.0, "gate_failures": []}


def test_candidate_submit_rejects_non_g2v_uri(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    result = task.execute_capability(
        CapabilityInvocation(
            "mcp_submit_call",
            {"tool": "candidate_submit", "args": {"graph_uri": "<missing>"}},
        ),
        [],
        variation,
        CapabilityExecutionContext("ep", "submit", {}),
    )
    assert result.success is False
    assert "g2v://graph" in (result.error or "")


def test_step_transition_auto_records_previous_phase(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    episode_context: dict = {}

    # Step 1 (explore_data): the agent invokes analysis_shell but never record_phase.
    task.execute_capability(
        CapabilityInvocation("phase_get", {"phase": "execute"}),
        [],
        variation,
        CapabilityExecutionContext("ep", "explore_data", episode_context),
    )
    assert episode_context["phase_records"] == {}

    # Step 2 (hypothesise): first capability call here triggers auto-record of explore_data.
    task.execute_capability(
        CapabilityInvocation("phase_get", {"phase": "explore_data"}),
        [],
        variation,
        CapabilityExecutionContext("ep", "hypothesise", episode_context),
    )
    assert episode_context["phase_records"]["explore_data"]["auto_recorded"] is True

    # Final-step backstop runs in finalize_episode.
    task._finalize_auto_record(episode_context)
    assert "hypothesise" in episode_context["phase_records"]
    assert episode_context["phase_records"]["hypothesise"]["auto_recorded"] is True


def test_phase_tool_rejects_out_of_phase_and_scoring_probe(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)
    result = task.execute_capability(
        CapabilityInvocation("mcp_execute_call", {"tool": "candidate_submit", "args": {}}),
        [],
        variation,
        CapabilityExecutionContext("ep", "execute", {}),
    )
    assert result.success is False
    result = task.execute_capability(
        CapabilityInvocation("mcp_refine_call", {"tool": "ic_score", "args": {}}),
        [],
        variation,
        CapabilityExecutionContext("ep", "refine", {}),
    )
    assert result.success is False
    assert result.output["criterion_probe"] is True


def test_populate_host_dataset_snapshot_and_episode_count(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)

    outcome = task.populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "bootstrap"
    assert outcome.episode_context["dataset_snapshot"]
    index = Path(variation.pool_dir) / "index.json"
    assert '"episode_count": 1' in index.read_text(encoding="utf-8")


def test_reward_uses_authoritative_score_not_agent_prediction(tmp_path: Path) -> None:
    task = _task(tmp_path)
    final = GeologyGraphState(
        workflow_kind="regular",
        pool_snapshot={"graph_ids": ["a", "b"]},
        dataset_snapshot={},
        phase_artifacts={"execute": {"status": "ok"}},
        terminal_artifacts={
            "candidate_submit": {"input": {}, "output": {}},
            "report_metric": {"input": {"predicted_score_bits": 100.0, "gate_failures": []}, "output": {}},
        },
        score_bits=42.0,
        structural_bits=12.0,
        fit_bits=30.0,
        passed_gates=True,
        admission_threshold=50.0,
        t_steady=40.0,
        calibration_error_bits=58.0,
    )
    reward = task.compute_reward(final, final, EpisodeArtifacts())

    assert reward.success is True
    assert reward.breakdown["agent_predicted_score_bits"] == 100.0
    assert reward.breakdown["calibration_error_bits"] == 58.0
    assert reward.value == 0.95


def test_bootstrap_workflow_steps_have_terminator_capabilities(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = task.list_variations()[0]
    assert isinstance(variation, GeologyGraphVariation)

    bootstrap = task.workflow(variation, {"pool_snapshot": {"graph_ids": []}})
    assert bootstrap is not None

    expected = {
        "explore_data": ("record_phase",),
        "hypothesise": ("record_phase",),
        "execute": ("record_phase",),
        "submit_seed": ("seed_graph_submit",),
    }
    for step in bootstrap.steps:
        assert step.terminator_capabilities == expected[step.name], (
            f"step {step.name!r}: expected {expected[step.name]}, got {step.terminator_capabilities}"
        )


def test_reward_failure_modes(tmp_path: Path) -> None:
    task = _task(tmp_path)
    regular = GeologyGraphState(
        workflow_kind="regular",
        pool_snapshot={"graph_ids": ["a", "b"]},
        dataset_snapshot={},
        phase_artifacts={"execute": {"status": "unparameterisable"}},
        terminal_artifacts={
            "candidate_submit": {"input": {}, "output": {}},
            "report_metric": {"input": {"predicted_score_bits": 10.0}, "output": {}},
        },
        score_bits=math.inf,
        passed_gates=False,
        admission_threshold=100.0,
        t_steady=50.0,
    )
    reward = task.compute_reward(regular, regular, EpisodeArtifacts())
    assert reward.value == 0.1
    assert reward.success is False

    bootstrap = GeologyGraphState(
        workflow_kind="bootstrap",
        pool_snapshot={"graph_ids": []},
        dataset_snapshot={},
        passed_gates=True,
        terminal_artifacts={
            "seed_submit": {"input": {"seed_graph_uri": "g2v://graph/x"}, "output": {}},
            "report_metric": {"input": {"predicted_passed_gates": True}, "output": {}},
        },
    )
    reward = task.compute_reward(bootstrap, bootstrap, EpisodeArtifacts())
    assert reward.value == 0.5
    assert reward.success is True

    bootstrap.terminal_artifacts.pop("report_metric")
    reward = task.compute_reward(bootstrap, bootstrap, EpisodeArtifacts())
    assert reward.value == 0.0
    assert "missing_report_metric" in reward.breakdown["gate_failures"]


def test_system_prompt_yaml_safe(tmp_path: Path) -> None:
    """Backslash+quote sequences in the system prompt break NAT's yaml_load
    (the wrapper drops one backslash, terminating the double-quoted scalar
    early). The host yaml.safe_dump round-trips them, so this guards against
    a class of breakage we can't otherwise catch on the host."""
    import yaml

    task = _task(tmp_path)
    variation = task.list_variations()[0]
    ec = {
        "workflow_kind": "bootstrap",
        "pool_snapshot": {
            "graph_ids": [],
            "min_pool_size": variation.min_pool_size,
            "episode_count": 0,
            "admission_count": 0,
        },
        "assigned_references": {},
        "phase_records": {},
        "terminal_records": {},
    }
    prompt = task.prompt_spec(variation, ec).system_instruction

    assert '\\"' not in prompt, (
        "system_prompt contains a literal backslash+quote sequence; NAT's "
        "yaml_load preprocesses double-quoted scalars in a way that drops one "
        "of the backslashes, so the scalar terminates at the bare quote and "
        "the YAML parse fails. Rewrite the offending prompt fragment to avoid "
        "embedded double-quoted strings."
    )

    dumped = yaml.safe_dump({"workflow": {"system_prompt": prompt}})
    assert yaml.safe_load(dumped)["workflow"]["system_prompt"] == prompt


def _variation(task: GeologyGraphTask, name: str) -> GeologyGraphVariation:
    for variation in task.list_variations():
        if variation.name == name:
            assert isinstance(variation, GeologyGraphVariation)
            return variation
    raise AssertionError(f"variation {name!r} not found")


def test_copper_variation_pool_grounds_prompt_in_copper_evidence(tmp_path: Path) -> None:
    task = _task(tmp_path)
    copper = _variation(task, "smolianova_copper")
    assert copper.objective_pool, "copper variation must expose grounded objectives"
    for objective in copper.objective_pool:
        assert "Cu" in objective or "copper" in objective.lower()

    outcome = task.populate([], copper)
    focus = outcome.episode_context["selected_objective"]
    assert focus and focus in copper.objective_pool

    env = task.prompt_spec(copper, outcome.episode_context).environment_context
    assert "Episode focus:" in env
    assert focus in env

    bootstrap = task.workflow(copper, outcome.episode_context)
    assert bootstrap is not None
    entry = next(step for step in bootstrap.steps if step.is_entry)
    assert focus in entry.prompt


def test_objective_selection_is_deterministic_per_workspace_seed(tmp_path: Path) -> None:
    from tasks.geology_graph import _select_objective

    copper = _variation(_task(tmp_path), "smolianova_copper")
    pool = copper.objective_pool

    first = _select_objective("smolianova_copper", pool, "episode_42_99")
    again = _select_objective("smolianova_copper", pool, "episode_42_99")
    assert first == again
    assert first in pool

    picks = {_select_objective("smolianova_copper", pool, f"episode_{i}_0") for i in range(40)}
    assert len(picks) > 1, "RNG should yield more than one objective across 40 seeds"

    other_variation = _select_objective("smolianova_basic", pool, "episode_42_99")
    assert other_variation != first or len(pool) == 1, (
        "different variation names should usually pick a different objective"
    )


def test_variation_without_pool_emits_no_focus_line(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(task, "smolianova_basic")
    # Force an empty pool to confirm the conditional renders correctly.
    variation.objective_pool = ()
    ec = {
        "workflow_kind": "bootstrap",
        "pool_snapshot": {"graph_ids": [], "min_pool_size": variation.min_pool_size},
        "assigned_references": {},
        "phase_records": {},
        "terminal_records": {},
        "selected_objective": None,
    }
    env = task.prompt_spec(variation, ec).environment_context
    assert "Episode focus:" not in env

    bootstrap = task.workflow(variation, ec)
    assert bootstrap is not None
    entry = next(step for step in bootstrap.steps if step.is_entry)
    assert "Episode focus:" not in entry.prompt
