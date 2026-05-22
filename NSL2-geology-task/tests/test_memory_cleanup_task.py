"""Tests for the memory-cleanup reference implementation.

Tests verify the task's structure and pure logic without Docker.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tasks.memory_cleanup import (
    MemoryCleanupTask,
    MemoryCleanupState,
    MemoryCleanupVariation,
)
from src.task.base import TaskEnvironmentError
from src.task.types import EpisodeArtifacts, Variation  # noqa: F401


@pytest.fixture
def task() -> MemoryCleanupTask:
    return MemoryCleanupTask({})


@pytest.fixture
def task_custom() -> MemoryCleanupTask:
    return MemoryCleanupTask(
        {
            "docker_compose_dir": "docker/custom",
            "tolerance": 0.1,
            "cleanup_paths": ["/tmp"],
            "success_threshold_kb": 10.0,
        }
    )


class TestTaskRegistration:
    def test_name(self, task):
        assert task.name == "memory-cleanup"

    def test_metric_metadata(self, task):
        assert task.metric_name == "space_freed_kb"
        assert task.metric_unit == "KB"
        assert task.higher_is_better is True

    def test_agent_service_name(self, task):
        assert task.agent_service_name == "service-a"


class TestVariations:
    def test_returns_five_variations(self, task):
        variations = task.list_variations()
        assert len(variations) == 5

    def test_variations_are_subclass(self, task):
        for v in task.list_variations():
            assert isinstance(v, MemoryCleanupVariation)
            assert isinstance(v, Variation)

    def test_variations_have_commands(self, task):
        for v in task.list_variations():
            assert isinstance(v, MemoryCleanupVariation)
            assert len(v.commands) > 0

    def test_variations_have_expected_kb(self, task):
        for v in task.list_variations():
            assert isinstance(v, MemoryCleanupVariation)
            assert v.expected_kb > 0


class TestPromptSpec:
    def test_has_run_python_capability(self, task):
        """Real MCP tools only — modes (investigator/explorer/recorder) live
        wholly on the harness side now."""
        spec = task.prompt_spec(task.list_variations()[0], {})
        names = {c.name for c in spec.capabilities}
        assert "run_python" in names

    def test_system_prompt_nonempty(self, task):
        spec = task.prompt_spec(task.list_variations()[0], {})
        assert len(spec.system_instruction) > 0

    def test_run_python_capability_runs_code(self, task):
        spec = task.prompt_spec(task.list_variations()[0], {})
        run_python = next(c for c in spec.capabilities if c.name == "run_python")
        assert run_python.runs_code is True

    def test_no_orchestrator_mode_annotations_on_capabilities(self, task):
        spec = task.prompt_spec(task.list_variations()[0], {})
        for cap in spec.capabilities:
            assert "orchestrator_modes" not in cap.annotations

    def test_system_instruction_describes_goal_not_tool_method(self, task):
        """system_instruction is rendered into agent.yaml.prompt.system,
        which sits directly above the chat template's <tool_call> schema
        injection. Phrasing like "Use the run_python tool to ..." conflicts
        with that schema by priming code-block emission, while the schema
        instructs <tool_call> wrapping. Run 20260508-yfngrb showed 0%
        tool-call rate vs ~70% pre-workflow on the same model. Keep
        system_instruction goal-focused; tool-invocation framing belongs
        in the profile preamble + chat template."""
        spec = task.prompt_spec(task.list_variations()[0], {})
        # Goal preserved.
        assert "free non-volatile storage" in spec.system_instruction
        # No prescriptive tool-method conflict with the chat template.
        assert "run_python" not in spec.system_instruction


class TestWorkflow:
    def test_declares_phased_cleanup_workflow(self, task):
        variation = task.list_variations()[0]
        workflow = task.workflow(variation, {})

        assert workflow is not None
        assert [step.name for step in workflow.steps] == [
            "explore_container",
            "plan_cleanup",
            "execute_cleanup",
        ]
        assert [step.context_mode for step in workflow.steps] == [
            "inherit",
            "inherit",
            "inherit",
        ]
        assert workflow.steps[0].next_steps == ("plan_cleanup",)
        assert workflow.steps[1].next_steps == ("execute_cleanup",)
        assert workflow.steps[2].next_steps == ()

    def test_workflow_allows_tools_only_during_explore_and_execute(self, task):
        variation = task.list_variations()[0]
        spec = task.prompt_spec(variation, {})
        workflow = task.workflow(variation, {})

        assert workflow is not None
        workflow.validate({cap.name for cap in spec.capabilities})
        by_name = {step.name: step for step in workflow.steps}
        assert by_name["explore_container"].capabilities == ("run_python",)
        assert by_name["explore_container"].inherit_all_capabilities is False
        assert by_name["plan_cleanup"].capabilities == ()
        assert by_name["plan_cleanup"].inherit_all_capabilities is False
        assert by_name["execute_cleanup"].capabilities == ("run_python",)
        assert by_name["execute_cleanup"].inherit_all_capabilities is False

    def test_episode_constraints_allow_no_tool_planning_step(self, task):
        constraints = task.episode_constraints(task.list_variations()[0], {})

        assert constraints.budgets.max_task_tool_calls == 15
        assert constraints.budgets.max_llm_turns == 18
        assert constraints.success.min_task_tool_calls_for_success == 1
        assert (
            constraints.step_overrides[
                "plan_cleanup"
            ].success.min_task_tool_calls_for_success
            == 0
        )

    def test_workflow_step_prompts_do_not_prescribe_tool_by_name(self, task):
        """Step prompts that contain "Use run_python to ..." prime
        code-biased models for markdown JSON emission instead of
        <tool_call> tags. The capability list and chat template schema
        are the canonical source of tool-name truth — step prompts should
        describe the action goal, not the implementation."""
        variation = task.list_variations()[0]
        workflow = task.workflow(variation, {})
        assert workflow is not None
        for step in workflow.steps:
            assert "Use run_python to" not in step.prompt, (
                f"step {step.name!r} still prescribes run_python by name"
            )


class TestParseResponse:
    def test_explorer_extracts_space_freed(self, task):
        raw = "Results: Freed 1024 KB of disk space"
        invs = task.parse_response(raw, invoked_capability="explorer")
        assert len(invs) == 1
        # parse_response returns a synthetic metric-report invocation; the
        # name is harness-internal, distinct from any declared MCP tool.
        assert invs[0].name == "explorer_metric_report"
        assert invs[0].input.get("space_freed_kb") == 1024.0

    def test_self_identifying_extracts_space_freed_without_phase_hint(self, task):
        raw = "Results: Freed 1024 KB of disk space"
        invs = task.parse_response(raw)
        assert len(invs) == 1
        assert invs[0].name == "metric_report"
        assert invs[0].input.get("space_freed_kb") == 1024.0

    def test_explorer_extracts_mb(self, task):
        raw = "Results: Freed 2.5 MB"
        invs = task.parse_response(raw, invoked_capability="explorer")
        assert invs[0].input.get("space_freed_kb") == 2.5 * 1024

    def test_explorer_returns_empty_on_no_kb(self, task):
        # Without KB mentions the task yields no space_freed invocation.
        raw = "Results: Nothing found"
        invs = task.parse_response(raw, invoked_capability="explorer")
        # May emit one with 0.0 if fallback regex hits; assert shape.
        assert all(isinstance(i.input, dict) for i in invs)

    def test_orchestrator_returns_empty(self, task):
        assert (
            task.parse_response("MODE: investigator", invoked_capability="orchestrator")
            == []
        )

    def test_recorder_returns_empty(self, task):
        assert task.parse_response("some text", invoked_capability="recorder") == []

    def test_does_not_raise(self, task):
        result = task.parse_response("", invoked_capability="explorer")
        assert isinstance(result, list)


class TestComputeRewardPurity:
    """compute_reward is pure — no Docker, no I/O."""

    def test_custom_threshold(self, task_custom):
        initial = MemoryCleanupState(
            used_kb={"c1": 105.0},
            filesystem_groups={"device:8:0": ["c1"]},
        )
        final = MemoryCleanupState(
            used_kb={"c1": 100.0},
            filesystem_groups={"device:8:0": ["c1"]},
        )
        artifacts = EpisodeArtifacts()
        reward = task_custom.compute_reward(initial, final, artifacts)
        assert reward.value == 5.0
        # threshold is 10.0, so 5.0 > 10.0 is False
        assert reward.success is False

    def test_deduplication_shared_filesystem(self, task):
        """Two containers on the same filesystem — space freed is counted once."""
        initial = MemoryCleanupState(
            used_kb={"c1": 200.0, "c2": 200.0},
            filesystem_groups={"device:8:0": ["c1", "c2"]},
        )
        final = MemoryCleanupState(
            used_kb={"c1": 100.0, "c2": 100.0},
            filesystem_groups={"device:8:0": ["c1", "c2"]},
        )
        artifacts = EpisodeArtifacts()
        reward = task.compute_reward(initial, final, artifacts)
        # Average: (200-100 + 200-100)/2 containers on same fs = 100.0
        assert reward.value == 100.0

    def test_deduplication_separate_filesystems(self, task):
        """Two containers on separate filesystems — space freed is summed."""
        initial = MemoryCleanupState(
            used_kb={"c1": 200.0, "c2": 300.0},
            filesystem_groups={"fs1": ["c1"], "fs2": ["c2"]},
        )
        final = MemoryCleanupState(
            used_kb={"c1": 100.0, "c2": 100.0},
            filesystem_groups={"fs1": ["c1"], "fs2": ["c2"]},
        )
        artifacts = EpisodeArtifacts()
        reward = task.compute_reward(initial, final, artifacts)
        # fs1: 200-100 = 100, fs2: 300-100 = 200, total = 300
        assert reward.value == 300.0


class TestPopulateExitCodeCheck:
    """populate must check exit codes from container commands."""

    def test_populate_reports_failure_on_nonzero_exit(self, task):
        """When a dd command fails (non-zero exit), populate reports success=False
        and surfaces the captured output as a diagnostic."""
        container = MagicMock()
        container.id = "container-abc123"
        container.name = "test-container"

        variation = task.list_variations()[0]
        # Sequence: mkdir, baseline du, dir-visibility wait, dd loop fails.
        # `set -ec` → first failing dd aborts the loop, only one dd exec runs.
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            side_effect=[
                (0, b""),                      # mkdir
                (0, b"__MEASURE_KB__=0\n"),    # baseline du
                (0, b""),                      # dir-visibility wait
                (1, b"dd: failed to open '/x/y.bak': No such file or directory"),
            ],
        ):
            outcome = task.populate([container], variation)
        assert len(outcome.results) == 1
        assert outcome.results[0].success is False
        msg = outcome.results[0].error_message
        assert "Population command failed" in msg
        # Captured stderr/stdout must surface so we don't lose diagnostics.
        assert "No such file or directory" in msg

    def test_populate_raises_on_mkdir_nonzero_exit(self, task):
        """mkdir must not silently swallow non-zero exits — otherwise dd's
        fail with no diagnostic. The captured output must be surfaced."""
        container = MagicMock()
        container.id = "container-mkdir-fail"
        container.name = "test-container"

        variation = task.list_variations()[0]
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(1, b"mkdir: cannot create directory: Read-only file system"),
        ):
            with pytest.raises(TaskEnvironmentError) as exc_info:
                task.populate([container], variation)
        assert "mkdir" in str(exc_info.value).lower()
        assert "Read-only file system" in str(exc_info.value)
        assert exc_info.value.container_ids == ["container-mkdir-fail"]


class TestPopulateNoGlobalSync:
    """populate must not call global ``sync`` — under heavy parallel I/O it
    blocks indefinitely (flushes ALL dirty pages system-wide), causing the
    120s exec timeout to fire and leaving zombie sync processes inside the
    container that prevent further docker exec setns operations."""

    def test_populate_does_not_call_global_sync(self, task):
        container = MagicMock()
        container.id = "container-no-sync"
        container.name = "test-container"

        variation = task.list_variations()[0]
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, b"__MEASURE_KB__=0\n"),
        ) as mock_exec:
            outcome = task.populate([container], variation)

        assert outcome.results[0].success is True
        for call in mock_exec.call_args_list:
            cmd = call.args[1]
            # Bare `sync` (whole-system flush) is forbidden. Per-fs `sync -f`
            # and per-file `dd ... conv=fdatasync` are fine.
            assert cmd != ["sync"], (
                f"populate must not call bare global sync; got: {cmd}"
            )
            if isinstance(cmd, list):
                joined = " ".join(cmd)
                assert " sync$" not in joined and not joined.endswith(" sync"), (
                    f"populate must not call bare global sync; got: {cmd}"
                )


class TestVariationDdHardening:
    """dd commands inside variations must use bounded per-file flushing and
    must NOT suppress dd's stderr — otherwise failures look like silent
    `exit 1` with no actionable diagnostic."""

    def test_dd_commands_keep_errors_visible(self, task):
        """`2>/dev/null` swallows real errors. Use `status=none` instead so
        normal progress noise is suppressed but errors still bubble up."""
        for variation in task.list_variations():
            assert isinstance(variation, MemoryCleanupVariation)
            for cmd in variation.commands:
                assert "2>/dev/null" not in cmd, (
                    f"variation {variation.name!r} suppresses dd stderr "
                    f"with `2>/dev/null` — replace with `status=none`: {cmd}"
                )
                assert "status=none" in cmd, (
                    f"variation {variation.name!r} dd commands must use "
                    f"`status=none` to suppress progress noise: {cmd}"
                )

    def test_dd_commands_use_per_file_fsync(self, task):
        """Each dd should flush its own file (`conv=fdatasync`), so we don't
        need a global `sync` afterwards. fdatasync is bounded to the single
        file's dirty data, not all dirty pages on the host."""
        for variation in task.list_variations():
            assert isinstance(variation, MemoryCleanupVariation)
            for cmd in variation.commands:
                assert "conv=fdatasync" in cmd, (
                    f"variation {variation.name!r} dd commands must include "
                    f"`conv=fdatasync` for per-file flushing: {cmd}"
                )


class TestPopulateUsesErrexit:
    """A dd loop with no errexit returns the LAST dd's exit code, hiding
    earlier failures. Use `sh -ec` so the first failure aborts immediately
    and the captured output points at the actual failing command."""

    def test_populate_runs_commands_with_errexit(self, task):
        container = MagicMock()
        container.id = "container-errexit"
        container.name = "test-container"

        variation = task.list_variations()[0]
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, b"__MEASURE_KB__=0\n"),
        ) as mock_exec:
            task.populate([container], variation)

        # Find the dd-loop calls (skip mkdir at index 0 and du measurement).
        dd_calls = [
            call for call in mock_exec.call_args_list
            if isinstance(call.args[1], list)
            and len(call.args[1]) >= 2
            and "dd " in (call.args[1][-1] if isinstance(call.args[1][-1], str) else "")
        ]
        assert dd_calls, "expected at least one dd-loop exec"
        for call in dd_calls:
            cmd = call.args[1]
            # Either `sh -ec '<cmd>'` or `sh -c 'set -e; <cmd>'`.
            uses_ec_flag = cmd[:2] == ["sh", "-ec"]
            uses_set_e = (
                cmd[:2] == ["sh", "-c"]
                and isinstance(cmd[2], str)
                and cmd[2].lstrip().startswith("set -e")
            )
            assert uses_ec_flag or uses_set_e, (
                f"populate must run dd commands with errexit so the first "
                f"failing dd aborts the loop; got: {cmd}"
            )


class TestPopulateDirVisibilityWait:
    """populate must absorb the overlay2 directory-entry visibility race
    between mkdir's exec_run and the subsequent dd exec_run.

    Symptom under 12-way parallel load: ``dd: failed to open '/.../foo.dat':
    No such file or directory`` even though the prior mkdir reported exit
    0. The mkdir really did succeed inside the container — but a later,
    independent exec_run did not yet see the new dirent because overlay2
    metadata propagation was queued behind heavy concurrent writes.

    Fix: a bounded ``[ -d ]`` poll inside its own exec_run before the dd
    loop starts. Cheap (≤ 1s), surgical, and re-uses the visibility
    machinery the kernel already provides.
    """

    def test_populate_runs_dir_visibility_wait_before_dd(self, task):
        container = MagicMock()
        container.id = "vis-container"
        container.name = "vis-container"

        variation = task.list_variations()[0]
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, b"__MEASURE_KB__=0\n"),
        ) as mock_exec:
            task.populate([container], variation)

        # Identify the visibility-wait call: it must precede every dd-loop
        # exec and reference the variation's created_dirs.
        calls = mock_exec.call_args_list
        wait_idx = None
        for i, call in enumerate(calls):
            cmd = call.args[1]
            if not isinstance(cmd, list) or len(cmd) < 3:
                continue
            script = cmd[-1] if isinstance(cmd[-1], str) else ""
            # The wait script polls with `[ -d ... ]` — never invokes dd.
            if "dd " in script:
                continue
            if "-d " not in script:
                continue
            for created_dir in variation.created_dirs:
                assert created_dir in script, (
                    f"visibility wait missing dir {created_dir!r} in: {script}"
                )
            wait_idx = i
            break

        assert wait_idx is not None, (
            "expected a dir-visibility wait exec_run between baseline "
            "measurement and the dd loop; found none in:\n"
            + "\n".join(repr(c.args[1]) for c in calls)
        )

        # Every dd-loop exec must come AFTER the visibility wait.
        for i, call in enumerate(calls):
            cmd = call.args[1]
            if not (isinstance(cmd, list) and len(cmd) >= 3
                    and isinstance(cmd[-1], str) and "dd " in cmd[-1]):
                continue
            assert i > wait_idx, (
                f"dd-loop exec at index {i} ran before the visibility wait "
                f"at index {wait_idx}"
            )

    def test_visibility_wait_skipped_when_no_created_dirs(self, task):
        """Variations without ``created_dirs`` have nothing to wait for —
        the wait exec is skipped to avoid a spurious empty-loop exec."""
        container = MagicMock()
        container.id = "no-dirs-container"
        container.name = "no-dirs-container"

        variation = MemoryCleanupVariation(
            name="no-dirs",
            description="nothing to wait for",
            commands=[
                "for i in $(seq 1 1); do dd if=/dev/zero of=/tmp/x bs=1K count=1 conv=fdatasync status=none; done",
            ],
            expected_kb=1.0,
            mkdir_cmd="",
            created_dirs=[],
        )
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, b"__MEASURE_KB__=0\n"),
        ) as mock_exec:
            task.populate([container], variation)

        for call in mock_exec.call_args_list:
            cmd = call.args[1]
            if isinstance(cmd, list) and len(cmd) >= 3 and isinstance(cmd[-1], str):
                # No standalone visibility-wait exec when there are no dirs.
                assert "-d " not in cmd[-1] or "dd " in cmd[-1], (
                    f"unexpected visibility-wait exec: {cmd}"
                )


class TestPopulateBaselineRetry:
    """The baseline measurement is the first du-after-populate operation
    in the episode. It runs while the populate phase across other slots
    is still actively writing — peak overlay2 contention. One retry
    (the prior contract) was insufficient under 12-way parallel load."""

    def test_populate_baseline_attempts_three_times_before_raising(self, task):
        container = MagicMock()
        container.id = "baseline-fail"
        container.name = "baseline-fail"

        variation = task.list_variations()[0]

        with patch.object(task, "_measure_population_kb", return_value=None) as mock_measure:
            with patch("tasks.memory_cleanup._exec_run_with_timeout", return_value=(0, b"")):
                with patch("tasks.memory_cleanup.time.sleep") as mock_sleep:
                    with pytest.raises(TaskEnvironmentError, match="Baseline measurement failed"):
                        task.populate([container], variation)

        # 3 attempts (1 initial + 2 retries), 2 sleeps in between.
        assert mock_measure.call_count == 3
        assert mock_sleep.call_count == 2

    def test_populate_baseline_succeeds_on_third_attempt(self, task):
        container = MagicMock()
        container.id = "baseline-eventual"
        container.name = "baseline-eventual"

        variation = task.list_variations()[0]

        with patch.object(
            task,
            "_measure_population_kb",
            side_effect=[None, None, 100.0],
        ) as mock_measure:
            with patch("tasks.memory_cleanup._exec_run_with_timeout", return_value=(0, b"")):
                with patch("tasks.memory_cleanup.time.sleep"):
                    outcome = task.populate([container], variation)

        assert mock_measure.call_count == 3
        assert outcome.results[0].success is True
        assert outcome.episode_context["baseline_kb"]["baseline-eventual"] == 100.0


class TestConfigInjection:
    def test_default_docker_compose_dir(self, task):
        assert task.docker_compose_dir == "docker/special-learn-compose"

    def test_override_docker_compose_dir(self, task_custom):
        assert task_custom.docker_compose_dir == "docker/custom"

    def test_default_tolerance(self, task):
        assert task._tolerance == 0.10

    def test_override_tolerance(self, task_custom):
        assert task_custom._tolerance == 0.1

    def test_default_success_threshold(self, task):
        assert task._success_threshold_kb == 0.0

    def test_override_success_threshold(self, task_custom):
        assert task_custom._success_threshold_kb == 10.0


class TestMeasurementTimeouts:
    """Measurement helpers must use _exec_run_with_timeout, not bare exec_run."""

    def test_measure_state_raises_when_measurement_fails(self, task):
        """_measure_state must raise after exhausting its retry budget.

        Budget is 5 attempts (initial + 4 retries) — under 12-way parallel
        load, the prior 3-attempt budget exhausted on transient overlay2
        EIO before kernel scheduling cleared the contention.
        """
        container = MagicMock()
        container.id = "hang-container"
        container.name = "hang-container"
        container.attrs = {"GraphDriver": {"Data": {}}}

        with patch.object(
            task, "_measure_population_kb", return_value=None
        ) as mock_measure:
            with patch("tasks.memory_cleanup.time.sleep") as mock_sleep:
                with pytest.raises(
                    TaskEnvironmentError, match="Used-space measurement failed"
                ):
                    task._measure_state([container])

        assert mock_measure.call_count == 5
        assert mock_sleep.call_count == 4

    def test_measure_state_retries_on_transient_failure(self, task):
        container = MagicMock()
        container.id = "retry-container"
        container.name = "retry-container"
        container.attrs = {"GraphDriver": {"Data": {}}}

        with patch.object(
            task,
            "_measure_population_kb",
            side_effect=[None, None, 42.0],
        ):
            with patch("tasks.memory_cleanup.time.sleep") as mock_sleep:
                result = task._measure_state([container])

        assert result.used_kb["retry-container"] == 42.0
        assert mock_sleep.call_count == 2

    def test_measure_state_raises_after_exhausted_retries(self, task):
        container = MagicMock()
        container.id = "fail-container"
        container.name = "fail-container"
        container.attrs = {"GraphDriver": {"Data": {}}}

        with patch.object(task, "_measure_population_kb", return_value=None):
            with patch("tasks.memory_cleanup.time.sleep"):
                with pytest.raises(TaskEnvironmentError, match="after 5 attempts"):
                    task._measure_state([container])

    def test_measure_state_uses_one_second_backoff_step(self, task):
        """Backoff is 1s/2s/3s/4s — wider than the prior 0.5s step. The
        narrow 0.5s/1.0s/1.5s window cleared too fast under heavy
        12-parallel overlay2 contention; widening buys time for the
        kernel writeback queue to drain."""
        container = MagicMock()
        container.id = "backoff-container"
        container.name = "backoff-container"
        container.attrs = {"GraphDriver": {"Data": {}}}

        with patch.object(task, "_measure_population_kb", return_value=None):
            with patch("tasks.memory_cleanup.time.sleep") as mock_sleep:
                with pytest.raises(TaskEnvironmentError):
                    task._measure_state([container])

        # 4 sleeps between 5 attempts: 1.0s, 2.0s, 3.0s, 4.0s.
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 3.0, 4.0]

    def test_measure_state_no_retry_on_first_success(self, task):
        container = MagicMock()
        container.id = "ok-container"
        container.name = "ok-container"
        container.attrs = {"GraphDriver": {"Data": {}}}

        with patch.object(
            task, "_measure_population_kb", return_value=100.0
        ) as mock_measure:
            with patch("tasks.memory_cleanup.time.sleep") as mock_sleep:
                task._measure_state([container])

        mock_measure.assert_called_once()
        mock_sleep.assert_not_called()

    def test_verify_population_retries_after_transient_measurement_failure(self, task):
        container = MagicMock()
        container.id = "container-1"
        container.name = "container-1"

        variation = task.list_variations()[0]
        episode_context = {"baseline_kb": {"container-1": 0.0}}

        with patch.object(
            task,
            "_measure_population_kb",
            side_effect=[None, variation.expected_kb],
        ):
            with patch("tasks.memory_cleanup.time.sleep") as mock_sleep:
                verified = task.verify_population(
                    [container], variation, episode_context
                )

        assert verified is True
        mock_sleep.assert_called_once_with(0.5)

    def test_populate_last_command_is_dd_not_global_sync(self, task):
        """Bare `sync` is forbidden (see TestPopulateNoGlobalSync). The last
        exec for a successful populate should be the final dd command, with
        per-file flushing baked in via `conv=fdatasync`."""
        container = MagicMock()
        container.id = "container-sync"
        container.name = "container-sync"

        variation = task.list_variations()[0]

        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, b"__MEASURE_KB__=0\n"),
        ) as mock_exec:
            outcome = task.populate([container], variation)

        assert outcome.results[0].success is True
        last_cmd = mock_exec.call_args_list[-1].args[1]
        assert last_cmd != ["sync"]
        # The final exec should be a dd-loop shell invocation.
        assert isinstance(last_cmd, list) and last_cmd[0] == "sh"
        assert "dd " in last_cmd[-1]

    def test_measure_population_kb_returns_none_on_timeout(self, task):
        container = MagicMock()
        container.id = "hang-container"
        container.name = "hang-container"

        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            side_effect=TaskEnvironmentError("Docker exec timed out after 120s"),
        ):
            result = task._measure_population_kb(container)

        assert result is None
