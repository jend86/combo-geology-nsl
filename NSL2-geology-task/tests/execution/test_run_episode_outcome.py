from pathlib import Path
from unittest.mock import MagicMock, patch

from src.execution import BackendRuntime
from src.execution.episode import (
    EpisodeOutcome,
    EpisodeRequest,
    _snapshot_episode_telemetry,
    run_episode,
)
from src.harness.base import HarnessError
from src.harness.transcript import HarnessTranscript
from src.task.base import TaskEnvironmentError
from src.task.types import EpisodeArtifacts, TaskReward, Variation
from src.typing.config import AppConfig


class _FakeRecorder:
    def __init__(self, **_kwargs):
        self.inference_records = [MagicMock()]
        self._counters = {"tool_calls": 0}
        self._labels = {}

    def snapshot_counters(self):
        return dict(self._counters)

    def snapshot_labels(self):
        return dict(self._labels)

    def bump_counter(self, key: str, by: int = 1):
        self._counters[key] = self._counters.get(key, 0) + by

    def set_label(self, key: str, value: str):
        self._labels[key] = value

    def log_state(self, category: str, payload: dict) -> None:
        pass


class _FakeRecorderWithToolCalls(_FakeRecorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counters = {"tool_calls": 1}


class _FakeTracedGenner:
    def __init__(self, **kwargs):
        self.inner = kwargs["inner"]
        self.recorder = kwargs["recorder"]
        self.cancel_event = kwargs["cancel_event"]
        self.episode_id = kwargs["episode_id"]
        self.budget_ledger = None

    def set_budget_ledger(self, ledger):
        self.budget_ledger = ledger


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        model_name="claude",
        code_host_cache_path=str(tmp_path / "cache"),
        container_ids=["container-a"],
        train_data_save_folder=str(tmp_path / "train-data"),
        harness={
            "name": "orchestrator_modes",
            "orchestrator_modes": {
                "orchestrator_prompt": "prompt {scratchpad_content}",
            },
        },
        observability={"enabled": False},
    )


def test_run_episode_returns_episode_outcome_with_legacy_prompt_responses(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.return_value = {}
    task.finalize_episode.return_value = TaskReward(
        value=1.0,
        success=True,
        breakdown={"reward": "ok"},
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-1",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )
    train_rows = [
        {
            "prompt": "prompt-1",
            "raw_response": "response-1",
            "timestamp": "2026-04-29T00:00:00",
            "interaction_type": "orchestrator",
            "success": True,
            "error_message": None,
        }
    ]
    transcript = HarnessTranscript(
        artifacts=EpisodeArtifacts(),
        llm_turns=1,
        termination_reason="done",
        termination_category="success",
        extra={"scratchpad_final": "scratch", "mode_history": ["planner"]},
    )

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorderWithToolCalls,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch(
            "src.execution.episode.construct_harness",
            return_value=MagicMock(run_episode=MagicMock(return_value=transcript)),
        ),
        patch("src.execution.episode.records_to_rows", return_value=train_rows),
    ):
        outcome = run_episode(runtime, request)

    assert isinstance(outcome, EpisodeOutcome)
    assert outcome.success is True
    assert outcome.prompt_responses == [
        {
            "prompt": "prompt-1",
            "raw_response": "response-1",
            "timestamp": "2026-04-29T00:00:00",
            "interaction_type": "orchestrator",
            "success": True,
            "error_message": None,
        }
    ]
    assert not hasattr(outcome, "episode_runtime_success")
    assert outcome.tool_calls_count == 1


def test_run_episode_returns_measurement_error_outcome(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.side_effect = TaskEnvironmentError(
        "Used-space measurement failed on container-a",
        container_ids=["container-a"],
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-measurement",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorder,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch("src.execution.episode.construct_harness"),
    ):
        outcome = run_episode(runtime, request)

    assert outcome.success is False
    assert outcome.partial is True
    assert outcome.score == 0.0
    assert outcome.tool_calls_count == 0
    assert outcome.error_category == "measurement_error"
    assert "Used-space measurement failed" in (outcome.error_message or "")
    assert "scratchpad_final" not in outcome.trajectory


def test_run_episode_marks_zero_tool_call_success_as_failure(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.return_value = {}
    task.finalize_episode.return_value = TaskReward(
        value=4832.0,
        success=True,
        breakdown={"reward": "artifact"},
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-spurious-success",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )
    train_rows = [
        {
            "prompt": "prompt-1",
            "raw_response": "response-1",
            "timestamp": "2026-04-29T00:00:00",
            "interaction_type": "orchestrator",
            "success": True,
            "error_message": None,
        }
    ]
    transcript = HarnessTranscript(
        artifacts=EpisodeArtifacts(),
        llm_turns=1,
        termination_reason="done",
        termination_category="success",
        extra={"scratchpad_final": "scratch", "mode_history": ["planner"]},
    )

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorder,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch(
            "src.execution.episode.construct_harness",
            return_value=MagicMock(run_episode=MagicMock(return_value=transcript)),
        ),
        patch("src.execution.episode.records_to_rows", return_value=train_rows),
    ):
        outcome = run_episode(runtime, request)

    assert outcome.score == 4832.0
    assert outcome.tool_calls_count == 0
    assert outcome.success is False


def test_run_episode_uses_budget_ledger_for_llm_turn_count(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.return_value = {}
    task.finalize_episode.return_value = TaskReward(
        value=0.0,
        success=False,
        breakdown={"reward": "ok"},
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-llm-turns",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )

    class _LedgerHarness:
        def run_episode(self, *, ctx):
            assert ctx.budget_ledger is not None
            ctx.budget_ledger.record_llm_turn()
            ctx.budget_ledger.record_llm_turn()
            ctx.budget_ledger.record_llm_turn()
            return HarnessTranscript(
                artifacts=EpisodeArtifacts(),
                llm_turns=0,
                termination_reason="done",
                termination_category="success",
                extra={},
            )

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorder,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch("src.execution.episode.construct_harness", return_value=_LedgerHarness()),
        patch("src.execution.episode.records_to_rows", return_value=[]),
    ):
        outcome = run_episode(runtime, request)

    assert outcome.llm_turns_count == 3
    assert outcome.trajectory["llm_turns"] == 3


def test_run_episode_uses_failure_extras_when_harness_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.return_value = {}
    task.finalize_episode.return_value = TaskReward(
        value=0.0,
        success=False,
        breakdown={"reward": "ok"},
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-harness-error",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )
    harness = MagicMock()
    harness.run_episode.side_effect = HarnessError("broken harness")
    harness.failure_extras.return_value = {
        "scratchpad_final": "scratch",
        "mode_history": ["planner"],
    }

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorderWithToolCalls,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch("src.execution.episode.construct_harness", return_value=harness),
        patch("src.execution.episode.records_to_rows", return_value=[]),
    ):
        outcome = run_episode(runtime, request)

    assert outcome.harness_error is True
    assert outcome.error_category == "harness_error"
    assert outcome.trajectory["extra"] == {
        "scratchpad_final": "scratch",
        "mode_history": ["planner"],
    }
    assert "scratchpad_final" not in outcome.trajectory


def test_run_episode_ignores_raising_failure_extras(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.return_value = {}
    task.finalize_episode.return_value = TaskReward(
        value=0.0,
        success=False,
        breakdown={"reward": "ok"},
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-harness-error-bad-extras",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )
    harness = MagicMock()
    harness.run_episode.side_effect = HarnessError("broken harness")
    harness.failure_extras.side_effect = RuntimeError("bad extras")

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorderWithToolCalls,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch("src.execution.episode.construct_harness", return_value=harness),
        patch("src.execution.episode.records_to_rows", return_value=[]),
    ):
        outcome = run_episode(runtime, request)

    assert outcome.harness_error is True
    assert outcome.trajectory["extra"] == {}
    assert "scratchpad_final" not in outcome.trajectory


def test_run_episode_publishes_framework_tool_call_telemetry(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    task = MagicMock()
    task.agent_service_name = "agent"
    task.metric_unit = "kb"
    task.prompt_spec.return_value = MagicMock()
    task.measure_initial_state.return_value = {}
    task.finalize_episode.return_value = TaskReward(
        value=0.0,
        success=False,
        breakdown={"reward": "ok"},
    )
    runtime = BackendRuntime(
        config=config,
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-telemetry",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
        telemetry_observer=MagicMock(),
    )

    class _HarnessWithoutToolTelemetry:
        def telemetry(self):
            return {"phase": "running"}

        def run_episode(self, ctx):
            ctx.recorder.bump_counter("tool_calls", 2)
            return HarnessTranscript(
                artifacts=EpisodeArtifacts(),
                llm_turns=1,
                termination_reason="done",
                termination_category="agent_failure",
                extra={},
            )

    with (
        patch(
            "src.execution.episode.resolve_event_recorder_class",
            return_value=_FakeRecorder,
        ),
        patch(
            "src.execution.episode.resolve_traced_genner_class",
            return_value=_FakeTracedGenner,
        ),
        patch(
            "src.execution.episode.construct_harness",
            return_value=_HarnessWithoutToolTelemetry(),
        ),
        patch("src.execution.episode.records_to_rows", return_value=[]),
    ):
        outcome = run_episode(runtime, request)

    assert outcome.tool_calls_count == 2
    request.telemetry_observer.assert_called_with(
        None,
        {"phase": "running", "tool_calls": "2"},
    )


def test_snapshot_episode_telemetry_uses_one_recorder_snapshot() -> None:
    class _CountingRecorder:
        def __init__(self):
            self.counter_snapshots = 0
            self.label_snapshots = 0

        def snapshot_counters(self):
            self.counter_snapshots += 1
            return {"turns": 1, "tool_calls": 2}

        def snapshot_labels(self):
            self.label_snapshots += 1
            return {"last_tool": "analyzer"}

    class _RecorderBackedHarness:
        def telemetry_from_recorder_snapshot(self, counters, labels):
            return {
                "turns": str(counters["turns"]),
                "last_tool": labels["last_tool"],
            }

    recorder = _CountingRecorder()

    telemetry = _snapshot_episode_telemetry(_RecorderBackedHarness(), recorder)

    assert telemetry == {
        "turns": "1",
        "last_tool": "analyzer",
        "tool_calls": "2",
    }
    assert recorder.counter_snapshots == 1
    assert recorder.label_snapshots == 1
