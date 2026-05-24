"""TDD tests: per-turn proposer transform + rewrite-output synthesis.

Mirrors the GeologyProposerRows precedent for feature_hypothesis. Two surfaces:

  1. ``FeatureHypothesisProposerRows`` (and its Kazakhstan twin) filter
     ``raw_training_rows`` down to non-``code`` workflow steps and reject any
     row whose ``workflow_step`` is None.
  2. ``_exec_submit_rewrite`` synthesizes one extra inference row carrying the
     rewrite agent's ``(prompt, response)`` pair (BIC verdict appended), via
     the recorder exposed on ``CapabilityExecutionContext``.

The transform contract: ``transform_export_rows`` returns ``EpisodeTrainingRows``
groups whose ``rows`` survive ``validate_training_row_groups``.
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from src.harness.recorder import EventRecorder, TrajectoryRecord
from src.task.types import CapabilityExecutionContext, CapabilityInvocation
from src.training_data.transforms import EpisodeTrainingRows
from tasks.feature_hypothesis import (
    FeatureHypothesisProposerRows,
    FeatureHypothesisTask,
    FeatureHypothesisVariation,
)
from tasks.feature_hypothesis_kazakhstan import (
    FeatureHypothesisKazakhstanProposerRows,
    FeatureHypothesisKazakhstanTask,
    FeatureHypothesisKazakhstanVariation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    episode_id: str = "ep-0",
    row_index: int = 0,
    workflow_step: str | None = "survey",
    actor_role: str | None = None,
) -> dict[str, Any]:
    return {
        "row_id": f"{episode_id}:{row_index}",
        "parent_row_id": None,
        "prompt": f"prompt-{row_index}",
        "raw_response": f"response-{row_index}",
        "interaction_type": "orchestrator",
        "source_interaction_type": "orchestrator",
        "timestamp": "2026-05-24T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": 0,
        "generation_id": 1,
        "episode_score": 1.0,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": actor_role,
        "record_meta": {},
    }


def _group(workflow_steps: list[str | None], episode_id: str = "ep-0") -> EpisodeTrainingRows:
    rows = [
        _row(episode_id=episode_id, row_index=i, workflow_step=step)
        for i, step in enumerate(workflow_steps)
    ]
    return EpisodeTrainingRows(
        episode_id=episode_id,
        episode_index=0,
        generation_id=1,
        episode_score=1.0,
        rows=rows,
    )


def _make_recorder(tmp_path: Path, episode_id: str = "ep-test") -> EventRecorder:
    return EventRecorder(episode_id, tmp_path / "events.jsonl")


def _task(tmp_path: Path) -> FeatureHypothesisTask:
    return FeatureHypothesisTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
        }
    )


def _variation(tmp_path: Path) -> FeatureHypothesisVariation:
    return FeatureHypothesisVariation(
        name="coe_fairbairn",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "coe_fairbairn"),
        kg_dir=str(tmp_path / "kg" / "coe_fairbairn"),
        min_features=0,
        crossbreed_enabled=False,
    )


def _kz_task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
        }
    )


def _kz_variation(tmp_path: Path) -> FeatureHypothesisKazakhstanVariation:
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=str(tmp_path / "kg" / "teniz_basin"),
        min_features=0,
        crossbreed_enabled=False,
    )


def _phase_records(*, admitted: bool = True, bic_delta: float = -1.5) -> dict[str, Any]:
    return {
        "hypothesise": {
            "hypothesis": "test hypothesis",
            "data_spec": {},
            "parent_experiments": [],
        },
        "code": {"result_summary": "ok"},
        "translate": {"feature_layer_name": "test_layer"},
        "evaluate": {
            "bic_delta": bic_delta,
            "admitted": admitted,
            "mutual_info": {},
            "masking_test_passed": True,
            "masking_test_improvement": 0.3,
            "masking_test_direction": "improvement",
            "stage_completed": "stage_2_completed",
        },
    }


# ---------------------------------------------------------------------------
# Transform: filter behaviour
# ---------------------------------------------------------------------------


class TestFeatureHypothesisProposerRows:
    def test_keeps_proposer_steps_drops_code(self) -> None:
        transform = FeatureHypothesisProposerRows()
        groups = [_group(["survey", "hypothesise", "code", "translate", "rewrite"])]

        result = transform.transform_export_rows(context=None, episodes=groups)

        kept = [r["workflow_step"] for r in result[0].rows]
        assert "code" not in kept
        assert kept == ["survey", "hypothesise", "translate", "rewrite"]

    def test_raises_on_null_workflow_step(self) -> None:
        transform = FeatureHypothesisProposerRows()
        groups = [_group(["survey", None, "rewrite"])]
        with pytest.raises(ValueError, match="workflow_step"):
            transform.transform_export_rows(context=None, episodes=groups)

    def test_config_lists_allowed_steps(self) -> None:
        transform = FeatureHypothesisProposerRows()
        config = transform.config()
        assert "included_workflow_steps" in config
        allowed = set(config["included_workflow_steps"])
        assert {"survey", "hypothesise", "translate", "rewrite"} <= allowed
        assert "code" not in allowed

    def test_name_is_stable_versioned(self) -> None:
        assert FeatureHypothesisProposerRows().name == "FeatureHypothesisProposerRows[v1]"

    def test_preserves_episode_score_after_filter(self) -> None:
        transform = FeatureHypothesisProposerRows()
        rows = [
            _row(row_index=0, workflow_step="survey"),
            _row(row_index=1, workflow_step="code"),
        ]
        groups = [
            EpisodeTrainingRows(
                episode_id="ep-0",
                episode_index=0,
                generation_id=1,
                episode_score=0.42,
                rows=rows,
            )
        ]
        result = transform.transform_export_rows(context=None, episodes=groups)
        assert result[0].episode_score == 0.42
        assert len(result[0].rows) == 1

    def test_crossbreed_episode_without_survey_passes_through(self) -> None:
        transform = FeatureHypothesisProposerRows()
        # crossbreed-mode episodes skip survey
        groups = [_group(["hypothesise", "code", "translate", "rewrite"])]

        result = transform.transform_export_rows(context=None, episodes=groups)
        kept = [r["workflow_step"] for r in result[0].rows]
        assert kept == ["hypothesise", "translate", "rewrite"]

    def test_kazakhstan_variant_drops_code(self) -> None:
        transform = FeatureHypothesisKazakhstanProposerRows()
        groups = [_group(["survey", "hypothesise", "code", "translate", "rewrite"])]
        result = transform.transform_export_rows(context=None, episodes=groups)
        kept = [r["workflow_step"] for r in result[0].rows]
        assert "code" not in kept


# ---------------------------------------------------------------------------
# Task wiring: training_data_transforms()
# ---------------------------------------------------------------------------


class TestTaskWiresTransform:
    def test_feature_hypothesis_task_returns_proposer_rows(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        transforms = tuple(task.training_data_transforms())
        assert len(transforms) == 1
        assert isinstance(transforms[0], FeatureHypothesisProposerRows)

    def test_kazakhstan_task_returns_proposer_rows(self, tmp_path: Path) -> None:
        task = _kz_task(tmp_path)
        transforms = tuple(task.training_data_transforms())
        assert len(transforms) == 1
        assert isinstance(transforms[0], FeatureHypothesisKazakhstanProposerRows)


# ---------------------------------------------------------------------------
# Rewrite-output synthesis: _exec_submit_rewrite records a synthetic row
# ---------------------------------------------------------------------------


class TestSubmitRewriteSynthesizesRow:
    """`_exec_submit_rewrite` must call ``recorder.record_inference()`` once
    with a synthetic record carrying the training_pair's prompt + response (BIC
    verdict appended). The synthetic record flows through ``records_to_rows``
    so the SFT export contains the rewriter's polished output, not just the
    transcript-level rewrite-step rows whose raw_response is empty for
    tool-call-only turns.
    """

    def _drive_submit(
        self,
        task: Any,
        variation: Any,
        recorder: EventRecorder | None,
        *,
        episode_id: str = "ep_synth_test",
        prompt: str = "the hypothesis prompt",
        response: str = "the analysis response",
        admitted: bool = True,
        bic_delta: float = -1.5,
    ) -> dict[str, Any]:
        kg_dir = Path(variation.kg_dir)
        store_dir = Path(variation.store_dir)
        kg_dir.mkdir(parents=True, exist_ok=True)
        store_dir.mkdir(parents=True, exist_ok=True)

        episode_context: dict[str, Any] = {
            "episode_id": episode_id,
            "store_dir": str(store_dir),
            "kg_dir": str(kg_dir),
            "grid_spec": variation.grid_spec,
            "workflow_kind": "survey",
            "phase_records": _phase_records(admitted=admitted, bic_delta=bic_delta),
        }
        ctx = CapabilityExecutionContext(
            episode_id=episode_id,
            workflow_step="rewrite",
            episode_context=episode_context,
            recorder=recorder,
        )
        task.execute_capability(
            CapabilityInvocation("submit_rewrite", {"prompt": prompt, "response": response}),
            [],
            variation,
            ctx,
        )
        return episode_context

    def test_records_synthetic_inference_when_recorder_present(
        self, tmp_path: Path
    ) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        recorder = _make_recorder(tmp_path)

        self._drive_submit(task, variation, recorder)

        records = recorder.inference_records
        synth = [r for r in records if r.meta.get("synthesized") is True]
        assert len(synth) == 1, "exactly one synthetic rewrite_output record"

    def test_synthetic_record_carries_training_pair(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        recorder = _make_recorder(tmp_path)
        prompt = "the agent-crafted prompt"
        response = "the agent-crafted response"

        self._drive_submit(
            task, variation, recorder, prompt=prompt, response=response, bic_delta=-1.2345
        )

        synth = [r for r in recorder.inference_records if r.meta.get("synthesized")][0]
        # Prompt = training_pair.prompt
        assert any(prompt in m.get("content", "") for m in synth.messages)
        # Response = training_pair.response + BIC verdict appended
        assert response in synth.response
        assert "-1.2345" in synth.response
        assert "Admitted" in synth.response

    def test_synthetic_record_meta_marks_workflow_step_rewrite(
        self, tmp_path: Path
    ) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        recorder = _make_recorder(tmp_path)

        self._drive_submit(task, variation, recorder)

        synth = [r for r in recorder.inference_records if r.meta.get("synthesized")][0]
        assert synth.meta.get("workflow_step") == "rewrite"
        assert synth.meta.get("actor_role") == "rewriter_output"

    def test_no_recorder_keeps_backward_compat(self, tmp_path: Path) -> None:
        # No recorder passed → existing behaviour: training_pairs.pkl + kg
        # still written, no crash, no synthetic record.
        task = _task(tmp_path)
        variation = _variation(tmp_path)

        ctx = self._drive_submit(task, variation, recorder=None)

        # terminal_record still set as before
        assert "terminal_record" in ctx
        assert "training_pair" in ctx["terminal_record"]

    def test_not_admitted_still_records_synthetic(self, tmp_path: Path) -> None:
        # Even admitted=False experiments are useful SFT material — the
        # rewriter's reasoning is the training signal, BIC is just the outcome
        # label.
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        recorder = _make_recorder(tmp_path)

        self._drive_submit(
            task, variation, recorder, admitted=False, bic_delta=0.5
        )

        synth = [r for r in recorder.inference_records if r.meta.get("synthesized")]
        assert len(synth) == 1
        assert "Not admitted" in synth[0].response

    def test_kazakhstan_variant_also_synthesizes(self, tmp_path: Path) -> None:
        task = _kz_task(tmp_path)
        variation = _kz_variation(tmp_path)
        recorder = _make_recorder(tmp_path)

        self._drive_submit(task, variation, recorder, prompt="kz prompt", response="kz response")

        synth = [r for r in recorder.inference_records if r.meta.get("synthesized")]
        assert len(synth) == 1
        assert "kz prompt" in synth[0].messages[0]["content"]
        assert "kz response" in synth[0].response
