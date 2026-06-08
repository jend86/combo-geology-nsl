"""Plateau-triggered survey interweaving for Kazakhstan feature hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

from src.task.types import EpisodeArtifacts
from tasks.feature_hypothesis_kazakhstan import (
    _KAZAKHSTAN_SOURCE_FILES,
    FeatureHypothesisKazakhstanState,
    FeatureHypothesisKazakhstanTask,
    FeatureHypothesisKazakhstanVariation,
)


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {"store_dir": str(tmp_path / "store_root"), "kg_dir": str(tmp_path / "kg_root")}
    )


def _variation(tmp_path: Path) -> FeatureHypothesisKazakhstanVariation:
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=str(tmp_path / "kg" / "teniz_basin"),
        bootstrap_permit_timeout_s=0.1,
    )


def _seed_crossbreed_ready(variation: FeatureHypothesisKazakhstanVariation) -> None:
    kg = Path(variation.kg_dir)
    kg.mkdir(parents=True, exist_ok=True)
    with (kg / "experiments.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(
                json.dumps(
                    {
                        "node_id": f"n{i}",
                        "hypothesis": f"seed hypothesis {i}",
                        "layer_name": f"layer_{i}",
                        "bic_delta": -(i + 1.0),
                        "admission_path": "normal",
                        "crossbreed_parent_eligible": True,
                    }
                )
                + "\n"
            )
    (kg / "file_rotation_state.json").write_text(
        json.dumps({"counts": {s["key"]: 2 for s in _KAZAKHSTAN_SOURCE_FILES}}),
        encoding="utf-8",
    )
    (kg / "greedy_init_complete.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )


def _write_interweave_state(kg: Path, failures: int) -> None:
    kg.mkdir(parents=True, exist_ok=True)
    (kg / "interweave_state.json").write_text(
        json.dumps({"consecutive_failed_crossbreed": failures}),
        encoding="utf-8",
    )


def _finalize_crossbreed(
    task: FeatureHypothesisKazakhstanTask,
    variation: FeatureHypothesisKazakhstanVariation,
    *,
    episode_id: str,
    admitted: bool,
    bic_delta: float,
    duplicate_rejected: bool = False,
) -> None:
    episode_context: dict = {
        "episode_id": episode_id,
        "workflow_kind": "crossbreed",
        "kg_dir": variation.kg_dir,
        "store_dir": variation.store_dir,
        "phase_records": {
            "evaluate": {
                "admitted": admitted,
                "bic_delta": bic_delta,
                "masking_test_passed": True,
                "masking_test_improvement": 0.001,
                "masking_test_direction": "improved",
                "stage_completed": "mae_bic_completed",
            }
        },
    }
    if duplicate_rejected:
        episode_context["duplicate_rejected"] = True

    initial = FeatureHypothesisKazakhstanState(
        episode_id=episode_id,
        workflow_kind="crossbreed",
    )
    task.finalize_episode([], initial, episode_context, EpisodeArtifacts())


def _finalize_interweave_survey(
    task: FeatureHypothesisKazakhstanTask,
    variation: FeatureHypothesisKazakhstanVariation,
    *,
    episode_id: str,
    admitted: bool,
):
    episode_context: dict = {
        "episode_id": episode_id,
        "workflow_kind": "survey",
        "interweave_bootstrap": True,
        "kg_dir": variation.kg_dir,
        "store_dir": variation.store_dir,
        "phase_records": {
            "evaluate": {
                "admitted": admitted,
                "bic_delta": -1.5 if admitted else 0.5,
                "masking_test_passed": True,
                "masking_test_improvement": 0.001,
                "masking_test_direction": "mae_delta",
                "stage_completed": "mae_bic_completed",
            }
        },
    }
    initial = FeatureHypothesisKazakhstanState(
        episode_id=episode_id,
        workflow_kind="survey",
    )
    return task.finalize_episode([], initial, episode_context, EpisodeArtifacts())


def test_below_plateau_threshold_keeps_crossbreed(tmp_path: Path) -> None:
    variation = _variation(tmp_path)
    _seed_crossbreed_ready(variation)
    assert variation.interweave_failed_episode_threshold == 30
    assert variation.interweave_survey_burst_episodes == 15
    _write_interweave_state(Path(variation.kg_dir), failures=29)

    outcome = _task(tmp_path).populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "crossbreed"
    assert outcome.episode_context.get("interweave_bootstrap") is not True


def test_plateau_threshold_enters_survey_burst(tmp_path: Path) -> None:
    variation = _variation(tmp_path)
    _seed_crossbreed_ready(variation)
    kg = Path(variation.kg_dir)
    _write_interweave_state(kg, failures=30)

    first = _task(tmp_path).populate([], variation)
    state_after_claim = json.loads((kg / "interweave_state.json").read_text())
    second = _task(tmp_path).populate([], variation)
    state_after_second = json.loads((kg / "interweave_state.json").read_text())

    assert first.episode_context["workflow_kind"] == "survey"
    assert first.episode_context["interweave_bootstrap"] is True
    assert first.episode_context["interweave_reason"] == "crossbreed_plateau"
    assert state_after_claim["consecutive_failed_crossbreed"] == 0
    assert state_after_claim["interweave_survey_remaining"] == 14
    assert second.episode_context["workflow_kind"] == "survey"
    assert second.episode_context.get("interweave_bootstrap") is True
    assert state_after_second["interweave_survey_remaining"] == 13


def test_failed_interweave_survey_continues_burst(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    _seed_crossbreed_ready(variation)
    kg = Path(variation.kg_dir)
    _write_interweave_state(kg, failures=30)

    first = task.populate([], variation)
    reward = _finalize_interweave_survey(
        task,
        variation,
        episode_id=first.episode_context["episode_id"],
        admitted=False,
    )
    second = task.populate([], variation)
    state_after_second = json.loads((kg / "interweave_state.json").read_text())

    assert reward.breakdown["bootstrap_active"] is False
    assert reward.breakdown["interweave_bootstrap"] is True
    assert second.episode_context["workflow_kind"] == "survey"
    assert second.episode_context.get("interweave_bootstrap") is True
    assert state_after_second["interweave_survey_remaining"] == 13


def test_successful_interweave_survey_ends_burst(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    _seed_crossbreed_ready(variation)
    kg = Path(variation.kg_dir)
    _write_interweave_state(kg, failures=30)

    first = task.populate([], variation)
    _finalize_interweave_survey(
        task,
        variation,
        episode_id=first.episode_context["episode_id"],
        admitted=True,
    )
    state_after_success = json.loads((kg / "interweave_state.json").read_text())
    second = task.populate([], variation)

    assert state_after_success["interweave_survey_remaining"] == 0
    assert second.episode_context["workflow_kind"] == "crossbreed"
    assert second.episode_context.get("interweave_bootstrap") is not True


def test_crossbreed_failure_streak_updates_on_finalize(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)

    _finalize_crossbreed(
        task,
        variation,
        episode_id="ep_fail_1",
        admitted=False,
        bic_delta=0.5,
    )
    state_after_failure = json.loads((kg / "interweave_state.json").read_text())

    _finalize_crossbreed(
        task,
        variation,
        episode_id="ep_success_1",
        admitted=True,
        bic_delta=-1.5,
    )
    state_after_success = json.loads((kg / "interweave_state.json").read_text())

    assert state_after_failure["consecutive_failed_crossbreed"] == 1
    assert state_after_success["consecutive_failed_crossbreed"] == 0


def test_crossbreed_lift_success_without_kg_admission_counts_toward_plateau(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)

    _finalize_crossbreed(
        task,
        variation,
        episode_id="ep_lift_only_1",
        admitted=True,
        bic_delta=1.5,
    )

    state = json.loads((kg / "interweave_state.json").read_text())
    assert state["consecutive_failed_crossbreed"] == 1


def test_duplicate_crossbreed_admit_counts_toward_plateau(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)

    _finalize_crossbreed(
        task,
        variation,
        episode_id="ep_dup_1",
        admitted=True,
        bic_delta=-1.5,
        duplicate_rejected=True,
    )

    state = json.loads((kg / "interweave_state.json").read_text())
    assert state["consecutive_failed_crossbreed"] == 1
