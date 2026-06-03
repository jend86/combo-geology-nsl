"""Invariant: an episode that lands a layer in the knowledge graph MUST be
counted as a successful episode.

`compute_reward` derives success from the scorer verdict (`final.admitted`),
but the survey-phase bypass admits distributed layers the scorer REJECTED
(admitted=False, positive BIC). Without correction those episodes were recorded
as failures (observed: 7 KG admits but total_successful=2). Successful episodes
must be a SUPERSET of episodes that admit a layer to the graph.

`_enforce_admission_success(reward, episode_context)` applies the override in
`finalize_episode` using the actual KG-admission outcome recorded by
`_exec_submit_rewrite` as `episode_context["layer_admitted_to_kg"]`.
"""

from __future__ import annotations

from pathlib import Path

from src.task.types import TaskReward
from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
            "dataset_dir": str(tmp_path / "data"),
        }
    )


def test_admission_forces_episode_success(tmp_path: Path) -> None:
    task = _task(tmp_path)
    # Scorer rejected it (success=False), but the layer entered the graph.
    base = TaskReward(value=0.3, success=False, breakdown={"stage_2_passed": False})
    out = task._enforce_admission_success(base, {"layer_admitted_to_kg": True})
    assert out.success is True
    assert out.breakdown.get("admitted_to_kg_forced_success") is True
    # value is preserved (success and reward magnitude are separate signals)
    assert out.value == 0.3


def test_no_admission_leaves_failure_untouched(tmp_path: Path) -> None:
    task = _task(tmp_path)
    base = TaskReward(value=0.3, success=False, breakdown={})
    # Episode did not admit a layer (or the key is absent) -> no override.
    assert task._enforce_admission_success(base, {"layer_admitted_to_kg": False}).success is False
    assert task._enforce_admission_success(base, {}).success is False


def test_already_successful_is_idempotent(tmp_path: Path) -> None:
    task = _task(tmp_path)
    base = TaskReward(value=1.0, success=True, breakdown={})
    out = task._enforce_admission_success(base, {"layer_admitted_to_kg": True})
    assert out.success is True
    assert out.value == 1.0
