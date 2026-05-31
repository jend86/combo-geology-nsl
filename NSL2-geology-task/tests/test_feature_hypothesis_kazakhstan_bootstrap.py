from __future__ import annotations

from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import (
    FeatureHypothesisKazakhstanTask,
    FeatureHypothesisKazakhstanVariation,
)


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {"store_dir": str(tmp_path / "store"), "kg_dir": str(tmp_path / "kg")}
    )


def _variation(tmp_path: Path) -> FeatureHypothesisKazakhstanVariation:
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=str(tmp_path / "kg" / "teniz_basin"),
    )


def test_default_bootstrap_uses_framework_parallelism_without_task_cap(
    tmp_path: Path,
) -> None:
    variation = _variation(tmp_path)

    outcome = _task(tmp_path).populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "survey"
    assert "bootstrap_permit_slot_id" not in outcome.episode_context
    assert not (Path(variation.kg_dir) / "bootstrap_state.json").exists()


def test_bootstrap_target_uses_all_slots_from_first_episode(tmp_path: Path) -> None:
    task = _task(tmp_path)

    assert task._bootstrap_target_active(
        bootstrap_episodes_seen=0,
        configured_slots=4,
        window_size=8,
        min_fraction=0.5,
    ) == 4


def test_bootstrap_permit_allows_full_cap_immediately(tmp_path: Path) -> None:
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg" / "teniz_basin"

    for index in range(4):
        assert task._acquire_bootstrap_permit(
            kg_dir,
            f"slot-{index}",
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
            timeout_s=0.0,
        ) is True
