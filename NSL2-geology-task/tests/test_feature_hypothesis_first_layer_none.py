from __future__ import annotations

from tasks.feature_hypothesis import FeatureHypothesisState, FeatureHypothesisTask


def test_first_layer_auto_reward_uses_none_bic_delta() -> None:
    task = FeatureHypothesisTask.__new__(FeatureHypothesisTask)
    final = FeatureHypothesisState(
        bic_delta=None,
        admitted=True,
        masking_test_passed=True,
        masking_test_improvement=0.0,
        masking_test_direction="first_layer_auto",
        stage_completed="first_layer_auto",
        admission_path="first_layer_auto",
    )

    reward = task.compute_reward(FeatureHypothesisState(), final, None)

    assert reward.success is True
    assert reward.value == 1.0
    assert reward.breakdown["first_layer_auto"] is True
    assert reward.breakdown["bic_delta"] is None


def test_base_task_kg_gate_allows_first_layer_auto_without_bic_delta() -> None:
    assert FeatureHypothesisTask._should_persist_to_kg(
        masking_test_passed=True,
        admitted=True,
        bic_delta=None,
        stage_completed="first_layer_auto",
        admission_path="first_layer_auto",
    )


def test_base_task_kg_gate_allows_current_mae_bic_stage() -> None:
    assert FeatureHypothesisTask._should_persist_to_kg(
        masking_test_passed=True,
        admitted=True,
        bic_delta=-1.0,
        stage_completed="mae_bic_completed",
    )


def test_base_task_kg_gate_blocks_none_bic_for_normal_path() -> None:
    assert not FeatureHypothesisTask._should_persist_to_kg(
        masking_test_passed=True,
        admitted=True,
        bic_delta=None,
        stage_completed="mae_bic_completed",
    )
