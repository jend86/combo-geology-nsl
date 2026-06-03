"""Tests for the kg-admission gate on FeatureHypothesisKazakhstanTask.

Regression spec: a mid-run audit (gen 0, run 20260524-rg26xw, ~T+1h after
the duplicate-handling cycle) found 30+ successful episodes with
``bic_delta`` between -40 000 and -76 000 yet ``experiments.jsonl`` frozen
at 3 rows. Root cause: ``voxel-features-mcp/voxel_features/scoring.py``
returns ``stage_completed="mae_bic_completed"`` on every successful
stage-2 path, but the kg gate was hardcoded to
``stage_completed == "stage_2_completed"``. The gate must accept both
strings so post-rewrite scoring keeps admitting.
"""

from __future__ import annotations

import pytest

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


_GATE = FeatureHypothesisKazakhstanTask._should_persist_to_kg


class TestStageCompletedAllowlist:
    def test_legacy_stage_2_completed_admits(self) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-1.0,
            stage_completed="stage_2_completed",
        )

    def test_current_mae_bic_completed_admits(self) -> None:
        # The post-rewrite scoring path: this assertion is the regression
        # pin — if it ever fails again, admissions silently stop.
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-76738.5,
            stage_completed="mae_bic_completed",
        )

    def test_first_layer_auto_admits_without_bic_delta(self) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=None,
            stage_completed="first_layer_auto",
            admission_path="first_layer_auto",
        )

    @pytest.mark.parametrize(
        "stage",
        ["", "stage_1_only", "aborted", "stage_2_partial", "unknown"],
    )
    def test_unrecognised_stage_blocks(self, stage: str) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-100.0,
            stage_completed=stage,
        )


class TestOtherGateConditions:
    def test_masking_test_failure_blocks(self) -> None:
        assert not _GATE(
            masking_test_passed=False,
            admitted=True,
            bic_delta=-100.0,
            stage_completed="mae_bic_completed",
        )

    def test_not_admitted_blocks(self) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=False,
            bic_delta=-100.0,
            stage_completed="mae_bic_completed",
        )

    def test_none_bic_blocks(self) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=None,
            stage_completed="mae_bic_completed",
        )

    @pytest.mark.parametrize("bic", [0.0, 0.001, 100.0])
    def test_non_negative_bic_blocks(self, bic: float) -> None:
        # bic_delta must be strictly negative (lower BIC = better).
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=bic,
            stage_completed="mae_bic_completed",
        )


class TestEvidenceParentage:
    def test_weak_raises_threshold_but_strong_does_not_lower_it(self) -> None:
        weak = {"proposal_evidence_tier": "weak"}
        mixed = {"proposal_evidence_tier": "mixed"}
        strong = {"proposal_evidence_tier": "strong"}

        assert FeatureHypothesisKazakhstanTask._parentage_threshold_for(weak) > (
            FeatureHypothesisKazakhstanTask._parentage_threshold_for(mixed)
        )
        assert FeatureHypothesisKazakhstanTask._parentage_threshold_for(strong) == pytest.approx(
            FeatureHypothesisKazakhstanTask._PARENTAGE_BASE_THRESHOLD
        )

    def test_parentage_uses_system_strength_not_agent_strength_claim(self) -> None:
        thin_strong_claim = {
            "proposal_evidence_tier": "strong",
            "evidence_strength": 0.1,
            "novelty_guard_passed": True,
            "provenance_guard_passed": True,
            "declared_nothing": False,
        }
        grounded_weak_claim = {
            "proposal_evidence_tier": "weak",
            "evidence_strength": 0.95,
            "novelty_guard_passed": True,
            "provenance_guard_passed": True,
            "declared_nothing": False,
        }

        assert FeatureHypothesisKazakhstanTask._is_crossbreed_parent_eligible(thin_strong_claim) is False
        assert FeatureHypothesisKazakhstanTask._is_crossbreed_parent_eligible(grounded_weak_claim) is True
