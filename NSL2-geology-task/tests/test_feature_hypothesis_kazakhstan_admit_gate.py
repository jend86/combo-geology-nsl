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

    def test_float_near_duplicate_is_not_parent_eligible(self) -> None:
        # Behavior change: float layers were previously exempt from the
        # near-duplicate gate (raw MAE had no normalised threshold). With the
        # normalized pairwise distance, a float layer practically duplicating a
        # pool member (min distance < 0.15) is barred from parentage even with
        # otherwise strong evidence.
        float_near_dup = {
            "layer_dtype": "float",
            "min_pairwise_distance_to_pool": 0.05,
            "proposal_evidence_tier": "mixed",
            "evidence_strength": 0.95,
            "novelty_guard_passed": True,
            "provenance_guard_passed": True,
            "declared_nothing": False,
        }
        assert FeatureHypothesisKazakhstanTask._is_crossbreed_parent_eligible(float_near_dup) is False

    def test_float_distinct_layer_is_parent_eligible(self) -> None:
        float_distinct = {
            "layer_dtype": "float",
            "min_pairwise_distance_to_pool": 0.5,
            "proposal_evidence_tier": "mixed",
            "evidence_strength": 0.95,
            "novelty_guard_passed": True,
            "provenance_guard_passed": True,
            "declared_nothing": False,
        }
        assert FeatureHypothesisKazakhstanTask._is_crossbreed_parent_eligible(float_distinct) is True

    def test_boolean_near_duplicate_remains_rejected(self) -> None:
        boolean_near_dup = {
            "layer_dtype": "boolean",
            "min_pairwise_distance_to_pool": 0.05,
            "proposal_evidence_tier": "mixed",
            "evidence_strength": 0.95,
            "novelty_guard_passed": True,
            "provenance_guard_passed": True,
            "declared_nothing": False,
        }
        assert FeatureHypothesisKazakhstanTask._is_crossbreed_parent_eligible(boolean_near_dup) is False


class TestFirstRootHardening:
    """The first free (first_layer_auto) admit anchors the whole KG and rides a
    BIC bypass, so it is held to a stricter bar than later layers:

    1. It can NEVER ride the creative_fallback override — an all-fallback first
       root is rejected regardless of allow_creative_fallback_admission.
    2. Triviality (single-op / uniform-value / low-entropy) is a HARD reject for
       the first root, whereas for later layers these stay telemetry-only.

    ``_first_root_admission_ok`` is a no-op for non-first-root records.
    """

    def _healthy_first_root(self) -> dict:
        # Multi-op, graded, artifact-backed first root — should pass.
        return {
            "admission_path": "first_layer_auto",
            "spatial_operation_provenance_count": 4,
            "coordinate_source_counts": {"artifact": 3, "creative_fallback": 1},
            "single_spatial_operation": False,
            "uniform_nonzero_value": False,
            "candidate_value_entropy": 2.3,
        }

    def test_non_first_root_is_unconstrained(self) -> None:
        # A "normal" admit that is single-op, uniform, all-fallback must NOT be
        # touched by this gate — later-layer triviality stays telemetry-only.
        trivial_normal = {
            "admission_path": "normal",
            "spatial_operation_provenance_count": 1,
            "coordinate_source_counts": {"creative_fallback": 1},
            "single_spatial_operation": True,
            "uniform_nonzero_value": True,
            "candidate_value_entropy": 0.0,
        }
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(trivial_normal) is True

    def test_healthy_first_root_admits(self) -> None:
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(self._healthy_first_root()) is True

    def test_all_creative_fallback_first_root_rejected_despite_override(self) -> None:
        record = self._healthy_first_root()
        record["spatial_operation_provenance_count"] = 3
        record["coordinate_source_counts"] = {"creative_fallback": 3}
        # Even with the override flag set on the record, the first root must not
        # ride it.
        record["allow_creative_fallback_admission"] = True
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(record) is False
        assert record["first_root_rejection_reason"] == "all_creative_fallback"

    def test_single_spatial_operation_first_root_rejected(self) -> None:
        record = self._healthy_first_root()
        record["single_spatial_operation"] = True
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(record) is False

    def test_uniform_value_first_root_rejected(self) -> None:
        record = self._healthy_first_root()
        record["uniform_nonzero_value"] = True
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(record) is False

    def test_low_entropy_first_root_rejected(self) -> None:
        record = self._healthy_first_root()
        record["candidate_value_entropy"] = 0.1
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(record) is False
        assert record["first_root_rejection_reason"] == "low_value_entropy"

    def test_partial_fallback_first_root_not_flagged_as_all_fallback(self) -> None:
        # 1 of 4 ops is fallback → not all_creative_fallback → passes (other
        # triviality fields are healthy).
        assert FeatureHypothesisKazakhstanTask._first_root_admission_ok(self._healthy_first_root()) is True
