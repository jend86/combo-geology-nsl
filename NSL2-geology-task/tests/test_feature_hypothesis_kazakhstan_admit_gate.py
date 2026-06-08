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

import numpy as np
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
            workflow_kind="crossbreed",
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


class TestSeedPhasePersistBypass:
    """Survey seeding is now explicit, not a blanket completed-scoring bypass."""

    def test_seed_phase_diverse_seed_bypasses_positive_bic(self) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
            admission_path="diverse_seed",
            seed_phase=True,
        )

    def test_seed_phase_no_longer_bypasses_stage1_failure(self) -> None:
        assert not _GATE(
            masking_test_passed=False,
            admitted=True,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
            admission_path="diverse_seed",
            seed_phase=True,
        )

    def test_seed_phase_still_requires_completed_scoring(self) -> None:
        for stage in ("", "aborted", "stage_2_partial", "stage_1_only"):
            assert not _GATE(
                masking_test_passed=True,
                admitted=True,
                bic_delta=3.38,
                stage_completed=stage,
                admission_path="diverse_seed",
                seed_phase=True,
            )

    def test_seed_phase_admits_natural_pass_too(self) -> None:
        # A survey candidate that *also* clears BIC naturally still admits.
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-50.0,
            stage_completed="mae_bic_completed",
            seed_phase=True,
        )

    def test_diverse_seed_persists_regardless_of_seed_phase(self) -> None:
        # The scorer emits diverse_seed only in the seed window, so the label (not
        # the racy seed_phase flag) governs: a validity-admitted +bic founder
        # persists even when the episode's seed_phase raced to False. (Survey
        # admits are parents regardless of score; the novelty guard blocks dups.)
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
            admission_path="diverse_seed",
            seed_phase=False,
        )

    def test_survey_positive_bic_success_persists_on_lift(self) -> None:
        # Survey is success/training-data oriented. Crossbreed admission adds the
        # raw-BIC criterion separately below.
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=0.264,
            stage_completed="mae_bic_completed",
            admission_path="normal",
            seed_phase=False,
            workflow_kind="survey",
        )

    def test_crossbreed_positive_bic_success_does_not_persist_to_kg(self) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=0.264,
            stage_completed="mae_bic_completed",
            admission_path="normal",
            seed_phase=False,
            workflow_kind="crossbreed",
        )

    def test_crossbreed_negative_bic_success_persists_to_kg(self) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-0.264,
            stage_completed="mae_bic_completed",
            admission_path="normal",
            seed_phase=False,
            workflow_kind="crossbreed",
        )

    def test_seed_phase_default_is_false(self) -> None:
        # Callers that don't pass seed_phase get the strict gate by default.
        assert not _GATE(
            masking_test_passed=True,
            admitted=False,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
        )


class TestSeedPhaseDetection:
    """``_in_seed_phase`` maps the episode's ``workflow_kind`` to the bypass.
    Survey (and a missing kind, matching the rest of the task's default) seeds;
    crossbreed does not.
    """

    def test_survey_is_seed_phase(self) -> None:
        assert FeatureHypothesisKazakhstanTask._in_seed_phase(
            {"workflow_kind": "survey"}
        ) is True

    def test_crossbreed_is_not_seed_phase(self) -> None:
        assert FeatureHypothesisKazakhstanTask._in_seed_phase(
            {"workflow_kind": "crossbreed"}
        ) is False

    def test_missing_workflow_kind_defaults_to_seed(self) -> None:
        # The task defaults workflow_kind to "survey" everywhere it is read;
        # the geometry/provenance floor still guards, so this is safe.
        assert FeatureHypothesisKazakhstanTask._in_seed_phase({}) is True


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

    @pytest.mark.parametrize("bic", [-50.0, 0.0, 0.001, 100.0])
    def test_bic_sign_does_not_gate_survey_success(self, bic: float) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=bic,
            stage_completed="mae_bic_completed",
            workflow_kind="survey",
        )

    @pytest.mark.parametrize("bic, expected", [(-50.0, True), (0.0, False), (0.001, False), (100.0, False)])
    def test_bic_sign_gates_crossbreed_admission(self, bic: float, expected: bool) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=bic,
            stage_completed="mae_bic_completed",
            workflow_kind="crossbreed",
        ) is expected


class TestQuarantineMirrorsAdmissionGate:
    def test_crossbreed_lift_success_positive_bic_is_quarantined_at_kg_gate(self) -> None:
        evaluate = {
            "masking_test_passed": True,
            "admitted": True,
            "bic_delta": 4.2,
            "stage_completed": "mae_bic_completed",
            "workflow_kind": "crossbreed",
        }

        assert FeatureHypothesisKazakhstanTask._should_quarantine_rejected_candidate(evaluate)
        assert FeatureHypothesisKazakhstanTask._rejection_stage(evaluate) == "kg_gate"


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


class TestSeedPhaseHardening:
    """Every SURVEY-phase admit seeds the KG (and rides the scorer bypass), so
    each is held to a GEOMETRY/PROVENANCE floor a single arbitrary central blob
    cannot clear (docs/design/scoring-colocation-monoculture-2026-06-03):

    1. An all-creative_fallback seed is rejected regardless of
       allow_creative_fallback_admission (the seed must rest on real provenance).
    2. A single-op seed (the single central blob that drives the co-location
       monoculture) is rejected.
    3. Value uniformity / low entropy are NOT gated — a *distributed* uniform
       layer is a perfectly good seed (the agent reliably places distributed
       coords but rarely grades values; an entropy floor deadlocked the run).
    4. In crossbreed (seed_phase=False) the floor is a no-op — the scorer
       governs and single-op/uniform are telemetry only.
    """

    def _healthy_seed(self) -> dict:
        # Multi-op, artifact-backed root — should pass even if uniform-valued.
        return {
            "spatial_operation_provenance_count": 4,
            "coordinate_source_counts": {"artifact": 3, "creative_fallback": 1},
            "single_spatial_operation": False,
        }

    def test_crossbreed_phase_is_unconstrained(self) -> None:
        # Outside survey, even a single-op all-fallback layer is no-op'd.
        trivial = {
            "spatial_operation_provenance_count": 1,
            "coordinate_source_counts": {"creative_fallback": 1},
            "single_spatial_operation": True,
        }
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            trivial, seed_phase=False
        ) is True
        assert trivial["first_root_rejection_reason"] == "none"

    def test_healthy_seed_admits(self) -> None:
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            self._healthy_seed(), seed_phase=True
        ) is True

    def test_distributed_uniform_seed_admits(self) -> None:
        # A multi-op uniform-valued, zero-entropy layer is a fine seed — it must
        # NOT be rejected (this is what deadlocked the prior run).
        record = self._healthy_seed()
        record["uniform_nonzero_value"] = True
        record["candidate_value_entropy"] = 0.0
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            record, seed_phase=True
        ) is True
        assert record["first_root_rejection_reason"] == "none"

    def test_all_creative_fallback_seed_rejected_despite_override(self) -> None:
        record = self._healthy_seed()
        record["spatial_operation_provenance_count"] = 3
        record["coordinate_source_counts"] = {"creative_fallback": 3}
        record["allow_creative_fallback_admission"] = True
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            record, seed_phase=True
        ) is False
        assert record["first_root_rejection_reason"] == "all_creative_fallback"

    def test_single_spatial_operation_seed_rejected(self) -> None:
        record = self._healthy_seed()
        record["single_spatial_operation"] = True
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            record, seed_phase=True
        ) is False
        assert record["first_root_rejection_reason"] == "single_spatial_operation"

    def test_single_array_operation_seed_is_not_single_geometry_blob(self) -> None:
        record = {
            "spatial_operation_provenance_count": 1,
            "coordinate_source_counts": {"artifact": 1},
            "geometry_kind_counts": {"array": 1},
        }
        values = np.zeros((4, 4, 2), dtype=float)
        values[:, :, :] = 0.5

        FeatureHypothesisKazakhstanTask._stamp_candidate_triviality(record, values=values)

        assert record["single_spatial_operation"] is False
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            record, seed_phase=True
        ) is True
        assert record["first_root_rejection_reason"] == "none"

    def test_partial_fallback_seed_passes(self) -> None:
        # 1 of 4 ops is fallback → not all_creative_fallback → passes.
        assert FeatureHypothesisKazakhstanTask._seed_phase_admission_ok(
            self._healthy_seed(), seed_phase=True
        ) is True
