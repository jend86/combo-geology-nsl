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


class TestEarlyRootPersistBypass:
    """The first K admits seed the KG and must NOT be gated by the co-location
    scorer: Stage-1 MAE and Stage-2 BIC both reject distributed layers at
    supports distinct from the seed (observed live: every post-seed candidate
    scored relative_mae~=1.0, bic_delta~=+3.38). When ``early_root=True`` the
    persist gate admits on the strength of *completed* scoring alone; the real
    quality gate is the geometry/provenance floor in ``_admit_with_dedup``
    (``_early_root_admission_ok``). See docs/design/scoring-colocation-
    monoculture-2026-06-03.md.
    """

    def test_early_root_bypasses_positive_bic(self) -> None:
        # The exact shape the live run stalled on: scorer rejected
        # (admitted=False, bic_delta>0) but within the first-K window it persists.
        assert _GATE(
            masking_test_passed=True,
            admitted=False,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
            early_root=True,
        )

    def test_early_root_bypasses_stage1_failure(self) -> None:
        # By the 3rd+ layer Stage-1 MAE is live and also rejects distributed
        # layers; the first-K window bypasses Stage-1 too.
        assert _GATE(
            masking_test_passed=False,
            admitted=False,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
            early_root=True,
        )

    def test_early_root_still_requires_completed_scoring(self) -> None:
        # Aborted/partial episodes must never seed the pool even inside the
        # window — the stage_completed allowlist is the one invariant kept.
        for stage in ("", "aborted", "stage_2_partial", "stage_1_only"):
            assert not _GATE(
                masking_test_passed=True,
                admitted=False,
                bic_delta=3.38,
                stage_completed=stage,
                early_root=True,
            )

    def test_early_root_admits_natural_pass_too(self) -> None:
        # A first-K candidate that *also* clears BIC naturally still admits.
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-50.0,
            stage_completed="mae_bic_completed",
            early_root=True,
        )

    def test_early_root_false_restores_strict_gate(self) -> None:
        # Once the pool holds >= K layers the bypass is off: a positive-BIC
        # rejected candidate is blocked exactly as before (regression pin).
        assert not _GATE(
            masking_test_passed=True,
            admitted=False,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
            early_root=False,
        )

    def test_early_root_default_is_false(self) -> None:
        # Callers that don't pass early_root get the strict gate by default.
        assert not _GATE(
            masking_test_passed=True,
            admitted=False,
            bic_delta=3.38,
            stage_completed="mae_bic_completed",
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


class TestEarlyRootHardening:
    """The first K (``_EARLY_ROOT_COUNT``) KG admits seed the KG and the very
    first rides a BIC bypass, so they are held to a GEOMETRY/PROVENANCE floor a
    single arbitrary central blob cannot clear (Approach 1 + B,
    docs/design/scoring-colocation-monoculture-2026-06-03):

    1. An all-creative_fallback early root is rejected regardless of
       allow_creative_fallback_admission (the seed must rest on real provenance).
    2. A single-op early root (the single central blob that drives the
       co-location monoculture) is rejected.
    3. Value uniformity / low entropy are NOT gated — a *distributed* uniform
       layer is a perfectly good seed (this is the Approach-1 relaxation).
    4. Once the pool already holds >= K layers the gate is a no-op.
    """

    def _healthy_early_root(self) -> dict:
        # Multi-op, artifact-backed root — should pass even if uniform-valued.
        return {
            "spatial_operation_provenance_count": 4,
            "coordinate_source_counts": {"artifact": 3, "creative_fallback": 1},
            "single_spatial_operation": False,
        }

    def test_pool_at_or_past_K_is_unconstrained(self) -> None:
        # Once K roots exist, even a single-op all-fallback layer is no-op'd.
        trivial = {
            "spatial_operation_provenance_count": 1,
            "coordinate_source_counts": {"creative_fallback": 1},
            "single_spatial_operation": True,
        }
        K = FeatureHypothesisKazakhstanTask._EARLY_ROOT_COUNT
        assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(trivial, pool_size=K) is True
        assert trivial["first_root_rejection_reason"] == "none"

    def test_healthy_early_root_admits(self) -> None:
        assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(
            self._healthy_early_root(), pool_size=0
        ) is True

    def test_distributed_uniform_early_root_admits(self) -> None:
        # Approach 1: a multi-op uniform-valued, zero-entropy layer is a fine
        # seed — it must NOT be rejected (this is what deadlocked the prior run).
        record = self._healthy_early_root()
        record["uniform_nonzero_value"] = True
        record["candidate_value_entropy"] = 0.0
        assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(record, pool_size=2) is True
        assert record["first_root_rejection_reason"] == "none"

    def test_all_creative_fallback_early_root_rejected_despite_override(self) -> None:
        record = self._healthy_early_root()
        record["spatial_operation_provenance_count"] = 3
        record["coordinate_source_counts"] = {"creative_fallback": 3}
        record["allow_creative_fallback_admission"] = True
        assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(record, pool_size=0) is False
        assert record["first_root_rejection_reason"] == "all_creative_fallback"

    def test_single_spatial_operation_early_root_rejected(self) -> None:
        record = self._healthy_early_root()
        record["single_spatial_operation"] = True
        assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(record, pool_size=3) is False
        assert record["first_root_rejection_reason"] == "single_spatial_operation"

    def test_window_applies_to_each_of_first_K(self) -> None:
        # A single-op layer is gated at every pool_size below K, not just 0.
        K = FeatureHypothesisKazakhstanTask._EARLY_ROOT_COUNT
        for ps in range(K):
            record = {"single_spatial_operation": True, "spatial_operation_provenance_count": 1,
                      "coordinate_source_counts": {"artifact": 1}}
            assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(record, pool_size=ps) is False

    def test_partial_fallback_early_root_passes(self) -> None:
        # 1 of 4 ops is fallback → not all_creative_fallback → passes.
        assert FeatureHypothesisKazakhstanTask._early_root_admission_ok(
            self._healthy_early_root(), pool_size=0
        ) is True
