from __future__ import annotations

import numpy as np

from voxel_features import scoring


def _blob(shape: tuple[int, int, int], center: tuple[int, int], radius: int = 2) -> np.ndarray:
    field = np.zeros(shape, dtype=np.float32)
    cx, cy = center
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = cx + dx
            y = cy + dy
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                field[x, y, :] = 1.0
    return field


def test_offset_candidate_lifts_existing_target_without_colocation() -> None:
    shape = (40, 40, 3)
    target = _blob(shape, (16, 20), radius=2)
    candidate = _blob(shape, (24, 20), radius=2)

    result = scoring.spatial_predictor_lift_score(
        [target.ravel()],
        ["target"],
        candidate.ravel(),
        shape,
        ridge_alpha=1e-2,
        null_permutations=0,
    )

    assert result["scoring_objective"] == "spatial_predictor_lift_v1"
    assert result["validity_passed"] is True
    assert result["candidate_predictor_lift_by_target"]["target"] > 0.0
    assert result["bic_delta"] < 0.0
    assert result["admitted"] is True


def test_clone_into_identical_pool_has_no_predictor_lift() -> None:
    shape = (40, 40, 3)
    target = _blob(shape, (20, 20), radius=3)
    clone = target.copy()

    result = scoring.spatial_predictor_lift_score(
        [target.ravel(), clone.ravel()],
        ["target", "clone"],
        clone.ravel(),
        shape,
        ridge_alpha=1e-2,
        null_permutations=0,
    )

    assert result["validity_passed"] is True
    assert result["candidate_predictor_lift_mean"] <= 1e-8
    assert result["admitted"] is False


def test_blanket_candidate_fails_self_validity_gate() -> None:
    shape = (30, 30, 2)
    target = _blob(shape, (15, 15), radius=2)
    blanket = np.ones(shape, dtype=np.float32)

    result = scoring.spatial_predictor_lift_score(
        [target.ravel()],
        ["target"],
        blanket.ravel(),
        shape,
        ridge_alpha=1e-2,
        null_permutations=0,
    )

    assert result["validity_passed"] is False
    assert result["masking_test_passed"] is False
    assert result["admitted"] is False
    assert result["masking_test_direction"] == "self_validity_gate"


def test_single_column_pillar_fails_self_validity_gate() -> None:
    """A candidate whose signal lives in a single (x,y) column self-predicts
    near-perfectly through its own vertical stack (Rv leak), so it would pass the
    self-validity gate despite having NO horizontal spatial structure — a
    geologically trivial borehole-like pillar. Calibration 2026-06-05 adds a
    minimum horizontal-support gate; assert it is rejected as invalid.
    """
    shape = (40, 40, 8)
    target = _blob(shape, (20, 20), radius=3)
    pillar = np.zeros(shape, dtype=np.float32)
    pillar[10, 10, :] = 1.0  # exactly one (x,y) column, all depths

    result = scoring.spatial_predictor_lift_score(
        [target.ravel()],
        ["target"],
        pillar.ravel(),
        shape,
        ridge_alpha=1e-2,
        null_permutations=0,
    )

    assert result["validity_passed"] is False
    assert result["admitted"] is False
    assert result["masking_test_direction"] == "self_validity_gate"


def test_two_pillars_fail_but_small_distributed_blob_passes_validity() -> None:
    """Boundary check on the min-support gate: two isolated columns (2 columns)
    are still rejected, but a compact 5-voxel-wide blob (well above the floor)
    remains a valid candidate.
    """
    shape = (40, 40, 8)
    two_pillars = np.zeros(shape, dtype=np.float32)
    two_pillars[8, 8, :] = 1.0
    two_pillars[30, 30, :] = 1.0
    assert scoring.self_validity_score(two_pillars.ravel(), shape) >= 1.0

    blob = _blob(shape, (20, 20), radius=2)  # ~13 columns
    assert scoring.self_validity_score(blob.ravel(), shape) < scoring._SPATIAL_TAU_SELF


# --- 2026-06-06: permutation-null-calibrated lift bar (Approach A) ----------------
# When the null is active, the admission bar is the (100 - null_percentile)th
# percentile of the LIFT a feature-scrambled candidate yields — a self-calibrating
# per-pool replacement for the hand-set _SPATIAL_ADMIT_MIN_LIFT (0.005).


def test_perm_null_emits_lift_distribution_and_nonneg_bar() -> None:
    """With null_permutations>0 the scorer emits a null LIFT distribution and the
    admission_threshold becomes that distribution's upper percentile (a NON-negative
    lift bar), replacing the hand-set 0.005. Pre-change: no null_lifts, and
    admission_threshold was the bic-percentile (<= 0)."""
    shape = (40, 40, 3)
    target = _blob(shape, (16, 20), radius=2)
    candidate = _blob(shape, (24, 20), radius=2)
    result = scoring.spatial_predictor_lift_score(
        [target.ravel()], ["target"], candidate.ravel(), shape,
        ridge_alpha=1e-2, null_permutations=40,
    )
    assert result["null_calibrated"] is True
    assert isinstance(result["permutation_null_lifts"], list)
    assert len(result["permutation_null_lifts"]) >= 1
    # the null only TIGHTENS: effective bar >= the hand-set floor, never below it
    assert result["admission_threshold"] >= scoring._SPATIAL_ADMIT_MIN_LIFT
    # the effective bar the decision used IS the null-calibrated threshold
    assert result["admit_min_lift"] == result["admission_threshold"]


def test_perm_null_admits_strong_offset_candidate() -> None:
    """A genuinely cross-predictive (offset, non-co-located) candidate beats the
    null's upper tail and is still admitted with the null active."""
    shape = (40, 40, 3)
    target = _blob(shape, (16, 20), radius=2)
    candidate = _blob(shape, (24, 20), radius=2)
    result = scoring.spatial_predictor_lift_score(
        [target.ravel()], ["target"], candidate.ravel(), shape,
        ridge_alpha=1e-2, null_permutations=40,
    )
    assert result["validity_passed"] is True
    assert result["candidate_predictor_lift_mean"] > result["admission_threshold"]
    assert result["admitted"] is True


def test_perm_null_rejects_clone_that_cannot_beat_its_own_null() -> None:
    """A clone of an existing layer has ~zero real lift; its real lift cannot exceed
    the upper tail of its own null distribution, so the null rejects it."""
    shape = (40, 40, 3)
    target = _blob(shape, (20, 20), radius=3)
    clone = target.copy()
    result = scoring.spatial_predictor_lift_score(
        [target.ravel(), clone.ravel()], ["target", "clone"], clone.ravel(), shape,
        ridge_alpha=1e-2, null_permutations=40,
    )
    assert result["admitted"] is False
    assert result["candidate_predictor_lift_mean"] <= result["admission_threshold"]


def test_perm_null_off_preserves_handset_bar() -> None:
    """null_permutations=0 → fall back to the hand-set _SPATIAL_ADMIT_MIN_LIFT;
    decision + telemetry unchanged from prior behaviour (regression guard)."""
    shape = (40, 40, 3)
    target = _blob(shape, (16, 20), radius=2)
    candidate = _blob(shape, (24, 20), radius=2)
    result = scoring.spatial_predictor_lift_score(
        [target.ravel()], ["target"], candidate.ravel(), shape,
        ridge_alpha=1e-2, null_permutations=0,
    )
    assert result["null_calibrated"] is False
    assert result["admit_min_lift"] == scoring._SPATIAL_ADMIT_MIN_LIFT
    assert result["admitted"] is True
