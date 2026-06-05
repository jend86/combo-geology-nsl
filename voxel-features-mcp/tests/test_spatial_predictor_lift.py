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
