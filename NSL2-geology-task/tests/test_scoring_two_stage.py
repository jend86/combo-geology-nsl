"""Regression tests for spatial predictor-lift scoring v1.

The live scorer now implements the D13 objective from
``docs/design/spatial-coherence-scoring-unified-2026-06-05.md``: full-field
cross-features, buffered spatial row CV, self-validity as a gate, and
candidate-as-predictor lift over existing targets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import voxel_features.scoring as scoring
from voxel_features.scoring import evaluate_new_layer
from voxel_features.store import GridSpec, VoxelStore


_GRID = GridSpec(
    origin=(0.0, 0.0, 0.0),
    maximum=(0.04, 0.04, 30.0),
    shape=(40, 40, 3),
)


def _store(tmp_path: Path) -> VoxelStore:
    return VoxelStore(tmp_path, _GRID)


def _blob(center: tuple[int, int], radius: int = 2) -> np.ndarray:
    field = np.zeros(_GRID.shape, dtype=np.float32)
    cx, cy = center
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = cx + dx
            y = cy + dy
            if 0 <= x < _GRID.shape[0] and 0 <= y < _GRID.shape[1]:
                field[x, y, :] = 1.0
    return field


def _seed_normal_pool(store: VoxelStore) -> None:
    store.add_layer("northwest", _blob((16, 20)), dtype="float")
    store.add_layer("east", _blob((24, 20)), dtype="float")
    store.add_layer("north", _blob((16, 28)), dtype="float")


def test_constant_target_relative_mae_is_neutral_not_zero() -> None:
    target = np.ones(10, dtype=float)
    predictor = np.arange(10, dtype=float)
    train_mask = np.array([True] * 5 + [False] * 5)
    test_mask = ~train_mask

    relative_mae = scoring.compute_out_of_sample_mae(
        target,
        [predictor],
        train_mask,
        test_mask,
    )

    assert relative_mae == pytest.approx(1.0)


def test_effective_samples_are_capped_at_10000(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (20_001, 1, 1)
    grid = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1.0, 1.0, 1.0), shape=shape)
    layers = [np.ones(shape, dtype=np.float32).ravel(), np.ones(shape, dtype=np.float32).ravel()]

    monkeypatch.setattr(
        scoring,
        "compute_pairwise_mae",
        lambda *args, **kwargs: np.array([[0.0, 0.5], [0.5, 0.0]]),
    )
    monkeypatch.setattr(scoring, "compute_moran_correction", lambda *args, **kwargs: 1.0)

    result = scoring.geological_coherence_score(layers, ["float", "float"], grid, shape, seed=7)

    assert result["n_effective_samples"] == 10_000


def test_first_layer_returns_stage1_fields(tmp_path: Path) -> None:
    store = _store(tmp_path / "first_layer")
    result = evaluate_new_layer(store, "only", _blob((20, 20)), "float", seed=7)

    assert result["admitted"] is True
    assert result["admission_path"] == "first_layer_auto"
    assert result["masking_test_direction"] == "first_layer_auto"
    assert result["stage_completed"] == "first_layer_auto"
    assert result["bic_delta"] is None
    assert result["scoring_objective"] == "spatial_predictor_lift_v1"


def test_diverse_seed_path_before_seed_target(tmp_path: Path) -> None:
    store = _store(tmp_path / "diverse_seed")
    evaluate_new_layer(store, "base", _blob((16, 20)), "float", seed=7)

    result = evaluate_new_layer(store, "offset", _blob((24, 20)), "float", seed=7)

    assert result["admitted"] is True
    assert result["admission_path"] == "diverse_seed"
    assert result["masking_test_passed"] is True
    assert result["validity_passed"] is True


def test_spatial_predictor_lift_admits_non_colocated_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path / "normal_admit")
    _seed_normal_pool(store)

    result = evaluate_new_layer(store, "central_bridge", _blob((20, 20)), "float", seed=7)

    assert result["admission_path"] == "normal"
    assert result["masking_test_direction"] == "candidate_predictor_lift"
    assert result["masking_test_passed"] is True
    assert result["candidate_predictor_lift_mean"] > 0.0
    assert result["bic_delta"] < 0.0
    assert result["admitted"] is True


def test_unrelated_self_coherent_blob_fails_normal_admission(tmp_path: Path) -> None:
    store = _store(tmp_path / "normal_reject")
    _seed_normal_pool(store)

    result = evaluate_new_layer(store, "far_blob", _blob((35, 35)), "float", seed=7)

    assert result["admission_path"] == "normal"
    assert result["validity_passed"] is True
    assert result["admitted"] is False
    assert result["bic_delta"] >= result["admission_threshold"]


def test_blanket_candidate_fails_validity_gate(tmp_path: Path) -> None:
    store = _store(tmp_path / "blanket")
    _seed_normal_pool(store)

    result = evaluate_new_layer(store, "blanket", np.ones(_GRID.shape, dtype=np.float32), "float", seed=7)

    assert result["validity_passed"] is False
    assert result["masking_test_passed"] is False
    assert result["masking_test_direction"] == "self_validity_gate"
    assert result["admitted"] is False


def test_clone_into_identical_pool_is_not_admitted() -> None:
    target = _blob((20, 20), radius=3)
    result = scoring.spatial_predictor_lift_score(
        [target.ravel(), target.ravel()],
        ["target", "clone"],
        target.ravel(),
        _GRID.shape,
        ridge_alpha=1e-2,
        null_permutations=0,
    )

    assert result["validity_passed"] is True
    assert result["candidate_predictor_lift_mean"] <= 1e-8
    assert result["masking_test_passed"] is False
    assert result["admitted"] is False


def test_scoring_deterministic_with_seed(tmp_path: Path) -> None:
    def _run() -> float:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoxelStore(Path(tmp), _GRID)
            _seed_normal_pool(store)
            return evaluate_new_layer(store, "central_bridge", _blob((20, 20)), "float", seed=123)[
                "bic_delta"
            ]

    assert _run() == pytest.approx(_run(), abs=1e-12)


def test_scoring_reports_spatial_observability_fields(tmp_path: Path) -> None:
    store = _store(tmp_path / "observability")
    _seed_normal_pool(store)

    result = evaluate_new_layer(store, "central_bridge", _blob((20, 20)), "float", seed=7)

    for key in (
        "n_spatial_folds",
        "n_signal_folds_by_target",
        "n_holdout_rows_by_target",
        "insufficient_evidence_by_target",
        "candidate_predictor_lift_by_target",
        "bic_delta_by_target",
        "self_relative_mae",
        "ridge_effective_dof_by_target",
    ):
        assert key in result
    assert result["n_effective_samples_after"] == result["n_effective_samples"]
    assert result["candidate_nonzero_voxels"] == int(np.count_nonzero(_blob((20, 20))))
