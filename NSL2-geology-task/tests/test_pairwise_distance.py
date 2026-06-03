"""Unit tests for ``voxel_features.scoring.pairwise_distance``.

Background: the crossbreed queue's orthogonality term was a broken
mutual_information() (unit mismatch — always returned 0). The current queue
score uses ``pairwise_distance`` instead.

Both branches now return a normalized dissimilarity in ``[0, 1]`` so the shared
``_NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.15`` carries the same meaning regardless
of dtype: Jaccard distance for boolean layers, and the magnitude-normalized L1
``sum|a-b| / (sum|a| + sum|b|)`` (Sørensen/Bray–Curtis) for non-boolean layers.
Raw MAE (unbounded, unit-dependent) was the old non-boolean contract; it made
jittered large-magnitude float duplicates read as "distinct" and distinct
small-magnitude float layers read as "near-duplicate". These tests pin the
normalized contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxel_features.scoring import pairwise_distance
from voxel_features.store import GridSpec, VoxelStore


_GRID = GridSpec(
    origin=(0.0, 0.0, 0.0),
    maximum=(0.02, 0.02, 20.0),
    shape=(4, 4, 2),
)


def _store_with_layers(
    tmp_path: Path,
    layers: dict[str, tuple[np.ndarray, str]],
) -> VoxelStore:
    store = VoxelStore(tmp_path, _GRID)
    for name, (values, dtype) in layers.items():
        store.add_layer(name=name, values=values, dtype=dtype)
    return store


def test_jaccard_identical_boolean_layers_returns_zero(tmp_path: Path) -> None:
    values = np.zeros(_GRID.shape, dtype=bool)
    values[0, 0, 0] = True
    values[1, 1, 1] = True
    store = _store_with_layers(tmp_path, {
        "a": (values, "boolean"),
        "b": (values.copy(), "boolean"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.0, abs=1e-9)


def test_jaccard_disjoint_boolean_layers_returns_one(tmp_path: Path) -> None:
    a = np.zeros(_GRID.shape, dtype=bool)
    a[0, 0, 0] = True
    b = np.zeros(_GRID.shape, dtype=bool)
    b[3, 3, 1] = True
    store = _store_with_layers(tmp_path, {
        "a": (a, "boolean"),
        "b": (b, "boolean"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(1.0, abs=1e-9)


def test_jaccard_partial_overlap_boolean(tmp_path: Path) -> None:
    # a has 3 trues at indices {(0,0,0), (1,1,1), (2,2,0)}.
    # b has 3 trues at indices {(0,0,0), (1,1,1), (3,3,1)}.
    # intersection: 2; union: 4 → Jaccard distance = 1 - 2/4 = 0.5.
    a = np.zeros(_GRID.shape, dtype=bool)
    b = np.zeros(_GRID.shape, dtype=bool)
    for idx in [(0, 0, 0), (1, 1, 1), (2, 2, 0)]:
        a[idx] = True
    for idx in [(0, 0, 0), (1, 1, 1), (3, 3, 1)]:
        b[idx] = True
    store = _store_with_layers(tmp_path, {
        "a": (a, "boolean"),
        "b": (b, "boolean"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.5, abs=1e-9)


def test_jaccard_empty_boolean_layers_returns_zero(tmp_path: Path) -> None:
    # Edge case: both layers are all-False. Union is 0; we define this as 0
    # distance (they're identical — both encode "nothing here"). Documented
    # in the pairwise_distance docstring.
    a = np.zeros(_GRID.shape, dtype=bool)
    b = np.zeros(_GRID.shape, dtype=bool)
    store = _store_with_layers(tmp_path, {
        "a": (a, "boolean"),
        "b": (b, "boolean"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.0, abs=1e-9)


def test_normalized_distance_for_constant_float_layers(tmp_path: Path) -> None:
    # Magnitude-normalized L1: sum|a-b| / (sum|a| + sum|b|).
    # Constant 0.7 vs 0.2 → |0.5| / (0.7 + 0.2) = 0.5 / 0.9 ≈ 0.5556.
    # (The old raw-MAE contract returned 0.5.)
    a = np.full(_GRID.shape, 0.7, dtype=np.float32)
    b = np.full(_GRID.shape, 0.2, dtype=np.float32)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (b, "float"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.5 / 0.9, abs=1e-5)


def test_mae_identical_float_layers_returns_zero(tmp_path: Path) -> None:
    a = np.full(_GRID.shape, 0.42, dtype=np.float32)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (a.copy(), "float"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.0, abs=1e-7)


def test_both_zero_float_layers_returns_zero(tmp_path: Path) -> None:
    # 0/0 guard: two all-zero float layers are identical → 0.0 (matches the
    # boolean union-empty convention).
    a = np.zeros(_GRID.shape, dtype=np.float32)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (a.copy(), "float"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.0, abs=1e-9)


def test_jittered_large_magnitude_float_is_near_duplicate(tmp_path: Path) -> None:
    # REGRESSION: a jittered clone of a large-magnitude float layer is a
    # practical duplicate. Old raw MAE (≈5) exceeded the 0.15 threshold and
    # wrongly passed it as "distinct"; the normalized metric (≈5/2000) keeps it
    # well below 0.15.
    rng = np.random.default_rng(7)
    a = np.full(_GRID.shape, 1000.0, dtype=np.float64)
    a += rng.normal(0.0, 1.0, size=_GRID.shape)
    b = a + rng.normal(0.0, 5.0, size=_GRID.shape)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (b, "float"),
    })
    dist = pairwise_distance(store, "a", "b")
    assert dist < 0.15
    assert 0.0 <= dist <= 1.0


def test_distinct_small_magnitude_float_is_not_near_duplicate(tmp_path: Path) -> None:
    # REGRESSION: two genuinely different small-magnitude float layers over
    # disjoint supports. Old raw MAE (≈0.05) fell below 0.15 and wrongly
    # collapsed them into one "near-duplicate"; the normalized metric returns
    # ≈1.0 (disjoint supports), well above 0.15.
    a = np.zeros(_GRID.shape, dtype=np.float32)
    b = np.zeros(_GRID.shape, dtype=np.float32)
    a[0:2, :, :] = 0.05
    b[2:4, :, :] = 0.05
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (b, "float"),
    })
    dist = pairwise_distance(store, "a", "b")
    assert dist > 0.15
    assert dist == pytest.approx(1.0, abs=1e-6)


def test_normalized_distance_bounded_in_unit_interval(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 500.0, size=_GRID.shape)
    b = rng.normal(0.0, 500.0, size=_GRID.shape)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (b, "float"),
    })
    dist = pairwise_distance(store, "a", "b")
    assert 0.0 <= dist <= 1.0


def test_symmetry_under_argument_swap(tmp_path: Path) -> None:
    # dist(a, b) == dist(b, a). Important because _enumerate_pairs looks up
    # the metric by the unordered (alphabetically-sorted) pair id.
    rng = np.random.default_rng(42)
    a = (rng.random(_GRID.shape) > 0.85).astype(bool)
    b = (rng.random(_GRID.shape) > 0.85).astype(bool)
    store = _store_with_layers(tmp_path, {
        "a": (a, "boolean"),
        "b": (b, "boolean"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(
        pairwise_distance(store, "b", "a"), abs=1e-9
    )


def test_symmetry_under_argument_swap_float(tmp_path: Path) -> None:
    rng = np.random.default_rng(43)
    a = rng.normal(0.0, 10.0, size=_GRID.shape)
    b = rng.normal(0.0, 10.0, size=_GRID.shape)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (b, "float"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(
        pairwise_distance(store, "b", "a"), abs=1e-9
    )
