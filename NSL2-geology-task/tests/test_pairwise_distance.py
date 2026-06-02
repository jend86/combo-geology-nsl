"""Unit tests for ``voxel_features.scoring.pairwise_distance``.

Background: the crossbreed queue's orthogonality term was a broken
mutual_information() (unit mismatch — always returned 0). The current queue
score uses ``pairwise_distance`` instead: Jaccard distance for boolean layers
and MAE for non-boolean. These tests pin the contract.
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


def test_mae_for_float_layers(tmp_path: Path) -> None:
    # Two float layers with a known mean-abs-error. shape=32 voxels;
    # |0.7 - 0.2| at every voxel → MAE = 0.5.
    a = np.full(_GRID.shape, 0.7, dtype=np.float32)
    b = np.full(_GRID.shape, 0.2, dtype=np.float32)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (b, "float"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.5, abs=1e-5)


def test_mae_identical_float_layers_returns_zero(tmp_path: Path) -> None:
    a = np.full(_GRID.shape, 0.42, dtype=np.float32)
    store = _store_with_layers(tmp_path, {
        "a": (a, "float"),
        "b": (a.copy(), "float"),
    })
    assert pairwise_distance(store, "a", "b") == pytest.approx(0.0, abs=1e-7)


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
