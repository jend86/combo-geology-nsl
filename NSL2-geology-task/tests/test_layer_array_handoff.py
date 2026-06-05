"""TDD for the code→translate ndarray transport (_load_layer_array).

Mirrors test_feature_points_handoff.py: the code phase leaves a top-level numpy
array (auto-saved as ``<varname>_array.npy``); the translate phase's
spatial_set_layer_array resolves it by varname (or, for 'auto', by grid shape)
and deposits it as a continuous layer.
"""
from __future__ import annotations

import numpy as np

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask as T


def test_load_layer_array_resolves_by_varname(tmp_path):
    arr = np.random.default_rng(1).random((200, 200, 8))
    np.save(tmp_path / "value_grid_array.npy", arr)
    got, path = T._load_layer_array(
        [str(tmp_path / "value_grid_array.npy")], str(tmp_path), artifact_name="value_grid"
    )
    assert got is not None
    np.testing.assert_array_equal(got, arr)


def test_load_layer_array_auto_prefers_grid_shape(tmp_path):
    """'auto' must pick the grid-shaped array even when an intermediate
    (non-grid) array was also captured from the code phase."""
    grid_arr = np.random.default_rng(2).random((200, 200, 8))
    other = np.arange(10.0)  # a non-grid intermediate exec also saved
    np.save(tmp_path / "scratch_array.npy", other)
    np.save(tmp_path / "value_grid_array.npy", grid_arr)
    got, path = T._load_layer_array(
        [str(tmp_path / "scratch_array.npy"), str(tmp_path / "value_grid_array.npy")],
        str(tmp_path),
        artifact_name="auto",
        expected_shape=(200, 200, 8),
    )
    assert got is not None
    np.testing.assert_array_equal(got, grid_arr)


def test_load_layer_array_missing_returns_none(tmp_path):
    got, path = T._load_layer_array([], str(tmp_path), artifact_name="nope")
    assert got is None
