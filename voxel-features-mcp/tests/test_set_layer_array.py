"""TDD for the array-write voxel path (spatial_set_layer_array).

The live run builds every layer as a binary point-mask (value ≡ 1.0) because the
only deposit paths flat-fill a single scalar per geometry. This tool lets the
code phase's precomputed continuous per-voxel field (kernel density / IDW /
distance / prospectivity) be deposited VERBATIM as one layer — no binarization.
"""
from __future__ import annotations

import numpy as np
import pytest

from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import GridSpec
from voxel_features.mcp.tools.spatial_tools import spatial_set_layer_array


TENIZ_GRID = GridSpec(
    origin=(66.5, 49.5, 0.0),
    maximum=(71.5, 52.5, 80.0),
    shape=(200, 200, 8),
    crs="EPSG:4326",
)


def test_set_layer_array_preserves_continuous_values(tmp_path):
    """A precomputed continuous field is deposited verbatim — the new array-write
    path must NOT binarize (the binary-mask root cause). Distinct float values
    survive the round-trip through add_layer/get_layer_values."""
    store = SpatialVoxelStore(tmp_path / "store", TENIZ_GRID)
    rng = np.random.default_rng(0)
    arr = rng.random((200, 200, 8)) * 0.9 + 0.1  # all nonzero, many distinct
    result = spatial_set_layer_array(store, name="grad", values=arr, dtype="float")
    assert result["success"] is True
    assert result["layer_name"] == "grad"
    assert result["nonzero_voxels"] == int((arr != 0).sum())
    stored = store.get_layer_values("grad")
    np.testing.assert_array_equal(stored, arr)
    assert len(np.unique(stored)) > 2  # genuinely continuous, not a 0/1 mask


def test_set_layer_array_rejects_shape_mismatch(tmp_path):
    """Shape != grid is rejected cleanly (success False); no layer is created."""
    store = SpatialVoxelStore(tmp_path / "store", TENIZ_GRID)
    bad = np.ones((10, 10, 5), dtype=float)
    result = spatial_set_layer_array(store, name="bad", values=bad, dtype="float")
    assert result["success"] is False
    assert "shape" in result["error"].lower()
    with pytest.raises(KeyError):
        store.get_layer_values("bad")


def test_set_layer_array_replaces_existing_layer(tmp_path):
    """Re-depositing under an existing scratch name replaces it (RMW-safe),
    rather than raising 'already exists' from add_layer."""
    store = SpatialVoxelStore(tmp_path / "store", TENIZ_GRID)
    a = np.full((200, 200, 8), 0.3)
    b = np.full((200, 200, 8), 0.7)
    assert spatial_set_layer_array(store, name="L", values=a)["success"] is True
    assert spatial_set_layer_array(store, name="L", values=b)["success"] is True
    np.testing.assert_array_equal(store.get_layer_values("L"), b)
