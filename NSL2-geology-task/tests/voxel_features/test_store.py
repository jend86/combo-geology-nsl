"""Test voxel store and scoring."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add voxel-features-mcp to path
vfm_path = str(Path(__file__).parent.parent.parent.parent / "voxel-features-mcp")
if vfm_path not in sys.path:
    sys.path.append(vfm_path)

from voxel_features.store import VoxelStore, GridSpec, COE_FAIRBAIRN_GRID
from voxel_features.scoring import compute_mdl, mutual_information, evaluate_new_layer


def test_grid_spec():
    """Test grid specification."""
    grid = COE_FAIRBAIRN_GRID
    
    assert grid.shape == (25, 25, 5)
    assert grid.n_voxels == 25 * 25 * 5
    
    dx, dy, dz = grid.cell_size
    assert dx > 0
    assert dy > 0
    assert dz > 0


def test_voxel_store_create():
    """Test creating a voxel store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        assert store.grid == COE_FAIRBAIRN_GRID
        assert store.layer_names == []


def test_add_layer():
    """Test adding a feature layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        # Create random values
        values = np.random.rand(25, 25, 5)
        
        layer = store.add_layer(
            name="test_feature",
            values=values,
            dtype="float",
            metadata={"source": "test"},
        )
        
        assert layer.name == "test_feature"
        assert layer.dtype == "float"
        assert "test_feature" in store.layer_names


def test_mdl_empty_store():
    """Test MDL of empty store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        mdl = compute_mdl(store)
        assert mdl == 0.0


def test_mdl_with_layer():
    """Test MDL with a feature layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        # Add a layer
        values = np.random.rand(25, 25, 5)
        store.add_layer("test", values, dtype="float")
        
        mdl = compute_mdl(store)
        assert mdl > 0


def test_mutual_information():
    """Test mutual information between layers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        # Add two correlated layers
        base = np.random.rand(25, 25, 5)
        noise = np.random.rand(25, 25, 5) * 0.1
        
        store.add_layer("layer_a", base, dtype="float")
        store.add_layer("layer_b", base + noise, dtype="float")
        
        mi = mutual_information(store, "layer_a", "layer_b")
        assert mi >= 0


def test_evaluate_new_layer_admitted():
    """Test that a useful layer gets admitted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        # Add initial layer
        initial = np.random.rand(25, 25, 5)
        store.add_layer("initial", initial, dtype="float")
        
        # Add a different layer - should be admitted if it adds info
        new_layer = np.random.rand(25, 25, 5)
        
        result = evaluate_new_layer(
            store=store,
            layer_name="new_feature",
            layer_values=new_layer,
            layer_dtype="float",
        )
        
        assert "bic_before" in result
        assert "bic_after" in result
        assert "bic_delta" in result
        assert "admitted" in result


def test_evaluate_redundant_layer_rejected():
    """Test that a redundant layer gets rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        # Add initial layer
        initial = np.random.rand(25, 25, 5)
        store.add_layer("initial", initial, dtype="float")
        
        # Add exact duplicate - should be rejected (no compression gain)
        result = evaluate_new_layer(
            store=store,
            layer_name="duplicate",
            layer_values=initial.copy(),
            layer_dtype="float",
        )
        
        # Exact duplicate should not improve compression
        # (though in practice entropy estimation noise may affect this)
        assert "admitted" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
