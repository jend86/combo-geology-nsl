#!/usr/bin/env python3
"""
Test BIC implementation in isolation with known good/bad feature layers.
"""

import numpy as np
import sys
from pathlib import Path

# Add the MCP module path
sys.path.append(str(Path(__file__).parent / "voxel-features-mcp"))

from voxel_features.storage import VoxelStore, Grid
from voxel_features import scoring

def create_test_store():
    """Create a test voxel store with known shape."""
    grid = Grid(
        longitude_range=(117.832, 117.973),
        latitude_range=(-27.441, -27.300),
        depth_range=(0, 80),
        shape=(200, 200, 8)  # nx, ny, nz
    )
    return VoxelStore(grid)

def test_bic_calculation():
    """Test BIC calculation with controlled scenarios."""
    
    print("🧪 Testing BIC Implementation in Isolation")
    print("=" * 50)
    
    # Create test store
    store = create_test_store()
    nx, ny, nz = store.grid.shape
    print(f"Grid shape: {nx} × {ny} × {nz} = {nx*ny*nz:,} voxels")
    
    # Test 1: First layer (should be admitted unconditionally)
    print("\n📊 Test 1: First Layer (Random)")
    random_layer = np.random.random((nx, ny, nz))
    result1 = scoring.evaluate_new_layer(
        store=store,
        layer_name="random_first",
        layer_values=random_layer,
        layer_dtype="float"
    )
    print(f"  BIC delta: {result1['bic_delta']:.2f}")
    print(f"  Admitted: {result1['admitted']}")
    print(f"  Expected: Always admitted (first layer)")
    
    # Test 2: Perfect predictor (should improve BIC significantly)  
    print("\n📊 Test 2: Perfect Predictor (Z-coordinate)")
    z_coords = np.zeros((nx, ny, nz))
    for z in range(nz):
        z_coords[:, :, z] = z / (nz - 1)  # Normalized depth
    
    result2 = scoring.evaluate_new_layer(
        store=store,
        layer_name="depth_predictor",
        layer_values=z_coords,
        layer_dtype="float"
    )
    print(f"  BIC delta: {result2['bic_delta']:.2f}")
    print(f"  Admitted: {result2['admitted']}")
    print(f"  Expected: Should be admitted (informative)")
    
    # Test 3: Noise layer (should hurt BIC)
    print("\n📊 Test 3: Pure Noise Layer")
    noise_layer = np.random.random((nx, ny, nz))
    result3 = scoring.evaluate_new_layer(
        store=store,
        layer_name="pure_noise",
        layer_values=noise_layer,
        layer_dtype="float"
    )
    print(f"  BIC delta: {result3['bic_delta']:.2f}")
    print(f"  Admitted: {result3['admitted']}")
    print(f"  Expected: Should be rejected (not informative)")
    
    # Test 4: Identical duplicate (should hurt BIC due to redundancy)
    print("\n📊 Test 4: Identical Duplicate")
    if "depth_predictor" in store.layer_names:
        duplicate_values = store.get_layer_values("depth_predictor")
        result4 = scoring.evaluate_new_layer(
            store=store,
            layer_name="duplicate_depth",
            layer_values=duplicate_values,
            layer_dtype="float"
        )
        print(f"  BIC delta: {result4['bic_delta']:.2f}")
        print(f"  Admitted: {result4['admitted']}")
        print(f"  Expected: Should be rejected (redundant)")
    
    # Test 5: Check BIC formula sanity
    print("\n📊 Test 5: BIC Formula Sanity Check")
    print(f"  Current layers in store: {list(store.layer_names)}")
    print(f"  Total voxels: {nx*ny*nz:,}")
    
    if len(store.layer_names) >= 2:
        # Get all layer values
        all_values = [store.get_layer_values(name).flatten() for name in store.layer_names]
        n_layers = len(all_values)
        n_voxels = all_values[0].shape[0]
        
        # Manual BIC check
        print(f"  Layers: {n_layers}")
        print(f"  Voxels used as sample size: {n_voxels:,}")
        print(f"  Parameters (n_layers²): {n_layers * n_layers}")
        
        # This is where we'd verify the user's fixes:
        # - Sample size should be based on effective degrees of freedom, not raw voxels
        # - Parameter count should reflect the actual model complexity
        print(f"  ⚠️  Check: Using voxels as sample size may inflate significance")
        print(f"  ⚠️  Check: Spatial correlation not accounted for")
    
    print("\n" + "=" * 50)
    print("🎯 BIC Test Summary:")
    print(f"  - Random first layer: {result1['admitted']} (expected: True)")
    print(f"  - Depth predictor: {result2['admitted']} (expected: True)")  
    print(f"  - Pure noise: {result3['admitted']} (expected: False)")
    if "depth_predictor" in store.layer_names:
        print(f"  - Duplicate layer: {result4['admitted']} (expected: False)")
    
    # Overall assessment
    expected_results = [True, True, False, False]
    actual_results = [result1['admitted'], result2['admitted'], result3['admitted']]
    if "depth_predictor" in store.layer_names:
        actual_results.append(result4['admitted'])
    
    matches = sum(a == e for a, e in zip(actual_results, expected_results[:len(actual_results)]))
    total = len(actual_results)
    
    print(f"\n🏆 BIC Implementation: {matches}/{total} tests match expected behavior")
    if matches == total:
        print("✅ BIC scoring appears to be working correctly!")
    else:
        print("❌ BIC scoring may need further debugging")
    
    return store

if __name__ == "__main__":
    test_store = test_bic_calculation()
