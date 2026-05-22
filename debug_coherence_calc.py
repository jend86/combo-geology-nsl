#!/usr/bin/env python3
"""
Debug geological coherence calculation to check if high BIC deltas are legitimate.
"""

import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "voxel-features-mcp"))
sys.path.append(str(Path(__file__).parent / "NSL2-geology-task" / "src"))

from voxel_features.store import VoxelStore, GridSpec
from voxel_features import scoring

def debug_coherence_calculation():
    """Debug the geological coherence calculation step by step."""
    
    print("🔬 Geological Coherence Debug")
    print("=" * 50)
    
    # Create test store matching your actual grid
    grid = GridSpec(
        origin=(117.832, -27.441, 0),
        maximum=(117.973, -27.300, 80),
        shape=(200, 200, 8)  # Same as actual system
    )
    import tempfile
    store = VoxelStore(tempfile.mkdtemp(), grid)
    nx, ny, nz = store.grid.shape
    print(f"Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels")
    
    # Test 1: Add a known good layer (depth gradient)
    print(f"\n📊 Test 1: Adding Depth Gradient Layer")
    depth_layer = np.zeros((nx, ny, nz))
    for z in range(nz):
        depth_layer[:, :, z] = z / (nz - 1)  # Perfect depth gradient
    
    result1 = scoring.evaluate_new_layer(
        store=store,
        layer_name="depth_gradient",
        layer_values=depth_layer,
        layer_dtype="float"
    )
    
    print(f"  BIC delta: {result1['bic_delta']:.6f}")
    print(f"  CV MSE after: {result1['cv_mse_after']:.6f}")
    print(f"  Admitted: {result1['admitted']}")
    print(f"  Expected: Should be admitted (first layer)")
    
    # Test 2: Add a layer with some geological correlation
    print(f"\n📊 Test 2: Adding Correlated Geological Layer")
    # Create a layer correlated with depth but with some geological variation
    geo_layer = np.random.random((nx, ny, nz))
    for z in range(nz):
        # Add depth correlation + geological noise
        depth_influence = (z / (nz - 1)) * 0.7
        geo_layer[:, :, z] = geo_layer[:, :, z] * 0.3 + depth_influence
    
    result2 = scoring.evaluate_new_layer(
        store=store,
        layer_name="geological_correlated",
        layer_values=geo_layer,
        layer_dtype="float"
    )
    
    print(f"  BIC delta: {result2['bic_delta']:.6f}")
    print(f"  CV MSE after: {result2['cv_mse_after']:.6f}")  
    print(f"  Admitted: {result2['admitted']}")
    print(f"  Expected: Should be admitted if coherence calculation works")
    
    # Test 3: Add pure noise (should be rejected)
    print(f"\n📊 Test 3: Adding Pure Noise Layer")
    noise_layer = np.random.random((nx, ny, nz))
    
    # Reset store to test noise properly
    store = VoxelStore(tempfile.mkdtemp(), grid)
    scoring.evaluate_new_layer(store, "depth_gradient", depth_layer, "float")
    
    result3 = scoring.evaluate_new_layer(
        store=store,
        layer_name="pure_noise",
        layer_values=noise_layer,
        layer_dtype="float"
    )
    
    print(f"  BIC delta: {result3['bic_delta']:.6f}")
    print(f"  CV MSE after: {result3['cv_mse_after']:.6f}")
    print(f"  Admitted: {result3['admitted']}")
    print(f"  Expected: Should be rejected (no coherence)")
    
    # Test 4: Check internal coherence calculation
    print(f"\n🔍 Internal Coherence Analysis")
    
    if len(store.layer_names) >= 1:
        # Get existing layers
        existing_names = list(store.layer_names)
        print(f"  Existing layers: {existing_names}")
        
        # Manual coherence calculation
        layer_values = [store.get_layer_values(name).flatten() for name in existing_names]
        layer_dtypes = [store.get_layer(name).dtype for name in existing_names]
        
        print(f"  Layer count: {len(layer_values)}")
        print(f"  Voxel count: {layer_values[0].shape[0]:,}")
        
        # Test with the new layer added
        test_values = layer_values + [noise_layer.flatten()]
        test_dtypes = layer_dtypes + ["float"]
        
        coherence_result = scoring.geological_coherence_score(
            test_values, test_dtypes, store.grid, store.grid.shape
        )
        
        print(f"  System coherence: {coherence_result['system_coherence']:.6f}")
        print(f"  Spatial correction: {coherence_result['spatial_correction']:.6f}")
        print(f"  BIC: {coherence_result['bic']:.6f}")
        print(f"  Total CV MSE: {coherence_result['total_cv_mse']:.6f}")
        
        # Check if spatial correction is too aggressive
        if coherence_result['spatial_correction'] < 0.1:
            print(f"  ⚠️  WARNING: Spatial correction very low ({coherence_result['spatial_correction']:.4f})")
            print(f"      This might be overly penalizing legitimate geological patterns")
        
        # Check coherence matrix
        if coherence_result['coherence_matrix'].size > 0:
            matrix = coherence_result['coherence_matrix']
            print(f"  R² matrix shape: {matrix.shape}")
            if matrix.shape[0] > 1:
                # Show off-diagonal correlations
                mask = ~np.eye(matrix.shape[0], dtype=bool)
                off_diag_values = matrix[mask]
                print(f"  Off-diagonal R² range: {off_diag_values.min():.4f} to {off_diag_values.max():.4f}")
                print(f"  Mean R²: {off_diag_values.mean():.4f}")
    
    # Summary
    print(f"\n🎯 Diagnosis Summary:")
    print(f"  Recent BIC delta: ~0.347 (similar to your actual results)")
    
    actual_deltas = [result1['bic_delta'], result2['bic_delta'], result3['bic_delta']]
    print(f"  Test deltas: {[f'{d:.3f}' for d in actual_deltas]}")
    
    # Check if all deltas are positive (indicating overly strict scoring)
    positive_count = sum(1 for d in actual_deltas[1:] if d > 0)  # Skip first layer
    total_tests = len(actual_deltas) - 1
    
    if positive_count == total_tests:
        print(f"  🚨 ISSUE: All layers rejected - scoring may be too strict")
        print(f"       Possible causes:")
        print(f"       - Spatial correction factor too aggressive")
        print(f"       - Normalization destroying geological signal")
        print(f"       - BIC penalty term too high")
    else:
        print(f"  ✅ GOOD: Scoring discriminates between layers")

if __name__ == "__main__":
    debug_coherence_calculation()
