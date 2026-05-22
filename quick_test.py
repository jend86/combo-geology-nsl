#!/usr/bin/env python3
"""
Quick test to verify geological interpolation optimization works.
"""

import numpy as np
import sys
from pathlib import Path
import time

# Add paths
sys.path.append(str(Path(__file__).parent / "voxel-features-mcp"))

from voxel_features.store import GridSpec  
from voxel_features import scoring

def quick_test():
    print("🚀 Quick Geological Interpolation Test")
    
    # Create test grid
    grid = GridSpec(
        origin=(117.832, -27.441, 0),
        maximum=(117.973, -27.300, 80),
        shape=(200, 200, 8)
    )
    
    print(f"Grid: {grid.shape} = {grid.n_voxels:,} voxels")
    
    # Test 1: Basic interpolation timing
    print("\n⏱️ Test 1: Performance Test")
    
    # Create minimal sparse pattern
    test_layer = np.zeros((200, 200, 8))
    test_layer[100, 100, 4] = 1.0  # Single point
    test_layer[110, 110, 4] = 0.8  # Second point
    test_layer_flat = test_layer.flatten()
    
    print(f"Original non-zero voxels: {np.count_nonzero(test_layer_flat)}")
    
    start_time = time.time()
    interpolated = scoring.compute_geological_interpolation(test_layer_flat, grid, (200, 200, 8))
    elapsed = time.time() - start_time
    
    print(f"Interpolation completed in {elapsed:.2f} seconds")
    print(f"Result non-zero voxels: {np.count_nonzero(interpolated)}")
    print(f"Expansion: {np.count_nonzero(interpolated) / np.count_nonzero(test_layer_flat):.1f}x")
    
    # Test 2: Default radius
    print("\n📏 Test 2: Default Radius Calculation")
    default_radius = scoring.get_default_influence_radius(grid)
    print(f"Default influence radius: {default_radius:.1f}m")
    
    # Test 3: ESA-BIC calculation
    print("\n📊 Test 3: ESA-BIC Calculation")
    
    # Simple ESA-BIC test
    test_bic = scoring.compute_esa_bic(
        system_coherence=0.5,
        spatial_correction=0.8,
        n_layers=2,
        effective_samples=100,
        total_voxels=grid.n_voxels
    )
    
    print(f"Test ESA-BIC: {test_bic:.3f}")
    
    print("\n✅ Quick test completed successfully!")
    
    if elapsed < 5.0:
        print("🎉 Performance: GOOD (< 5 seconds)")
    else:
        print("⚠️ Performance: SLOW (> 5 seconds)")
    
    return elapsed < 10.0  # Return success if under 10 seconds

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Ready for full testing!")
    else:
        print("\n❌ Still too slow, needs more optimization")
