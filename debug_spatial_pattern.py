#!/usr/bin/env python3
"""
Debug the actual spatial pattern that led to BIC rejection.
Recreate the exact spatial operations from the logs.
"""

import numpy as np
import sys
from pathlib import Path
import tempfile

# Add paths
sys.path.append(str(Path(__file__).parent / "voxel-features-mcp"))

from voxel_features.store import VoxelStore, GridSpec  
from voxel_features.spatial import SpatialVoxelStore
from voxel_features import scoring

def debug_rejected_spatial_pattern():
    """Debug the exact spatial pattern that was rejected with BIC delta 0.347."""
    
    print("🔍 Spatial Pattern Rejection Analysis")
    print("=" * 55)
    
    # Recreate the exact grid used in the system
    grid = GridSpec(
        origin=(117.832, -27.441, 0),
        maximum=(117.973, -27.300, 80),
        shape=(200, 200, 8)
    )
    
    print(f"Grid: {grid.shape} = {grid.n_voxels:,} voxels")
    print(f"Bounds: lon {grid.origin[0]:.3f}-{grid.maximum[0]:.3f}")
    print(f"        lat {grid.origin[1]:.3f}-{grid.maximum[1]:.3f}")
    print(f"        depth {grid.origin[2]:.1f}-{grid.maximum[2]:.1f}m")
    
    # Create spatial store (same as workflow uses)
    store = SpatialVoxelStore(tempfile.mkdtemp(), grid)
    
    # Step 1: Recreate the exact spatial operations from logs
    print(f"\n📍 Recreating Spatial Operations:")
    
    # Point feature: lon=117.911, lat=-27.407, depth=45m, value=0.85, radius=200m
    print(f"  1. Adding point: (117.911, -27.407, 45m), value=0.85, radius=200m")
    from voxel_features.mcp.tools.spatial_tools import spatial_add_point
    
    point_result = spatial_add_point(
        store=store,
        name="mineralization_potential",
        longitude=117.911,
        latitude=-27.407,
        depth_m=45.0,
        value=0.85,
        radius_m=200.0
    )
    print(f"     → {point_result['affected_voxels']} voxels affected")
    
    # Line feature: (117.911,-27.407,0m) to (117.913,-27.406,60m), value=1.0, width=75m  
    print(f"  2. Adding line: (117.911,-27.407,0m) to (117.913,-27.406,60m), value=1.0, width=75m")
    from voxel_features.mcp.tools.spatial_tools import spatial_add_line
    
    line_result = spatial_add_line(
        store=store,
        name="mineralization_potential",
        start_longitude=117.911,
        start_latitude=-27.407,
        start_depth_m=0.0,
        end_longitude=117.913,
        end_latitude=-27.406,
        end_depth_m=60.0,
        value=1.0,
        width_m=75.0
    )
    print(f"     → {line_result['affected_voxels']} voxels affected")
    
    # Step 2: Examine the created layer
    print(f"\n🔬 Layer Analysis:")
    if "mineralization_potential" in store.layer_names:
        layer_values = store.get_layer_values("mineralization_potential")
        
        print(f"  Layer shape: {layer_values.shape}")
        print(f"  Non-zero voxels: {np.count_nonzero(layer_values):,}")
        print(f"  Coverage: {np.count_nonzero(layer_values)/layer_values.size*100:.3f}%")
        
        unique_values = np.unique(layer_values[layer_values > 0])
        print(f"  Unique values: {unique_values}")
        print(f"  Value range: {layer_values.min():.3f} to {layer_values.max():.3f}")
        
        # Check for mixed values issue
        if len(unique_values) > 1:
            print(f"  ⚠️  MIXED VALUES: Layer has {len(unique_values)} different values")
            for val in unique_values:
                count = np.sum(layer_values == val)
                print(f"     Value {val:.2f}: {count:,} voxels")
    
    # Step 3: Test BIC evaluation step by step
    print(f"\n📊 BIC Evaluation Debug:")
    
    # First check if this is truly the first layer
    regular_store = VoxelStore(tempfile.mkdtemp(), grid)
    existing_layers = list(regular_store.layer_names)
    print(f"  Existing layers in store: {len(existing_layers)}")
    
    if len(existing_layers) == 0:
        print(f"  → This is the FIRST layer (should be auto-admitted)")
    
    # Extract the layer and evaluate it
    layer_3d = store.get_layer_values("mineralization_potential")
    result = scoring.evaluate_new_layer(
        store=regular_store,
        layer_name="mineralization_potential",
        layer_values=layer_3d,
        layer_dtype="float"
    )
    
    print(f"  BIC delta: {result['bic_delta']:.6f}")
    print(f"  CV MSE after: {result['cv_mse_after']:.6f}")
    print(f"  Admitted: {result['admitted']}")
    print(f"  Mutual info: {result['mutual_info']}")
    
    # Step 4: Compare with a better spatial pattern
    print(f"\n🆚 Comparison with Better Pattern:")
    
    # Create a denser, more geological pattern
    store2 = SpatialVoxelStore(tempfile.mkdtemp(), grid)
    
    # Add multiple points creating a coherent geological trend
    center_lon, center_lat = 117.9, -27.37
    for i in range(5):
        offset_lon = center_lon + (i - 2) * 0.01  # 5 points across ~1km
        offset_lat = center_lat + (i - 2) * 0.005
        
        spatial_add_point(
            store=store2,
            name="coherent_pattern",
            longitude=offset_lon,
            latitude=offset_lat,
            depth_m=20.0,
            value=0.8,  # Consistent value
            radius_m=150.0
        )
    
    coherent_layer = store2.get_layer_values("coherent_pattern")
    coherent_result = scoring.evaluate_new_layer(
        store=VoxelStore(tempfile.mkdtemp(), grid),
        layer_name="coherent_pattern", 
        layer_values=coherent_layer,
        layer_dtype="float"
    )
    
    print(f"  Coherent pattern coverage: {np.count_nonzero(coherent_layer)/coherent_layer.size*100:.3f}%")
    print(f"  Coherent BIC delta: {coherent_result['bic_delta']:.6f}")
    print(f"  Coherent admitted: {coherent_result['admitted']}")
    
    # Step 5: Diagnosis
    print(f"\n🎯 Diagnosis:")
    sparse_coverage = np.count_nonzero(layer_values)/layer_values.size*100
    
    issues_found = []
    if sparse_coverage < 1.0:
        issues_found.append(f"Very sparse coverage ({sparse_coverage:.3f}%)")
    if len(unique_values) > 1:
        issues_found.append(f"Mixed values in same layer ({unique_values})")
    if result['bic_delta'] > 0.3:
        issues_found.append(f"High BIC penalty ({result['bic_delta']:.3f})")
    
    if issues_found:
        print(f"  🚨 Issues identified:")
        for issue in issues_found:
            print(f"     - {issue}")
        
        print(f"\n  🔧 Potential fixes:")
        print(f"     - Increase spatial coverage (more points/lines)")
        print(f"     - Use consistent values within same layer") 
        print(f"     - Create denser, more connected patterns")
        print(f"     - Ensure geological coherence across space")
    else:
        print(f"  ✅ Pattern structure looks reasonable")
        print(f"     Issue may be in geological coherence calculation")

if __name__ == "__main__":
    debug_rejected_spatial_pattern()
