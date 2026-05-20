#!/usr/bin/env python3
"""Test script for spatial feature hypothesis integration."""

import sys
import os
sys.path.append('/home/jen/Desktop/geonsl/voxel-features-mcp')

import tempfile
import json
from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import COE_FAIRBAIRN_GRID
from voxel_features.mcp.tools.spatial_tools import (
    spatial_add_point, spatial_add_line, spatial_coord_to_voxel, spatial_query_region
)

def test_spatial_integration():
    """Test the complete spatial integration workflow."""
    print("🔗 Testing Spatial Integration Workflow...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Creating spatial store in {tmpdir}")
        store = SpatialVoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        print(f"🗺️  Grid: {store.grid.shape} ({store.grid.n_voxels:,} voxels)")
        print(f"📏 Cell size: {[f'{c:.6f}' for c in store.grid.cell_size]}")
        
        # Test MCP spatial tools
        print("\n🛠️ Testing MCP Spatial Tools...")
        
        # Test coordinate validation
        print("\n📍 Testing coordinate validation...")
        coord_result = spatial_coord_to_voxel(
            store, longitude=117.9, latitude=-27.41, depth_m=40
        )
        print(f"   Coordinate mapping result: {coord_result}")
        
        if not coord_result["success"]:
            print("   ❌ Coordinate validation failed!")
            return False
        
        # Test point feature creation
        print("\n⚡ Testing spatial point creation...")
        point_result = spatial_add_point(
            store,
            name="integration_test_copper",
            longitude=117.92,
            latitude=-27.407,
            depth_m=45,
            value=0.85,
            radius_m=120,
            dtype="float",
            metadata={"test": "integration", "element": "Cu"}
        )
        print(f"   Point result: {point_result}")
        
        if not point_result["success"]:
            print("   ❌ Point feature creation failed!")
            return False
        
        # Test line feature creation  
        print("\n📏 Testing spatial line creation...")
        line_result = spatial_add_line(
            store,
            name="integration_test_fault",
            start_longitude=117.91,
            start_latitude=-27.408,
            start_depth_m=5,
            end_longitude=117.913,
            end_latitude=-27.405,
            end_depth_m=65,
            value=1.0,
            width_m=80,
            dtype="boolean",
            metadata={"type": "fault_zone", "orientation": "NE-SW"}
        )
        print(f"   Line result: {line_result}")
        
        if not line_result["success"]:
            print("   ❌ Line feature creation failed!")
            return False
        
        # Test spatial query
        print("\n🔍 Testing spatial query...")
        query_result = spatial_query_region(
            store,
            center_longitude=117.92,
            center_latitude=-27.407,
            center_depth_m=45,
            radius_m=200
        )
        print(f"   Query found {query_result.get('affected_voxels', 0)} voxels")
        print(f"   Layers found: {list(query_result.get('layer_samples', {}).keys())}")
        
        if not query_result["success"]:
            print("   ❌ Spatial query failed!")
            return False
        
        # Verify feature layer properties
        print("\n📊 Testing feature layer properties...")
        layers = store.list_layers()
        print(f"   Created {len(layers)} layers")
        
        for layer in layers:
            layer_name = layer["name"]
            layer_values = store.get_layer_values(layer_name)
            non_zero = (layer_values != 0).sum()
            memory_mb = layer_values.nbytes / (1024 * 1024)
            
            print(f"   Layer '{layer_name}':")
            print(f"     Shape: {layer_values.shape}")
            print(f"     Non-zero voxels: {non_zero:,}")
            print(f"     Memory: {memory_mb:.1f} MB")
            print(f"     Value range: [{layer_values.min():.3f}, {layer_values.max():.3f}]")
        
        # Test BIC scoring compatibility
        print("\n🎯 Testing BIC scoring compatibility...")
        try:
            from voxel_features.scoring import evaluate_new_layer
            
            # Create a test layer for BIC evaluation
            import numpy as np
            test_values = np.random.rand(200, 200, 8) * 0.1  # Small random values
            
            bic_result = evaluate_new_layer(
                store=store,
                layer_name="test_bic_layer", 
                layer_values=test_values,
                layer_dtype="float"
            )
            
            print(f"   BIC evaluation result: {bic_result}")
            print(f"   BIC delta: {bic_result.get('bic_delta', 'N/A')}")
            print(f"   Admitted: {bic_result.get('admitted', 'N/A')}")
            
            if "bic_delta" not in bic_result:
                print("   ❌ BIC scoring integration failed!")
                return False
            
        except Exception as e:
            print(f"   ❌ BIC scoring test failed: {e}")
            return False
        
        # Test voxel resolution calculations
        print("\n📐 Testing resolution calculations...")
        origin = store.grid.origin
        maximum = store.grid.maximum
        shape = store.grid.shape
        
        # Calculate actual geographic span per voxel
        lon_span_deg = (maximum[0] - origin[0]) / shape[0]
        lat_span_deg = (maximum[1] - origin[1]) / shape[1] 
        depth_span_m = (maximum[2] - origin[2]) / shape[2]
        
        # Convert to meters (approximate)
        lat_center = (origin[1] + maximum[1]) / 2
        lon_span_m = lon_span_deg * 111320 * abs(math.cos(math.radians(lat_center)))
        lat_span_m = lat_span_deg * 111320
        
        print(f"   Actual voxel resolution:")
        print(f"     Longitude: {lon_span_m:.0f}m ({lon_span_deg:.6f}°)")
        print(f"     Latitude: {lat_span_m:.0f}m ({lat_span_deg:.6f}°)")
        print(f"     Depth: {depth_span_m:.1f}m")
        print(f"     Total area per voxel: {(lon_span_m * lat_span_m / 10000):.1f} hectares")
        
        print("\n✅ All spatial integration tests passed!")
        return True

if __name__ == "__main__":
    import math
    success = test_spatial_integration()
    sys.exit(0 if success else 1)
