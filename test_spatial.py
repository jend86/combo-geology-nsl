#!/usr/bin/env python3
"""Test script for spatial voxel functionality."""

import sys
import os
sys.path.append('/home/jen/Desktop/geonsl/voxel-features-mcp')

import tempfile
import numpy as np
from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import COE_FAIRBAIRN_GRID

def test_spatial_functionality():
    """Test the spatial voxel store functionality."""
    print("🧪 Testing Spatial Voxel Store...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create spatial store
        print(f"📁 Creating spatial store in {tmpdir}")
        store = SpatialVoxelStore(tmpdir, COE_FAIRBAIRN_GRID)
        
        print(f"🗺️  Grid: {store.grid.shape} voxels ({store.grid.n_voxels:,} total)")
        print(f"📐 Cell size: {store.grid.cell_size}")
        
        # Test coordinate conversion
        print("\n🔄 Testing coordinate conversion...")
        lon, lat, depth = 117.9, -27.41, 40  # Center of study area
        try:
            x, y, z = store.coord_to_voxel_indices(lon, lat, depth)
            print(f"   {lon}, {lat}, {depth}m → voxel indices {x}, {y}, {z}")
            
            # Convert back
            lon2, lat2, depth2 = store.voxel_indices_to_coord(x, y, z)
            print(f"   Voxel center: {lon2:.6f}, {lat2:.6f}, {depth2:.1f}m")
        except Exception as e:
            print(f"   ❌ Coordinate conversion failed: {e}")
            return False
        
        # Test point feature
        print("\n⚡ Testing point feature creation...")
        try:
            result = store.add_point_feature(
                name="test_copper_anomaly",
                longitude=117.92,
                latitude=-27.41, 
                depth=45,
                value=0.8,
                radius_m=100,
                dtype="float",
                metadata={"source": "test_drill_hole"}
            )
            print(f"   Point feature result: {result}")
            
            if result["success"]:
                print(f"   ✅ Created feature affecting {result['affected_voxels']} voxels")
                
                # Verify the layer was created
                layers = store.list_layers()
                print(f"   📋 Store now has {len(layers)} layers")
            else:
                print(f"   ❌ Point feature creation failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"   ❌ Point feature test failed: {e}")
            return False
        
        # Test line feature  
        print("\n📏 Testing line feature creation...")
        try:
            result = store.add_line_feature(
                name="test_fault",
                start_coords=(117.91, -27.407, 0),
                end_coords=(117.913, -27.406, 60),
                value=1.0,
                width_m=50,
                dtype="boolean",
                metadata={"type": "fault_zone"}
            )
            print(f"   Line feature result: {result}")
            
            if result["success"]:
                print(f"   ✅ Created line feature affecting {result['affected_voxels']} voxels")
            else:
                print(f"   ❌ Line feature creation failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"   ❌ Line feature test failed: {e}")
            return False
            
        # Test query  
        print("\n🔍 Testing spatial query...")
        try:
            query_result = store.get_voxels_in_sphere(117.92, -27.41, 45, 150)
            print(f"   Query found {len(query_result)} voxels within 150m of test point")
        except Exception as e:
            print(f"   ❌ Spatial query failed: {e}")
            return False
        
        # Test memory usage
        print("\n💾 Testing memory usage...")
        try:
            test_layer = store.get_layer_values("test_copper_anomaly")
            memory_mb = test_layer.nbytes / (1024 * 1024)
            print(f"   Layer memory: {memory_mb:.1f} MB")
            print(f"   Layer shape: {test_layer.shape}")
            print(f"   Non-zero voxels: {np.count_nonzero(test_layer)}")
        except Exception as e:
            print(f"   ❌ Memory test failed: {e}")
            return False
        
        print("\n✅ All spatial tests passed!")
        return True

if __name__ == "__main__":
    success = test_spatial_functionality()
    sys.exit(0 if success else 1)
