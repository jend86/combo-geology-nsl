#!/usr/bin/env python3
"""
Test script to validate Kazakhstan grid coordinates and voxel mapping.
"""

import numpy as np

# Kazakhstan Teniz Basin grid specification
KAZAKHSTAN_TENIZ_GRID = {
    "origin": [66.5, 49.5, 0.0],      # 66°30'E, 49°30'N, 0m depth
    "maximum": [71.5, 52.5, 80.0],    # 71°30'E, 52°30'N, 80m depth  
    "shape": [200, 200, 8],            # ~1.75km x 1.75km x 10m resolution
    "crs": "EPSG:4326",
}

def coord_to_voxel(longitude, latitude, depth_m, grid_spec):
    """Convert geographic coordinates to voxel indices."""
    origin = grid_spec["origin"]
    maximum = grid_spec["maximum"]
    shape = grid_spec["shape"]
    
    # Calculate fractional position within grid bounds
    lon_frac = (longitude - origin[0]) / (maximum[0] - origin[0])
    lat_frac = (latitude - origin[1]) / (maximum[1] - origin[1])
    depth_frac = (depth_m - origin[2]) / (maximum[2] - origin[2])
    
    # Convert to voxel indices
    lon_idx = int(lon_frac * shape[0])
    lat_idx = int(lat_frac * shape[1])
    depth_idx = int(depth_frac * shape[2])
    
    # Check bounds
    in_bounds = (
        0 <= lon_idx < shape[0] and
        0 <= lat_idx < shape[1] and
        0 <= depth_idx < shape[2]
    )
    
    return (lon_idx, lat_idx, depth_idx), in_bounds

def test_kazakhstan_grid():
    """Test Kazakhstan coordinate mapping."""
    
    print("🔬 Kazakhstan Grid Mapping Test")
    print("="*50)
    
    # Grid info
    origin = KAZAKHSTAN_TENIZ_GRID["origin"]
    maximum = KAZAKHSTAN_TENIZ_GRID["maximum"]
    shape = KAZAKHSTAN_TENIZ_GRID["shape"]
    
    print(f"Grid bounds: {origin[0]:.1f}°-{maximum[0]:.1f}°E, {origin[1]:.1f}°-{maximum[1]:.1f}°N, {origin[2]:.0f}-{maximum[2]:.0f}m")
    print(f"Grid shape: {shape[0]}×{shape[1]}×{shape[2]} = {np.prod(shape):,} voxels")
    
    # Calculate resolution
    lon_res = (maximum[0] - origin[0]) / shape[0]  # degrees per voxel
    lat_res = (maximum[1] - origin[1]) / shape[1]  # degrees per voxel
    depth_res = (maximum[2] - origin[2]) / shape[2]  # meters per voxel
    
    # Convert degrees to km (approximately)
    km_per_degree_lon = 111.32 * np.cos(np.radians((origin[1] + maximum[1]) / 2))
    km_per_degree_lat = 110.54
    
    print(f"Voxel resolution: {lon_res*km_per_degree_lon:.2f}km × {lat_res*km_per_degree_lat:.2f}km × {depth_res:.0f}m")
    print()
    
    # Test coordinates from Kazakhstan data
    test_coords = [
        # Grid corners
        (origin[0], origin[1], 0.0, "Grid origin (SW corner)"),
        (maximum[0], maximum[1], maximum[2], "Grid maximum (NE corner)"),
        
        # Grid center
        ((origin[0] + maximum[0])/2, (origin[1] + maximum[1])/2, 40.0, "Grid center"),
        
        # Sample Kazakhstan prospect coordinates from the data we saw
        (68.046, 51.997, 10.0, "Sovetskoe prospect"),
        (68.029, 51.963, 10.0, "Kirei prospect"),
        (68.358, 51.995, 10.0, "Teniz Basin prospect"),
        
        # Edge cases
        (66.5, 49.5, 0.0, "Exact origin"),
        (71.5, 52.5, 80.0, "Exact maximum"),
        (69.0, 51.0, 40.0, "Mid-basin"),
        
        # Out of bounds tests
        (66.0, 49.0, 0.0, "Out of bounds (too far SW)"),
        (72.0, 53.0, 0.0, "Out of bounds (too far NE)"),
    ]
    
    print("🎯 Coordinate Mapping Tests:")
    print("-" * 80)
    print(f"{'Coordinate':25s} {'Voxel (i,j,k)':15s} {'In Bounds':10s} {'Description':20s}")
    print("-" * 80)
    
    for lon, lat, depth, desc in test_coords:
        voxel, in_bounds = coord_to_voxel(lon, lat, depth, KAZAKHSTAN_TENIZ_GRID)
        status = "✅ YES" if in_bounds else "❌ NO"
        coord_str = f"({lon:.3f},{lat:.3f},{depth:.0f})"
        voxel_str = f"({voxel[0]},{voxel[1]},{voxel[2]})"
        
        print(f"{coord_str:25s} {voxel_str:15s} {status:10s} {desc:20s}")
    
    print("-" * 80)
    
    # Summary statistics
    in_bounds_coords = [(lon, lat, depth) for lon, lat, depth, desc in test_coords 
                       if coord_to_voxel(lon, lat, depth, KAZAKHSTAN_TENIZ_GRID)[1]]
    
    print(f"\n📊 Test Results:")
    print(f"  • Total coordinates tested: {len(test_coords)}")
    print(f"  • In bounds: {len(in_bounds_coords)}")
    print(f"  • Grid coverage area: {(maximum[0]-origin[0]) * km_per_degree_lon * (maximum[1]-origin[1]) * km_per_degree_lat:,.0f} km²")
    print(f"  • Voxel volume: {lon_res*km_per_degree_lon * lat_res*km_per_degree_lat * depth_res/1000:.3f} km³")
    
    print("\n✅ Kazakhstan grid mapping test completed!")

if __name__ == "__main__":
    test_kazakhstan_grid()
