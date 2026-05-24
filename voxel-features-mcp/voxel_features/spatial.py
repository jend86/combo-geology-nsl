"""Spatial extensions for VoxelStore with coordinate mapping and geometric operations."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .store import VoxelStore, GridSpec


class SpatialVoxelStore(VoxelStore):
    """
    VoxelStore extended with spatial database capabilities.
    
    Provides coordinate-based feature creation and geometric operations
    while maintaining compatibility with existing BIC scoring system.
    """
    
    def __init__(
        self,
        store_path: Path | str,
        grid: GridSpec | None = None,
        *,
        read_only_overlay: Path | str | None = None,
    ):
        super().__init__(store_path, grid, read_only_overlay=read_only_overlay)
        self._spatial_db_path = self.store_path / "spatial.db"
        self._init_spatial_database()
    
    def _init_spatial_database(self):
        """Initialize SQLite database with spatial extensions."""
        self._spatial_conn = sqlite3.connect(str(self._spatial_db_path))
        cursor = self._spatial_conn.cursor()
        
        # Enable spatial extensions (SpatiaLite)
        try:
            cursor.execute("SELECT load_extension('mod_spatialite')")
        except sqlite3.OperationalError:
            # Fallback for systems without SpatiaLite module
            pass
        
        # Create spatial metadata table for coordinate operations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spatial_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                coordinates TEXT NOT NULL,
                parameters TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._spatial_conn.commit()
    
    def coord_to_voxel_indices(self, longitude: float, latitude: float, depth: float) -> tuple[int, int, int]:
        """
        Convert geographic coordinates to voxel grid indices.
        
        Args:
            longitude: Longitude in degrees
            latitude: Latitude in degrees 
            depth: Depth in meters
            
        Returns:
            Tuple of (x_idx, y_idx, z_idx) voxel indices
            
        Raises:
            ValueError: If coordinates are outside grid bounds
        """
        grid = self.grid
        
        # Validate bounds
        if not (grid.origin[0] <= longitude <= grid.maximum[0]):
            raise ValueError(f"Longitude {longitude} outside grid bounds [{grid.origin[0]}, {grid.maximum[0]}]")
        if not (grid.origin[1] <= latitude <= grid.maximum[1]):
            raise ValueError(f"Latitude {latitude} outside grid bounds [{grid.origin[1]}, {grid.maximum[1]}]")
        if not (grid.origin[2] <= depth <= grid.maximum[2]):
            raise ValueError(f"Depth {depth} outside grid bounds [{grid.origin[2]}, {grid.maximum[2]}]")
        
        # Convert to grid indices
        x_idx = int((longitude - grid.origin[0]) / (grid.maximum[0] - grid.origin[0]) * grid.shape[0])
        y_idx = int((latitude - grid.origin[1]) / (grid.maximum[1] - grid.origin[1]) * grid.shape[1])
        z_idx = int((depth - grid.origin[2]) / (grid.maximum[2] - grid.origin[2]) * grid.shape[2])
        
        # Clamp to valid range
        x_idx = max(0, min(x_idx, grid.shape[0] - 1))
        y_idx = max(0, min(y_idx, grid.shape[1] - 1))
        z_idx = max(0, min(z_idx, grid.shape[2] - 1))
        
        return x_idx, y_idx, z_idx
    
    def voxel_indices_to_coord(self, x_idx: int, y_idx: int, z_idx: int) -> tuple[float, float, float]:
        """
        Convert voxel indices to geographic coordinates (voxel center).
        
        Args:
            x_idx: X voxel index
            y_idx: Y voxel index 
            z_idx: Z voxel index
            
        Returns:
            Tuple of (longitude, latitude, depth) coordinates
        """
        grid = self.grid
        
        # Calculate voxel center coordinates
        longitude = grid.origin[0] + (x_idx + 0.5) * (grid.maximum[0] - grid.origin[0]) / grid.shape[0]
        latitude = grid.origin[1] + (y_idx + 0.5) * (grid.maximum[1] - grid.origin[1]) / grid.shape[1]
        depth = grid.origin[2] + (z_idx + 0.5) * (grid.maximum[2] - grid.origin[2]) / grid.shape[2]
        
        return longitude, latitude, depth
    
    def meters_to_degrees(self, meters: float, latitude: float) -> tuple[float, float]:
        """
        Convert distance in meters to degrees at given latitude.
        
        Returns:
            Tuple of (longitude_degrees, latitude_degrees)
        """
        # Approximate conversion (good enough for local areas)
        lat_deg = meters / 111320  # 1 degree latitude ≈ 111.32 km
        lon_deg = meters / (111320 * math.cos(math.radians(latitude)))  # Adjust for latitude
        return lon_deg, lat_deg
    
    def get_voxels_in_sphere(self, longitude: float, latitude: float, depth: float, radius_m: float) -> list[tuple[int, int, int]]:
        """
        Get all voxel indices within a sphere of given radius.
        
        Args:
            longitude: Center longitude
            latitude: Center latitude
            depth: Center depth
            radius_m: Radius in meters
            
        Returns:
            List of (x_idx, y_idx, z_idx) tuples for affected voxels
        """
        # Convert radius to grid units
        lon_deg, lat_deg = self.meters_to_degrees(radius_m, latitude)
        depth_units = radius_m  # Depth is already in meters
        
        # Get center voxel
        center_x, center_y, center_z = self.coord_to_voxel_indices(longitude, latitude, depth)
        
        # Calculate search radius in voxel units
        grid = self.grid
        radius_x = int(lon_deg / (grid.maximum[0] - grid.origin[0]) * grid.shape[0]) + 1
        radius_y = int(lat_deg / (grid.maximum[1] - grid.origin[1]) * grid.shape[1]) + 1
        radius_z = int(depth_units / (grid.maximum[2] - grid.origin[2]) * grid.shape[2]) + 1
        
        affected_voxels = []
        
        for x in range(max(0, center_x - radius_x), min(grid.shape[0], center_x + radius_x + 1)):
            for y in range(max(0, center_y - radius_y), min(grid.shape[1], center_y + radius_y + 1)):
                for z in range(max(0, center_z - radius_z), min(grid.shape[2], center_z + radius_z + 1)):
                    # Check if voxel center is within sphere
                    voxel_lon, voxel_lat, voxel_depth = self.voxel_indices_to_coord(x, y, z)
                    
                    # Calculate distance
                    lon_dist_deg, lat_dist_deg = self.meters_to_degrees(1.0, latitude)
                    lon_dist = (voxel_lon - longitude) / lon_dist_deg
                    lat_dist = (voxel_lat - latitude) / lat_dist_deg
                    depth_dist = voxel_depth - depth
                    
                    distance = math.sqrt(lon_dist**2 + lat_dist**2 + depth_dist**2)
                    
                    if distance <= radius_m:
                        affected_voxels.append((x, y, z))
        
        return affected_voxels
    
    def add_point_feature(
        self,
        name: str,
        longitude: float,
        latitude: float,
        depth: float,
        value: float,
        radius_m: float = 100,
        dtype: Literal["float", "categorical", "boolean"] = "float",
        combination_rule: Literal["replace", "max", "add", "mean"] = "max",
        **kwargs
    ) -> dict[str, Any]:
        """
        Add a feature at a geographic point with specified radius.
        
        Args:
            name: Feature layer name
            longitude: Center longitude
            latitude: Center latitude
            depth: Center depth in meters
            value: Feature value
            radius_m: Radius of effect in meters
            dtype: Data type for the layer
            combination_rule: How to combine with existing values
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Get or create layer array
            if name in self.layer_names:
                layer_values = self.get_layer_values(name).copy()
            else:
                layer_values = np.zeros(self.grid.shape, dtype=float)
            
            # Get affected voxels
            affected_voxels = self.get_voxels_in_sphere(longitude, latitude, depth, radius_m)
            
            # Apply values based on combination rule
            for x, y, z in affected_voxels:
                if combination_rule == "replace":
                    layer_values[x, y, z] = value
                elif combination_rule == "max":
                    layer_values[x, y, z] = max(layer_values[x, y, z], value)
                elif combination_rule == "add":
                    layer_values[x, y, z] += value
                elif combination_rule == "mean":
                    layer_values[x, y, z] = (layer_values[x, y, z] + value) / 2
            
            # Update or create layer
            if name in self.layer_names:
                self.remove_layer(name)
            
            layer = self.add_layer(name=name, values=layer_values, dtype=dtype, **kwargs)
            
            # Log spatial operation
            self._log_spatial_operation("point", name, f"{longitude},{latitude},{depth}", f"radius_m={radius_m},value={value}")
            
            return {
                "success": True,
                "operation": "point_feature",
                "affected_voxels": len(affected_voxels),
                "layer_name": name,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "point_feature",
            }
    
    def add_line_feature(
        self,
        name: str,
        start_coords: tuple[float, float, float],
        end_coords: tuple[float, float, float], 
        value: float,
        width_m: float = 50,
        dtype: Literal["float", "categorical", "boolean"] = "float",
        combination_rule: Literal["replace", "max", "add", "mean"] = "max",
        **kwargs
    ) -> dict[str, Any]:
        """
        Add a feature along a line (e.g., fault, vein) with specified width.
        
        Args:
            name: Feature layer name
            start_coords: (longitude, latitude, depth) start point
            end_coords: (longitude, latitude, depth) end point
            value: Feature value
            width_m: Width of line feature in meters
            dtype: Data type for the layer
            combination_rule: How to combine with existing values
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Get or create layer array
            if name in self.layer_names:
                layer_values = self.get_layer_values(name).copy()
            else:
                layer_values = np.zeros(self.grid.shape, dtype=float)
            
            # Sample points along the line
            num_samples = max(10, int(np.linalg.norm(np.array(end_coords) - np.array(start_coords)) * 1000))  # Sample every ~1m
            
            affected_voxels = set()
            for i in range(num_samples):
                t = i / (num_samples - 1)
                
                # Interpolate position
                lon = start_coords[0] + t * (end_coords[0] - start_coords[0])
                lat = start_coords[1] + t * (end_coords[1] - start_coords[1])
                depth = start_coords[2] + t * (end_coords[2] - start_coords[2])
                
                # Add voxels in cylinder around this point
                voxels = self.get_voxels_in_sphere(lon, lat, depth, width_m / 2)
                affected_voxels.update(voxels)
            
            # Apply values
            for x, y, z in affected_voxels:
                if combination_rule == "replace":
                    layer_values[x, y, z] = value
                elif combination_rule == "max":
                    layer_values[x, y, z] = max(layer_values[x, y, z], value)
                elif combination_rule == "add":
                    layer_values[x, y, z] += value
                elif combination_rule == "mean":
                    layer_values[x, y, z] = (layer_values[x, y, z] + value) / 2
            
            # Update or create layer
            if name in self.layer_names:
                self.remove_layer(name)
                
            layer = self.add_layer(name=name, values=layer_values, dtype=dtype, **kwargs)
            
            # Log spatial operation
            coords_str = f"{start_coords[0]},{start_coords[1]},{start_coords[2]};{end_coords[0]},{end_coords[1]},{end_coords[2]}"
            self._log_spatial_operation("line", name, coords_str, f"width_m={width_m},value={value}")
            
            return {
                "success": True,
                "operation": "line_feature",
                "affected_voxels": len(affected_voxels),
                "layer_name": name,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "line_feature",
            }
    
    def _log_spatial_operation(self, operation_type: str, feature_name: str, coordinates: str, parameters: str):
        """Log spatial operation to database for debugging/tracing."""
        cursor = self._spatial_conn.cursor()
        cursor.execute(
            "INSERT INTO spatial_operations (operation_type, feature_name, coordinates, parameters) VALUES (?, ?, ?, ?)",
            (operation_type, feature_name, coordinates, parameters)
        )
        self._spatial_conn.commit()
    
    def get_spatial_operations(self) -> list[dict[str, Any]]:
        """Get history of spatial operations."""
        cursor = self._spatial_conn.cursor()
        cursor.execute("SELECT * FROM spatial_operations ORDER BY timestamp DESC")
        
        operations = []
        for row in cursor.fetchall():
            operations.append({
                "id": row[0],
                "operation_type": row[1],
                "feature_name": row[2], 
                "coordinates": row[3],
                "parameters": row[4],
                "timestamp": row[5],
            })
        
        return operations
    
    def __del__(self):
        """Clean up spatial database connection."""
        if hasattr(self, '_spatial_conn'):
            self._spatial_conn.close()
