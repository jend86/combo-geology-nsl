"""Spatial extensions for VoxelStore with coordinate mapping and geometric operations."""

from __future__ import annotations

import math
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .store import VoxelStore, GridSpec


# Per-store-path locks serialize the read-modify-write that point/line adds perform on the
# scratch layer. Each capability call builds a FRESH SpatialVoxelStore instance for the same
# scratch_dir and the capability bridge dispatches calls via asyncio.to_thread (concurrent
# threads), so without this lock concurrent adds clobber each other (lost updates) and the
# remove->re-add window is transiently empty for a racing reader. Keyed by str(store_path) so
# all instances for one episode's scratch share the lock; different episodes never contend.
_STORE_RMW_LOCKS: dict[str, threading.RLock] = {}
_STORE_RMW_LOCKS_GUARD = threading.Lock()


def _store_rmw_lock(store_path) -> threading.RLock:
    key = str(store_path)
    with _STORE_RMW_LOCKS_GUARD:
        lk = _STORE_RMW_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _STORE_RMW_LOCKS[key] = lk
        return lk


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
                source_file TEXT,
                source_excerpt TEXT,
                coordinate_source TEXT NOT NULL DEFAULT 'creative_fallback',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for column_name, column_type in (
            ("source_file", "TEXT"),
            ("source_excerpt", "TEXT"),
            ("coordinate_source", "TEXT NOT NULL DEFAULT 'creative_fallback'"),
            ("operation_group_id", "TEXT"),
            ("record_id", "TEXT"),
            ("affected_voxels", "INTEGER"),
        ):
            try:
                cursor.execute(f"ALTER TABLE spatial_operations ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        
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
        
        # A point ALWAYS occupies its containing voxel, even when radius_m is below
        # the voxel half-width. On the coarse Teniz grid (~1.7 km voxels) a 100 m
        # point sits far from every voxel CENTER, so the center-distance test below
        # would otherwise return ZERO voxels and the record would be silently dropped
        # as "outside grid bounds" (2026-06-05: ~98-100% of in-bounds records lost
        # this way). Seeding the center voxel makes sub-voxel points/lines rasterize.
        affected_voxels = [(center_x, center_y, center_z)]

        for x in range(max(0, center_x - radius_x), min(grid.shape[0], center_x + radius_x + 1)):
            for y in range(max(0, center_y - radius_y), min(grid.shape[1], center_y + radius_y + 1)):
                for z in range(max(0, center_z - radius_z), min(grid.shape[2], center_z + radius_z + 1)):
                    if (x, y, z) == (center_x, center_y, center_z):
                        continue  # already claimed above
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
    
    def _accumulate_voxels(
        self,
        name: str,
        affected_voxels,
        value: float,
        combination_rule: str,
        dtype: str,
        **kwargs,
    ):
        """Atomically accumulate ``value`` into layer ``name`` at ``affected_voxels``.

        Serialized per store_path AND disk-truthful: reads the current scratch layer from disk
        rather than the per-instance (possibly stale) in-memory index, so concurrent
        fresh-instance adds from the capability bridge's threads accumulate instead of
        clobbering each other. Replaces the previous non-atomic read -> remove_layer ->
        add_layer dance that lost updates and left a transient-empty window for racing readers.
        """
        with _store_rmw_lock(self.store_path):
            scratch_npy = self._layers_dir / f"{name}.npy"
            if scratch_npy.exists():
                layer_values = np.load(scratch_npy).copy()
            elif name in self.layer_names:  # overlay (admitted) base, if any
                layer_values = self.get_layer_values(name).copy()
            else:
                layer_values = np.zeros(self.grid.shape, dtype=float)

            for x, y, z in affected_voxels:
                if combination_rule == "replace":
                    layer_values[x, y, z] = value
                elif combination_rule == "max":
                    layer_values[x, y, z] = max(layer_values[x, y, z], value)
                elif combination_rule == "add":
                    layer_values[x, y, z] += value
                elif combination_rule == "mean":
                    layer_values[x, y, z] = (layer_values[x, y, z] + value) / 2

            # Clean re-add regardless of stale in-memory state.
            self._layers.pop(name, None)
            if scratch_npy.exists():
                scratch_npy.unlink()
            return self.add_layer(name=name, values=layer_values, dtype=dtype, **kwargs)

    def set_layer_array(
        self,
        name: str,
        values: np.ndarray,
        dtype: str = "float",
        metadata: dict[str, Any] | None = None,
        hypothesis_uri: str | None = None,
        experiment_id: str | None = None,
        source_file: str | None = None,
        source_excerpt: str | None = None,
        coordinate_source: Literal[
            "geonames", "web", "artifact", "creative_fallback"
        ] = "creative_fallback",
    ):
        """Deposit a FULL precomputed per-voxel value array as layer ``name``.

        The geometry tools flat-fill a single scalar across each shape's voxels;
        this writes an arbitrary CONTINUOUS field (the array the code phase
        computed — kernel density / IDW / distance / prospectivity) VERBATIM, with
        no flat-fill and no binarization. RMW-safe (serialized per store_path) and
        disk-truthful, mirroring ``_accumulate_voxels``: pops any stale
        in-memory/scratch layer then re-adds. Shape is validated against the grid
        (also enforced by ``add_layer``).
        """
        arr = np.asarray(values)
        if dtype == "float":
            arr = arr.astype(float, copy=False)
        if arr.shape != self.grid.shape:
            raise ValueError(
                f"Array shape {tuple(arr.shape)} does not match grid shape {tuple(self.grid.shape)}"
            )
        nz = arr[arr != 0]
        with _store_rmw_lock(self.store_path):
            self._layers.pop(name, None)
            scratch_npy = self._layers_dir / f"{name}.npy"
            if scratch_npy.exists():
                scratch_npy.unlink()
            layer = self.add_layer(
                name=name,
                values=arr,
                dtype=dtype,
                metadata=metadata,
                hypothesis_uri=hypothesis_uri,
                experiment_id=experiment_id,
            )
            with self._spatial_conn:
                self._spatial_conn.execute(
                    "DELETE FROM spatial_operations WHERE feature_name = ?",
                    (name,),
                )
                self._log_spatial_operation(
                    "array",
                    name,
                    (
                        f"grid_origin={tuple(self.grid.origin)};"
                        f"grid_maximum={tuple(self.grid.maximum)};"
                        f"shape={tuple(arr.shape)}"
                    ),
                    (
                        f"dtype={dtype},shape={tuple(arr.shape)},"
                        f"nonzero_voxels={int(nz.size)},"
                        f"value_min={float(arr.min()) if arr.size else 0.0},"
                        f"value_max={float(arr.max()) if arr.size else 0.0},"
                        f"distinct_nonzero_values={int(np.unique(nz).size) if nz.size else 0}"
                    ),
                    source_file=source_file,
                    source_excerpt=source_excerpt,
                    coordinate_source=coordinate_source,
                    affected_voxels=int(arr.size),
                    commit=False,
                )
            return layer

    @staticmethod
    def _combine_values(current, value: float, combination_rule: str):
        if combination_rule == "replace":
            return value
        if combination_rule == "max":
            return np.maximum(current, value)
        if combination_rule == "add":
            return current + value
        if combination_rule == "mean":
            return (current + value) / 2
        raise ValueError(f"Unsupported combination_rule: {combination_rule}")

    def _apply_region_into(
        self,
        layer_values: np.ndarray,
        region,
        value: float,
        combination_rule: str,
    ) -> int:
        """Fold a voxel list/set or slice tuple into an in-memory layer array."""
        if isinstance(region, tuple) and len(region) == 3 and all(isinstance(s, slice) for s in region):
            before = layer_values[region]
            affected = int(before.size)
            if affected == 0:
                return 0
            layer_values[region] = self._combine_values(before, value, combination_rule)
            return affected

        voxels = list(region or [])
        if not voxels:
            return 0
        xs, ys, zs = zip(*voxels)
        if combination_rule == "replace":
            layer_values[xs, ys, zs] = value
        elif combination_rule == "max":
            layer_values[xs, ys, zs] = np.maximum(layer_values[xs, ys, zs], value)
        elif combination_rule == "add":
            layer_values[xs, ys, zs] += value
        elif combination_rule == "mean":
            layer_values[xs, ys, zs] = (layer_values[xs, ys, zs] + value) / 2
        else:
            raise ValueError(f"Unsupported combination_rule: {combination_rule}")
        return len(voxels)

    def _coord_in_bounds(self, longitude: float, latitude: float, depth: float) -> bool:
        grid = self.grid
        return (
            grid.origin[0] <= longitude <= grid.maximum[0]
            and grid.origin[1] <= latitude <= grid.maximum[1]
            and grid.origin[2] <= depth <= grid.maximum[2]
        )

    def _clamp_coord(self, longitude: float, latitude: float, depth: float) -> tuple[float, float, float]:
        grid = self.grid
        return (
            max(grid.origin[0], min(float(longitude), grid.maximum[0])),
            max(grid.origin[1], min(float(latitude), grid.maximum[1])),
            max(grid.origin[2], min(float(depth), grid.maximum[2])),
        )

    def _box_region(
        self,
        min_longitude: float,
        min_latitude: float,
        min_depth_m: float,
        max_longitude: float,
        max_latitude: float,
        max_depth_m: float,
        *,
        bounds_policy: Literal["skip", "clip", "fail"] = "clip",
    ) -> tuple[slice, slice, slice] | None:
        grid = self.grid
        lon0, lon1 = sorted((float(min_longitude), float(max_longitude)))
        lat0, lat1 = sorted((float(min_latitude), float(max_latitude)))
        dep0, dep1 = sorted((float(min_depth_m), float(max_depth_m)))
        values = (
            (lon0, grid.origin[0], grid.maximum[0], "Longitude"),
            (lon1, grid.origin[0], grid.maximum[0], "Longitude"),
            (lat0, grid.origin[1], grid.maximum[1], "Latitude"),
            (lat1, grid.origin[1], grid.maximum[1], "Latitude"),
            (dep0, grid.origin[2], grid.maximum[2], "Depth"),
            (dep1, grid.origin[2], grid.maximum[2], "Depth"),
        )
        outside = [(value, lower, upper, label) for value, lower, upper, label in values if value < lower or value > upper]
        if outside:
            if bounds_policy == "fail":
                value, lower, upper, label = outside[0]
                raise ValueError(f"{label} {value} outside grid bounds [{lower}, {upper}]")
            if bounds_policy == "skip":
                return None

        # For clipping, no overlap means no voxels rather than a clamped point.
        if lon1 < grid.origin[0] or lon0 > grid.maximum[0]:
            return None
        if lat1 < grid.origin[1] or lat0 > grid.maximum[1]:
            return None
        if dep1 < grid.origin[2] or dep0 > grid.maximum[2]:
            return None

        lon0, lat0, dep0 = self._clamp_coord(lon0, lat0, dep0)
        lon1, lat1, dep1 = self._clamp_coord(lon1, lat1, dep1)
        ix0, iy0, iz0 = self.coord_to_voxel_indices(lon0, lat0, dep0)
        ix1, iy1, iz1 = self.coord_to_voxel_indices(lon1, lat1, dep1)
        x0, x1 = sorted((ix0, ix1))
        y0, y1 = sorted((iy0, iy1))
        z0, z1 = sorted((iz0, iz1))
        region = (
            slice(x0, min(grid.shape[0], x1 + 1)),
            slice(y0, min(grid.shape[1], y1 + 1)),
            slice(z0, min(grid.shape[2], z1 + 1)),
        )
        if any((s.stop or 0) <= (s.start or 0) for s in region):
            return None
        return region

    def _region_affected_voxels(self, region) -> int:
        if isinstance(region, tuple) and len(region) == 3 and all(isinstance(s, slice) for s in region):
            total = 1
            for s in region:
                total *= max(0, (s.stop or 0) - (s.start or 0))
            return int(total)
        return len(region or [])

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
        source_file: str | None = None,
        source_excerpt: str | None = None,
        coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
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
            # Get affected voxels, then atomically accumulate them into the layer
            # (locked + disk-truthful; see _accumulate_voxels).
            affected_voxels = self.get_voxels_in_sphere(longitude, latitude, depth, radius_m)
            layer = self._accumulate_voxels(
                name, affected_voxels, value, combination_rule, dtype, **kwargs
            )

            # Log spatial operation
            self._log_spatial_operation(
                "point",
                name,
                f"{longitude},{latitude},{depth}",
                f"radius_m={radius_m},value={value}",
                source_file=source_file,
                source_excerpt=source_excerpt,
                coordinate_source=coordinate_source,
            )
            
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
        source_file: str | None = None,
        source_excerpt: str | None = None,
        coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
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
            
            # Atomically accumulate (locked + disk-truthful; see _accumulate_voxels).
            layer = self._accumulate_voxels(
                name, affected_voxels, value, combination_rule, dtype, **kwargs
            )
            
            # Log spatial operation
            coords_str = f"{start_coords[0]},{start_coords[1]},{start_coords[2]};{end_coords[0]},{end_coords[1]},{end_coords[2]}"
            self._log_spatial_operation(
                "line",
                name,
                coords_str,
                f"width_m={width_m},value={value}",
                source_file=source_file,
                source_excerpt=source_excerpt,
                coordinate_source=coordinate_source,
            )
            
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

    def add_box_feature(
        self,
        name: str,
        min_longitude: float,
        min_latitude: float,
        min_depth_m: float,
        max_longitude: float,
        max_latitude: float,
        max_depth_m: float,
        value: float,
        dtype: Literal["float", "categorical", "boolean"] = "float",
        combination_rule: Literal["replace", "max", "add", "mean"] = "max",
        source_file: str | None = None,
        source_excerpt: str | None = None,
        coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
        **kwargs,
    ) -> dict[str, Any]:
        """Add an axis-aligned box feature, clipping partial OOB extents to the grid."""
        try:
            region = self._box_region(
                min_longitude,
                min_latitude,
                min_depth_m,
                max_longitude,
                max_latitude,
                max_depth_m,
                bounds_policy="clip",
            )
            if region is None:
                return {
                    "success": False,
                    "error": "Box does not overlap grid bounds",
                    "operation": "box_feature",
                }

            affected_voxels = self._region_affected_voxels(region)
            with _store_rmw_lock(self.store_path):
                scratch_npy = self._layers_dir / f"{name}.npy"
                if scratch_npy.exists():
                    layer_values = np.load(scratch_npy).copy()
                elif name in self.layer_names:
                    layer_values = self.get_layer_values(name).copy()
                else:
                    layer_values = np.zeros(self.grid.shape, dtype=float)
                self._apply_region_into(layer_values, region, value, combination_rule)
                self._layers.pop(name, None)
                if scratch_npy.exists():
                    scratch_npy.unlink()
                self.add_layer(name=name, values=layer_values, dtype=dtype, **kwargs)

            coords_str = (
                f"{min_longitude},{min_latitude},{min_depth_m};"
                f"{max_longitude},{max_latitude},{max_depth_m}"
            )
            self._log_spatial_operation(
                "box",
                name,
                coords_str,
                f"value={value}",
                source_file=source_file,
                source_excerpt=source_excerpt,
                coordinate_source=coordinate_source,
                affected_voxels=affected_voxels,
            )

            return {
                "success": True,
                "operation": "box_feature",
                "affected_voxels": affected_voxels,
                "layer_name": name,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": "box_feature",
            }

    # Coordinate column-name aliases agents emit instead of the canonical
    # longitude/latitude/depth_m. DataFrames from geopandas expose geometry.x/.y,
    # short forms (lon/lat) and capitalized headers (Longitude) are common; a hard
    # record["longitude"] raised KeyError('longitude') and skipped EVERY such
    # record -> empty layer (the dominant 2026-06-05 empty-layer cause:
    # 'Skipped record N: longitude'). Resolve aliases case-insensitively instead.
    # x/y are degrees here (grid CRS is EPSG:4326); a projected x/y in metres would
    # fall outside grid bounds and skip on the bounds check, same as before.
    _COORD_REQUIRED = object()
    _LON_ALIASES = ("longitude", "lon", "long", "lng", "x")
    _LAT_ALIASES = ("latitude", "lat", "y")
    _DEPTH_ALIASES = ("depth_m", "depth", "depth_meters", "z")

    @staticmethod
    def _coord_value(
        record: dict[str, Any],
        aliases: tuple[str, ...],
        default: Any = _COORD_REQUIRED,
    ) -> Any:
        lowered = {str(k).lower(): v for k, v in record.items()}
        for alias in aliases:
            val = lowered.get(alias)
            if val is not None:
                return val
        if default is not SpatialVoxelStore._COORD_REQUIRED:
            return default
        raise KeyError(aliases[0])

    def _record_region(
        self,
        record: dict[str, Any],
        kind: str,
        bounds_policy: Literal["skip", "clip", "fail"],
    ):
        if kind == "point":
            lon = float(self._coord_value(record, self._LON_ALIASES))
            lat = float(self._coord_value(record, self._LAT_ALIASES))
            # depth is the most-omitted axis; default to surface (0 m) rather than
            # dropping the record on a shallow 80 m grid.
            depth = float(self._coord_value(record, self._DEPTH_ALIASES, default=0.0))
            if not self._coord_in_bounds(lon, lat, depth):
                if bounds_policy == "fail":
                    self.coord_to_voxel_indices(lon, lat, depth)
                return None
            return self.get_voxels_in_sphere(
                lon,
                lat,
                depth,
                float(record.get("radius_m", 100.0) or 100.0),
            )
        if kind == "line":
            start = (
                float(record["start_longitude"]),
                float(record["start_latitude"]),
                float(record["start_depth_m"]),
            )
            end = (
                float(record["end_longitude"]),
                float(record["end_latitude"]),
                float(record["end_depth_m"]),
            )
            if not (self._coord_in_bounds(*start) and self._coord_in_bounds(*end)):
                if bounds_policy == "fail":
                    self.coord_to_voxel_indices(*start)
                    self.coord_to_voxel_indices(*end)
                if bounds_policy == "skip":
                    return None
                start = self._clamp_coord(*start)
                end = self._clamp_coord(*end)
            width_m = float(record.get("width_m", 50.0) or 50.0)
            num_samples = max(10, int(np.linalg.norm(np.array(end) - np.array(start)) * 1000))
            affected_voxels = set()
            for i in range(num_samples):
                t = i / (num_samples - 1)
                lon = start[0] + t * (end[0] - start[0])
                lat = start[1] + t * (end[1] - start[1])
                depth = start[2] + t * (end[2] - start[2])
                affected_voxels.update(self.get_voxels_in_sphere(lon, lat, depth, width_m / 2))
            return affected_voxels
        if kind == "box":
            return self._box_region(
                float(record["lon_min"]),
                float(record["lat_min"]),
                float(record["depth_min_m"]),
                float(record["lon_max"]),
                float(record["lat_max"]),
                float(record["depth_max_m"]),
                bounds_policy=bounds_policy,
            )
        raise ValueError(f"Unsupported geometry_kind: {kind}")

    def _record_coordinates_and_parameters(
        self,
        record: dict[str, Any],
        kind: str,
        value: float,
    ) -> tuple[str, str]:
        if kind == "point":
            lon = self._coord_value(record, self._LON_ALIASES, default=None)
            lat = self._coord_value(record, self._LAT_ALIASES, default=None)
            depth = self._coord_value(record, self._DEPTH_ALIASES, default=0.0)
            coords = f"{lon},{lat},{depth}"
            params = f"radius_m={record.get('radius_m', 100.0)},value={value}"
            return coords, params
        if kind == "line":
            coords = (
                f"{record.get('start_longitude')},{record.get('start_latitude')},{record.get('start_depth_m')};"
                f"{record.get('end_longitude')},{record.get('end_latitude')},{record.get('end_depth_m')}"
            )
            params = f"width_m={record.get('width_m', 50.0)},value={value}"
            return coords, params
        if kind == "box":
            coords = (
                f"{record.get('lon_min')},{record.get('lat_min')},{record.get('depth_min_m')};"
                f"{record.get('lon_max')},{record.get('lat_max')},{record.get('depth_max_m')}"
            )
            return coords, f"value={value}"
        return "", f"value={value}"

    def add_geometry_batch(
        self,
        name: str,
        records: list[dict[str, Any]],
        *,
        mode: Literal["replace_layer", "accumulate_layer"] = "replace_layer",
        dtype: Literal["float", "categorical", "boolean"] = "float",
        combination_rule: Literal["replace", "max", "add", "mean"] = "max",
        max_records: int = 5000,
        bounds_policy: Literal["skip", "clip", "fail"] = "skip",
        metadata: dict[str, Any] | None = None,
        hypothesis_uri: str | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Materialize point/line/box geometry records into one layer in one locked write."""
        operation_group_id = uuid.uuid4().hex
        records_seen = len(records or [])
        warnings: list[str] = []
        if mode not in {"replace_layer", "accumulate_layer"}:
            return {"success": False, "operation": "geometry_batch", "error": f"Unsupported mode: {mode}"}
        if bounds_policy not in {"skip", "clip", "fail"}:
            return {
                "success": False,
                "operation": "geometry_batch",
                "error": f"Unsupported bounds_policy: {bounds_policy}",
            }

        skipped_for_cap = max(0, records_seen - int(max_records))
        batch_records = list(records or [])[: int(max_records)]
        if skipped_for_cap:
            warnings.append(f"Skipped {skipped_for_cap} records beyond max_records={max_records}")

        try:
            with _store_rmw_lock(self.store_path):
                scratch_npy = self._layers_dir / f"{name}.npy"
                if mode == "accumulate_layer" and scratch_npy.exists():
                    layer_values = np.load(scratch_npy).copy()
                elif mode == "accumulate_layer" and name in self.layer_names:
                    layer_values = self.get_layer_values(name).copy()
                else:
                    layer_values = np.zeros(self.grid.shape, dtype=float)

                applied_records: list[dict[str, Any]] = []
                geometry_kind_counts: dict[str, int] = {}
                coordinate_source_counts: dict[str, int] = {}
                applied_values: list[float] = []
                affected_total = 0

                for idx, record in enumerate(batch_records):
                    rec = dict(record or {})
                    kind = str(rec.get("geometry_kind") or rec.get("geometry_type") or "point").strip().lower()
                    record_id = str(rec.get("record_id") or idx)
                    try:
                        raw_value = rec.get("value", 1.0)
                        try:
                            value = float(raw_value)
                        except (TypeError, ValueError):
                            # Non-numeric value (e.g. a categorical suite-name string
                            # the agent mis-mapped into `value`): treat as PRESENCE
                            # (1.0) rather than dropping the record. The voxel scorer
                            # is numeric/presence-oriented, so presence at distributed
                            # coords is the meaningful signal; dropping these produced
                            # ~85% degenerate empty layers in the 2026-06-03 run.
                            value = 1.0
                            warnings.append(
                                f"Record {record_id}: non-numeric value {raw_value!r} "
                                "coerced to presence 1.0"
                            )
                        rule = str(rec.get("combination_rule") or combination_rule)
                        region = self._record_region(rec, kind, bounds_policy)
                        if region is None:
                            warnings.append(f"Skipped record {record_id}: outside grid bounds")
                            continue
                        if self._region_affected_voxels(region) == 0:
                            warnings.append(
                                f"Skipped record {record_id}: geometry too small to "
                                "intersect any voxel (grid voxel ~1.7 km; raise radius_m/width_m)"
                            )
                            continue
                        affected = self._apply_region_into(layer_values, region, value, rule)
                        if affected == 0:
                            warnings.append(f"Skipped record {record_id}: affected no voxels")
                            continue
                        coords, params = self._record_coordinates_and_parameters(rec, kind, value)
                        coordinate_source = str(rec.get("coordinate_source") or "creative_fallback")
                        applied_records.append(
                            {
                                "kind": kind,
                                "record_id": record_id,
                                "coordinates": coords,
                                "parameters": params,
                                "source_file": rec.get("source_file") or None,
                                "source_excerpt": rec.get("source_excerpt") or None,
                                "coordinate_source": coordinate_source,
                                "affected_voxels": affected,
                            }
                        )
                        geometry_kind_counts[kind] = geometry_kind_counts.get(kind, 0) + 1
                        coordinate_source_counts[coordinate_source] = coordinate_source_counts.get(coordinate_source, 0) + 1
                        affected_total += affected
                        applied_values.append(value)
                    except Exception as exc:  # noqa: BLE001
                        if bounds_policy == "fail":
                            raise
                        warnings.append(f"Skipped record {record_id}: {exc}")

                with self._spatial_conn:
                    if mode == "replace_layer":
                        self._spatial_conn.execute(
                            "DELETE FROM spatial_operations WHERE feature_name = ?",
                            (name,),
                        )
                    self._layers.pop(name, None)
                    if scratch_npy.exists():
                        scratch_npy.unlink()
                    self.add_layer(
                        name=name,
                        values=layer_values,
                        dtype=dtype,
                        metadata=metadata,
                        hypothesis_uri=hypothesis_uri,
                        experiment_id=experiment_id,
                    )
                    for applied in applied_records:
                        self._log_spatial_operation(
                            applied["kind"],
                            name,
                            applied["coordinates"],
                            applied["parameters"],
                            source_file=applied["source_file"],
                            source_excerpt=applied["source_excerpt"],
                            coordinate_source=applied["coordinate_source"],
                            operation_group_id=operation_group_id,
                            record_id=applied["record_id"],
                            affected_voxels=applied["affected_voxels"],
                            commit=False,
                        )

            nonzero_values = layer_values[layer_values != 0]
            unique_values_approx = int(np.unique(nonzero_values).size) if nonzero_values.size else 0
            return {
                "success": True,
                "operation": "geometry_batch",
                "operation_group_id": operation_group_id,
                "layer_name": name,
                "records_seen": records_seen,
                "records_applied": len(applied_records),
                "records_skipped": records_seen - len(applied_records),
                "affected_voxels": affected_total,
                "geometry_kind_counts": dict(sorted(geometry_kind_counts.items())),
                "coordinate_source_counts": dict(sorted(coordinate_source_counts.items())),
                "value_min": min(applied_values) if applied_values else None,
                "value_max": max(applied_values) if applied_values else None,
                "unique_values_approx": unique_values_approx,
                "warnings": warnings,
            }
        except Exception as e:
            return {
                "success": False,
                "operation": "geometry_batch",
                "operation_group_id": operation_group_id,
                "layer_name": name,
                "records_seen": records_seen,
                "records_applied": 0,
                "records_skipped": records_seen,
                "affected_voxels": 0,
                "geometry_kind_counts": {},
                "coordinate_source_counts": {},
                "warnings": warnings,
                "error": str(e),
            }
    
    def _log_spatial_operation(
        self,
        operation_type: str,
        feature_name: str,
        coordinates: str,
        parameters: str,
        *,
        source_file: str | None = None,
        source_excerpt: str | None = None,
        coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
        operation_group_id: str | None = None,
        record_id: str | None = None,
        affected_voxels: int | None = None,
        commit: bool = True,
    ):
        """Log spatial operation to database for debugging/tracing."""
        cursor = self._spatial_conn.cursor()
        cursor.execute(
            """
            INSERT INTO spatial_operations (
                operation_type, feature_name, coordinates, parameters,
                source_file, source_excerpt, coordinate_source,
                operation_group_id, record_id, affected_voxels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_type,
                feature_name,
                coordinates,
                parameters,
                source_file,
                source_excerpt,
                coordinate_source,
                operation_group_id,
                record_id,
                affected_voxels,
            ),
        )
        if commit:
            self._spatial_conn.commit()
    
    def get_spatial_operations(self) -> list[dict[str, Any]]:
        """Get history of spatial operations."""
        cursor = self._spatial_conn.cursor()
        cursor.execute(
            """
            SELECT id, operation_type, feature_name, coordinates, parameters,
                   source_file, source_excerpt, coordinate_source,
                   operation_group_id, record_id, affected_voxels, timestamp
            FROM spatial_operations
            ORDER BY timestamp DESC
            """
        )
        
        operations = []
        for row in cursor.fetchall():
            operations.append({
                "id": row[0],
                "operation_type": row[1],
                "feature_name": row[2], 
                "coordinates": row[3],
                "parameters": row[4],
                "source_file": row[5],
                "source_excerpt": row[6],
                "coordinate_source": row[7],
                "operation_group_id": row[8],
                "record_id": row[9],
                "affected_voxels": row[10],
                "timestamp": row[11],
            })
        
        return operations
    
    def __del__(self):
        """Clean up spatial database connection."""
        if hasattr(self, '_spatial_conn'):
            self._spatial_conn.close()
