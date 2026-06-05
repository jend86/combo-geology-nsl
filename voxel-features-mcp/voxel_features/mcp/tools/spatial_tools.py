"""Spatial command tools for geographic feature creation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from voxel_features.spatial import SpatialVoxelStore


def spatial_add_point(
    store: SpatialVoxelStore,
    name: str,
    longitude: float,
    latitude: float, 
    depth_m: float,
    value: float,
    radius_m: float = 100,
    dtype: Literal["float", "categorical", "boolean"] = "float",
    combination_rule: Literal["replace", "max", "add", "mean"] = "max",
    metadata: dict[str, Any] | None = None,
    hypothesis_uri: str | None = None,
    experiment_id: str | None = None,
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
) -> dict[str, Any]:
    """
    Add a point feature at geographic coordinates with radius of effect.
    
    Args:
        name: Unique layer name (e.g., "copper_anomaly_dhXYZ")
        longitude: Longitude in degrees 
        latitude: Latitude in degrees
        depth_m: Depth in meters below surface
        value: Feature value (e.g., 0.8 for 80% copper probability)
        radius_m: Radius of effect in meters (default: 100m)
        dtype: Data type - "float", "categorical", or "boolean"
        combination_rule: How to combine with existing values - "replace", "max", "add", "mean"
        metadata: Optional metadata dict
        hypothesis_uri: Optional URI linking to hypothesis
        experiment_id: Optional experiment ID
    
    Returns:
        Dictionary with success status and operation details
        
    Example:
        # Add high-grade copper zone at drill hole location
        spatial_add_point(
            store, "copper_grade_dh001", 117.9186, -27.4077, 45,
            value=0.85, radius_m=150, dtype="float"
        )
    """
    try:
        result = store.add_point_feature(
            name=name,
            longitude=longitude,
            latitude=latitude,
            depth=depth_m,
            value=value,
            radius_m=radius_m,
            dtype=dtype,
            combination_rule=combination_rule,
            metadata=metadata,
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
            source_file=source_file,
            source_excerpt=source_excerpt,
            coordinate_source=coordinate_source,
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_add_point",
        }


def spatial_add_line(
    store: SpatialVoxelStore,
    name: str,
    start_longitude: float,
    start_latitude: float,
    start_depth_m: float,
    end_longitude: float,
    end_latitude: float,
    end_depth_m: float,
    value: float,
    width_m: float = 50,
    dtype: Literal["float", "categorical", "boolean"] = "float",
    combination_rule: Literal["replace", "max", "add", "mean"] = "max",
    metadata: dict[str, Any] | None = None,
    hypothesis_uri: str | None = None,
    experiment_id: str | None = None,
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
) -> dict[str, Any]:
    """
    Add a line feature between two geographic points (e.g., fault, vein).
    
    Args:
        name: Unique layer name (e.g., "fault_zone_main")
        start_longitude: Start longitude in degrees
        start_latitude: Start latitude in degrees  
        start_depth_m: Start depth in meters
        end_longitude: End longitude in degrees
        end_latitude: End latitude in degrees
        end_depth_m: End depth in meters
        value: Feature value
        width_m: Width of the line feature in meters (default: 50m)
        dtype: Data type - "float", "categorical", or "boolean"
        combination_rule: How to combine with existing values
        metadata: Optional metadata dict
        hypothesis_uri: Optional URI linking to hypothesis
        experiment_id: Optional experiment ID
    
    Returns:
        Dictionary with success status and operation details
        
    Example:
        # Add fault zone from surface to 60m depth
        spatial_add_line(
            store, "fault_main", 
            117.911, -27.407, 0,  # start coords
            117.913, -27.406, 60, # end coords
            value=1.0, width_m=75, dtype="boolean"
        )
    """
    try:
        start_coords = (start_longitude, start_latitude, start_depth_m)
        end_coords = (end_longitude, end_latitude, end_depth_m)
        
        result = store.add_line_feature(
            name=name,
            start_coords=start_coords,
            end_coords=end_coords,
            value=value,
            width_m=width_m,
            dtype=dtype,
            combination_rule=combination_rule,
            metadata=metadata,
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
            source_file=source_file,
            source_excerpt=source_excerpt,
            coordinate_source=coordinate_source,
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_add_line",
        }


def spatial_add_box(
    store: SpatialVoxelStore,
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
    metadata: dict[str, Any] | None = None,
    hypothesis_uri: str | None = None,
    experiment_id: str | None = None,
    source_file: str | None = None,
    source_excerpt: str | None = None,
    coordinate_source: Literal["geonames", "web", "artifact", "creative_fallback"] = "creative_fallback",
) -> dict[str, Any]:
    """Add an axis-aligned box feature with explicit depth bounds."""
    try:
        return store.add_box_feature(
            name=name,
            min_longitude=min_longitude,
            min_latitude=min_latitude,
            min_depth_m=min_depth_m,
            max_longitude=max_longitude,
            max_latitude=max_latitude,
            max_depth_m=max_depth_m,
            value=value,
            dtype=dtype,
            combination_rule=combination_rule,
            metadata=metadata,
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
            source_file=source_file,
            source_excerpt=source_excerpt,
            coordinate_source=coordinate_source,
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_add_box",
        }


def spatial_upsert_geometry_batch(
    store: SpatialVoxelStore,
    name: str,
    records: list[dict[str, Any]],
    mode: Literal["replace_layer", "accumulate_layer"] = "replace_layer",
    dtype: Literal["float", "categorical", "boolean"] = "float",
    combination_rule: Literal["replace", "max", "add", "mean"] = "max",
    max_records: int = 5000,
    bounds_policy: Literal["skip", "clip", "fail"] = "skip",
    metadata: dict[str, Any] | None = None,
    hypothesis_uri: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Materialize many point/line/box records into one layer in one write."""
    try:
        return store.add_geometry_batch(
            name=name,
            records=records,
            mode=mode,
            dtype=dtype,
            combination_rule=combination_rule,
            max_records=max_records,
            bounds_policy=bounds_policy,
            metadata=metadata,
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_upsert_geometry_batch",
        }


def spatial_set_layer_array(
    store: SpatialVoxelStore,
    name: str,
    values: np.ndarray,
    dtype: Literal["float", "categorical", "boolean"] = "float",
    metadata: dict[str, Any] | None = None,
    hypothesis_uri: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Deposit a FULL precomputed per-voxel value array as one layer.

    Unlike the geometry tools (which flat-fill ONE scalar across each shape's
    voxels), this writes an arbitrary CONTINUOUS field — the array the code phase
    computed (kernel density, IDW/interpolation, distance-to-contact, a redox/
    grade gradient, prospectivity 0..1). Values are preserved verbatim (no
    binarization). Shape must match the grid; an existing scratch layer of the
    same name is replaced (RMW-safe).
    """
    try:
        arr = np.asarray(values)
        store.set_layer_array(
            name=name,
            values=arr,
            dtype=dtype,
            metadata=metadata,
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
        )
        nz = arr[arr != 0]
        return {
            "success": True,
            "operation": "set_layer_array",
            "layer_name": name,
            "shape": list(arr.shape),
            "nonzero_voxels": int(nz.size),
            "value_min": float(arr.min()) if arr.size else 0.0,
            "value_max": float(arr.max()) if arr.size else 0.0,
            "distinct_nonzero_values": int(np.unique(nz).size) if nz.size else 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_set_layer_array",
        }


def spatial_query_region(
    store: SpatialVoxelStore,
    center_longitude: float,
    center_latitude: float,
    center_depth_m: float,
    radius_m: float,
) -> dict[str, Any]:
    """
    Query existing feature layers within a geographic region.
    
    Args:
        center_longitude: Center longitude
        center_latitude: Center latitude
        center_depth_m: Center depth in meters
        radius_m: Query radius in meters
    
    Returns:
        Dictionary with layers and values in the region
    """
    try:
        # Get voxels in the region
        affected_voxels = store.get_voxels_in_sphere(
            center_longitude, center_latitude, center_depth_m, radius_m
        )
        
        # Sample values from all layers
        layer_samples = {}
        for layer_name in store.layer_names:
            layer_values = store.get_layer_values(layer_name)
            samples = []
            
            for x, y, z in affected_voxels[:50]:  # Limit to 50 samples
                value = float(layer_values[x, y, z])
                if value != 0:  # Only include non-zero values
                    coord = store.voxel_indices_to_coord(x, y, z)
                    samples.append({
                        "coordinates": coord,
                        "value": value,
                        "voxel_indices": (x, y, z),
                    })
            
            if samples:
                layer_samples[layer_name] = samples
        
        return {
            "success": True,
            "operation": "spatial_query_region", 
            "center": (center_longitude, center_latitude, center_depth_m),
            "radius_m": radius_m,
            "affected_voxels": len(affected_voxels),
            "layer_samples": layer_samples,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_query_region",
        }


def spatial_get_operations_log(store: SpatialVoxelStore) -> dict[str, Any]:
    """
    Get history of spatial operations for debugging/review.
    
    Returns:
        Dictionary with operation history
    """
    try:
        operations = store.get_spatial_operations()
        
        return {
            "success": True,
            "operations": operations,
            "count": len(operations),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_get_operations_log",
        }


def spatial_coord_to_voxel(
    store: SpatialVoxelStore,
    longitude: float,
    latitude: float,
    depth_m: float,
) -> dict[str, Any]:
    """
    Convert geographic coordinates to voxel indices (utility function).
    
    Args:
        longitude: Longitude in degrees
        latitude: Latitude in degrees  
        depth_m: Depth in meters
    
    Returns:
        Dictionary with voxel indices and validation
    """
    try:
        voxel_indices = store.coord_to_voxel_indices(longitude, latitude, depth_m)
        voxel_center = store.voxel_indices_to_coord(*voxel_indices)
        
        return {
            "success": True,
            "input_coordinates": (longitude, latitude, depth_m),
            "voxel_indices": voxel_indices,
            "voxel_center_coordinates": voxel_center,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": "spatial_coord_to_voxel",
        }
