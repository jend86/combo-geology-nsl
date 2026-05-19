"""Feature layer CRUD tools."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from voxel_features.store import VoxelStore


def feature_create(
    store: VoxelStore,
    name: str,
    values: list[list[list[float]]] | list[float],
    dtype: Literal["float", "categorical", "boolean"] = "float",
    metadata: dict[str, Any] | None = None,
    hypothesis_uri: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """
    Add a new feature layer to the voxel store.
    
    Args:
        name: Unique name for the layer (e.g., "au_kriged_surface")
        values: 3D array of values matching grid shape, or flat list
        dtype: Data type - "float", "categorical", or "boolean"
        metadata: Optional metadata dict
        hypothesis_uri: Optional URI linking to hypothesis
        experiment_id: Optional experiment ID
    
    Returns:
        Layer info including content_hash
    """
    # Convert values to numpy array
    arr = np.array(values)
    
    # Reshape if flat
    if arr.ndim == 1:
        arr = arr.reshape(store.grid.shape)
    
    try:
        layer = store.add_layer(
            name=name,
            values=arr,
            dtype=dtype,
            metadata=metadata,
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
        )
        return {
            "success": True,
            "layer": layer.to_dict(),
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
        }


def feature_get(
    store: VoxelStore,
    name: str,
    include_values: bool = False,
) -> dict[str, Any]:
    """
    Get a feature layer by name.
    
    Args:
        name: Layer name
        include_values: If True, include the full values array
    
    Returns:
        Layer info, optionally with values
    """
    try:
        layer = store.get_layer(name)
        result = layer.to_dict()
        
        if include_values:
            values = store.get_layer_values(name)
            result["values"] = values.tolist()
        
        return {"success": True, "layer": result}
    except KeyError as e:
        return {"success": False, "error": str(e)}


def feature_list(store: VoxelStore) -> dict[str, Any]:
    """
    List all feature layers with metadata.
    
    Returns:
        List of layer info dicts (without values)
    """
    return {
        "success": True,
        "layers": store.list_layers(),
        "grid": store.grid.to_dict(),
    }


def feature_delete(
    store: VoxelStore,
    name: str,
) -> dict[str, Any]:
    """
    Remove a feature layer from the store.
    
    Args:
        name: Layer name to remove
    
    Returns:
        Success status
    """
    try:
        store.remove_layer(name)
        return {"success": True, "removed": name}
    except KeyError as e:
        return {"success": False, "error": str(e)}
