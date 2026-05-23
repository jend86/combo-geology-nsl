"""Scoring tools for MDL and Mutual Information."""

from __future__ import annotations

from typing import Any, Literal
import time

import numpy as np

from voxel_features.store import VoxelStore
from voxel_features import scoring


def scoring_compute_mdl(store: VoxelStore) -> dict[str, Any]:
    """
    Compute the current MDL (Minimum Description Length) of the voxel store.
    
    Returns:
        mdl: Total bits needed to describe the store
        n_layers: Number of feature layers
        n_voxels: Total voxels in grid
    """
    mdl = scoring.compute_mdl(store)
    return {
        "success": True,
        "mdl": mdl,
        "n_layers": len(store.layer_names),
        "n_voxels": store.grid.n_voxels,
    }


def scoring_mutual_information(
    store: VoxelStore,
    layer_a: str,
    layer_b: str,
) -> dict[str, Any]:
    """
    Compute mutual information between two layers.
    
    Args:
        layer_a: First layer name
        layer_b: Second layer name
    
    Returns:
        mutual_info: Bits of shared information
    """
    try:
        mi = scoring.mutual_information(store, layer_a, layer_b)
        return {
            "success": True,
            "layer_a": layer_a,
            "layer_b": layer_b,
            "mutual_info": mi,
        }
    except KeyError as e:
        return {"success": False, "error": str(e)}


def scoring_marginal_contribution(
    store: VoxelStore,
    layer_name: str,
) -> dict[str, Any]:
    """
    Compute how much a layer contributes to compression.
    
    Positive value = layer is useful (MDL would increase without it)
    Negative value = layer is harmful (MDL would decrease without it)
    
    Args:
        layer_name: Layer to evaluate
    
    Returns:
        contribution: MDL change if layer were removed
    """
    try:
        contribution = scoring.marginal_contribution(store, layer_name)
        return {
            "success": True,
            "layer_name": layer_name,
            "contribution": contribution,
            "useful": contribution > 0,
        }
    except KeyError as e:
        return {"success": False, "error": str(e)}


def scoring_evaluate_layer(
    store: VoxelStore,
    name: str,
    values: list[list[list[float]]] | list[float],
    dtype: Literal["float", "categorical", "boolean"] = "float",
) -> dict[str, Any]:
    """
    Evaluate adding a new layer to the store.
    
    This is the automated scoring step that runs after Translate phase.
    The layer is added if it improves compression, rolled back if not.
    
    Args:
        name: Proposed layer name
        values: 3D array of values
        dtype: Data type
    
    Returns:
        mdl_before: MDL before adding layer
        mdl_after: MDL after adding layer
        mdl_delta: Change in MDL (negative = improved)
        mutual_info: MI with each existing layer
        admitted: Whether layer was kept
    """
    # Convert values to numpy array
    arr = np.array(values)
    if arr.ndim == 1:
        arr = arr.reshape(store.grid.shape)
    
    # Validate shape
    if arr.shape != store.grid.shape:
        return {
            "success": False,
            "error": f"Shape {arr.shape} does not match grid {store.grid.shape}",
        }
    
    result = scoring.evaluate_new_layer(
        store=store,
        layer_name=name,
        layer_values=arr,
        layer_dtype=dtype,
    )
    
    return {
        "success": True,
        **result,
    }


def scoring_create_feature_layer(
    store: VoxelStore,
    name: str,
    dtype: Literal["float", "categorical", "boolean"] = "float",
) -> dict[str, Any]:
    """
    Extract spatial layer and evaluate with BIC scoring.
    
    This extracts values from an existing spatial layer created by spatial_add_point 
    or spatial_add_line operations, then evaluates whether it improves compression.
    
    Layer names are automatically made unique by appending timestamps to prevent
    collisions when models accidentally give different useful layers the same name.
    
    Args:
        name: Base name for spatial layer to evaluate
        dtype: Data type for evaluation
    
    Returns:
        bic_delta: Change in BIC (negative = improved)
        admitted: Whether layer was kept in store
        mutual_info: MI with each existing layer
        layer_name: Unique name assigned to the evaluated layer
    """
    try:
        # Check if this is a spatial store that supports get_layer_values
        if not hasattr(store, 'get_layer_values'):
            return {
                "success": False,
                "error": f"Store does not support spatial layer extraction",
            }
        
        # Check if layer exists
        if not hasattr(store, 'layer_names') or name not in store.layer_names:
            return {
                "success": False,
                "error": f"Spatial layer '{name}' not found in store",
            }
        
        # Extract layer values from spatial operations
        arr = store.get_layer_values(name)
        
        # Create unique layer name with timestamp to prevent collisions
        timestamp = int(time.time() * 1000)  # millisecond precision
        unique_name = f"{name}_{timestamp}"
        
        # Remove the original spatial layer (temporary)
        store.remove_layer(name)
        
        # Run BIC evaluation with unique name
        result = scoring.evaluate_new_layer(
            store=store,
            layer_name=unique_name,
            layer_values=arr,
            layer_dtype=dtype,
        )
        
        # Update result to reflect the unique name used
        result["layer_name"] = unique_name
        
        return {
            "success": True,
            **result,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to evaluate spatial layer '{name}': {str(e)}",
        }
