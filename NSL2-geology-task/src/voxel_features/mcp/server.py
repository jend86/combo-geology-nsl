"""MCP server for voxel feature store.

Exposes tools for:
- Feature layer CRUD
- MDL/MI scoring
- Spatial point/line operations
- Sandboxed Python execution
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from voxel_features.store import VoxelStore
from voxel_features.spatial import SpatialVoxelStore
from voxel_features.mcp.tools.feature_tools import (
    feature_create, feature_get, feature_list, feature_delete
)
from voxel_features.mcp.tools.scoring_tools import (
    scoring_compute_mdl, scoring_mutual_information,
    scoring_marginal_contribution, scoring_evaluate_layer,
    scoring_create_feature_layer
)
from voxel_features.mcp.tools.spatial_tools import (
    spatial_add_point, spatial_add_line, spatial_query_region,
    spatial_get_operations_log, spatial_coord_to_voxel
)
from voxel_features.mcp.tools.execution_tools import (
    execution_submit, execution_status, execution_results, 
    execution_cancel, execution_reset_session
)


# Global state (initialized on first use)
_store: SpatialVoxelStore | None = None


def _get_store() -> SpatialVoxelStore:
    """Get or create the spatial voxel store."""
    global _store
    if _store is None:
        store_path = Path(os.environ.get("VFM_STORE_PATH", "/tmp/voxel-features"))
        grid_json = os.environ.get("VFM_GRID_SPEC")
        
        if grid_json:
            from voxel_features.store import GridSpec
            grid = GridSpec.from_dict(json.loads(grid_json))
        else:
            from voxel_features.store import COE_FAIRBAIRN_GRID
            grid = COE_FAIRBAIRN_GRID
        
        _store = SpatialVoxelStore(store_path, grid)
    return _store


# Define tools
TOOLS = [
    # Feature tools
    Tool(
        name="feature.create",
        description="Add a new feature layer to the voxel store",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique layer name"},
                "values": {
                    "type": "array",
                    "description": "3D array of values matching grid shape (25x25x5)",
                },
                "dtype": {
                    "type": "string",
                    "enum": ["float", "categorical", "boolean"],
                    "default": "float",
                },
                "metadata": {"type": "object", "description": "Optional metadata"},
                "hypothesis_uri": {"type": "string"},
                "experiment_id": {"type": "string"},
            },
            "required": ["name", "values"],
        },
    ),
    Tool(
        name="feature.get",
        description="Get a feature layer by name",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "include_values": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="feature.list",
        description="List all feature layers with metadata",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="feature.delete",
        description="Remove a feature layer from the store",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
    # Scoring tools
    Tool(
        name="scoring.compute_mdl",
        description="Compute total MDL (bits) of the voxel store",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="scoring.mutual_information",
        description="Compute mutual information between two layers",
        inputSchema={
            "type": "object",
            "properties": {
                "layer_a": {"type": "string"},
                "layer_b": {"type": "string"},
            },
            "required": ["layer_a", "layer_b"],
        },
    ),
    Tool(
        name="scoring.marginal_contribution",
        description="Compute how much a layer contributes to compression",
        inputSchema={
            "type": "object",
            "properties": {"layer_name": {"type": "string"}},
            "required": ["layer_name"],
        },
    ),
    Tool(
        name="scoring.evaluate_layer",
        description="Evaluate adding a new layer (automated scoring, admits or rejects)",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "values": {"type": "array"},
                "dtype": {
                    "type": "string",
                    "enum": ["float", "categorical", "boolean"],
                    "default": "float",
                },
            },
            "required": ["name", "values"],
        },
    ),
    Tool(
        name="scoring.create_feature_layer",
        description="Extract spatial layer and evaluate with BIC scoring",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string", 
                    "description": "Name of existing spatial layer to evaluate"
                },
                "dtype": {
                    "type": "string",
                    "enum": ["float", "categorical", "boolean"],
                    "default": "float",
                    "description": "Data type for evaluation"
                },
            },
            "required": ["name"],
        },
    ),
    # Spatial tools
    Tool(
        name="spatial.add_point",
        description="Add a point feature at geographic coordinates with radius",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Feature layer name"},
                "longitude": {"type": "number", "description": "Longitude in degrees"},
                "latitude": {"type": "number", "description": "Latitude in degrees"},
                "depth_m": {"type": "number", "description": "Depth in meters"},
                "value": {"type": "number", "description": "Feature value"},
                "radius_m": {"type": "number", "default": 100, "description": "Radius of effect in meters"},
                "dtype": {"type": "string", "enum": ["float", "categorical", "boolean"], "default": "float"},
                "combination_rule": {"type": "string", "enum": ["replace", "max", "add", "mean"], "default": "max"},
                "metadata": {"type": "object", "description": "Optional metadata"},
                "hypothesis_uri": {"type": "string"},
                "experiment_id": {"type": "string"},
            },
            "required": ["name", "longitude", "latitude", "depth_m", "value"],
        },
    ),
    Tool(
        name="spatial.add_line",
        description="Add a line feature between two points (e.g. fault, vein)",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Feature layer name"},
                "start_longitude": {"type": "number", "description": "Start longitude in degrees"},
                "start_latitude": {"type": "number", "description": "Start latitude in degrees"},
                "start_depth_m": {"type": "number", "description": "Start depth in meters"},
                "end_longitude": {"type": "number", "description": "End longitude in degrees"},
                "end_latitude": {"type": "number", "description": "End latitude in degrees"},
                "end_depth_m": {"type": "number", "description": "End depth in meters"},
                "value": {"type": "number", "description": "Feature value"},
                "width_m": {"type": "number", "default": 50, "description": "Width of line in meters"},
                "dtype": {"type": "string", "enum": ["float", "categorical", "boolean"], "default": "float"},
                "combination_rule": {"type": "string", "enum": ["replace", "max", "add", "mean"], "default": "max"},
                "metadata": {"type": "object", "description": "Optional metadata"},
                "hypothesis_uri": {"type": "string"},
                "experiment_id": {"type": "string"},
            },
            "required": ["name", "start_longitude", "start_latitude", "start_depth_m", 
                        "end_longitude", "end_latitude", "end_depth_m", "value"],
        },
    ),
    Tool(
        name="spatial.query_region",
        description="Query existing features within a geographic region",
        inputSchema={
            "type": "object",
            "properties": {
                "center_longitude": {"type": "number", "description": "Center longitude in degrees"},
                "center_latitude": {"type": "number", "description": "Center latitude in degrees"},
                "center_depth_m": {"type": "number", "description": "Center depth in meters"},
                "radius_m": {"type": "number", "description": "Query radius in meters"},
            },
            "required": ["center_longitude", "center_latitude", "center_depth_m", "radius_m"],
        },
    ),
    Tool(
        name="spatial.coord_to_voxel",
        description="Convert geographic coordinates to voxel indices",
        inputSchema={
            "type": "object",
            "properties": {
                "longitude": {"type": "number", "description": "Longitude in degrees"},
                "latitude": {"type": "number", "description": "Latitude in degrees"},
                "depth_m": {"type": "number", "description": "Depth in meters"},
            },
            "required": ["longitude", "latitude", "depth_m"],
        },
    ),
    Tool(
        name="spatial.get_operations_log",
        description="Get history of spatial operations for debugging",
        inputSchema={"type": "object", "properties": {}},
    ),
    # Execution tools
    Tool(
        name="execution.submit",
        description="Submit code for async execution with budget control",
        inputSchema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout_s": {"type": "integer", "default": 300, "description": "Execution timeout in seconds"},
                "session_id": {"type": "string", "description": "Optional session ID for budget tracking"},
                "max_attempts": {"type": "integer", "default": 3, "description": "Maximum execution attempts for this session"},
            },
            "required": ["code"],
        },
    ),
    Tool(
        name="execution.status",
        description="Check status and progress of async execution",
        inputSchema={
            "type": "object",
            "properties": {
                "execution_id": {"type": "string", "description": "Execution ID to check"},
            },
            "required": ["execution_id"],
        },
    ),
    Tool(
        name="execution.results",
        description="Get results and artifacts from completed execution",
        inputSchema={
            "type": "object",
            "properties": {
                "execution_id": {"type": "string", "description": "Execution ID to get results for"},
            },
            "required": ["execution_id"],
        },
    ),
    Tool(
        name="execution.cancel",
        description="Cancel a running execution",
        inputSchema={
            "type": "object",
            "properties": {
                "execution_id": {"type": "string", "description": "Execution ID to cancel"},
            },
            "required": ["execution_id"],
        },
    ),
    Tool(
        name="execution.reset_session",
        description="Reset execution budget for a session (admin tool)",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to reset (optional)"},
            },
        },
    ),
]


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route tool calls to implementations."""
    store = _get_store()

    # Feature tools
    if name == "feature.create":
        return feature_create(store, **arguments)
    elif name == "feature.get":
        return feature_get(store, **arguments)
    elif name == "feature.list":
        return feature_list(store)
    elif name == "feature.delete":
        return feature_delete(store, **arguments)
    
    # Scoring tools
    elif name == "scoring.compute_mdl":
        return scoring_compute_mdl(store)
    elif name == "scoring.mutual_information":
        return scoring_mutual_information(store, **arguments)
    elif name == "scoring.marginal_contribution":
        return scoring_marginal_contribution(store, **arguments)
    elif name == "scoring.evaluate_layer":
        return scoring_evaluate_layer(store, **arguments)
    elif name == "scoring.create_feature_layer":
        return scoring_create_feature_layer(store, **arguments)

    # Spatial tools
    elif name == "spatial.add_point":
        return spatial_add_point(store, **arguments)
    elif name == "spatial.add_line":
        return spatial_add_line(store, **arguments)
    elif name == "spatial.query_region":
        return spatial_query_region(store, **arguments)
    elif name == "spatial.coord_to_voxel":
        return spatial_coord_to_voxel(store, **arguments)
    elif name == "spatial.get_operations_log":
        return spatial_get_operations_log(store)
    
    # Execution tools
    elif name == "execution.submit":
        return execution_submit(**arguments)
    elif name == "execution.status":
        return execution_status(**arguments)
    elif name == "execution.results":
        return execution_results(**arguments)
    elif name == "execution.cancel":
        return execution_cancel(**arguments)
    elif name == "execution.reset_session":
        return execution_reset_session(**arguments)
    
    else:
        return {"success": False, "error": f"Unknown tool: {name}"}


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("voxel-features-mcp")
    
    @server.list_tools()
    async def list_tools():
        return TOOLS
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        result = await handle_tool_call(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    return server


async def run_server():
    """Run the MCP server."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
