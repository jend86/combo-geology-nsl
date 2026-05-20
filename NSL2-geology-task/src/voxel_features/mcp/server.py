"""MCP server for voxel feature store.

Exposes tools for:
- Feature layer CRUD
- MDL/MI scoring
- Experiment recording and crossbreeding
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from voxel_features.store import VoxelStore, GridSpec, COE_FAIRBAIRN_GRID
from voxel_features.knowledge_graph import KnowledgeGraph
from voxel_features.mcp.tools import (
    feature_create,
    feature_get,
    feature_list,
    feature_delete,
    scoring_compute_mdl,
    scoring_mutual_information,
    scoring_marginal_contribution,
    scoring_evaluate_layer,
    experiment_record,
    experiment_get,
    experiment_list_admitted,
    experiment_get_crossbreed_pairs,
    experiment_export_training,
)


# Global state (initialized on first use)
_store: VoxelStore | None = None
_kg: KnowledgeGraph | None = None


def _get_store() -> VoxelStore:
    """Get or create the voxel store."""
    global _store
    if _store is None:
        store_path = Path(os.environ.get("VFM_STORE_PATH", "/tmp/voxel-features"))
        grid_json = os.environ.get("VFM_GRID_SPEC")
        
        if grid_json:
            grid = GridSpec.from_dict(json.loads(grid_json))
        else:
            grid = COE_FAIRBAIRN_GRID
        
        _store = VoxelStore(store_path, grid)
    return _store


def _get_kg() -> KnowledgeGraph:
    """Get or create the knowledge graph."""
    global _kg
    if _kg is None:
        kg_path = Path(os.environ.get("VFM_KG_PATH", "/tmp/voxel-features/knowledge"))
        _kg = KnowledgeGraph(kg_path)
    return _kg


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
    # Experiment tools
    Tool(
        name="experiment.record",
        description="Record a completed experiment with hypothesis and results",
        inputSchema={
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "rationale": {"type": "string"},
                "data_spec": {"type": "object"},
                "code_executed": {"type": "string"},
                "result_summary": {"type": "string"},
                "feature_layer_name": {"type": "string"},
                "mdl_before": {"type": "number"},
                "mdl_after": {"type": "number"},
                "mdl_delta": {"type": "number"},
                "mutual_info": {"type": "object"},
                "admitted": {"type": "boolean"},
                "parent_experiments": {"type": "array", "items": {"type": "string"}},
                "episode_id": {"type": "string"},
                "variation_name": {"type": "string"},
            },
            "required": ["hypothesis", "rationale", "code_executed", "result_summary"],
        },
    ),
    Tool(
        name="experiment.get",
        description="Get an experiment record by ID",
        inputSchema={
            "type": "object",
            "properties": {"experiment_id": {"type": "string"}},
            "required": ["experiment_id"],
        },
    ),
    Tool(
        name="experiment.list_admitted",
        description="List all admitted experiments (for crossbreeding)",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="experiment.get_crossbreed_pairs",
        description="Get pairs of experiments for crossbreeding with prompts",
        inputSchema={
            "type": "object",
            "properties": {"max_pairs": {"type": "integer", "default": 5}},
        },
    ),
    Tool(
        name="experiment.export_training",
        description="Export experiments as JSONL training data",
        inputSchema={
            "type": "object",
            "properties": {"output_path": {"type": "string"}},
            "required": ["output_path"],
        },
    ),
]


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route tool calls to implementations."""
    store = _get_store()
    kg = _get_kg()
    
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
    
    # Experiment tools
    elif name == "experiment.record":
        return experiment_record(kg, **arguments)
    elif name == "experiment.get":
        return experiment_get(kg, **arguments)
    elif name == "experiment.list_admitted":
        return experiment_list_admitted(kg)
    elif name == "experiment.get_crossbreed_pairs":
        return experiment_get_crossbreed_pairs(kg, **arguments)
    elif name == "experiment.export_training":
        return experiment_export_training(kg, **arguments)
    
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
