"""Smoke tests for the FastMCP server wire-up."""
from __future__ import annotations

import asyncio

import pytest


class TestServerRegistration:
    def test_tools_are_registered(self) -> None:
        from graph_to_voxel.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        expected = {
            "graph_branch", "graph_apply_patch", "graph_commit",
            "graph_query", "graph_subgraph", "graph_diff", "graph_provenance",
            "engine_run", "engine_run_preview",
            "voxel_sample", "voxel_stats", "voxel_export",
            "ic_score", "ic_score_from_graphs",
            "hypothesis_create", "hypothesis_list", "hypothesis_get", "hypothesis_update",
            "experiment_submit", "experiment_claim", "experiment_update",
            "experiment_complete", "experiment_refuse", "experiment_cancel",
            "experiment_review", "experiment_get", "experiment_list",
            "job_status", "job_cancel", "job_list",
            "data_ingest", "data_preview", "data_list",
            "workspace_describe", "workspace_gc", "actions_query",
            "candidate_submit", "seed_graph_submit", "refine_commit",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"

    def test_resource_templates_registered(self) -> None:
        from graph_to_voxel.mcp.server import mcp

        templates = asyncio.run(mcp.list_resource_templates())
        uris = {t.uriTemplate for t in templates}
        assert "g2v://graph/{graph_id}" in uris
        assert "g2v://experiment/{experiment_id}" in uris
        assert "g2v://job/{job_id}" in uris
        assert "g2v://field/{field_id}" in uris

    def test_tool_count(self) -> None:
        from graph_to_voxel.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        assert len(tools) >= 38
