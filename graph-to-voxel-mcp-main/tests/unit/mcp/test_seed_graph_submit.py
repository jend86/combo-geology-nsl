"""Tests for the seed_graph_submit MCP helper."""
from __future__ import annotations

import json
from pathlib import Path

from graph_to_voxel.mcp.tools.graph_tools import seed_graph_submit
from graph_to_voxel.mcp.workspace.models import GraphRecord
from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from tests.fixtures.toy_graphs import two_unit_horizontal


def _store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def test_seed_graph_submit_ingests_graph_and_returns_seed_uri(tmp_path: Path) -> None:
    store = _store(tmp_path)
    content_text = json.dumps(two_unit_horizontal().to_dict())

    result = seed_graph_submit(
        store,
        filename="seed.json",
        content_text=content_text,
        message="bootstrap seed",
    )

    assert result["graph_uri"].startswith("g2v://graph/")
    assert result["seed_graph_uri"] == result["graph_uri"]
    assert result["node_count"] > 0
    rec = store.get_resource(result["seed_graph_uri"])
    assert isinstance(rec, GraphRecord)
    assert rec.message == "bootstrap seed"


def test_seed_graph_submit_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    content_text = json.dumps(two_unit_horizontal().to_dict())

    first = seed_graph_submit(store, filename="a.json", content_text=content_text)
    second = seed_graph_submit(store, filename="b.json", content_text=content_text)

    assert first["seed_graph_uri"] == second["seed_graph_uri"]
    assert first["from_cache"] is False
    assert second["from_cache"] is True
