"""Tests for workspace and candidate MCP tools (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.workspace_tools import workspace_describe, workspace_gc, actions_query
from graph_to_voxel.mcp.tools.candidate_tools import candidate_submit
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


# ---------------------------------------------------------------------------
# workspace_describe
# ---------------------------------------------------------------------------

class TestWorkspaceDescribe:
    def test_describe_known_graph(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = workspace_describe(store, uri)
        assert result["uri"] == uri
        assert result["kind"] == "graph"
        assert "size_bytes" in result

    def test_describe_unknown_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            workspace_describe(store, "g2v://graph/doesnotexist")


# ---------------------------------------------------------------------------
# workspace_gc
# ---------------------------------------------------------------------------

class TestWorkspaceGC:
    def test_gc_dry_run_returns_report(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = workspace_gc(store, policy={"dry_run": True})
        assert "evicted" in result
        assert "retained" in result
        assert "bytes_freed" in result

    def test_gc_dry_run_does_not_evict(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = workspace_gc(store, policy={"dry_run": True, "kinds": ["graph"]})
        # Dry run: resource should still be accessible
        store.get_resource(uri)


# ---------------------------------------------------------------------------
# actions_query
# ---------------------------------------------------------------------------

class TestActionsQuery:
    def test_query_empty_returns_empty(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = actions_query(store)
        assert "actions" in result
        assert result["actions"] == []

    def test_query_returns_logged_actions(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.log_action(agent_id="a1", role="proposer", capability="graph:read", tool="graph_query")
        result = actions_query(store)
        assert len(result["actions"]) == 1
        assert result["actions"][0]["agent_id"] == "a1"

    def test_query_filter_by_agent(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.log_action(agent_id="a1", role="proposer", capability="graph:read", tool="graph_query")
        store.log_action(agent_id="a2", role="executor", capability="engine:run", tool="engine_run")
        result = actions_query(store, agent_id="a1")
        assert all(a["agent_id"] == "a1" for a in result["actions"])

    def test_query_respects_limit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        for i in range(10):
            store.log_action(agent_id="a", role="r", capability="c", tool=f"t{i}")
        result = actions_query(store, limit=3)
        assert len(result["actions"]) <= 3


# ---------------------------------------------------------------------------
# candidate_submit
# ---------------------------------------------------------------------------

class TestCandidateSubmit:
    def test_submit_returns_candidate_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g_a = make_simple_graph()
        g_b = make_simple_graph()
        g_b.metadata["x"] = "ref_b"
        g_c = make_simple_graph()
        g_c.metadata["x"] = "candidate"

        uri_a = store.register_graph(g_a)
        uri_b = store.register_graph(g_b)
        uri_c = store.register_graph(g_c)

        result = candidate_submit(
            store,
            graph_uri=uri_c,
            reference_pair=(uri_a, uri_b),
        )
        assert "candidate_uri" in result
        assert result["candidate_uri"].startswith("g2v://candidate/")

    def test_candidate_is_retrievable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g_a = make_simple_graph()
        g_b = make_simple_graph()
        g_b.metadata["x"] = "ref_b"
        g_c = make_simple_graph()
        g_c.metadata["x"] = "candidate"

        uri_a = store.register_graph(g_a)
        uri_b = store.register_graph(g_b)
        uri_c = store.register_graph(g_c)

        result = candidate_submit(store, graph_uri=uri_c, reference_pair=(uri_a, uri_b))
        rec = store.get_resource(result["candidate_uri"])
        assert rec.kind == "candidate"

    def test_unknown_graph_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        with pytest.raises(Exception):
            candidate_submit(
                store,
                graph_uri="g2v://graph/doesnotexist",
                reference_pair=(uri, uri),
            )
