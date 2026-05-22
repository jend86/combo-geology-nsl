"""Tests for graph MCP tools (written before implementation per TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.graph_tools import (
    graph_branch,
    graph_apply_patch,
    graph_commit,
    graph_query,
    graph_subgraph,
    graph_diff,
    graph_provenance,
    refine_commit,
)
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


# ---------------------------------------------------------------------------
# graph_branch
# ---------------------------------------------------------------------------

class TestGraphBranch:
    def test_branch_returns_scratch_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = graph_branch(store, uri)
        assert result["scratch_uri"].startswith("g2v://scratch/")
        assert result["head_rev_uri"].endswith("@rev/0")

    def test_branch_from_unknown_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            graph_branch(store, "g2v://graph/nonexistent")

    def test_scratch_starts_with_same_content(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = graph_branch(store, uri)
        loaded = store.load_graph(result["head_rev_uri"])
        assert loaded.unit_catalog() == g.unit_catalog()


# ---------------------------------------------------------------------------
# graph_apply_patch
# ---------------------------------------------------------------------------

class TestGraphApplyPatch:
    def test_update_node_metadata(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        branch = graph_branch(store, uri)
        scratch_uri = branch["scratch_uri"]

        from graph_to_voxel.schema.nodes import StratigraphicUnit
        head = store.load_graph(branch["head_rev_uri"])
        unit_node_id = next(n.id for n in head.nodes() if isinstance(n, StratigraphicUnit))

        result = graph_apply_patch(
            store,
            scratch_uri,
            [{"op": "update_node", "node_id": unit_node_id, "patch": {"metadata": {"tag": "v2"}}}],
        )
        assert result["head_rev_uri"].endswith("@rev/1")
        assert result["validation_report"]["count"] == 1

    def test_invalid_patch_raises_and_rolls_back(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        branch = graph_branch(store, uri)
        scratch_uri = branch["scratch_uri"]

        from graph_to_voxel.schema.nodes import StratigraphicUnit
        head = store.load_graph(branch["head_rev_uri"])
        unit_node_ids = [n.id for n in head.nodes() if isinstance(n, StratigraphicUnit)]

        with pytest.raises(Exception):
            graph_apply_patch(
                store,
                scratch_uri,
                [{"op": "remove_node", "node_id": nid} for nid in unit_node_ids],
            )
        # rev should still be 0
        assert store.get_scratch_record(scratch_uri).head_rev == 0

    def test_empty_patch_creates_new_rev(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        branch = graph_branch(store, uri)
        result = graph_apply_patch(store, branch["scratch_uri"], [])
        assert result["head_rev_uri"].endswith("@rev/1")


# ---------------------------------------------------------------------------
# graph_commit
# ---------------------------------------------------------------------------

class TestGraphCommit:
    def test_commit_returns_immutable_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        branch = graph_branch(store, uri)
        committed = graph_commit(store, branch["scratch_uri"])
        assert committed["graph_uri"].startswith("g2v://graph/")

    def test_committed_graph_is_loadable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        branch = graph_branch(store, uri)
        result = graph_commit(store, branch["scratch_uri"])
        loaded = store.load_graph(result["graph_uri"])
        assert loaded.unit_catalog() == g.unit_catalog()

    def test_double_commit_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        branch = graph_branch(store, uri)
        graph_commit(store, branch["scratch_uri"])
        with pytest.raises(Exception):
            graph_commit(store, branch["scratch_uri"])


# ---------------------------------------------------------------------------
# refine_commit
# ---------------------------------------------------------------------------

class TestRefineCommit:
    def test_refine_commit_returns_committed_graph_and_patch_report(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)

        from graph_to_voxel.schema.nodes import StratigraphicUnit
        unit_id = next(n.id for n in g.nodes() if isinstance(n, StratigraphicUnit))

        result = refine_commit(
            store,
            uri,
            [{"op": "update_node", "node_id": unit_id, "patch": {"metadata": {"tag": "v2"}}}],
            message="regular refine",
        )

        assert result["graph_uri"].startswith("g2v://graph/")
        assert result["scratch_uri"].startswith("g2v://scratch/")
        assert result["head_rev_uri"].endswith("@rev/1")
        assert result["validation_report"]["count"] == 1
        loaded = store.load_graph(result["graph_uri"])
        assert loaded.get_node(unit_id).metadata["tag"] == "v2"

    def test_refine_commit_rejects_invalid_patch_without_commit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        uri = store.register_graph(make_simple_graph())

        with pytest.raises(Exception):
            refine_commit(store, uri, [{"op": "remove_node", "node_id": "missing"}])


# ---------------------------------------------------------------------------
# graph_query
# ---------------------------------------------------------------------------

class TestGraphQuery:
    def test_query_all_nodes_bounded(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = graph_query(store, uri, selector={}, limit=5)
        assert "nodes" in result
        assert len(result["nodes"]) <= 5

    def test_query_by_kind(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = graph_query(store, uri, selector={"kind": "StratigraphicUnit"}, limit=100)
        assert len(result["nodes"]) > 0
        assert all(n.get("kind") == "stratigraphic_unit" for n in result["nodes"])

    def test_query_edges(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = graph_query(store, uri, selector={"include_edges": True}, limit=100)
        assert "edges" in result


# ---------------------------------------------------------------------------
# graph_subgraph
# ---------------------------------------------------------------------------

class TestGraphSubgraph:
    def test_subgraph_respects_radius(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        from graph_to_voxel.schema.nodes import StratigraphicUnit
        seed = next(n.id for n in g.nodes() if isinstance(n, StratigraphicUnit))
        result = graph_subgraph(store, uri, seed_nodes=[seed], radius=1, limit=50)
        assert seed in [n["id"] for n in result["nodes"]]

    def test_subgraph_respects_limit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        seeds = g.node_ids()[:1]
        result = graph_subgraph(store, uri, seed_nodes=seeds, radius=10, limit=2)
        assert len(result["nodes"]) <= 2


# ---------------------------------------------------------------------------
# graph_diff
# ---------------------------------------------------------------------------

class TestGraphDiff:
    def test_diff_identical_graphs(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        result = graph_diff(store, uri, uri)
        assert result["added_nodes"] == []
        assert result["removed_nodes"] == []
        assert result["changed_nodes"] == []

    def test_diff_detects_committed_edit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri_a = store.register_graph(g)
        branch = graph_branch(store, uri_a)

        from graph_to_voxel.schema.nodes import StratigraphicUnit
        head = store.load_graph(branch["head_rev_uri"])
        unit_id = next(n.id for n in head.nodes() if isinstance(n, StratigraphicUnit))
        graph_apply_patch(
            store, branch["scratch_uri"],
            [{"op": "update_node", "node_id": unit_id, "patch": {"metadata": {"tag": "v2"}}}],
        )
        result_b = graph_commit(store, branch["scratch_uri"])
        diff = graph_diff(store, uri_a, result_b["graph_uri"])
        assert unit_id in diff["changed_nodes"]


# ---------------------------------------------------------------------------
# graph_provenance
# ---------------------------------------------------------------------------

class TestGraphProvenance:
    def test_provenance_returns_dict(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        from graph_to_voxel.schema.nodes import StratigraphicUnit
        node_id = next(n.id for n in g.nodes() if isinstance(n, StratigraphicUnit))
        result = graph_provenance(store, uri, node_id)
        assert "node_id" in result
