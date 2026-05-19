"""Tests for hypothesis MCP tools (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.hypothesis_tools import (
    hypothesis_create,
    hypothesis_list,
    hypothesis_get,
    hypothesis_update,
)
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


# ---------------------------------------------------------------------------
# hypothesis_create
# ---------------------------------------------------------------------------

class TestHypothesisCreate:
    def test_create_returns_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = hypothesis_create(store, statement="Rocks are layered.")
        assert "hypothesis_uri" in result
        assert result["hypothesis_uri"].startswith("g2v://hypothesis/")

    def test_create_with_graph_refs(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        result = hypothesis_create(
            store,
            statement="Rocks are layered.",
            graph_refs=[graph_uri],
        )
        assert result["hypothesis_uri"].startswith("g2v://hypothesis/")

    def test_created_record_is_retrievable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = hypothesis_create(store, statement="Test hypothesis.")
        rec = store.get_resource(result["hypothesis_uri"])
        assert rec.kind == "hypothesis"

    def test_statement_is_stored(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = hypothesis_create(store, statement="My unique statement 12345.")
        from graph_to_voxel.mcp.workspace.models import HypothesisRecord
        rec = store.get_resource(result["hypothesis_uri"])
        assert isinstance(rec, HypothesisRecord)
        assert rec.statement == "My unique statement 12345."

    def test_create_with_rationale(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = hypothesis_create(
            store,
            statement="Test.",
            rationale="Because of observations X, Y, Z.",
        )
        from graph_to_voxel.mcp.workspace.models import HypothesisRecord
        rec = store.get_resource(result["hypothesis_uri"])
        assert isinstance(rec, HypothesisRecord)
        assert rec.rationale == "Because of observations X, Y, Z."


# ---------------------------------------------------------------------------
# hypothesis_list
# ---------------------------------------------------------------------------

class TestHypothesisList:
    def test_list_returns_created_hypotheses(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hypothesis_create(store, statement="H1")
        hypothesis_create(store, statement="H2")
        result = hypothesis_list(store)
        assert "hypotheses" in result
        assert len(result["hypotheses"]) >= 2

    def test_list_empty_when_none(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = hypothesis_list(store)
        assert result["hypotheses"] == []

    def test_list_respects_limit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        for i in range(5):
            hypothesis_create(store, statement=f"H{i}")
        result = hypothesis_list(store, limit=3)
        assert len(result["hypotheses"]) <= 3


# ---------------------------------------------------------------------------
# hypothesis_get
# ---------------------------------------------------------------------------

class TestHypothesisGet:
    def test_get_returns_record(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        created = hypothesis_create(store, statement="Get this one.")
        result = hypothesis_get(store, created["hypothesis_uri"])
        assert result["statement"] == "Get this one."
        assert result["hypothesis_uri"] == created["hypothesis_uri"]

    def test_get_unknown_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            hypothesis_get(store, "g2v://hypothesis/doesnotexist")


# ---------------------------------------------------------------------------
# hypothesis_update
# ---------------------------------------------------------------------------

class TestHypothesisUpdate:
    def test_update_adds_graph_ref(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        created = hypothesis_create(store, statement="Base hypothesis.")

        result = hypothesis_update(
            store,
            created["hypothesis_uri"],
            graph_refs=[graph_uri],
        )
        assert graph_uri in result["graph_refs"]

    def test_update_statement(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        created = hypothesis_create(store, statement="Old statement.")
        result = hypothesis_update(
            store,
            created["hypothesis_uri"],
            statement="New statement.",
        )
        assert result["statement"] == "New statement."
