"""Tests for graph_ingest MCP tool (TDD: written before implementation)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from graph_to_voxel.mcp.tools.graph_tools import graph_ingest
from graph_to_voxel.mcp.workspace.models import GraphRecord
from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from tests.fixtures.toy_graphs import two_unit_horizontal


def _store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def _graph_json_text() -> str:
    return json.dumps(two_unit_horizontal().to_dict())


class TestGraphIngestText:
    def test_creates_graph_uri(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = graph_ingest(store, filename="g.json", content_text=_graph_json_text())
        assert result["graph_uri"].startswith("g2v://graph/")
        assert result["from_cache"] is False
        assert result["node_count"] > 0
        assert result["edge_count"] >= 0
        assert isinstance(result["unit_catalog"], list)

    def test_registers_graph_record(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = graph_ingest(store, filename="g.json", content_text=_graph_json_text())
        rec = store.get_resource(result["graph_uri"])
        assert isinstance(rec, GraphRecord)

    def test_graph_is_loadable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = graph_ingest(store, filename="g.json", content_text=_graph_json_text())
        loaded = store.load_graph(result["graph_uri"])
        assert loaded.unit_catalog() == two_unit_horizontal().unit_catalog()


class TestGraphIngestBase64:
    def test_base64_matches_text(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        text = _graph_json_text()
        b64 = base64.b64encode(text.encode()).decode()
        r_text = graph_ingest(store, filename="a.json", content_text=text)
        r_b64 = graph_ingest(store, filename="b.json", content_base64=b64)
        assert r_text["graph_uri"] == r_b64["graph_uri"]


class TestGraphIngestIdempotency:
    def test_second_ingest_is_cached(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        text = _graph_json_text()
        r1 = graph_ingest(store, filename="a.json", content_text=text)
        r2 = graph_ingest(store, filename="b.json", content_text=text)
        assert r1["graph_uri"] == r2["graph_uri"]
        assert r1["from_cache"] is False
        assert r2["from_cache"] is True


class TestGraphIngestValidation:
    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            graph_ingest(store, filename="bad.json", content_text="{not json")

    def test_schema_mismatch_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(Exception):
            graph_ingest(
                store,
                filename="bad.json",
                content_text=json.dumps({"nodes": [{"kind": "BogusNode", "id": "x"}], "edges": []}),
            )

    def test_neither_payload_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            graph_ingest(store, filename="x.json")

    def test_both_payloads_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        text = _graph_json_text()
        b64 = base64.b64encode(text.encode()).decode()
        with pytest.raises(ValueError):
            graph_ingest(store, filename="x.json", content_text=text, content_base64=b64)


class TestGraphIngestTags:
    def test_filename_recorded_in_tags(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = graph_ingest(store, filename="myfile.json", content_text=_graph_json_text())
        rec = store.get_resource(result["graph_uri"])
        assert rec.tags.get("source_filename") == "myfile.json"

    def test_user_tags_override_filename(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = graph_ingest(
            store,
            filename="orig.json",
            content_text=_graph_json_text(),
            tags={"source_filename": "override.json", "scenario": "alpha"},
        )
        rec = store.get_resource(result["graph_uri"])
        assert rec.tags["source_filename"] == "override.json"
        assert rec.tags["scenario"] == "alpha"

    def test_message_propagated(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = graph_ingest(
            store, filename="g.json", content_text=_graph_json_text(), message="baseline A"
        )
        rec = store.get_resource(result["graph_uri"])
        assert rec.message == "baseline A"
