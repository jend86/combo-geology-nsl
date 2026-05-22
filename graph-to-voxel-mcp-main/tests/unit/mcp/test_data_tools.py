"""Tests for data MCP tools (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.data_tools import data_ingest, data_preview, data_list


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


# ---------------------------------------------------------------------------
# data_ingest
# ---------------------------------------------------------------------------

class TestDataIngest:
    def test_ingest_bytes_returns_data_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = data_ingest(store, content=b"hello world", filename="test.txt")
        assert "data_uri" in result
        assert result["data_uri"].startswith("g2v://data/")

    def test_ingest_creates_retrievable_record(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = data_ingest(store, content=b"csv,data", filename="data.csv", media_type="text/csv")
        rec = store.get_resource(result["data_uri"])
        assert rec.kind == "data"

    def test_ingest_idempotent_same_content(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        r1 = data_ingest(store, content=b"same", filename="a.txt")
        r2 = data_ingest(store, content=b"same", filename="b.txt")
        assert r1["data_uri"] == r2["data_uri"]

    def test_ingest_from_file(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        f = tmp_path / "test.bin"
        f.write_bytes(b"binary data")
        result = data_ingest(store, file_path=str(f), filename="test.bin")
        assert result["data_uri"].startswith("g2v://data/")

    def test_ingest_requires_content_or_file(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises((ValueError, TypeError)):
            data_ingest(store, filename="nodata.txt")


# ---------------------------------------------------------------------------
# data_preview
# ---------------------------------------------------------------------------

class TestDataPreview:
    def test_preview_returns_dict(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = data_ingest(store, content=b"a,b\n1,2\n3,4", filename="data.csv", media_type="text/csv")
        preview = data_preview(store, result["data_uri"])
        assert "data_uri" in preview
        assert "preview_text" in preview

    def test_preview_unknown_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            data_preview(store, "g2v://data/doesnotexist")


# ---------------------------------------------------------------------------
# data_list
# ---------------------------------------------------------------------------

class TestDataList:
    def test_list_includes_ingested_data(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        data_ingest(store, content=b"d1", filename="d1.txt")
        data_ingest(store, content=b"d2", filename="d2.txt")
        result = data_list(store)
        assert len(result["data"]) >= 2

    def test_list_empty_when_none(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = data_list(store)
        assert result["data"] == []

    def test_list_respects_limit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        for i in range(5):
            data_ingest(store, content=f"d{i}".encode(), filename=f"d{i}.txt")
        result = data_list(store, limit=3)
        assert len(result["data"]) <= 3
