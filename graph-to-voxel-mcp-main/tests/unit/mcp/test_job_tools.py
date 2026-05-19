"""Tests for job MCP tools (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.job_tools import job_status, job_cancel, job_list


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def _make_job(store: WorkspaceStore) -> str:
    return store.register_job(job_type="test_job", input_uris=[])


# ---------------------------------------------------------------------------
# job_status
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_status_returns_dict(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        job_uri = _make_job(store)
        result = job_status(store, job_uri)
        assert "job_uri" in result
        assert "status" in result
        assert "progress" in result

    def test_status_starts_queued(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        job_uri = _make_job(store)
        result = job_status(store, job_uri)
        assert result["status"] == "queued"

    def test_unknown_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            job_status(store, "g2v://job/doesnotexist")


# ---------------------------------------------------------------------------
# job_cancel
# ---------------------------------------------------------------------------

class TestJobCancel:
    def test_cancel_sets_cancelling(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        job_uri = _make_job(store)
        result = job_cancel(store, job_uri)
        assert result["status"] in ("cancelling", "cancelled")

    def test_cancel_completed_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        job_uri = _make_job(store)
        store.update_job(job_uri, status="completed")
        with pytest.raises(Exception):
            job_cancel(store, job_uri)

    def test_unknown_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            job_cancel(store, "g2v://job/doesnotexist")


# ---------------------------------------------------------------------------
# job_list
# ---------------------------------------------------------------------------

class TestJobList:
    def test_list_includes_created_jobs(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        _make_job(store)
        _make_job(store)
        result = job_list(store)
        assert len(result["jobs"]) >= 2

    def test_list_filter_by_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        j1 = _make_job(store)
        _make_job(store)
        store.update_job(j1, status="running")
        result = job_list(store, status="running")
        uris = [j["job_uri"] for j in result["jobs"]]
        assert j1 in uris

    def test_list_respects_limit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        for _ in range(5):
            _make_job(store)
        result = job_list(store, limit=3)
        assert len(result["jobs"]) <= 3
