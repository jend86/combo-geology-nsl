"""Tests for engine and voxel MCP tools (TDD — written before implementation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.engine_tools import (
    engine_run,
    engine_run_preview,
    voxel_sample,
    voxel_stats,
    voxel_export,
)
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def _tiny_field_spec() -> dict:
    """4x4x4 field spec matching two_unit_horizontal (z_interface=5.0)."""
    return {
        "grid_origin": (0.0, 0.0, 0.0),
        "grid_maximum": (10.0, 10.0, 10.0),
        "grid_shape": (4, 4, 4),
    }


# ---------------------------------------------------------------------------
# engine_run — cache hit
# ---------------------------------------------------------------------------

class TestEngineRun:
    def test_cache_hit_returns_field_uri(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        content_hash = graph_uri.split("/")[-1]

        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(
            graph_content_hash=content_hash,
            grid_origin=(0.0, 0.0, 0.0),
            grid_maximum=(10.0, 10.0, 10.0),
            grid_shape=(4, 4, 4),
        )
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = engine_run(store, graph_uri, _tiny_field_spec())
        assert result["field_uri"] == field_uri
        assert result.get("from_cache") is True

    def test_cache_miss_returns_job_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)

        result = engine_run(store, graph_uri, _tiny_field_spec())
        assert "job_uri" in result
        assert result["job_uri"].startswith("g2v://job/")

    def test_scratch_ref_is_snapshotted(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)

        result = engine_run(store, scratch_uri, _tiny_field_spec())
        # should not raise — scratch is snapshotted to immutable graph
        assert "job_uri" in result or "field_uri" in result

    def test_unknown_graph_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            engine_run(store, "g2v://graph/doesnotexist", _tiny_field_spec())


# ---------------------------------------------------------------------------
# engine_run_preview
# ---------------------------------------------------------------------------

class TestEngineRunPreview:
    def test_cache_hit_returns_field_uri(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        content_hash = graph_uri.split("/")[-1]

        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(
            graph_content_hash=content_hash,
            grid_origin=(0.0, 0.0, 0.0),
            grid_maximum=(10.0, 10.0, 10.0),
            grid_shape=(4, 4, 4),
        )
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = engine_run_preview(store, graph_uri, _tiny_field_spec())
        assert result["field_uri"] == field_uri
        assert result.get("from_cache") is True

    def test_over_budget_returns_job(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)

        # Budget of only 1 voxel forces job path
        result = engine_run_preview(
            store, graph_uri, _tiny_field_spec(), preview_budget={"max_voxels": 1}
        )
        assert "job_uri" in result

    def test_within_budget_runs_sync(self, tmp_path: Path) -> None:
        """Small grid + LoopStructural available → synchronous field returned."""
        store = make_store(tmp_path)
        g = make_simple_graph(z_interface=5.0)
        graph_uri = store.register_graph(g)

        spec = {
            "grid_origin": (0.0, 0.0, 0.0),
            "grid_maximum": (10.0, 10.0, 10.0),
            "grid_shape": (4, 4, 4),
        }
        result = engine_run_preview(store, graph_uri, spec, preview_budget={"max_voxels": 500_000})
        assert "field_uri" in result
        assert result["field_uri"].startswith("g2v://field/")
        assert result.get("from_cache") is False


# ---------------------------------------------------------------------------
# voxel_sample
# ---------------------------------------------------------------------------

class TestVoxelSample:
    def test_sample_returns_list_of_points(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = voxel_sample(store, field_uri, points=[(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)])
        assert "samples" in result
        assert len(result["samples"]) == 2
        assert "point" in result["samples"][0]
        assert "unit_probs" in result["samples"][0]
        assert "entropy" in result["samples"][0]

    def test_sample_respects_limit(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        points = [(float(i), 0.0, 0.0) for i in range(10)]
        result = voxel_sample(store, field_uri, points=points, limit=3)
        assert len(result["samples"]) <= 3

    def test_unit_probs_sum_to_one(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = voxel_sample(store, field_uri, points=[(0.5, 0.5, 0.5)])
        probs = result["samples"][0]["unit_probs"]
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_unknown_field_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            voxel_sample(store, "g2v://field/doesnotexist", points=[(0, 0, 0)])


# ---------------------------------------------------------------------------
# voxel_stats
# ---------------------------------------------------------------------------

class TestVoxelStats:
    def test_stats_returns_summary(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = voxel_stats(store, field_uri)
        assert "unit_coverage" in result
        assert "mean_entropy" in result
        assert "domain_fraction" in result
        assert "shape" in result

    def test_stats_unit_coverage_sums_to_one(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = voxel_stats(store, field_uri)
        total = sum(result["unit_coverage"].values())
        assert abs(total - 1.0) < 1e-5

    def test_unknown_field_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            voxel_stats(store, "g2v://field/doesnotexist")


# ---------------------------------------------------------------------------
# voxel_export
# ---------------------------------------------------------------------------

class TestVoxelExport:
    def test_export_zarr_returns_data_uri(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        result = voxel_export(store, field_uri, format="zarr")
        assert "data_uri" in result
        assert result["data_uri"].startswith("g2v://data/")

    def test_export_unknown_format_raises(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        from graph_to_voxel.mcp.workspace.models import FieldRunSpec
        spec = FieldRunSpec(graph_content_hash="abc", grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
        field_uri = store.register_field(minimal_voxel_field, spec)

        with pytest.raises(ValueError, match="format"):
            voxel_export(store, field_uri, format="geotiff")

    def test_unknown_field_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            voxel_export(store, "g2v://field/doesnotexist", format="zarr")
