"""Tests for IC scoring MCP tools (TDD — written before implementation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.ic_tools import ic_score, ic_score_from_graphs
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def _register_two_fields(store: WorkspaceStore, tmp_path: Path, minimal_voxel_field):
    """Register two graphs and two identical minimal fields for testing."""
    from graph_to_voxel.mcp.workspace.models import FieldRunSpec

    ga = make_simple_graph()
    gb = make_simple_graph()
    gb.metadata["x"] = "different"  # make content hash different

    uri_a = store.register_graph(ga)
    uri_b = store.register_graph(gb)

    hash_a = uri_a.split("/")[-1]
    hash_b = uri_b.split("/")[-1]

    spec_a = FieldRunSpec(graph_content_hash=hash_a, grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))
    spec_b = FieldRunSpec(graph_content_hash=hash_b, grid_origin=(0,0,0), grid_maximum=(1,1,1), grid_shape=(2,2,2))

    field_uri_a = store.register_field(minimal_voxel_field, spec_a)
    field_uri_b = store.register_field(minimal_voxel_field, spec_b)

    return uri_a, field_uri_a, uri_b, field_uri_b


# ---------------------------------------------------------------------------
# ic_score
# ---------------------------------------------------------------------------

class TestIcScore:
    def test_score_returns_score_dict(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        uri_a, field_a, uri_b, field_b = _register_two_fields(store, tmp_path, minimal_voxel_field)

        # candidate is uri_a, references are uri_b and uri_a (self-reference ok for test)
        result = ic_score(
            store,
            candidate_graph_uri=uri_a,
            candidate_field_uri=field_a,
            reference_a_graph_uri=uri_b,
            reference_a_field_uri=field_b,
            reference_b_graph_uri=uri_a,
            reference_b_field_uri=field_a,
        )
        assert "score_bits" in result
        assert "score_uri" in result
        assert isinstance(result["score_bits"], float)

    def test_score_uri_is_registered(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        uri_a, field_a, uri_b, field_b = _register_two_fields(store, tmp_path, minimal_voxel_field)

        result = ic_score(
            store,
            candidate_graph_uri=uri_a,
            candidate_field_uri=field_a,
            reference_a_graph_uri=uri_b,
            reference_a_field_uri=field_b,
            reference_b_graph_uri=uri_a,
            reference_b_field_uri=field_a,
        )
        rec = store.get_resource(result["score_uri"])
        assert rec.kind == "score"

    def test_score_includes_breakdown(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        uri_a, field_a, uri_b, field_b = _register_two_fields(store, tmp_path, minimal_voxel_field)

        result = ic_score(
            store,
            candidate_graph_uri=uri_a,
            candidate_field_uri=field_a,
            reference_a_graph_uri=uri_b,
            reference_a_field_uri=field_b,
            reference_b_graph_uri=uri_a,
            reference_b_field_uri=field_a,
        )
        assert "structural_bits" in result
        assert "fit_bits" in result

    def test_unknown_field_uri_raises(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        uri_a, field_a, uri_b, field_b = _register_two_fields(store, tmp_path, minimal_voxel_field)

        with pytest.raises(Exception):
            ic_score(
                store,
                candidate_graph_uri=uri_a,
                candidate_field_uri="g2v://field/doesnotexist",
                reference_a_graph_uri=uri_b,
                reference_a_field_uri=field_b,
                reference_b_graph_uri=uri_a,
                reference_b_field_uri=field_a,
            )

    def test_unknown_graph_uri_raises(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        uri_a, field_a, uri_b, field_b = _register_two_fields(store, tmp_path, minimal_voxel_field)

        with pytest.raises(Exception):
            ic_score(
                store,
                candidate_graph_uri="g2v://graph/doesnotexist",
                candidate_field_uri=field_a,
                reference_a_graph_uri=uri_b,
                reference_a_field_uri=field_b,
                reference_b_graph_uri=uri_a,
                reference_b_field_uri=field_a,
            )


# ---------------------------------------------------------------------------
# ic_score_from_graphs
# ---------------------------------------------------------------------------

class TestIcScoreFromGraphs:
    def test_always_returns_job_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        ga = make_simple_graph()
        gb = make_simple_graph()
        gb.metadata["x"] = "different"
        gc = make_simple_graph()
        gc.metadata["x"] = "third"

        uri_a = store.register_graph(ga)
        uri_b = store.register_graph(gb)
        uri_c = store.register_graph(gc)

        field_spec = {
            "grid_origin": (0.0, 0.0, 0.0),
            "grid_maximum": (10.0, 10.0, 10.0),
            "grid_shape": (4, 4, 4),
        }
        result = ic_score_from_graphs(
            store,
            candidate_graph_uri=uri_a,
            reference_a_graph_uri=uri_b,
            reference_b_graph_uri=uri_c,
            field_spec=field_spec,
        )
        assert "job_uri" in result
        assert result["job_uri"].startswith("g2v://job/")

    def test_job_is_registered(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        ga = make_simple_graph()
        gb = make_simple_graph()
        gb.metadata["x"] = "different"
        gc = make_simple_graph()
        gc.metadata["x"] = "third"

        uri_a = store.register_graph(ga)
        uri_b = store.register_graph(gb)
        uri_c = store.register_graph(gc)

        field_spec = {
            "grid_origin": (0.0, 0.0, 0.0),
            "grid_maximum": (10.0, 10.0, 10.0),
            "grid_shape": (4, 4, 4),
        }
        result = ic_score_from_graphs(
            store,
            candidate_graph_uri=uri_a,
            reference_a_graph_uri=uri_b,
            reference_b_graph_uri=uri_c,
            field_spec=field_spec,
        )
        rec = store.get_resource(result["job_uri"])
        assert rec.kind == "job"

    def test_unknown_graph_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            ic_score_from_graphs(
                store,
                candidate_graph_uri="g2v://graph/doesnotexist",
                reference_a_graph_uri="g2v://graph/doesnotexist2",
                reference_b_graph_uri="g2v://graph/doesnotexist3",
                field_spec={
                    "grid_origin": (0.0, 0.0, 0.0),
                    "grid_maximum": (10.0, 10.0, 10.0),
                    "grid_shape": (4, 4, 4),
                },
            )
