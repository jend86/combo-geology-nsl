from __future__ import annotations

from datetime import UTC, datetime

import pytest

from graph_to_voxel.graph.core import Graph
from graph_to_voxel.mcp import G2VMcpServer
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Contact, StratigraphicUnit
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import PointUncertainty
from graph_to_voxel.workspace import FieldRunSpec, FileWorkspace


def _prov() -> Provenance:
    return Provenance(
        source="workspace-test",
        confidence=1.0,
        timestamp=datetime(2026, 5, 12, tzinfo=UTC),
        agent="pytest",
    )


def _unit(unit_id: str) -> dict:
    return StratigraphicUnit(
        id=unit_id,
        unit_id=unit_id,
        series_id="main",
        topology="layer",
        provenance=_prov(),
    ).model_dump(mode="json")


def _contact(contact_id: str, a: str, b: str) -> dict:
    return Contact(
        id=contact_id,
        position=(PointUncertainty(value=0.0), PointUncertainty(value=0.0), PointUncertainty(value=1.0)),
        between=(a, b),
        provenance=_prov(),
    ).model_dump(mode="json")


def _edge(edge_id: str, kind: EdgeKind, source: str, target: str) -> dict:
    return GraphEdge(
        id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_prov(),
    ).model_dump(mode="json")


def test_workspace_stores_graph_snapshots_as_content_addressed_resources(tmp_path):
    workspace = FileWorkspace(tmp_path / "workspace")
    graph = Graph.from_dict({"nodes": [_unit("u1"), _unit("u2")], "edges": []})

    graph_uri = workspace.put_graph(graph, agent_id="pytest")
    same_uri = workspace.put_graph(Graph.from_dict(graph.to_dict()), agent_id="pytest")

    assert graph_uri == same_uri
    assert graph_uri.startswith("g2v://graph/")

    description = workspace.describe(graph_uri)
    assert description["type"] == "graph"
    assert description["content_hash"] == graph_uri.rsplit("/", 1)[-1]
    assert description["summary"]["node_count"] == 2
    assert workspace.load_graph(graph_uri).node_ids() == ["u1", "u2"]
    assert workspace.query_actions(tool_name="graph.put")


def test_scratch_graph_patch_is_transactional_and_creates_immutable_revisions(tmp_path):
    workspace = FileWorkspace(tmp_path / "workspace")
    base_uri = workspace.put_graph(Graph(), agent_id="pytest")
    branch = workspace.branch_graph(base_uri, agent_id="pytest")

    result = workspace.apply_graph_patch(
        branch["scratch_uri"],
        operations=[
            {"op": "add_node", "node": _contact("c1", "u1", "u2")},
            {"op": "add_node", "node": _unit("u1")},
            {"op": "add_node", "node": _unit("u2")},
            {"op": "add_edge", "edge": _edge("e1", EdgeKind.OVERLIES, "u1", "u2")},
        ],
        agent_id="pytest",
    )

    assert result["head_rev_uri"].endswith("@rev/1")
    assert result["validation_report"]["valid"] is True
    assert set(workspace.load_graph(result["head_rev_uri"]).node_ids()) == {"c1", "u1", "u2"}

    with pytest.raises(ValueError, match="cycle|self-loop"):
        workspace.apply_graph_patch(
            branch["scratch_uri"],
            operations=[{"op": "add_edge", "edge": _edge("bad", EdgeKind.OVERLIES, "u1", "u1")}],
            agent_id="pytest",
        )

    assert workspace.scratch_head(branch["scratch_uri"]) == result["head_rev_uri"]


def test_experiment_submit_snapshots_scratch_graph_refs(tmp_path):
    workspace = FileWorkspace(tmp_path / "workspace")
    base_uri = workspace.put_graph(Graph.from_dict({"nodes": [_unit("u1")], "edges": []}), agent_id="pytest")
    branch = workspace.branch_graph(base_uri, agent_id="pytest")
    workspace.apply_graph_patch(
        branch["scratch_uri"],
        operations=[{"op": "add_node", "node": _unit("u2")}],
        agent_id="pytest",
    )

    experiment_uri = workspace.submit_experiment(
        {
            "id": "exp-1",
            "graph_refs": [branch["scratch_uri"]],
            "data_refs": [],
            "field_refs": [],
            "procedure_uri": "g2v://procedure/build_field",
            "procedure_params": {"grid": {"shape": [2, 2, 2]}},
            "success_criteria": {"criteria": [{"id": "field-builds", "metric": "status", "equals": "success"}]},
            "budget": {"time_s": 30, "memory_mb": 512, "storage_mb": 10, "gpu_count": 0},
        },
        agent_id="pytest",
    )

    body = workspace.read_resource(experiment_uri)
    frozen_ref = body["spec"]["graph_refs"][0]

    assert body["state"] == "queued"
    assert frozen_ref.startswith("g2v://graph/")
    assert body["graph_ref_resolution"][0]["original_ref"] == branch["scratch_uri"]
    assert workspace.load_graph(frozen_ref).node_ids() == ["u1", "u2"]


def test_data_ingest_is_limited_to_configured_roots_and_hashes_content(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "assays.csv"
    source.write_text("sample,cu\nA,0.2\n", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("nope\n", encoding="utf-8")

    workspace = FileWorkspace(tmp_path / "workspace", input_roots=[input_root])
    data_uri = workspace.ingest_data(source, agent_id="pytest")

    description = workspace.describe(data_uri)
    assert description["type"] == "data"
    assert description["content_hash"] == data_uri.rsplit("/", 1)[-1]
    assert description["metadata"]["source_path"] == "assays.csv"
    assert workspace.preview_data(data_uri, rows=1)["rows"] == [["sample", "cu"]]

    with pytest.raises(PermissionError):
        workspace.ingest_data(outside, agent_id="pytest")


def test_jobs_are_persistent(tmp_path):
    workspace = FileWorkspace(tmp_path / "workspace")
    graph_uri = workspace.put_graph(Graph.from_dict({"nodes": [_unit("u1")], "edges": []}), agent_id="pytest")

    job_uri = workspace.create_job(
        kind="engine.run",
        input_uris=[graph_uri],
        budget={"time_s": 10, "memory_mb": 256, "gpu_count": 0},
        agent_id="pytest",
    )
    assert workspace.job_status(job_uri)["state"] == "queued"
    assert workspace.cancel_job(job_uri, reason="test cancellation", agent_id="pytest")["state"] == "cancelled"

    reopened = FileWorkspace(tmp_path / "workspace")
    assert reopened.job_status(job_uri)["state"] == "cancelled"


def test_field_run_spec_cache_key_includes_all_output_affecting_options():
    base = FieldRunSpec(
        graph_content_hash="abc123",
        grid={"bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], "shape": [2, 2, 2]},
        bandwidth=None,
        subgrid_factor=1,
        min_membership=0.05,
        epsg=28350,
        engine_name="loopstructural",
        engine_version="v0",
        options={"batch_size": 1000},
        prior_field_hash=None,
        drop_threshold=0.5,
    )
    changed = base.model_copy(update={"bandwidth": 10.0})
    reordered_options = base.model_copy(update={"options": {"batch_size": 1000}})

    assert base.cache_key() != changed.cache_key()
    assert base.cache_key() == reordered_options.cache_key()
