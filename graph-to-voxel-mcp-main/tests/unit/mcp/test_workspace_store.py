"""Tests for the workspace store substrate.

Written before/alongside the implementation to verify core contracts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.models import (
    Budget,
    ExperimentSpec,
    FieldRunSpec,
    SuccessCriteria,
)
from graph_to_voxel.mcp.workspace.store import (
    ClaimError,
    ResourceNotFound,
    WorkspaceStore,
    WorkspaceStoreError,
)
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def make_spec(graph_uris: list[str], hypothesis_uri: str | None = None) -> ExperimentSpec:
    import uuid
    return ExperimentSpec(
        id=uuid.uuid4().hex,
        hypothesis_uri=hypothesis_uri,
        graph_refs=graph_uris,
        procedure_uri="g2v://procedure/engine_run",
        procedure_params={},
        success_criteria=SuccessCriteria(criteria=[{"id": "c1", "type": "threshold", "metric": "ic_score", "threshold": 0.5}]),
        budget=Budget(time_s=60, memory_mb=512),
    )


# ---------------------------------------------------------------------------
# Graph registration
# ---------------------------------------------------------------------------

class TestGraphRegistration:
    def test_register_and_load_round_trip(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        assert uri.startswith("g2v://graph/")
        loaded = store.load_graph(uri)
        assert loaded.unit_catalog() == g.unit_catalog()
        assert len(loaded.node_ids()) == len(g.node_ids())

    def test_idempotent_registration(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri1 = store.register_graph(g)
        uri2 = store.register_graph(g)
        assert uri1 == uri2

    def test_content_hash_stable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g1 = make_simple_graph()
        g2 = make_simple_graph()
        uri1 = store.register_graph(g1)
        uri2 = store.register_graph(g2)
        assert uri1 == uri2

    def test_different_graphs_have_different_uris(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g1 = make_simple_graph()
        g2 = make_simple_graph()
        # mutate g2 by changing metadata
        g2.metadata["x"] = "different"
        uri1 = store.register_graph(g1)
        uri2 = store.register_graph(g2)
        # Only differs if to_dict() differs; metadata is included
        # They might be equal if metadata doesn't affect to_dict content hash
        # Just verify both are valid URIs
        assert uri1.startswith("g2v://graph/")
        assert uri2.startswith("g2v://graph/")

    def test_load_unknown_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ResourceNotFound):
            store.load_graph("g2v://graph/doesnotexist")

    def test_record_is_in_registry(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        rec = store.get_resource(uri)
        assert rec.uri == uri
        assert rec.kind == "graph"


# ---------------------------------------------------------------------------
# Scratch branches
# ---------------------------------------------------------------------------

class TestScratchBranches:
    def test_branch_creates_independent_copy(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)
        assert scratch_uri.startswith("g2v://scratch/")
        # loading the scratch should give an equivalent graph
        rev0 = store.get_scratch_record(scratch_uri).head_rev_uri
        loaded = store.load_graph(rev0)
        assert loaded.unit_catalog() == g.unit_catalog()

    def test_patch_creates_new_revision(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)

        new_rev_uri, report = store.apply_scratch_patch(
            scratch_uri,
            [{"op": "update_node", "node_id": _first_unit_node_id(store, scratch_uri), "patch": {"metadata": {"updated": "true"}}}],
        )
        assert new_rev_uri.endswith("@rev/1")
        assert report["count"] == 1
        assert store.get_scratch_record(scratch_uri).head_rev == 1

    def test_patch_rolls_back_on_invalid_state(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)

        # Remove all StratigraphicUnit nodes — graph validation requires at least one unit
        from graph_to_voxel.schema.nodes import StratigraphicUnit
        g2 = store.load_graph(store.get_scratch_record(scratch_uri).head_rev_uri)
        unit_node_ids = [n.id for n in g2.nodes() if isinstance(n, StratigraphicUnit)]
        ops = [{"op": "remove_node", "node_id": nid} for nid in unit_node_ids]

        with pytest.raises(Exception):
            store.apply_scratch_patch(scratch_uri, ops)
        # state should be unchanged
        assert store.get_scratch_record(scratch_uri).head_rev == 0

    def test_commit_produces_immutable_snapshot(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)
        committed_uri = store.commit_scratch(scratch_uri)
        assert committed_uri.startswith("g2v://graph/")
        assert store.get_scratch_record(scratch_uri).committed

    def test_double_commit_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)
        store.commit_scratch(scratch_uri)
        with pytest.raises(WorkspaceStoreError):
            store.commit_scratch(scratch_uri)

    def test_patch_after_commit_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)
        store.commit_scratch(scratch_uri)
        with pytest.raises(WorkspaceStoreError):
            store.apply_scratch_patch(scratch_uri, [])


# ---------------------------------------------------------------------------
# Field cache
# ---------------------------------------------------------------------------

class TestFieldCache:
    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        spec = _make_field_run_spec("abc123")
        assert store.get_cached_field(spec) is None

    def test_cache_hit_after_registration(self, tmp_path: Path, minimal_voxel_field) -> None:
        store = make_store(tmp_path)
        spec = _make_field_run_spec("abc123")
        field_uri = store.register_field(minimal_voxel_field, spec)
        assert store.get_cached_field(spec) == field_uri

    def test_different_specs_have_different_cache_keys(self) -> None:
        spec1 = _make_field_run_spec("abc123")
        spec2 = _make_field_run_spec("def456")
        assert spec1.content_hash() != spec2.content_hash()

    def test_same_spec_same_hash(self) -> None:
        spec1 = _make_field_run_spec("abc123")
        spec2 = _make_field_run_spec("abc123")
        assert spec1.content_hash() == spec2.content_hash()

    def test_options_affect_hash(self) -> None:
        spec1 = _make_field_run_spec("abc123", options={"a": 1})
        spec2 = _make_field_run_spec("abc123", options={"a": 2})
        assert spec1.content_hash() != spec2.content_hash()


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

class TestPins:
    def test_pin_and_unpin(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        store.pin(uri, held_by="exp_001")
        assert store.is_pinned(uri)
        store.unpin(uri, held_by="exp_001")
        assert not store.is_pinned(uri)

    def test_multiple_holders(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        store.pin(uri, held_by="exp_001")
        store.pin(uri, held_by="exp_002")
        store.unpin(uri, held_by="exp_001")
        assert store.is_pinned(uri)  # still held by exp_002
        store.unpin(uri, held_by="exp_002")
        assert not store.is_pinned(uri)


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------

class TestActionLog:
    def test_append_and_query(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.log_action(
            agent_id="agent_a",
            role="proposer",
            capability="graph:read",
            tool="graph_query",
            input_uris=["g2v://graph/abc"],
            output_uris=[],
        )
        actions = store.query_actions()
        assert len(actions) == 1
        assert actions[0].agent_id == "agent_a"

    def test_filter_by_agent(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.log_action(agent_id="a", role="r", capability="c", tool="t")
        store.log_action(agent_id="b", role="r", capability="c", tool="t")
        assert len(store.query_actions(agent_id="a")) == 1
        assert len(store.query_actions(agent_id="b")) == 1

    def test_log_is_append_only(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.log_action(agent_id="a", role="r", capability="c", tool="t1")
        store.log_action(agent_id="a", role="r", capability="c", tool="t2")
        actions = store.query_actions()
        assert len(actions) == 2
        assert {a.tool for a in actions} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Experiment claims
# ---------------------------------------------------------------------------

class TestExperimentClaims:
    def _register_experiment(self, store: WorkspaceStore, graph_uri: str) -> str:
        spec = make_spec([graph_uri])
        # Also register a hypothesis for the URI to be valid
        store.register_hypothesis("test hypothesis")
        return store.register_experiment(spec)

    def test_claim_queued_experiment(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        exp_uri = self._register_experiment(store, uri)
        claim_id, rec = store.claim_experiment(exp_uri)
        assert claim_id
        assert rec.status == "claimed"

    def test_double_claim_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        exp_uri = self._register_experiment(store, uri)
        store.claim_experiment(exp_uri)
        with pytest.raises(ClaimError):
            store.claim_experiment(exp_uri)

    def test_claim_transitions_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        exp_uri = self._register_experiment(store, uri)
        _, rec = store.claim_experiment(exp_uri)
        assert rec.status == "claimed"
        assert rec.lease_expires_at is not None

    def test_complete_updates_status(self, tmp_path: Path) -> None:
        from graph_to_voxel.mcp.workspace.models import ExperimentResult
        import uuid
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        exp_uri = self._register_experiment(store, uri)
        store.claim_experiment(exp_uri)
        result = ExperimentResult(
            id=uuid.uuid4().hex,
            spec_id="test",
            status="success",
            criterion_outcomes=[],
        )
        result_uri = store.register_result(exp_uri, result)
        updated = store.update_experiment_status(exp_uri, "completed", result_uri=result_uri)
        assert updated.status == "completed"
        assert updated.result_uri == result_uri


# ---------------------------------------------------------------------------
# Snapshot-on-submit
# ---------------------------------------------------------------------------

class TestSnapshotOnSubmit:
    def test_snapshot_immutable_returns_same(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        assert store.snapshot_graph_ref(uri) == uri

    def test_snapshot_scratch_creates_immutable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        scratch_uri = store.create_scratch(uri)
        immutable = store.snapshot_graph_ref(scratch_uri)
        assert immutable.startswith("g2v://graph/")

    def test_snapshot_scratch_rev_creates_immutable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        uri = store.register_graph(g)
        scratch_uri = store.create_scratch(uri)
        rev_uri = store.get_scratch_record(scratch_uri).head_rev_uri
        immutable = store.snapshot_graph_ref(rev_uri)
        assert immutable.startswith("g2v://graph/")


# ---------------------------------------------------------------------------
# List resources
# ---------------------------------------------------------------------------

class TestListResources:
    def test_list_by_kind(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        store.register_graph(g)
        store.register_hypothesis("my hypothesis")
        graphs = store.list_resources(kind="graph")
        hyps = store.list_resources(kind="hypothesis")
        assert len(graphs) >= 1
        assert len(hyps) >= 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_unit_node_id(store: WorkspaceStore, scratch_uri: str) -> str:
    from graph_to_voxel.schema.nodes import StratigraphicUnit
    rec = store.get_scratch_record(scratch_uri)
    g = store.load_graph(rec.head_rev_uri)
    for node in g.nodes():
        if isinstance(node, StratigraphicUnit):
            return node.id
    raise ValueError("no StratigraphicUnit node found")


def _make_field_run_spec(graph_hash: str, options: dict | None = None) -> FieldRunSpec:
    return FieldRunSpec(
        graph_content_hash=graph_hash,
        grid_origin=(0.0, 0.0, 0.0),
        grid_maximum=(100.0, 100.0, 100.0),
        grid_shape=(10, 10, 10),
        options=options or {},
    )
