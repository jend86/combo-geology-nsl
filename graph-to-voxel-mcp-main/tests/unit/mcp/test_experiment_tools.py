"""Tests for experiment MCP tools (TDD)."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.mcp.tools.experiment_tools import (
    experiment_submit,
    experiment_claim,
    experiment_update,
    experiment_complete,
    experiment_refuse,
    experiment_cancel,
    experiment_review,
    experiment_get,
    experiment_list,
)
from tests.fixtures.toy_graphs import two_unit_horizontal as make_simple_graph


def make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")


def _make_experiment(store: WorkspaceStore) -> tuple[str, str]:
    """Return (hypothesis_uri, graph_uri)."""
    g = make_simple_graph()
    graph_uri = store.register_graph(g)
    hyp_uri = store.register_hypothesis("Test hypothesis")
    return hyp_uri, graph_uri


# ---------------------------------------------------------------------------
# experiment_submit
# ---------------------------------------------------------------------------

class TestExperimentSubmit:
    def test_submit_returns_experiment_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        result = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[{"id": "c1", "type": "threshold", "metric": "ic_score", "threshold": 0.5}],
            hypothesis_uri=hyp_uri,
        )
        assert "experiment_uri" in result
        assert result["experiment_uri"].startswith("g2v://experiment/")

    def test_submit_queued_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        result = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[{"id": "c1", "type": "threshold", "metric": "ic_score", "threshold": 0.5}],
        )
        rec = store.get_experiment_record(result["experiment_uri"])
        assert rec.status == "queued"

    def test_submit_snapshots_scratch_refs(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        scratch_uri = store.create_scratch(graph_uri)

        result = experiment_submit(
            store,
            graph_refs=[scratch_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        rec = store.get_experiment_record(result["experiment_uri"])
        # scratch refs should have been snapshotted to immutable graph URIs
        assert all(ref.startswith("g2v://graph/") for ref in rec.spec.graph_refs)

    def test_unknown_graph_uri_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(Exception):
            experiment_submit(
                store,
                graph_refs=["g2v://graph/doesnotexist"],
                procedure_uri="g2v://procedure/engine_run",
                procedure_params={},
                success_criteria=[],
            )


# ---------------------------------------------------------------------------
# experiment_claim
# ---------------------------------------------------------------------------

class TestExperimentClaim:
    def test_claim_transitions_to_claimed(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        result = experiment_claim(store, submitted["experiment_uri"])
        assert result["status"] == "claimed"
        assert "claim_id" in result

    def test_double_claim_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        experiment_claim(store, submitted["experiment_uri"])
        with pytest.raises(Exception):
            experiment_claim(store, submitted["experiment_uri"])


# ---------------------------------------------------------------------------
# experiment_update / complete / refuse / cancel
# ---------------------------------------------------------------------------

class TestExperimentLifecycle:
    def _submit_and_claim(self, store: WorkspaceStore) -> tuple[str, str]:
        g = make_simple_graph()
        graph_uri = store.register_graph(g)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        exp_uri = submitted["experiment_uri"]
        claim_result = experiment_claim(store, exp_uri)
        return exp_uri, claim_result["claim_id"]

    def test_update_status_to_running(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        exp_uri, _ = self._submit_and_claim(store)
        result = experiment_update(store, exp_uri, status="running")
        assert result["status"] == "running"

    def test_complete_sets_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        exp_uri, _ = self._submit_and_claim(store)
        result = experiment_complete(
            store,
            exp_uri,
            outcome="success",
            criterion_outcomes=[],
        )
        assert result["status"] == "completed"

    def test_refuse_sets_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        exp_uri, _ = self._submit_and_claim(store)
        result = experiment_refuse(store, exp_uri, reason="Budget exceeded")
        assert result["status"] == "refused"

    def test_cancel_sets_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        result = experiment_cancel(store, submitted["experiment_uri"])
        assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# experiment_review
# ---------------------------------------------------------------------------

class TestExperimentReview:
    def test_review_returns_review_uri(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        exp_uri = submitted["experiment_uri"]
        experiment_claim(store, exp_uri)
        experiment_complete(store, exp_uri, outcome="success", criterion_outcomes=[])

        result = experiment_review(
            store,
            exp_uri,
            status="accepted",
            notes="Looks good",
        )
        assert "review_uri" in result
        assert result["review_uri"].startswith("g2v://review/")


# ---------------------------------------------------------------------------
# experiment_get / list
# ---------------------------------------------------------------------------

class TestExperimentGetList:
    def test_get_returns_experiment_dict(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        result = experiment_get(store, submitted["experiment_uri"])
        assert result["experiment_uri"] == submitted["experiment_uri"]
        assert "status" in result

    def test_list_returns_submitted_experiments(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        for _ in range(3):
            experiment_submit(
                store,
                graph_refs=[graph_uri],
                procedure_uri="g2v://procedure/engine_run",
                procedure_params={},
                success_criteria=[],
            )
        result = experiment_list(store)
        assert len(result["experiments"]) >= 3

    def test_list_filter_by_status(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        hyp_uri, graph_uri = _make_experiment(store)
        submitted = experiment_submit(
            store,
            graph_refs=[graph_uri],
            procedure_uri="g2v://procedure/engine_run",
            procedure_params={},
            success_criteria=[],
        )
        exp_uri = submitted["experiment_uri"]
        experiment_claim(store, exp_uri)

        result = experiment_list(store, status="claimed")
        uris = [e["experiment_uri"] for e in result["experiments"]]
        assert exp_uri in uris
