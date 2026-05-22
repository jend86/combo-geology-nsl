"""Experiment MCP tools: submit, claim, update, complete, refuse, cancel, review."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def experiment_submit(
    store: "WorkspaceStore",
    graph_refs: list[str],
    procedure_uri: str,
    procedure_params: dict[str, Any],
    success_criteria: list[dict[str, Any]],
    *,
    hypothesis_uri: str | None = None,
    data_refs: list[str] | None = None,
    field_refs: list[str] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit an experiment by snapshotting all graph refs and queuing the spec.

    Capability: experiment:write
    """
    from graph_to_voxel.mcp.workspace.models import Budget, ExperimentSpec, SuccessCriteria

    # Validate and snapshot all graph refs
    snapshotted = [store.snapshot_graph_ref(ref) for ref in graph_refs]
    for ref in snapshotted:
        store.get_resource(ref)  # raises ResourceNotFound if not in registry

    spec = ExperimentSpec(
        id=uuid.uuid4().hex,
        hypothesis_uri=hypothesis_uri,
        graph_refs=snapshotted,
        data_refs=data_refs or [],
        field_refs=field_refs or [],
        procedure_uri=procedure_uri,
        procedure_params=procedure_params,
        success_criteria=SuccessCriteria(criteria=success_criteria),
        budget=Budget(**(budget or {})),
    )
    exp_uri = store.register_experiment(spec)
    return {"experiment_uri": exp_uri, "status": "queued"}


def experiment_claim(
    store: "WorkspaceStore",
    experiment_uri: str,
    lease_s: int = 300,
) -> dict[str, Any]:
    """Claim an experiment for execution (TTL lease).

    Capability: experiment:execute
    """
    claim_id, rec = store.claim_experiment(experiment_uri, lease_s=lease_s)
    return {
        "experiment_uri": experiment_uri,
        "claim_id": claim_id,
        "status": rec.status,
        "lease_expires_at": rec.lease_expires_at.isoformat() if rec.lease_expires_at else None,
    }


def experiment_update(
    store: "WorkspaceStore",
    experiment_uri: str,
    status: str,
) -> dict[str, Any]:
    """Update experiment status (e.g. queued → running).

    Capability: experiment:execute
    """
    from graph_to_voxel.mcp.workspace.models import ExperimentStatus
    rec = store.update_experiment_status(experiment_uri, status)  # type: ignore[arg-type]
    return {"experiment_uri": experiment_uri, "status": rec.status}


def experiment_complete(
    store: "WorkspaceStore",
    experiment_uri: str,
    outcome: str,
    criterion_outcomes: list[dict[str, Any]],
    *,
    artefact_refs: list[str] | None = None,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Mark an experiment as completed and record its result.

    Capability: experiment:execute
    """
    from graph_to_voxel.mcp.workspace.models import CriterionOutcome, ExperimentResult

    outcomes = [CriterionOutcome(**o) for o in criterion_outcomes]
    result = ExperimentResult(
        id=uuid.uuid4().hex,
        spec_id=experiment_uri,
        status=outcome,  # type: ignore[arg-type]
        criterion_outcomes=outcomes,
        artefact_refs=artefact_refs or [],
        caveats=caveats or [],
    )
    result_uri = store.register_result(experiment_uri, result)
    rec = store.update_experiment_status(experiment_uri, "completed", result_uri=result_uri)
    return {"experiment_uri": experiment_uri, "status": rec.status, "result_uri": result_uri}


def experiment_refuse(
    store: "WorkspaceStore",
    experiment_uri: str,
    reason: str,
) -> dict[str, Any]:
    """Refuse an experiment (e.g. budget exceeded, validation failed).

    Capability: experiment:execute
    """
    rec = store.update_experiment_status(experiment_uri, "refused")
    return {"experiment_uri": experiment_uri, "status": rec.status, "reason": reason}


def experiment_cancel(
    store: "WorkspaceStore",
    experiment_uri: str,
) -> dict[str, Any]:
    """Cancel a queued or claimed experiment.

    Capability: experiment:write
    """
    rec = store.update_experiment_status(experiment_uri, "cancelled")
    return {"experiment_uri": experiment_uri, "status": rec.status}


def experiment_review(
    store: "WorkspaceStore",
    experiment_uri: str,
    status: str,
    *,
    notes: str | None = None,
    criterion_assessments: list[dict[str, Any]] | None = None,
    candidate_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Submit a review for a completed experiment.

    Capability: experiment:review
    """
    from graph_to_voxel.mcp.workspace.models import ExperimentReview

    review = ExperimentReview(
        experiment_uri=experiment_uri,
        status=status,  # type: ignore[arg-type]
        criterion_assessments=criterion_assessments or [],
        candidate_refs=candidate_refs or [],
        notes=notes,
    )
    review_uri = store.register_review(experiment_uri, review)
    return {"review_uri": review_uri, "experiment_uri": experiment_uri, "status": status}


def experiment_get(
    store: "WorkspaceStore",
    experiment_uri: str,
) -> dict[str, Any]:
    """Get the current state of an experiment.

    Capability: experiment:read
    """
    rec = store.get_experiment_record(experiment_uri)
    return {
        "experiment_uri": rec.uri,
        "status": rec.status,
        "graph_refs": rec.spec.graph_refs,
        "procedure_uri": rec.spec.procedure_uri,
        "result_uri": rec.result_uri,
    }


def experiment_list(
    store: "WorkspaceStore",
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List registered experiments, optionally filtered by status.

    Capability: experiment:read
    """
    from graph_to_voxel.mcp.workspace.models import ExperimentRecord

    records = store.list_resources(kind="experiment")
    exps: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, ExperimentRecord):
            continue
        try:
            full = store.get_experiment_record(r.uri)
        except Exception:
            full = r  # type: ignore[assignment]
        if status is not None and full.status != status:
            continue
        exps.append({
            "experiment_uri": full.uri,
            "status": full.status,
        })
        if len(exps) >= limit:
            break

    return {"experiments": exps, "total": len(exps)}
