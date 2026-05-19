"""g2v-mcp: FastMCP server wiring all graph-to-voxel tools and resources."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from graph_to_voxel.mcp.workspace.store import WorkspaceStore

# ---------------------------------------------------------------------------
# Store initialisation
# ---------------------------------------------------------------------------

_store: WorkspaceStore | None = None


def _get_store() -> WorkspaceStore:
    global _store
    if _store is None:
        root = Path(os.environ.get("G2V_WORKSPACE_ROOT", "./workspace"))
        _store = WorkspaceStore(root)
    return _store


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "g2v-mcp",
    instructions=(
        "Graph-to-Voxel MCP server. Provides tools for managing geological graphs, "
        "running voxel field generation, IC scoring, and experiment tracking. "
        "All URIs use the g2v:// scheme. Resources are immutable and content-addressed. "
        "Use graph_branch → graph_apply_patch → graph_commit to iterate on geological models."
    ),
)


TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"name": "workspace.describe", "capability": "graph:read", "side_effects": "read", "budget_class": "sync", "resource_types": ["resource"]},
    {"name": "workspace.gc", "capability": "admin:gc", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["resource"]},
    {"name": "actions.query", "capability": "graph:read", "side_effects": "read", "budget_class": "sync", "resource_types": ["action-log"]},
    {"name": "data.ingest", "capability": "data:ingest", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["data"]},
    {"name": "data.preview", "capability": "data:read", "side_effects": "read", "budget_class": "sync", "resource_types": ["data"]},
    {"name": "graph.ingest", "capability": "graph:ingest", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["graph"]},
    {"name": "seed_graph.submit", "capability": "graph:ingest", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["graph"]},
    {"name": "graph.branch", "capability": "graph:edit", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["graph", "scratch"]},
    {"name": "graph.apply_patch", "capability": "graph:edit", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["scratch"]},
    {"name": "graph.commit", "capability": "graph:commit", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["scratch", "graph"]},
    {"name": "refine.commit", "capability": "graph:commit", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["graph", "scratch"]},
    {"name": "engine.run_preview", "capability": "engine:preview", "side_effects": "compute", "budget_class": "sync-or-job", "resource_types": ["graph", "field", "job"]},
    {"name": "engine.run", "capability": "engine:run", "side_effects": "compute", "budget_class": "job", "resource_types": ["graph", "field", "job"]},
    {"name": "job.status", "capability": "experiment:execute", "side_effects": "read", "budget_class": "sync", "resource_types": ["job"]},
    {"name": "job.cancel", "capability": "experiment:execute", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["job"]},
    {"name": "experiment.submit", "capability": "experiment:submit", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["experiment"]},
    {"name": "experiment.claim", "capability": "experiment:execute", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["experiment"]},
    {"name": "experiment.review", "capability": "experiment:review", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["experiment", "review"]},
    {"name": "candidate.submit", "capability": "graph:commit", "side_effects": "mutate", "budget_class": "sync", "resource_types": ["candidate"]},
    {"name": "ic.score", "capability": "ic:score", "side_effects": "compute", "budget_class": "sync", "resource_types": ["graph", "field", "score"]},
    {"name": "ic.score_from_graphs", "capability": "ic:score", "side_effects": "compute", "budget_class": "job", "resource_types": ["graph", "field", "score", "job"]},
)


RESOURCE_TEMPLATES: tuple[str, ...] = (
    "g2v://graph/{hash}",
    "g2v://scratch/{id}@rev/{n}",
    "g2v://field/{hash}",
    "g2v://data/{hash}",
    "g2v://hypothesis/{id}",
    "g2v://procedure/{name}",
    "g2v://experiment/{id}",
    "g2v://experiment/{id}/result",
    "g2v://candidate/{id}",
    "g2v://job/{id}",
    "g2v://action-log/{id}",
)


class G2VMcpServer:
    """In-process facade for tests and non-transport integrations."""

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    def tool_manifest(self) -> list[dict[str, Any]]:
        return [dict(tool) for tool in TOOL_DEFINITIONS]

    def resource_templates(self) -> list[str]:
        return list(RESOURCE_TEMPLATES)

# ---------------------------------------------------------------------------
# Resource templates — durable reads via MCP Resources
# ---------------------------------------------------------------------------


@mcp.resource("g2v://graph/{graph_id}")
def resource_graph(graph_id: str) -> str:
    uri = f"g2v://graph/{graph_id}"
    try:
        rec = _get_store().get_resource(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("g2v://scratch/{scratch_id}")
def resource_scratch(scratch_id: str) -> str:
    uri = f"g2v://scratch/{scratch_id}"
    try:
        rec = _get_store().get_resource(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("g2v://field/{field_id}")
def resource_field(field_id: str) -> str:
    uri = f"g2v://field/{field_id}"
    try:
        rec = _get_store().get_resource(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("g2v://experiment/{experiment_id}")
def resource_experiment(experiment_id: str) -> str:
    uri = f"g2v://experiment/{experiment_id}"
    try:
        rec = _get_store().get_experiment_record(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("g2v://job/{job_id}")
def resource_job(job_id: str) -> str:
    uri = f"g2v://job/{job_id}"
    try:
        rec = _get_store().get_job_record(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("g2v://hypothesis/{hypothesis_id}")
def resource_hypothesis(hypothesis_id: str) -> str:
    uri = f"g2v://hypothesis/{hypothesis_id}"
    try:
        rec = _get_store().get_hypothesis_record(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("g2v://data/{data_id}")
def resource_data(data_id: str) -> str:
    uri = f"g2v://data/{data_id}"
    try:
        rec = _get_store().get_data_record(uri)
        return rec.model_dump_json(indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Graph tools — Capability: graph:read / graph:edit / graph:commit
# ---------------------------------------------------------------------------

@mcp.tool()
def graph_ingest(
    filename: str,
    content_text: str | None = None,
    content_base64: str | None = None,
    message: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Parse a graph JSON payload into an immutable g2v://graph/<hash> snapshot.

    Provide exactly one of content_text or content_base64.

    Capability: graph:ingest
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_ingest as _f
    return _f(
        _get_store(),
        filename=filename,
        content_text=content_text,
        content_base64=content_base64,
        message=message,
        tags=tags,
    )


@mcp.tool()
def seed_graph_submit(
    filename: str = "seed.json",
    content_text: str | None = None,
    content_base64: str | None = None,
    message: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Ingest a bootstrap seed graph and return seed_graph_uri.

    Provide exactly one of content_text or content_base64.

    Capability: graph:ingest
    """
    from graph_to_voxel.mcp.tools.graph_tools import seed_graph_submit as _f
    return _f(
        _get_store(),
        filename=filename,
        content_text=content_text,
        content_base64=content_base64,
        message=message,
        tags=tags,
    )


@mcp.tool()
def graph_branch(graph_uri: str) -> dict[str, Any]:
    """Create a mutable scratch branch from an immutable graph snapshot.

    Capability: graph:edit
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_branch as _f
    return _f(_get_store(), graph_uri)


@mcp.tool()
def graph_apply_patch(scratch_uri: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a transactional patch to a scratch graph. Rolls back on failure.

    Capability: graph:edit
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_apply_patch as _f
    return _f(_get_store(), scratch_uri, operations)


@mcp.tool()
def graph_commit(scratch_uri: str, message: str | None = None) -> dict[str, Any]:
    """Commit a scratch graph to an immutable snapshot.

    Capability: graph:commit
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_commit as _f
    return _f(_get_store(), scratch_uri, message=message)


@mcp.tool()
def refine_commit(
    graph_uri: str,
    operations: list[dict[str, Any]],
    message: str | None = None,
) -> dict[str, Any]:
    """Branch, patch, and commit a graph in one regular-workflow call.

    Capability: graph:commit
    """
    from graph_to_voxel.mcp.tools.graph_tools import refine_commit as _f
    return _f(_get_store(), graph_uri, operations, message=message)


@mcp.tool()
def graph_query(
    graph_uri: str,
    selector: dict[str, Any],
    limit: int = 50,
) -> dict[str, Any]:
    """Bounded node/edge query. selector keys: kind, node_id, include_edges.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_query as _f
    return _f(_get_store(), graph_uri, selector, limit=limit)


@mcp.tool()
def graph_subgraph(
    graph_uri: str,
    seed_nodes: list[str],
    radius: int,
    limit: int = 100,
) -> dict[str, Any]:
    """BFS-bounded subgraph around seed nodes.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_subgraph as _f
    return _f(_get_store(), graph_uri, seed_nodes, radius, limit=limit)


@mcp.tool()
def graph_diff(graph_uri_a: str, graph_uri_b: str, limit: int = 200) -> dict[str, Any]:
    """Structured diff between two graph snapshots.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_diff as _f
    return _f(_get_store(), graph_uri_a, graph_uri_b, limit=limit)


@mcp.tool()
def graph_provenance(graph_uri: str, node_id: str) -> dict[str, Any]:
    """Return provenance information for a node.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.graph_tools import graph_provenance as _f
    return _f(_get_store(), graph_uri, node_id)


# ---------------------------------------------------------------------------
# Engine / voxel tools
# ---------------------------------------------------------------------------

@mcp.tool()
def engine_run(
    graph_ref: str,
    field_spec: dict[str, Any],
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return cached field_uri or create a job. field_spec: grid_origin, grid_maximum, grid_shape.

    Capability: engine:run
    """
    from graph_to_voxel.mcp.tools.engine_tools import engine_run as _f
    return _f(_get_store(), graph_ref, field_spec, budget=budget)


@mcp.tool()
def engine_run_preview(
    graph_ref: str,
    field_spec: dict[str, Any],
    preview_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run sync within preview budget or return job_uri.

    Capability: engine:preview
    """
    from graph_to_voxel.mcp.tools.engine_tools import engine_run_preview as _f
    return _f(_get_store(), graph_ref, field_spec, preview_budget=preview_budget)


@mcp.tool()
def voxel_sample(
    field_uri: str,
    points: list[list[float]],
    limit: int | None = None,
) -> dict[str, Any]:
    """Sample a voxel field at (x,y,z) points.

    Capability: engine:preview
    """
    from graph_to_voxel.mcp.tools.engine_tools import voxel_sample as _f
    pts = [tuple(p) for p in points]  # type: ignore[misc]
    return _f(_get_store(), field_uri, pts, limit=limit)


@mcp.tool()
def voxel_stats(field_uri: str, region: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summary statistics for a voxel field.

    Capability: engine:preview
    """
    from graph_to_voxel.mcp.tools.engine_tools import voxel_stats as _f
    return _f(_get_store(), field_uri, region=region)


@mcp.tool()
def voxel_export(field_uri: str, format: str = "zarr") -> dict[str, Any]:
    """Export a voxel field to a data resource.

    Capability: engine:preview
    """
    from graph_to_voxel.mcp.tools.engine_tools import voxel_export as _f
    return _f(_get_store(), field_uri, format=format)


# ---------------------------------------------------------------------------
# IC tools
# ---------------------------------------------------------------------------

@mcp.tool()
def ic_score(
    candidate_graph_uri: str,
    candidate_field_uri: str,
    reference_a_graph_uri: str,
    reference_a_field_uri: str,
    reference_b_graph_uri: str,
    reference_b_field_uri: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a candidate. All six field/graph URIs must exist.

    Capability: ic:score
    """
    from graph_to_voxel.mcp.tools.ic_tools import ic_score as _f
    return _f(
        _get_store(),
        candidate_graph_uri=candidate_graph_uri,
        candidate_field_uri=candidate_field_uri,
        reference_a_graph_uri=reference_a_graph_uri,
        reference_a_field_uri=reference_a_field_uri,
        reference_b_graph_uri=reference_b_graph_uri,
        reference_b_field_uri=reference_b_field_uri,
        config=config,
    )


@mcp.tool()
def ic_score_from_graphs(
    candidate_graph_uri: str,
    reference_a_graph_uri: str,
    reference_b_graph_uri: str,
    field_spec: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build fields and score. Always async (job_uri).

    Capability: ic:score
    """
    from graph_to_voxel.mcp.tools.ic_tools import ic_score_from_graphs as _f
    return _f(
        _get_store(),
        candidate_graph_uri=candidate_graph_uri,
        reference_a_graph_uri=reference_a_graph_uri,
        reference_b_graph_uri=reference_b_graph_uri,
        field_spec=field_spec,
        config=config,
    )


# ---------------------------------------------------------------------------
# Hypothesis tools
# ---------------------------------------------------------------------------

@mcp.tool()
def hypothesis_create(
    statement: str,
    graph_refs: list[str] | None = None,
    data_refs: list[str] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create a geological hypothesis resource.

    Capability: experiment:submit
    """
    from graph_to_voxel.mcp.tools.hypothesis_tools import hypothesis_create as _f
    return _f(_get_store(), statement, graph_refs=graph_refs, data_refs=data_refs, rationale=rationale)


@mcp.tool()
def hypothesis_list(limit: int = 50) -> dict[str, Any]:
    """List registered hypotheses.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.hypothesis_tools import hypothesis_list as _f
    return _f(_get_store(), limit=limit)


@mcp.tool()
def hypothesis_get(hypothesis_uri: str) -> dict[str, Any]:
    """Retrieve a hypothesis by URI.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.hypothesis_tools import hypothesis_get as _f
    return _f(_get_store(), hypothesis_uri)


@mcp.tool()
def hypothesis_update(
    hypothesis_uri: str,
    statement: str | None = None,
    graph_refs: list[str] | None = None,
    data_refs: list[str] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Update a hypothesis.

    Capability: experiment:submit
    """
    from graph_to_voxel.mcp.tools.hypothesis_tools import hypothesis_update as _f
    return _f(_get_store(), hypothesis_uri, statement=statement, graph_refs=graph_refs,
              data_refs=data_refs, rationale=rationale)


# ---------------------------------------------------------------------------
# Experiment tools
# ---------------------------------------------------------------------------

@mcp.tool()
def experiment_submit(
    graph_refs: list[str],
    procedure_uri: str,
    procedure_params: dict[str, Any],
    success_criteria: list[dict[str, Any]],
    hypothesis_uri: str | None = None,
    data_refs: list[str] | None = None,
    field_refs: list[str] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit an experiment spec. Snapshots all graph refs.

    Capability: experiment:submit
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_submit as _f
    return _f(
        _get_store(),
        graph_refs=graph_refs,
        procedure_uri=procedure_uri,
        procedure_params=procedure_params,
        success_criteria=success_criteria,
        hypothesis_uri=hypothesis_uri,
        data_refs=data_refs,
        field_refs=field_refs,
        budget=budget,
    )


@mcp.tool()
def experiment_claim(experiment_uri: str, lease_s: int = 300) -> dict[str, Any]:
    """Claim an experiment for execution.

    Capability: experiment:execute
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_claim as _f
    return _f(_get_store(), experiment_uri, lease_s=lease_s)


@mcp.tool()
def experiment_update(experiment_uri: str, status: str) -> dict[str, Any]:
    """Update experiment status.

    Capability: experiment:execute
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_update as _f
    return _f(_get_store(), experiment_uri, status)


@mcp.tool()
def experiment_complete(
    experiment_uri: str,
    outcome: str,
    criterion_outcomes: list[dict[str, Any]],
    artefact_refs: list[str] | None = None,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Mark an experiment completed.

    Capability: experiment:execute
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_complete as _f
    return _f(_get_store(), experiment_uri, outcome, criterion_outcomes,
              artefact_refs=artefact_refs, caveats=caveats)


@mcp.tool()
def experiment_refuse(experiment_uri: str, reason: str) -> dict[str, Any]:
    """Refuse an experiment.

    Capability: experiment:execute
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_refuse as _f
    return _f(_get_store(), experiment_uri, reason)


@mcp.tool()
def experiment_cancel(experiment_uri: str) -> dict[str, Any]:
    """Cancel a queued or claimed experiment.

    Capability: experiment:submit
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_cancel as _f
    return _f(_get_store(), experiment_uri)


@mcp.tool()
def experiment_review(
    experiment_uri: str,
    status: str,
    notes: str | None = None,
    criterion_assessments: list[dict[str, Any]] | None = None,
    candidate_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Submit a structured review for a completed experiment.

    Capability: experiment:review
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_review as _f
    return _f(_get_store(), experiment_uri, status, notes=notes,
              criterion_assessments=criterion_assessments, candidate_refs=candidate_refs)


@mcp.tool()
def experiment_get(experiment_uri: str) -> dict[str, Any]:
    """Get current state of an experiment.

    Capability: experiment:submit
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_get as _f
    return _f(_get_store(), experiment_uri)


@mcp.tool()
def experiment_list(status: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List experiments.

    Capability: experiment:submit
    """
    from graph_to_voxel.mcp.tools.experiment_tools import experiment_list as _f
    return _f(_get_store(), status=status, limit=limit)


# ---------------------------------------------------------------------------
# Job tools
# ---------------------------------------------------------------------------

@mcp.tool()
def job_status(job_uri: str) -> dict[str, Any]:
    """Return status and progress of a job.

    Capability: engine:run
    """
    from graph_to_voxel.mcp.tools.job_tools import job_status as _f
    return _f(_get_store(), job_uri)


@mcp.tool()
def job_cancel(job_uri: str) -> dict[str, Any]:
    """Request cancellation of a running or pending job.

    Capability: engine:run
    """
    from graph_to_voxel.mcp.tools.job_tools import job_cancel as _f
    return _f(_get_store(), job_uri)


@mcp.tool()
def job_list(status: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List jobs.

    Capability: engine:run
    """
    from graph_to_voxel.mcp.tools.job_tools import job_list as _f
    return _f(_get_store(), status=status, limit=limit)


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------

@mcp.tool()
def data_ingest(
    filename: str,
    content_base64: str | None = None,
    content_text: str | None = None,
    media_type: str = "application/octet-stream",
    crs: str | None = None,
) -> dict[str, Any]:
    """Ingest content as an immutable data resource. Provide content_base64 or content_text.

    Capability: data:ingest
    """
    import base64

    if content_base64 is not None:
        content = base64.b64decode(content_base64)
    elif content_text is not None:
        content = content_text.encode("utf-8")
    else:
        raise ValueError("Provide content_base64 or content_text")

    from graph_to_voxel.mcp.tools.data_tools import data_ingest as _f
    return _f(_get_store(), filename=filename, content=content, media_type=media_type, crs=crs)


@mcp.tool()
def data_preview(data_uri: str, max_bytes: int = 1024) -> dict[str, Any]:
    """Return a small preview of a data resource.

    Capability: data:read
    """
    from graph_to_voxel.mcp.tools.data_tools import data_preview as _f
    return _f(_get_store(), data_uri, max_bytes=max_bytes)


@mcp.tool()
def data_list(media_type: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List registered data resources.

    Capability: data:read
    """
    from graph_to_voxel.mcp.tools.data_tools import data_list as _f
    return _f(_get_store(), media_type=media_type, limit=limit)


# ---------------------------------------------------------------------------
# Workspace / admin tools
# ---------------------------------------------------------------------------

@mcp.tool()
def workspace_describe(uri: str) -> dict[str, Any]:
    """Describe any workspace resource by URI.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.workspace_tools import workspace_describe as _f
    return _f(_get_store(), uri)


@mcp.tool()
def workspace_gc(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run (or dry-run) garbage collection.

    Capability: admin:gc
    """
    from graph_to_voxel.mcp.tools.workspace_tools import workspace_gc as _f
    return _f(_get_store(), policy=policy)


@mcp.tool()
def actions_query(
    agent_id: str | None = None,
    tool: str | None = None,
    capability: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Query the append-only provenance action log.

    Capability: graph:read
    """
    from graph_to_voxel.mcp.tools.workspace_tools import actions_query as _f
    return _f(_get_store(), agent_id=agent_id, tool=tool, capability=capability,
              limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Candidate tools
# ---------------------------------------------------------------------------

@mcp.tool()
def candidate_submit(
    graph_uri: str,
    reference_pair: list[str],
    evidence_refs: list[str] | None = None,
    score_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Submit a graph as a candidate against a reference pair (exactly 2 URIs).

    Capability: graph:commit
    """
    from graph_to_voxel.mcp.tools.candidate_tools import candidate_submit as _f
    if len(reference_pair) != 2:
        raise ValueError("reference_pair must contain exactly 2 URIs")
    return _f(_get_store(), graph_uri, (reference_pair[0], reference_pair[1]),
              evidence_refs=evidence_refs, score_refs=score_refs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the g2v-mcp server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
