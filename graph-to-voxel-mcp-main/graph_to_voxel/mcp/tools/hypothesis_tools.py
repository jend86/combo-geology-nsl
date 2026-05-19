"""Hypothesis MCP tools: create, list, get, update."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def hypothesis_create(
    store: "WorkspaceStore",
    statement: str,
    *,
    graph_refs: list[str] | None = None,
    data_refs: list[str] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create a new hypothesis resource.

    Capability: hypothesis:write
    """
    uri = store.register_hypothesis(
        statement,
        graph_refs=graph_refs,
        data_refs=data_refs,
        rationale=rationale,
    )
    return {"hypothesis_uri": uri, "statement": statement}


def hypothesis_list(
    store: "WorkspaceStore",
    limit: int = 50,
) -> dict[str, Any]:
    """List registered hypotheses.

    Capability: hypothesis:read
    """
    records = store.list_resources(kind="hypothesis")
    items = [
        {"hypothesis_uri": r.uri, "statement": getattr(r, "statement", "")}
        for r in records[:limit]
    ]
    return {"hypotheses": items, "total": len(records)}


def hypothesis_get(
    store: "WorkspaceStore",
    hypothesis_uri: str,
) -> dict[str, Any]:
    """Retrieve a hypothesis record.

    Capability: hypothesis:read
    """
    rec = store.get_hypothesis_record(hypothesis_uri)
    return {
        "hypothesis_uri": rec.uri,
        "statement": rec.statement,
        "graph_refs": rec.graph_refs,
        "data_refs": rec.data_refs,
        "rationale": rec.rationale,
    }


def hypothesis_update(
    store: "WorkspaceStore",
    hypothesis_uri: str,
    *,
    statement: str | None = None,
    graph_refs: list[str] | None = None,
    data_refs: list[str] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Update an existing hypothesis record.

    Capability: hypothesis:write
    """
    rec = store.update_hypothesis(
        hypothesis_uri,
        statement=statement,
        graph_refs=graph_refs,
        data_refs=data_refs,
        rationale=rationale,
    )
    return {
        "hypothesis_uri": rec.uri,
        "statement": rec.statement,
        "graph_refs": rec.graph_refs,
        "data_refs": rec.data_refs,
        "rationale": rec.rationale,
    }
