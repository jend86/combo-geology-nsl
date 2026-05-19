"""Candidate MCP tools: submit."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def candidate_submit(
    store: "WorkspaceStore",
    graph_uri: str,
    reference_pair: tuple[str, str],
    *,
    evidence_refs: list[str] | None = None,
    score_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Submit a graph as a candidate against a reference pair.

    Capability: graph:commit
    """
    # Validate graph exists
    store.get_resource(graph_uri)

    candidate_uri = store.register_candidate(
        graph_uri=graph_uri,
        reference_pair=reference_pair,
        evidence_refs=evidence_refs or [],
        score_refs=score_refs or [],
    )
    return {"candidate_uri": candidate_uri, "graph_uri": graph_uri}
