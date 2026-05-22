"""Workspace and action-log MCP tools."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def workspace_describe(
    store: "WorkspaceStore",
    uri: str,
) -> dict[str, Any]:
    """Return metadata for any workspace resource by URI.

    Capability: graph:read / data:read (depends on resource kind)
    """
    rec = store.get_resource(uri)
    return {
        "uri": rec.uri,
        "kind": rec.kind,
        "size_bytes": rec.size_bytes,
        "created_at": rec.created_at.isoformat(),
        "content_hash": rec.content_hash,
        "tags": rec.tags,
        "pinned_by": rec.pinned_by,
    }


def workspace_gc(
    store: "WorkspaceStore",
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run (or dry-run) garbage collection according to policy.

    Capability: admin:gc
    """
    from graph_to_voxel.mcp.workspace.gc import GCPolicy, WorkspaceGC

    pol = GCPolicy(**(policy or {}))
    gc = WorkspaceGC(store)
    report = gc.run(pol)
    return {
        "evicted": report.evicted,
        "retained": report.retained,
        "bytes_freed": report.bytes_freed,
        "dry_run": pol.dry_run,
    }


def actions_query(
    store: "WorkspaceStore",
    agent_id: str | None = None,
    tool: str | None = None,
    capability: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Query the append-only action log.

    Capability: graph:read (read-only provenance)
    """
    actions = store.query_actions(
        agent_id=agent_id,
        tool=tool,
        capability=capability,
        limit=limit,
        offset=offset,
    )
    return {
        "actions": [
            {
                "id": a.id,
                "agent_id": a.agent_id,
                "role": a.role,
                "capability": a.capability,
                "tool": a.tool,
                "timestamp": a.timestamp.isoformat(),
                "input_uris": a.input_uris,
                "output_uris": a.output_uris,
                "error": a.error,
            }
            for a in actions
        ],
        "total": len(actions),
    }
