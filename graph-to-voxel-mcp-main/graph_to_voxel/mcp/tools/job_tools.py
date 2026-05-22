"""Job MCP tools: status, cancel, list."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def job_status(
    store: "WorkspaceStore",
    job_uri: str,
) -> dict[str, Any]:
    """Return the current status and progress of a job.

    Capability: job:read
    """
    rec = store.get_job_record(job_uri)
    return {
        "job_uri": job_uri,
        "status": rec.status,
        "progress": rec.progress,
        "current_step": rec.current_step,
        "job_type": rec.job_type,
        "result_refs": rec.result_refs,
        "error": rec.error,
    }


def job_cancel(
    store: "WorkspaceStore",
    job_uri: str,
) -> dict[str, Any]:
    """Request cancellation of a running or pending job.

    Capability: job:write
    """
    rec = store.get_job_record(job_uri)
    if rec.status in _TERMINAL_STATUSES:
        raise ValueError(f"Cannot cancel job with terminal status {rec.status!r}")

    updated = store.update_job(job_uri, status="cancelling")
    return {"job_uri": job_uri, "status": updated.status}


def job_list(
    store: "WorkspaceStore",
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List jobs, optionally filtered by status or type.

    Capability: job:read
    """
    from graph_to_voxel.mcp.workspace.models import JobRecord

    records = store.list_resources(kind="job")
    jobs: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, JobRecord):
            continue
        try:
            full = store.get_job_record(r.uri)
        except Exception:
            full = r  # type: ignore[assignment]
        if status is not None and full.status != status:
            continue
        if job_type is not None and full.job_type != job_type:
            continue
        jobs.append({"job_uri": full.uri, "status": full.status, "job_type": full.job_type})
        if len(jobs) >= limit:
            break

    return {"jobs": jobs, "total": len(jobs)}
