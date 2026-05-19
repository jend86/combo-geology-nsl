"""Data MCP tools: evidence-resource registration and preview.

These tools register bytes/files as immutable content-hashed workspace
resources (`g2v://data/<hash>`) and expose a small typed preview. They are
*not* a raw-dataset analysis surface — grep, SQL, Python evaluation,
in-place transforms, reprojection, and PDF extraction are out of scope and
belong in a consumer-controlled analysis image (see
`docs/design/08-mcp-scope.md` §5.4 and §7.2, and
`docs/design/09-docker-runtime.md` §5.6 for the split-image recipe).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def data_ingest(
    store: "WorkspaceStore",
    *,
    filename: str,
    content: bytes | None = None,
    file_path: str | None = None,
    media_type: str = "application/octet-stream",
    crs: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Register raw bytes or a local file as an immutable evidence resource.

    Returns a content-hashed `g2v://data/<hash>` URI that other tools
    (graph_apply_patch evidence refs, candidate_submit, etc.) can reference.
    Not a raw-dataset analysis tool: there is no grep, SQL, transform, or
    in-place edit — those belong in a consumer-controlled analysis container.

    Capability: data:write
    """
    if content is None and file_path is None:
        raise ValueError("Either content or file_path must be provided")

    if content is None:
        with open(file_path, "rb") as f:  # type: ignore[arg-type]
            content = f.read()

    preview_text = ""
    if media_type.startswith("text/"):
        try:
            preview_text = content[:500].decode("utf-8", errors="replace")
        except Exception:
            pass

    uri = store.register_data(
        source_path_rel=filename,
        media_type=media_type,
        crs=crs,
        preview_text=preview_text,
        raw_bytes=content,
        tags=tags,
    )
    return {"data_uri": uri, "filename": filename, "media_type": media_type}


def data_preview(
    store: "WorkspaceStore",
    data_uri: str,
    max_bytes: int = 1024,
) -> dict[str, Any]:
    """Return a small typed preview of a registered evidence resource.

    Returns the stored media type, size, optional CRS, detected schema, and a
    bounded preview text slice. Full reads should use the MCP resource read
    path. Not a full-file analysis tool — consumers that need to scan an
    entire dataset should mount it read-only in a separate analysis container.

    Capability: data:read
    """
    rec = store.get_data_record(data_uri)
    return {
        "data_uri": data_uri,
        "media_type": rec.media_type,
        "size_bytes": rec.size_bytes,
        "preview_text": rec.preview_text[:max_bytes],
        "crs": rec.crs,
        "detected_schema": rec.detected_schema,
    }


def data_list(
    store: "WorkspaceStore",
    media_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List registered evidence resources, optionally filtered by media type.

    Returns metadata only (URI, media type, size). The body of each resource
    is reachable through `data_preview` or the MCP resource read path.

    Capability: data:read
    """
    from graph_to_voxel.mcp.workspace.models import DataRecord

    records = store.list_resources(kind="data")
    items: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, DataRecord):
            continue
        if media_type is not None and r.media_type != media_type:
            continue
        items.append({
            "data_uri": r.uri,
            "media_type": r.media_type,
            "size_bytes": r.size_bytes,
        })
        if len(items) >= limit:
            break

    return {"data": items, "total": len(items)}
