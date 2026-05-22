from __future__ import annotations

import base64 as _b64
import json
from collections import deque
from typing import Any

from graph_to_voxel.graph.core import Graph
from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def graph_ingest(
    store: WorkspaceStore,
    *,
    filename: str,
    content_text: str | None = None,
    content_base64: str | None = None,
    message: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Parse a graph JSON payload into an immutable g2v://graph/<hash> snapshot.

    Capability: graph:ingest
    """
    if (content_text is None) == (content_base64 is None):
        raise ValueError("provide exactly one of content_text or content_base64")

    if content_text is not None:
        raw = content_text
    else:
        assert content_base64 is not None  # narrowed by XOR check above
        try:
            raw = _b64.b64decode(content_base64, validate=True).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"invalid base64 payload: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc

    graph = Graph.from_dict(data)

    merged_tags: dict[str, str] = {"source_filename": filename}
    if tags:
        merged_tags.update(tags)

    existed = False
    candidate_uri = f"g2v://graph/{store._content_hash(json.dumps(graph.to_dict(), sort_keys=True))}"
    try:
        store.get_resource(candidate_uri)
        existed = True
    except Exception:
        existed = False

    graph_uri = store.register_graph(graph, message=message, tags=merged_tags)

    return {
        "graph_uri": graph_uri,
        "from_cache": existed,
        "node_count": len(graph.node_ids()),
        "edge_count": len(graph.get_edges()),
        "unit_catalog": list(graph.unit_catalog()),
    }


def seed_graph_submit(
    store: WorkspaceStore,
    *,
    filename: str = "seed.json",
    content_text: str | None = None,
    content_base64: str | None = None,
    message: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Ingest a bootstrap seed graph and return the URI under seed terminology.

    This is intentionally a thin wrapper over graph_ingest so seed submission
    uses the same parsing, validation, caching, and persistence path as normal
    graph ingestion.
    """
    merged_tags: dict[str, str] = {"submission_kind": "bootstrap_seed"}
    if tags:
        merged_tags.update(tags)
    result = graph_ingest(
        store,
        filename=filename,
        content_text=content_text,
        content_base64=content_base64,
        message=message,
        tags=merged_tags,
    )
    result["seed_graph_uri"] = result["graph_uri"]
    return result


def graph_branch(store: WorkspaceStore, graph_uri: str) -> dict[str, str]:
    scratch_uri = store.create_scratch(graph_uri)
    scratch = store.get_scratch_record(scratch_uri)
    return {"scratch_uri": scratch_uri, "head_rev_uri": scratch.head_rev_uri}


def graph_apply_patch(
    store: WorkspaceStore,
    scratch_uri: str,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    head_rev_uri, report = store.apply_scratch_patch(scratch_uri, operations)
    return {"head_rev_uri": head_rev_uri, "validation_report": report}


def graph_commit(store: WorkspaceStore, scratch_uri: str, message: str | None = None) -> dict[str, str]:
    return {"graph_uri": store.commit_scratch(scratch_uri, message=message)}


def refine_commit(
    store: WorkspaceStore,
    graph_uri: str,
    operations: list[dict[str, Any]],
    message: str | None = None,
) -> dict[str, Any]:
    """Branch, patch, and commit a graph in one regular-workflow call."""
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    branch = graph_branch(store, graph_uri)
    patched = graph_apply_patch(store, branch["scratch_uri"], operations)
    committed = graph_commit(store, branch["scratch_uri"], message=message)
    return {
        "graph_uri": committed["graph_uri"],
        "scratch_uri": branch["scratch_uri"],
        "head_rev_uri": patched["head_rev_uri"],
        "validation_report": patched["validation_report"],
    }


def graph_query(
    store: WorkspaceStore,
    graph_uri: str,
    selector: dict[str, Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    graph = store.load_graph(graph_uri)
    selector = selector or {}
    max_items = 100 if limit is None else max(0, limit)
    kind = _normalise_kind(selector.get("kind"))
    nodes = []
    for node in graph.nodes():
        raw = node.model_dump(mode="json", exclude_none=True)
        if kind is not None and raw.get("kind") != kind:
            continue
        nodes.append(raw)
        if len(nodes) >= max_items:
            break
    result: dict[str, Any] = {"nodes": nodes, "truncated": len(nodes) >= max_items}
    if selector.get("include_edges"):
        result["edges"] = [edge.model_dump(mode="json", exclude_none=True) for edge in graph.get_edges()[:max_items]]
    return result


def graph_subgraph(
    store: WorkspaceStore,
    graph_uri: str,
    seed_nodes: list[str],
    radius: int,
    limit: int | None = None,
) -> dict[str, Any]:
    graph = store.load_graph(graph_uri)
    max_items = 100 if limit is None else max(0, limit)
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in graph.node_ids()}
    for edge in graph.get_edges():
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.source)

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in seed_nodes)
    while queue and len(visited) < max_items:
        node_id, depth = queue.popleft()
        if node_id in visited or node_id not in adjacency:
            continue
        visited.add(node_id)
        if depth >= radius:
            continue
        for neighbour in sorted(adjacency[node_id]):
            if neighbour not in visited:
                queue.append((neighbour, depth + 1))

    nodes = [
        graph.get_node(node_id).model_dump(mode="json", exclude_none=True)
        for node_id in graph.node_ids()
        if node_id in visited
    ][:max_items]
    selected = {node["id"] for node in nodes}
    edges = [
        edge.model_dump(mode="json", exclude_none=True)
        for edge in graph.get_edges()
        if edge.source in selected and edge.target in selected
    ]
    return {"nodes": nodes, "edges": edges, "truncated": len(visited) > len(nodes)}


def graph_diff(
    store: WorkspaceStore,
    graph_uri_a: str,
    graph_uri_b: str,
    limit: int | None = None,
) -> dict[str, Any]:
    graph_a = store.load_graph(graph_uri_a)
    graph_b = store.load_graph(graph_uri_b)
    max_items = 100 if limit is None else max(0, limit)
    nodes_a = {node.id: node.model_dump(mode="json", exclude_none=True) for node in graph_a.nodes()}
    nodes_b = {node.id: node.model_dump(mode="json", exclude_none=True) for node in graph_b.nodes()}
    added_nodes = sorted(set(nodes_b) - set(nodes_a))[:max_items]
    removed_nodes = sorted(set(nodes_a) - set(nodes_b))[:max_items]
    changed_nodes = sorted(
        node_id for node_id in set(nodes_a) & set(nodes_b) if nodes_a[node_id] != nodes_b[node_id]
    )[:max_items]
    edges_a = {_edge_key(edge.model_dump(mode="json", exclude_none=True)) for edge in graph_a.get_edges()}
    edges_b = {_edge_key(edge.model_dump(mode="json", exclude_none=True)) for edge in graph_b.get_edges()}
    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "changed_nodes": changed_nodes,
        "added_edges": sorted(edges_b - edges_a)[:max_items],
        "removed_edges": sorted(edges_a - edges_b)[:max_items],
    }


def graph_provenance(store: WorkspaceStore, graph_uri: str, node_id: str) -> dict[str, Any]:
    node = store.load_graph(graph_uri).get_node(node_id)
    return {
        "node_id": node_id,
        "provenance": node.provenance.model_dump(mode="json"),
        "metadata": dict(node.metadata),
    }


def _normalise_kind(kind: Any) -> str | None:
    if kind is None:
        return None
    aliases = {
        "StratigraphicUnit": "stratigraphic_unit",
        "Contact": "contact",
        "Orientation": "orientation",
        "Fault": "fault",
        "ObservationPoint": "observation_point",
        "Location": "location",
        "Sample": "sample",
        "Series": "series",
    }
    value = str(kind)
    return aliases.get(value, value.lower())


def _edge_key(edge: dict[str, Any]) -> str:
    return str(edge.get("id") or (edge.get("kind"), edge.get("source"), edge.get("target")))


__all__ = [
    "graph_apply_patch",
    "graph_branch",
    "graph_commit",
    "graph_diff",
    "graph_ingest",
    "graph_provenance",
    "graph_query",
    "refine_commit",
    "seed_graph_submit",
    "graph_subgraph",
]
