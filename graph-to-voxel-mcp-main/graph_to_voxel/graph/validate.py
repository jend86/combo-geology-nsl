from __future__ import annotations

from collections import defaultdict

import networkx as nx

from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Contact, Fault, Location, Sample, StratigraphicUnit


class GraphValidationError(ValueError):
    """Raised when the graph is valid JSON but invalid geology for v1."""


def validate_graph(nodes: dict[str, object], edges: list[GraphEdge]) -> None:
    _validate_unique_unit_ids(nodes)
    _validate_edge_endpoints(nodes, edges)
    _validate_contacts(nodes)
    _validate_fault_surface_points(nodes)
    _validate_within_edges(nodes, edges)
    _validate_at_edges(nodes, edges)
    _validate_overlies_dag(nodes, edges)
    _validate_member_of_series_function(edges)


def _validate_unique_unit_ids(nodes: dict[str, object]) -> None:
    seen: dict[str, str] = {}
    for node in nodes.values():
        if isinstance(node, StratigraphicUnit):
            if node.unit_id in seen and seen[node.unit_id] != node.id:
                raise GraphValidationError(f"duplicate StratigraphicUnit unit_id {node.unit_id!r}")
            seen[node.unit_id] = node.id


def _validate_edge_endpoints(nodes: dict[str, object], edges: list[GraphEdge]) -> None:
    for edge in edges:
        if edge.source not in nodes:
            raise GraphValidationError(f"edge {edge.id or edge.kind.value} source {edge.source!r} is missing")
        if edge.target not in nodes:
            raise GraphValidationError(f"edge {edge.id or edge.kind.value} target {edge.target!r} is missing")


def _validate_contacts(nodes: dict[str, object]) -> None:
    unit_ids = {node.unit_id for node in nodes.values() if isinstance(node, StratigraphicUnit)}
    for node in nodes.values():
        if isinstance(node, Contact):
            missing = [unit_id for unit_id in node.between if unit_id not in unit_ids]
            if missing:
                raise GraphValidationError(
                    f"Contact {node.id!r} references missing unit_id(s): {', '.join(missing)}"
                )


def _validate_fault_surface_points(nodes: dict[str, object]) -> None:
    contact_ids = {node.id for node in nodes.values() if isinstance(node, Contact)}
    for node in nodes.values():
        if isinstance(node, Fault):
            missing = [contact_id for contact_id in node.surface_points if contact_id not in contact_ids]
            if missing:
                raise GraphValidationError(
                    f"Fault {node.id!r} references missing Contact id(s): {', '.join(missing)}"
                )


def _validate_within_edges(nodes: dict[str, object], edges: list[GraphEdge]) -> None:
    for edge in edges:
        if edge.kind is not EdgeKind.WITHIN:
            continue
        if not isinstance(nodes[edge.source], Sample):
            raise GraphValidationError(f"WITHIN edge {edge.id or edge.kind.value} source must be a Sample")
        if not isinstance(nodes[edge.target], StratigraphicUnit):
            raise GraphValidationError(
                f"WITHIN edge {edge.id or edge.kind.value} target must be a StratigraphicUnit"
            )


def _validate_at_edges(nodes: dict[str, object], edges: list[GraphEdge]) -> None:
    at_by_sample: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        if edge.kind is not EdgeKind.AT:
            continue
        if not isinstance(nodes[edge.source], Sample):
            raise GraphValidationError(f"AT edge {edge.id or edge.kind.value} source must be a Sample")
        if not isinstance(nodes[edge.target], Location):
            raise GraphValidationError(f"AT edge {edge.id or edge.kind.value} target must be a Location")
        at_by_sample[edge.source].append(edge)
    for node_id, node in nodes.items():
        if not isinstance(node, Sample):
            continue
        count = len(at_by_sample.get(node_id, []))
        if count != 1:
            raise GraphValidationError(f"Sample {node_id!r} must have exactly one outgoing AT edge")


def _validate_overlies_dag(nodes: dict[str, object], edges: list[GraphEdge]) -> None:
    graph = nx.DiGraph()
    for edge in edges:
        if edge.kind is not EdgeKind.OVERLIES:
            continue
        if edge.source == edge.target:
            raise GraphValidationError(f"OVERLIES self-loop on {edge.source!r}")
        source = _unit_id_for_node(nodes, edge.source)
        target = _unit_id_for_node(nodes, edge.target)
        graph.add_edge(source, target, edge=edge.id)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        formatted = " -> ".join(source for source, _target in cycle)
        raise GraphValidationError(f"OVERLIES cycle detected: {formatted}")


def _validate_member_of_series_function(edges: list[GraphEdge]) -> None:
    targets_by_source: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.kind is EdgeKind.MEMBER_OF_SERIES:
            targets_by_source[edge.source].add(edge.target)
    for source, targets in targets_by_source.items():
        if len(targets) > 1:
            raise GraphValidationError(f"unit {source!r} belongs to multiple series: {sorted(targets)}")


def _unit_id_for_node(nodes: dict[str, object], node_id: str) -> str:
    node = nodes[node_id]
    if isinstance(node, StratigraphicUnit):
        return node.unit_id
    return node_id
