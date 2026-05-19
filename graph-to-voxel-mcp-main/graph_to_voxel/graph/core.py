from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from graph_to_voxel.graph.validate import GraphValidationError, validate_graph
from graph_to_voxel.schema import GraphDocument
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import (
    AnyNode,
    Contact,
    Fault,
    Location,
    NodeValue,
    Orientation,
    PositionWithUncertainty,
    Sample,
    StratigraphicUnit,
)
from graph_to_voxel.schema.uncertainty import UncertaintyValue, sample_to_point


class RealisationInfeasible(GraphValidationError):
    """Raised when a sampled graph cannot satisfy the v1 ontology."""


class Graph:
    def __init__(
        self,
        nodes: Iterable[NodeValue] | None = None,
        edges: Iterable[GraphEdge] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._nodes: dict[str, NodeValue] = {}
        self._edges: list[GraphEdge] = []
        self.metadata = metadata or {}
        for node in nodes or []:
            self._nodes[node.id] = node
        for edge in edges or []:
            self._edges.append(self._edge_with_id(edge))
        self.validate()

    @classmethod
    def from_document(cls, document: GraphDocument) -> Graph:
        return cls(document.node_values, document.edges, metadata=document.metadata)

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        return cls.from_document(GraphDocument.model_validate(data))

    @classmethod
    def from_file(cls, path: str | Path) -> Graph:
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_document(self) -> GraphDocument:
        nodes: list[AnyNode] = []
        samples_with_locations = {edge.source for edge in self.get_edges(EdgeKind.AT)}
        for node in self._nodes.values():
            if isinstance(node, Sample) and node.id in samples_with_locations:
                node = node.model_copy(update={"position": None})
            nodes.append(AnyNode(root=node))
        return GraphDocument(
            nodes=nodes,
            edges=list(self._edges),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict:
        return self.to_document().model_dump(mode="json", exclude_none=True)

    def add_node(self, node: NodeValue) -> None:
        if (
            isinstance(node, Sample)
            and node.position is not None
            and not any(edge.kind is EdgeKind.AT and edge.source == node.id for edge in self._edges)
        ):
            self._add_legacy_positioned_sample(node)
            return
        previous = self._nodes.get(node.id)
        self._nodes[node.id] = node
        try:
            self.validate()
        except Exception:
            if previous is None:
                self._nodes.pop(node.id, None)
            else:
                self._nodes[node.id] = previous
            raise

    def add_edge(self, edge: GraphEdge) -> None:
        edge = self._edge_with_id(edge)
        self._edges.append(edge)
        try:
            self.validate()
        except Exception:
            self._edges.pop()
            raise

    def add_sample(
        self,
        *,
        unit_id: str,
        location: Location | str,
        analyte: str,
        unit_of_measure: str,
        value: UncertaintyValue,
        sample_id: str | None = None,
        provenance: object | None = None,
        p_exists: float = 1.0,
        metadata: dict | None = None,
    ) -> Sample:
        """Create a Sample and wire its AT and WITHIN edges atomically."""
        if isinstance(location, str):
            loc_node = self.get_node(location)
            if not isinstance(loc_node, Location):
                raise TypeError("location must reference a Location node")
            location_node = loc_node
        else:
            location_node = location
        sample_provenance = provenance or location_node.provenance
        if sample_id is None:
            sample_id = f"sample__{location_node.id}__{analyte}"
        sample = Sample(
            id=sample_id,
            analyte=analyte,
            unit_of_measure=unit_of_measure,
            value=value,
            p_exists=p_exists,
            provenance=sample_provenance,
            metadata=metadata or {},
        )
        unit_node_id = self._node_id_for_unit_id(unit_id)
        previous_node = self._nodes.get(sample.id)
        previous_location = self._nodes.get(location_node.id)
        edge_count = len(self._edges)
        self._nodes[location_node.id] = location_node
        self._nodes[sample.id] = sample
        self._edges.append(
            self._edge_with_id(
                GraphEdge(
                    kind=EdgeKind.AT,
                    source=sample.id,
                    target=location_node.id,
                    provenance=sample_provenance,
                )
            )
        )
        self._edges.append(
            self._edge_with_id(
                GraphEdge(
                    kind=EdgeKind.WITHIN,
                    source=sample.id,
                    target=unit_node_id,
                    provenance=sample_provenance,
                )
            )
        )
        try:
            self.validate()
        except Exception:
            self._edges = self._edges[:edge_count]
            if previous_node is None:
                self._nodes.pop(sample.id, None)
            else:
                self._nodes[sample.id] = previous_node
            if previous_location is None:
                self._nodes.pop(location_node.id, None)
            else:
                self._nodes[location_node.id] = previous_location
            raise
        return sample

    def get_node(self, node_id: str) -> NodeValue:
        return self._nodes[node_id]

    def node(self, node_id: str) -> NodeValue:
        return self.get_node(node_id)

    def node_ids(self) -> list[str]:
        return list(self._nodes)

    def nodes(self) -> list[NodeValue]:
        return list(self._nodes.values())

    def get_edges(self, kind: EdgeKind | None = None) -> list[GraphEdge]:
        if kind is None:
            return list(self._edges)
        return [edge for edge in self._edges if edge.kind is kind]

    def validate(self) -> None:
        validate_graph(self._nodes, self._edges)

    def unit_catalog(self) -> list[str]:
        return [node.unit_id for node in self._nodes.values() if isinstance(node, StratigraphicUnit)]

    def unit_node_by_unit_id(self) -> dict[str, StratigraphicUnit]:
        return {
            node.unit_id: node
            for node in self._nodes.values()
            if isinstance(node, StratigraphicUnit)
        }

    def samples_for_unit(self, unit_id: str) -> list[Sample]:
        unit_node_ids = {
            node.id
            for node in self._nodes.values()
            if isinstance(node, StratigraphicUnit) and node.unit_id == unit_id
        }
        if not unit_node_ids:
            return []
        samples: list[Sample] = []
        for edge in self.get_edges(EdgeKind.WITHIN):
            if edge.target not in unit_node_ids:
                continue
            node = self._nodes[edge.source]
            if isinstance(node, Sample):
                samples.append(node)
        return samples

    def location_for(self, sample: Sample) -> Location:
        edges = [edge for edge in self.get_edges(EdgeKind.AT) if edge.source == sample.id]
        if len(edges) != 1:
            raise GraphValidationError(f"Sample {sample.id!r} must have exactly one outgoing AT edge")
        node = self.get_node(edges[0].target)
        if not isinstance(node, Location):
            raise GraphValidationError(f"Sample {sample.id!r} AT target is not a Location")
        return node

    def samples_at(self, location_id: str) -> list[Sample]:
        samples: list[Sample] = []
        for edge in self.get_edges(EdgeKind.AT):
            if edge.target != location_id:
                continue
            node = self.get_node(edge.source)
            if isinstance(node, Sample):
                samples.append(node)
        return samples

    def position_array(self, node: object) -> np.ndarray:
        if isinstance(node, Sample) and node.id in self._nodes:
            return self.location_for(node).position_array()
        position_array = getattr(node, "position_array", None)
        if position_array is None:
            raise TypeError("node does not expose a position")
        warnings.warn(
            "Graph.position_array(node) is deprecated; use graph.location_for(sample).position_array() "
            "for samples or node.position_array() for positioned nodes",
            DeprecationWarning,
            stacklevel=2,
        )
        return position_array()

    def position_with_std(self, node: object) -> PositionWithUncertainty:
        if isinstance(node, Sample) and node.id in self._nodes:
            return self.location_for(node).position_with_std()
        position_with_std = getattr(node, "position_with_std", None)
        if position_with_std is None:
            raise TypeError("node does not expose a position")
        warnings.warn(
            "Graph.position_with_std(node) is deprecated; use graph.location_for(sample).position_with_std() "
            "for samples or node.position_with_std() for positioned nodes",
            DeprecationWarning,
            stacklevel=2,
        )
        return position_with_std()

    def stratigraphic_order(self) -> list[str]:
        graph = nx.DiGraph()
        for unit_id in self.unit_catalog():
            graph.add_node(unit_id)
        for edge in self.get_edges(EdgeKind.OVERLIES):
            source = self._unit_id_for_node_id(edge.source)
            target = self._unit_id_for_node_id(edge.target)
            graph.add_edge(source, target)
        if not graph.edges:
            return self.unit_catalog()
        rank = {
            unit.unit_id: int(unit.metadata.get("chronology_rank", 0))
            for unit in self.unit_node_by_unit_id().values()
        }
        return list(nx.lexicographical_topological_sort(graph, key=lambda unit_id: (rank.get(unit_id, 0), unit_id)))

    def realise(self, rng: np.random.Generator) -> Graph:
        kept_nodes: dict[str, NodeValue] = {}
        absent_unit_ids: set[str] = set()

        for node in self._nodes.values():
            if rng.uniform() > node.p_exists:
                if isinstance(node, StratigraphicUnit):
                    absent_unit_ids.add(node.unit_id)
                continue
            kept_nodes[node.id] = _realise_node(node, rng)

        for edge in self.get_edges(EdgeKind.AT):
            if edge.source in kept_nodes and edge.target not in kept_nodes:
                kept_nodes.pop(edge.source, None)

        for node_id, node in list(kept_nodes.items()):
            if isinstance(node, Contact) and any(unit_id in absent_unit_ids for unit_id in node.between):
                kept_nodes.pop(node_id)

        for node_id, node in list(kept_nodes.items()):
            if isinstance(node, Fault):
                surface_points = [contact_id for contact_id in node.surface_points if contact_id in kept_nodes]
                if len(surface_points) < 3:
                    kept_nodes.pop(node_id)
                elif len(surface_points) != len(node.surface_points):
                    kept_nodes[node_id] = node.model_copy(update={"surface_points": surface_points})

        kept_edges = []
        for edge in self._edges:
            if edge.source not in kept_nodes or edge.target not in kept_nodes:
                continue
            if rng.uniform() > edge.p_exists:
                continue
            kept_edges.append(deepcopy(edge))

        samples_with_at = {edge.source for edge in kept_edges if edge.kind is EdgeKind.AT}
        for node_id, node in list(kept_nodes.items()):
            if isinstance(node, Sample) and node_id not in samples_with_at:
                kept_nodes.pop(node_id)

        locations_by_sample = {edge.source: edge.target for edge in kept_edges if edge.kind is EdgeKind.AT}
        for sample_id, location_id in locations_by_sample.items():
            sample = kept_nodes.get(sample_id)
            location = kept_nodes.get(location_id)
            if isinstance(sample, Sample) and isinstance(location, Location):
                kept_nodes[sample_id] = sample.model_copy(deep=True, update={"position": location.position})

        try:
            return Graph(kept_nodes.values(), kept_edges, metadata=deepcopy(self.metadata))
        except GraphValidationError as exc:
            raise RealisationInfeasible(str(exc)) from exc

    def _edge_with_id(self, edge: GraphEdge) -> GraphEdge:
        if edge.id is not None:
            return edge
        return edge.model_copy(update={"id": f"e{len(self._edges) + 1}"})

    def _unit_id_for_node_id(self, node_id: str) -> str:
        node = self._nodes[node_id]
        if isinstance(node, StratigraphicUnit):
            return node.unit_id
        return node_id

    def _node_id_for_unit_id(self, unit_id: str) -> str:
        for node in self._nodes.values():
            if isinstance(node, StratigraphicUnit) and node.unit_id == unit_id:
                return node.id
        raise KeyError(unit_id)

    def _add_legacy_positioned_sample(self, sample: Sample) -> None:
        loc_id = self._location_id_for_position(sample.position) or f"loc__{sample.id}"
        existing_location = self._nodes.get(loc_id)
        location = (
            existing_location
            if isinstance(existing_location, Location)
            else Location(
                id=loc_id,
                position=sample.position,
                p_exists=sample.p_exists,
                provenance=sample.provenance,
                metadata={"migrated_from": sample.id},
            )
        )
        previous_sample = self._nodes.get(sample.id)
        previous_location = self._nodes.get(loc_id)
        edge_count = len(self._edges)
        self._nodes[loc_id] = location
        self._nodes[sample.id] = sample
        self._edges.append(
            self._edge_with_id(
                GraphEdge(
                    kind=EdgeKind.AT,
                    source=sample.id,
                    target=loc_id,
                    provenance=sample.provenance,
                )
            )
        )
        try:
            self.validate()
        except Exception:
            self._edges = self._edges[:edge_count]
            if previous_sample is None:
                self._nodes.pop(sample.id, None)
            else:
                self._nodes[sample.id] = previous_sample
            if previous_location is None:
                self._nodes.pop(loc_id, None)
            else:
                self._nodes[loc_id] = previous_location
            raise

    def _location_id_for_position(self, position: object) -> str | None:
        key = _position_key(position)
        if key is None:
            return None
        for node in self._nodes.values():
            if isinstance(node, Location) and _position_key(node.position) == key:
                return node.id
        return None


def _realise_node(node: NodeValue, rng: np.random.Generator) -> NodeValue:
    if isinstance(node, Contact):
        position = tuple(sample_to_point(component, rng) for component in node.position)
        return node.model_copy(deep=True, update={"position": position})
    if isinstance(node, Orientation):
        position = tuple(sample_to_point(component, rng) for component in node.position)
        return node.model_copy(deep=True, update={"position": position})
    if isinstance(node, Location):
        position = tuple(sample_to_point(component, rng) for component in node.position)
        return node.model_copy(deep=True, update={"position": position})
    if isinstance(node, Sample):
        return node.model_copy(
            deep=True,
            update={"value": sample_to_point(node.value, rng)},
        )
    return node.model_copy(deep=True)


EntityGraph = Graph


def _position_key(position: object) -> tuple[float, float, float] | None:
    if not isinstance(position, tuple) or len(position) != 3:
        return None
    values = []
    for component in position:
        value = getattr(component, "nominal", lambda: None)()
        if isinstance(value, tuple) or value is None:
            return None
        values.append(round(float(value), 9))
    return tuple(values)  # type: ignore[return-value]
