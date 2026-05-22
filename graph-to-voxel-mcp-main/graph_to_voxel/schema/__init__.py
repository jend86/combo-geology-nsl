from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import (
    AnyNode,
    Contact,
    Fault,
    NodeValue,
    Location,
    ObservationPoint,
    Orientation,
    PositionWithUncertainty,
    Sample,
    Series,
    StratigraphicUnit,
)
from graph_to_voxel.schema.provenance import DerivationSpec, Provenance
from graph_to_voxel.schema.uncertainty import (
    CategoricalUncertainty,
    DistributionUncertainty,
    GaussianUncertainty,
    IntervalUncertainty,
    OrientationUncertainty,
    PointUncertainty,
    Uncertainty,
    UncertaintyValue,
)


class GraphDocument(BaseModel):
    nodes: list[AnyNode]
    edges: list[GraphEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_v16_samples(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            return data
        edges = list(data.get("edges", []))
        existing_at_sources = {
            edge.get("source")
            for edge in edges
            if isinstance(edge, dict) and str(edge.get("kind", "")).lower() in {"at", "edgekind.at"}
        }
        migrated = 0
        new_nodes: list[Any] = []
        new_edges: list[Any] = list(edges)
        for raw in nodes:
            inner = raw.get("root", raw) if isinstance(raw, dict) else None
            if not isinstance(inner, dict):
                new_nodes.append(raw)
                continue
            kind = inner.get("kind")
            if kind in ("sample", "Sample") and "position" in inner and inner.get("id") not in existing_at_sources:
                sample = dict(inner)
                position = sample.pop("position")
                sample_id = str(sample["id"])
                loc_id = f"loc__{sample_id}"
                provenance = sample["provenance"]
                new_nodes.append(
                    {
                        "kind": "location",
                        "id": loc_id,
                        "position": position,
                        "p_exists": sample.get("p_exists", 1.0),
                        "provenance": provenance,
                        "metadata": {"migrated_from": sample_id},
                    }
                )
                new_nodes.append(sample)
                new_edges.append(
                    {
                        "id": f"at__{sample_id}",
                        "kind": "at",
                        "source": sample_id,
                        "target": loc_id,
                        "provenance": provenance,
                        "p_exists": 1.0,
                    }
                )
                migrated += 1
            else:
                new_nodes.append(raw)
        if migrated == 0:
            return data
        metadata = dict(data.get("metadata", {}))
        metadata["v17_migration"] = {
            "samples_migrated": migrated,
            "policy": "auto_lifted_to_location",
        }
        return {**data, "nodes": new_nodes, "edges": new_edges, "metadata": metadata}

    @field_validator("nodes", mode="before")
    @classmethod
    def _parse_nodes(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [AnyNode.model_validate(item) for item in value]
        return value

    @property
    def node_values(self) -> list[NodeValue]:
        return [node.root for node in self.nodes]


__all__ = [
    "AnyNode",
    "CategoricalUncertainty",
    "Contact",
    "DistributionUncertainty",
    "DerivationSpec",
    "EdgeKind",
    "Fault",
    "GaussianUncertainty",
    "GraphDocument",
    "GraphEdge",
    "IntervalUncertainty",
    "Location",
    "NodeValue",
    "ObservationPoint",
    "Orientation",
    "OrientationUncertainty",
    "PointUncertainty",
    "PositionWithUncertainty",
    "Provenance",
    "Sample",
    "Series",
    "StratigraphicUnit",
    "Uncertainty",
    "UncertaintyValue",
]
