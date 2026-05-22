"""Tests for edge schema and graph validation — must FAIL before implementation."""
import pytest
from pydantic import ValidationError
from datetime import datetime, timezone

from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Sample, StratigraphicUnit
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import GaussianUncertainty, PointUncertainty
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.graph.validate import GraphValidationError


def _prov():
    return Provenance(
        source="test",
        confidence=1.0,
        timestamp=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


# ── Enum round-trips and GeoSciML URI ─────────────────────────────────────────

def test_edge_kind_values_have_geosciml_uri():
    for kind in EdgeKind:
        assert kind.geosciml_uri.startswith("http"), f"{kind} missing URI"


def test_overlies_round_trip_via_name():
    k = EdgeKind["OVERLIES"]
    assert k == EdgeKind.OVERLIES


def test_in_contact_with_round_trip():
    k = EdgeKind["IN_CONTACT_WITH"]
    assert k == EdgeKind.IN_CONTACT_WITH


def test_within_round_trip():
    k = EdgeKind["WITHIN"]
    assert k == EdgeKind.WITHIN
    assert "cgi" in k.geosciml_uri


def test_contains_edge_kind_not_added():
    assert "CONTAINS" not in EdgeKind.__members__


# ── GraphEdge model ───────────────────────────────────────────────────────────

def test_graph_edge_round_trip():
    e = GraphEdge(
        id="e1",
        kind=EdgeKind.OVERLIES,
        source="u1",
        target="u2",
        p_exists=0.9,
        provenance=_prov(),
    )
    data = e.model_dump()
    e2 = GraphEdge.model_validate(data)
    assert e2 == e


def test_graph_edge_p_exists_out_of_range():
    with pytest.raises(ValidationError):
        GraphEdge(
            id="e2",
            kind=EdgeKind.OVERLIES,
            source="u1",
            target="u2",
            p_exists=1.5,
            provenance=_prov(),
        )


def test_within_requires_stratigraphic_unit_target():
    g = Graph()
    sample = Sample(
        id="s1",
        position=(PointUncertainty(value=1.0),) * 3,
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=GaussianUncertainty(mean=0.8, std=0.1),
        provenance=_prov(),
    )
    target_sample = sample.model_copy(update={"id": "s2"})
    g.add_node(sample)
    g.add_node(target_sample)

    with pytest.raises(GraphValidationError, match="WITHIN"):
        g.add_edge(GraphEdge(kind=EdgeKind.WITHIN, source="s1", target="s2", provenance=_prov()))


def test_within_accepts_sample_to_stratigraphic_unit():
    g = Graph()
    g.add_node(StratigraphicUnit(id="u1", unit_id="intrusion", series_id="s1", topology="layer", provenance=_prov()))
    g.add_node(
        Sample(
            id="s1",
            position=(PointUncertainty(value=1.0),) * 3,
            analyte="Cu",
            unit_of_measure="wt_pct",
            value=GaussianUncertainty(mean=0.8, std=0.1),
            provenance=_prov(),
        )
    )

    g.add_edge(GraphEdge(kind=EdgeKind.WITHIN, source="s1", target="u1", provenance=_prov()))

    assert len(g.get_edges(EdgeKind.WITHIN)) == 1
