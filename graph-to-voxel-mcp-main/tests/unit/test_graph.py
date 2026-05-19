"""Tests for Graph wrapper and ontology validators — must FAIL before implementation."""
import pytest
from datetime import datetime, timezone

from graph_to_voxel.schema.uncertainty import GaussianUncertainty, PointUncertainty
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.nodes import Sample, StratigraphicUnit, Contact
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.graph.validate import GraphValidationError


def _prov():
    return Provenance(
        source="test",
        confidence=1.0,
        timestamp=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def _unit(uid, series="s1"):
    return StratigraphicUnit(id=uid, unit_id=uid, series_id=series, topology="layer", p_exists=1.0, provenance=_prov())


def _edge(eid, kind, src, tgt, p=1.0):
    return GraphEdge(id=eid, kind=kind, source=src, target=tgt, p_exists=p, provenance=_prov())


# ── Basic graph operations ────────────────────────────────────────────────────

def test_add_node_and_get():
    g = Graph()
    u = _unit("u1")
    g.add_node(u)
    assert g.get_node("u1") == u


def test_add_edge_and_get():
    g = Graph()
    g.add_node(_unit("u1"))
    g.add_node(_unit("u2"))
    e = _edge("e1", EdgeKind.OVERLIES, "u1", "u2")
    g.add_edge(e)
    edges = g.get_edges(kind=EdgeKind.OVERLIES)
    assert any(e2.id == "e1" for e2 in edges)


def test_node_ids():
    g = Graph()
    g.add_node(_unit("a"))
    g.add_node(_unit("b"))
    assert set(g.node_ids()) == {"a", "b"}


# ── Ontology validation: OVERLIES must be DAG ────────────────────────────────

def test_overlies_cycle_rejected():
    g = Graph()
    g.add_node(_unit("u1"))
    g.add_node(_unit("u2"))
    g.add_node(_unit("u3"))
    g.add_edge(_edge("e1", EdgeKind.OVERLIES, "u1", "u2"))
    g.add_edge(_edge("e2", EdgeKind.OVERLIES, "u2", "u3"))
    with pytest.raises(GraphValidationError, match="cycle"):
        g.add_edge(_edge("e3", EdgeKind.OVERLIES, "u3", "u1"))


def test_overlies_self_loop_rejected():
    g = Graph()
    g.add_node(_unit("u1"))
    with pytest.raises(GraphValidationError):
        g.add_edge(_edge("e1", EdgeKind.OVERLIES, "u1", "u1"))


# ── Ontology validation: MEMBER_OF_SERIES is function (one series per unit) ──

def test_member_of_series_multiplicity_rejected():
    from graph_to_voxel.schema.nodes import Series
    g = Graph()
    g.add_node(_unit("u1"))
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(Series(id="s2", p_exists=1.0, provenance=_prov()))
    g.add_edge(_edge("e1", EdgeKind.MEMBER_OF_SERIES, "u1", "s1"))
    with pytest.raises(GraphValidationError, match="series"):
        g.add_edge(_edge("e2", EdgeKind.MEMBER_OF_SERIES, "u1", "s2"))


# ── Ontology validation: Contact.between references valid unit_ids ────────────

def test_contact_dangling_reference_rejected():
    g = Graph()
    pos = (GaussianUncertainty(mean=0.0, std=1.0),) * 3
    c = Contact(id="c1", position=pos, between=("nonexistent_a", "nonexistent_b"), p_exists=1.0, provenance=_prov())
    with pytest.raises(GraphValidationError, match="unit"):
        g.add_node(c)  # validation runs on add


# ── realise() determinism ─────────────────────────────────────────────────────

def test_realise_deterministic():
    import numpy as np
    g = Graph()
    pos = (GaussianUncertainty(mean=100.0, std=5.0),
           GaussianUncertainty(mean=200.0, std=5.0),
           GaussianUncertainty(mean=-50.0, std=2.0))
    g.add_node(_unit("u1"))
    g.add_node(_unit("u2"))
    c = Contact(id="c1", position=pos, between=("u1", "u2"), p_exists=1.0, provenance=_prov())
    g.add_node(c)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    g1 = g.realise(rng1)
    g2 = g.realise(rng2)

    c1 = g1.get_node("c1")
    c2 = g2.get_node("c1")
    # positions are now Point (realised), so values should match
    assert c1.position[0].value == c2.position[0].value


def test_realise_point_unchanged():
    import numpy as np
    g = Graph()
    pos = (PointUncertainty(value=42.0),) * 3
    g.add_node(_unit("u1"))
    g.add_node(_unit("u2"))
    c = Contact(id="c1", position=pos, between=("u1", "u2"), p_exists=1.0, provenance=_prov())
    g.add_node(c)
    rng = np.random.default_rng(0)
    g2 = g.realise(rng)
    assert g2.get_node("c1").position[0].value == 42.0


# ── JSON IO ───────────────────────────────────────────────────────────────────

def test_graph_json_round_trip(tmp_path):
    from graph_to_voxel.graph.io import save_graph, load_graph
    g = Graph()
    g.add_node(_unit("u1"))
    g.add_node(_unit("u2"))
    g.add_edge(_edge("e1", EdgeKind.OVERLIES, "u1", "u2"))
    path = tmp_path / "test.json"
    save_graph(g, path)
    g2 = load_graph(path)
    assert set(g2.node_ids()) == {"u1", "u2"}
    assert len(list(g2.get_edges(kind=EdgeKind.OVERLIES))) == 1


def test_samples_for_unit_uses_within_edges():
    g = Graph()
    g.add_node(_unit("u1"))
    g.add_node(_unit("u2"))
    sample = Sample(
        id="s1",
        position=(PointUncertainty(value=1.0), PointUncertainty(value=2.0), PointUncertainty(value=3.0)),
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=GaussianUncertainty(mean=0.8, std=0.1),
        provenance=_prov(),
    )
    g.add_node(sample)
    g.add_edge(_edge("e1", EdgeKind.WITHIN, "s1", "u1"))

    assert g.samples_for_unit("u1") == [sample]
    assert g.samples_for_unit("u2") == []


def test_position_array_uses_nominal_values():
    g = Graph()
    sample = Sample(
        id="s1",
        position=(
            PointUncertainty(value=1.0),
            GaussianUncertainty(mean=2.0, std=0.1),
            PointUncertainty(value=3.0),
        ),
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=GaussianUncertainty(mean=0.8, std=0.1),
        provenance=_prov(),
    )

    with pytest.warns(DeprecationWarning, match="position_array"):
        assert g.position_array(sample).tolist() == [1.0, 2.0, 3.0]


def test_position_with_std_wrapper_delegates_with_deprecation():
    g = Graph()
    sample = Sample(
        id="s1",
        position=(
            PointUncertainty(value=1.0),
            GaussianUncertainty(mean=2.0, std=0.25),
            PointUncertainty(value=3.0),
        ),
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=GaussianUncertainty(mean=0.8, std=0.1),
        provenance=_prov(),
    )

    with pytest.warns(DeprecationWarning, match="position_with_std"):
        position = g.position_with_std(sample)

    assert position.nominal.tolist() == [1.0, 2.0, 3.0]
    assert position.std.tolist() == [0.0, 0.25, 0.0]
    assert position.covariance is None


def test_realise_samples_sample_position_and_value():
    import numpy as np

    g = Graph()
    sample = Sample(
        id="s1",
        position=(GaussianUncertainty(mean=1.0, std=0.1),) * 3,
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=GaussianUncertainty(mean=0.8, std=0.1),
        provenance=_prov(),
    )
    g.add_node(sample)

    realised = g.realise(np.random.default_rng(0)).get_node("s1")

    assert isinstance(realised.position[0], PointUncertainty)
    assert isinstance(realised.value, PointUncertainty)


def test_realise_co_located_samples_share_auto_lifted_location():
    import numpy as np

    g = Graph()
    position = (PointUncertainty(value=1.0), PointUncertainty(value=2.0), PointUncertainty(value=3.0))
    for sample_id, analyte in [("s_cu", "Cu"), ("s_s", "S")]:
        g.add_node(
            Sample(
                id=sample_id,
                position=position,
                analyte=analyte,
                unit_of_measure="wt_pct",
                value=GaussianUncertainty(mean=0.8, std=0.1),
                provenance=_prov(),
            )
        )

    realised = g.realise(np.random.default_rng(0))

    assert g.location_for(g.get_node("s_cu")).id == g.location_for(g.get_node("s_s")).id
    assert realised.position_array(realised.get_node("s_cu")).tolist() == realised.position_array(realised.get_node("s_s")).tolist()


def test_realise_does_not_warn_without_co_located_samples(recwarn):
    import numpy as np

    g = Graph()
    g.add_node(
        Sample(
            id="s1",
            position=(PointUncertainty(value=1.0), PointUncertainty(value=2.0), PointUncertainty(value=3.0)),
            analyte="Cu",
            unit_of_measure="wt_pct",
            value=GaussianUncertainty(mean=0.8, std=0.1),
            provenance=_prov(),
        )
    )

    g.realise(np.random.default_rng(0))

    assert not recwarn
