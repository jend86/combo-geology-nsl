"""Hand-authored toy graphs of known geometry for integration and golden tests."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from graph_to_voxel.schema.uncertainty import PointUncertainty, OrientationUncertainty, GaussianUncertainty
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.nodes import StratigraphicUnit, Series, Contact, Orientation
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.graph.core import Graph


def _prov() -> Provenance:
    return Provenance(
        source="fixture",
        confidence=1.0,
        timestamp=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def _pt(v: float) -> PointUncertainty:
    return PointUncertainty(value=v)


def two_unit_horizontal(z_interface: float = 500.0) -> Graph:
    """Two units separated by a horizontal interface at z=z_interface.

    Domain: x=[0,1000], y=[0,1000], z=[0,1000] (metres).
    Unit 'above' occupies z > z_interface; unit 'below' occupies z < z_interface.
    Interface is pinned by 5 contact points and 1 horizontal orientation.
    """
    g = Graph()

    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="above", unit_id="above", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="below", unit_id="below", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))

    g.add_edge(GraphEdge(id="e_series_above", kind=EdgeKind.MEMBER_OF_SERIES, source="above", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_series_below", kind=EdgeKind.MEMBER_OF_SERIES, source="below", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_overlies", kind=EdgeKind.OVERLIES, source="above", target="below", provenance=_prov()))

    # 5 contact points on the interface plane
    xs = [200.0, 400.0, 600.0, 800.0, 500.0]
    ys = [200.0, 200.0, 500.0, 800.0, 600.0]
    for i, (x, y) in enumerate(zip(xs, ys)):
        c = Contact(
            id=f"c{i}",
            position=(_pt(x), _pt(y), _pt(z_interface)),
            between=("above", "below"),
            p_exists=1.0,
            provenance=_prov(),
        )
        g.add_node(c)

    # 1 horizontal orientation (dip=0 means horizontal)
    o = Orientation(
        id="o0",
        position=(_pt(500.0), _pt(500.0), _pt(z_interface)),
        dip=OrientationUncertainty(dip_mean=0.0, dip_kappa=1e4, azimuth_mean=0.0, azimuth_kappa=1.0),
        for_unit="above",
        p_exists=1.0,
        provenance=_prov(),
    )
    g.add_node(o)

    return g


def tilted_layer(dip_deg: float = 30.0, dip_azimuth: float = 90.0) -> Graph:
    """Two units separated by a planar interface at dip=dip_deg / azimuth=dip_azimuth.

    The interface passes through the centre of a 1000x1000x1000m domain.
    Contact points are spread along the interface plane.
    """
    g = Graph()

    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="hanging", unit_id="hanging", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="footwall", unit_id="footwall", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))

    g.add_edge(GraphEdge(id="e_series_h", kind=EdgeKind.MEMBER_OF_SERIES, source="hanging", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_series_f", kind=EdgeKind.MEMBER_OF_SERIES, source="footwall", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_overlies", kind=EdgeKind.OVERLIES, source="hanging", target="footwall", provenance=_prov()))

    # Compute contact points on the tilted plane passing through (500,500,500).
    # Normal to plane: n = (sin(dip)*cos(az), sin(dip)*sin(az), cos(dip))  [geological convention]
    dip_r = np.deg2rad(dip_deg)
    az_r = np.deg2rad(dip_azimuth)
    normal = np.array([
        np.sin(dip_r) * np.cos(az_r),
        np.sin(dip_r) * np.sin(az_r),
        np.cos(dip_r),
    ])
    centre = np.array([500.0, 500.0, 500.0])

    # 5 contact points: displace in-plane from centre
    offsets_xy = [(-200, -200), (200, -200), (200, 200), (-200, 200), (0, 0)]
    for i, (dx, dy) in enumerate(offsets_xy):
        pt = centre.copy()
        pt[0] += dx
        pt[1] += dy
        # z such that dot(pt - centre, normal) = 0
        if abs(normal[2]) > 1e-9:
            pt[2] = centre[2] - (normal[0] * dx + normal[1] * dy) / normal[2]
        c = Contact(
            id=f"c{i}",
            position=(_pt(pt[0]), _pt(pt[1]), _pt(pt[2])),
            between=("hanging", "footwall"),
            p_exists=1.0,
            provenance=_prov(),
        )
        g.add_node(c)

    o = Orientation(
        id="o0",
        position=(_pt(500.0), _pt(500.0), _pt(500.0)),
        dip=OrientationUncertainty(
            dip_mean=dip_deg, dip_kappa=1e4,
            azimuth_mean=dip_azimuth, azimuth_kappa=1.0,
        ),
        for_unit="hanging",
        p_exists=1.0,
        provenance=_prov(),
    )
    g.add_node(o)

    return g


def cycle_graph() -> Graph:
    """A deliberately invalid graph with a cycle in OVERLIES — for negative tests."""
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    for uid in ("u1", "u2", "u3"):
        g.add_node(StratigraphicUnit(id=uid, unit_id=uid, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
        g.add_edge(GraphEdge(id=f"e_s_{uid}", kind=EdgeKind.MEMBER_OF_SERIES, source=uid, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e12", kind=EdgeKind.OVERLIES, source="u1", target="u2", provenance=_prov()))
    g.add_edge(GraphEdge(id="e23", kind=EdgeKind.OVERLIES, source="u2", target="u3", provenance=_prov()))
    # Caller must catch GraphValidationError when adding e31
    return g


def three_unit_valid_realised() -> Graph:
    """Three units A→B→C with contacts in the correct vertical order.

    A-B contact at z=700, B-C contact at z=300 — A is on top, C is at bottom.
    All positions are PointUncertainty (as if already realised).
    """
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    for uid in ("A", "B", "C"):
        g.add_node(StratigraphicUnit(id=uid, unit_id=uid, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
        g.add_edge(GraphEdge(id=f"es_{uid}", kind=EdgeKind.MEMBER_OF_SERIES, source=uid, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="eAB", kind=EdgeKind.OVERLIES, source="A", target="B", provenance=_prov()))
    g.add_edge(GraphEdge(id="eBC", kind=EdgeKind.OVERLIES, source="B", target="C", provenance=_prov()))
    # A-B contact at z=700 (upper boundary)
    g.add_node(Contact(id="cAB", position=(_pt(500.0), _pt(500.0), _pt(700.0)), between=("A", "B"), p_exists=1.0, provenance=_prov()))
    # B-C contact at z=300 (lower boundary)
    g.add_node(Contact(id="cBC", position=(_pt(500.0), _pt(500.0), _pt(300.0)), between=("B", "C"), p_exists=1.0, provenance=_prov()))
    return g


def three_unit_crossing_realised() -> Graph:
    """Three units A→B→C where the A-B contact is BELOW the B-C contact (crossing).

    A-B contact at z=200 (too low), B-C contact at z=700 (too high).
    Stratigraphic order is violated — stratigraphic_constrain must reject this.
    """
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    for uid in ("A", "B", "C"):
        g.add_node(StratigraphicUnit(id=uid, unit_id=uid, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
        g.add_edge(GraphEdge(id=f"es_{uid}", kind=EdgeKind.MEMBER_OF_SERIES, source=uid, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="eAB", kind=EdgeKind.OVERLIES, source="A", target="B", provenance=_prov()))
    g.add_edge(GraphEdge(id="eBC", kind=EdgeKind.OVERLIES, source="B", target="C", provenance=_prov()))
    # A-B contact at z=200 (should be top but is at bottom — crossing!)
    g.add_node(Contact(id="cAB", position=(_pt(500.0), _pt(500.0), _pt(200.0)), between=("A", "B"), p_exists=1.0, provenance=_prov()))
    # B-C contact at z=700 (should be bottom but is at top — crossing!)
    g.add_node(Contact(id="cBC", position=(_pt(500.0), _pt(500.0), _pt(700.0)), between=("B", "C"), p_exists=1.0, provenance=_prov()))
    return g


def crossing_uncertainty_graph() -> Graph:
    """Three units A→B→C with identical Gaussian uncertainty on both contact depths.

    Both A-B and B-C contacts have mean z=500 with std=300. Since the means are equal,
    ~50% of realisations will have the boundaries in the wrong order (layer crossings).
    Used to assert that stratigraphic_constrain catches these and n_rejected > 0.
    """
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    for uid in ("A", "B", "C"):
        g.add_node(StratigraphicUnit(id=uid, unit_id=uid, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
        g.add_edge(GraphEdge(id=f"es_{uid}", kind=EdgeKind.MEMBER_OF_SERIES, source=uid, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="eAB", kind=EdgeKind.OVERLIES, source="A", target="B", provenance=_prov()))
    g.add_edge(GraphEdge(id="eBC", kind=EdgeKind.OVERLIES, source="B", target="C", provenance=_prov()))

    gauss_ab = GaussianUncertainty(mean=600.0, std=200.0)
    gauss_bc = GaussianUncertainty(mean=400.0, std=200.0)
    # 3 contact points per boundary so the plane fit is well-determined
    for i, (x, y) in enumerate([(300.0, 300.0), (700.0, 300.0), (500.0, 700.0)]):
        g.add_node(Contact(
            id=f"cAB{i}", position=(_pt(x), _pt(y), gauss_ab),
            between=("A", "B"), p_exists=1.0, provenance=_prov(),
        ))
        g.add_node(Contact(
            id=f"cBC{i}", position=(_pt(x), _pt(y), gauss_bc),
            between=("B", "C"), p_exists=1.0, provenance=_prov(),
        ))
    return g


def sparse_existence_graph() -> Graph:
    """A unit with p_exists=0.05 — used by sense-check #6 test."""
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="common", unit_id="common", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="rare", unit_id="rare", series_id="s1", topology="layer", p_exists=0.05, provenance=_prov()))
    g.add_edge(GraphEdge(id="e_sc", kind=EdgeKind.MEMBER_OF_SERIES, source="common", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_sr", kind=EdgeKind.MEMBER_OF_SERIES, source="rare", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_ov", kind=EdgeKind.OVERLIES, source="common", target="rare", provenance=_prov()))
    return g
