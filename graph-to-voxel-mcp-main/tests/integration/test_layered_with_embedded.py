"""Integration tests for mixed layered+embedded body graphs.

These tests cover the §1.x gaps surfaced by the v1.7 stress-test (doc 05).
Each test must FAIL before the corresponding fix lands.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest

from graph_to_voxel.engine.loopstructural import (
    AutoAnchoredWarning,
    FaultsIgnoredWarning,
    GridSpec,
    InsufficientUnitDataError,
    PolarityIgnoredOnEmbeddedWarning,
    build_voxel_field,
)
from graph_to_voxel.analyses.checks import check_voxel_stratigraphic_order
from graph_to_voxel.graph import EntityGraph
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Contact, Fault, StratigraphicUnit
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import GaussianUncertainty, PointUncertainty
from graph_to_voxel.voxel.ensemble import run_ensemble


def _prov() -> Provenance:
    return Provenance(
        source="stress-test",
        confidence=1.0,
        timestamp=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )


def _pt(v: float) -> PointUncertainty:
    return PointUncertainty(value=v)


def _gauss(mean: float, std: float) -> GaussianUncertainty:
    return GaussianUncertainty(mean=mean, std=std)


def _unit(
    unit_id: str,
    anchor: tuple[float, float, float] | None = None,
    *,
    topology: str | None = None,
    chronology_rank: int | None = None,
) -> StratigraphicUnit:
    metadata = {"chronology_rank": chronology_rank} if chronology_rank is not None else {}
    return StratigraphicUnit(
        id=unit_id,
        unit_id=unit_id,
        series_id="s1",
        topology=topology or ("embedded" if anchor is not None else "layer"),
        anchor_inside=anchor,
        provenance=_prov(),
        metadata=metadata,
    )


def _contact(
    cid: str,
    pos: tuple[float, float, float],
    between: tuple[str, str],
    polarity: int | None = None,
) -> Contact:
    return Contact(
        id=cid,
        position=(_pt(pos[0]), _pt(pos[1]), _pt(pos[2])),
        between=between,
        provenance=_prov(),
        polarity=polarity,
    )


def _overlies(src: str, tgt: str, edge_id: str | None = None, **meta) -> GraphEdge:
    return GraphEdge(
        id=edge_id or f"e_{src}__{tgt}",
        kind=EdgeKind.OVERLIES,
        source=src,
        target=tgt,
        provenance=_prov(),
        metadata=meta,
    )


def _sphere_contacts(
    centre: tuple[float, float, float],
    radius: float,
    between: tuple[str, str],
    prefix: str,
) -> list[Contact]:
    """8 axis-aligned contacts on a sphere surface."""
    cx, cy, cz = centre
    directions = [
        (radius, 0, 0), (-radius, 0, 0),
        (0, radius, 0), (0, -radius, 0),
        (0, 0, radius), (0, 0, -radius),
        (radius * 0.7, radius * 0.7, 0), (-radius * 0.7, -radius * 0.7, 0),
    ]
    return [
        _contact(f"{prefix}_{i}", (cx + dx, cy + dy, cz + dz), between)
        for i, (dx, dy, dz) in enumerate(directions)
    ]


# ── §1.1 — Layered unit count stable after adding embedded body ───────────────


def _make_cap_basement_graph(with_intrusion: bool) -> EntityGraph:
    """
    cap (younger) conformably overlies basement (older).
    Cap-basement interface is a horizontal plane at z=7.
    Optionally add an anchored intrusion embedded in basement centred at (5,5,3).
    """
    nodes: list = [
        _unit("cap"),
        _unit("basement"),
        # cap-basement interface: 5 contacts at z=7
        _contact("cb0", (2.0, 2.0, 7.0), ("cap", "basement")),
        _contact("cb1", (8.0, 2.0, 7.0), ("cap", "basement")),
        _contact("cb2", (8.0, 8.0, 7.0), ("cap", "basement")),
        _contact("cb3", (2.0, 8.0, 7.0), ("cap", "basement")),
        _contact("cb4", (5.0, 5.0, 7.0), ("cap", "basement")),
    ]
    edges: list[GraphEdge] = [_overlies("cap", "basement")]

    if with_intrusion:
        nodes.append(_unit("intrusion", anchor=(5.0, 5.0, 3.0)))
        nodes.extend(
            _sphere_contacts((5.0, 5.0, 3.0), 1.5, ("intrusion", "basement"), "ci")
        )
        edges.append(_overlies("intrusion", "basement"))

    return EntityGraph(nodes=nodes, edges=edges)


def test_layered_with_embedded_no_unit_leakage() -> None:
    """§1.1: adding an anchored embedded body must not shift voxels to unrelated layered unit."""
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field1 = build_voxel_field(_make_cap_basement_graph(with_intrusion=False), grid, bandwidth=0.3)
        field2 = build_voxel_field(_make_cap_basement_graph(with_intrusion=True), grid, bandwidth=0.3)

    cap_idx1 = field1.unit_ids.index("cap")
    cap_idx2 = field2.unit_ids.index("cap")
    cap_count1 = int(np.count_nonzero(field1.most_likely_unit == cap_idx1))
    cap_count2 = int(np.count_nonzero(field2.most_likely_unit == cap_idx2))

    # Cap voxel count must not grow when an embedded body is added to basement.
    # Allow ≤30 voxels of drift (sigmoid tails); the bug shifts ~100 voxels.
    delta = cap_count2 - cap_count1
    assert delta <= 30, (
        f"cap grew by {delta} voxels after adding intrusion "
        f"(step1={cap_count1}, step2={cap_count2}); plane pollution suspected"
    )


# ── §1.2 — Declared embedded bodies auto-anchor or fail cleanly ──────────────


def test_embedded_unit_without_anchor_auto_anchors() -> None:
    """§1.2: declared embedded closed-envelope unit without anchor auto-anchors."""
    centre = (5.0, 5.0, 5.0)
    radius = 2.0

    def _make_graph(anchor: tuple[float, float, float] | None) -> EntityGraph:
        return EntityGraph(
            nodes=[
                _unit("host"),
                _unit("intrusion", anchor=anchor, topology="embedded"),
                *_sphere_contacts(centre, radius, ("intrusion", "host"), "ci"),
            ],
            edges=[_overlies("intrusion", "host")],
        )

    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(11, 11, 11))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field_anchored = build_voxel_field(_make_graph(anchor=centre), grid, bandwidth=0.3)
    with pytest.warns(AutoAnchoredWarning):
        field_auto = build_voxel_field(_make_graph(anchor=None), grid, bandwidth=0.3)

    intr_idx_a = field_anchored.unit_ids.index("intrusion")
    intr_idx_auto = field_auto.unit_ids.index("intrusion")
    np.testing.assert_allclose(
        field_auto.support_membership[intr_idx_auto],
        field_anchored.support_membership[intr_idx_a],
        atol=0.2,
    )


def test_embedded_unit_open_envelope_raises() -> None:
    """§1.2: declared embedded unit without anchor and open/coplanar contacts raises cleanly."""
    nodes: list = [
        _unit("host"),
        _unit("intrusion", topology="embedded"),
        _contact("ci0", (2.0, 2.0, 5.0), ("intrusion", "host")),
        _contact("ci1", (8.0, 2.0, 5.0), ("intrusion", "host")),
        _contact("ci2", (5.0, 8.0, 5.0), ("intrusion", "host")),
    ]
    edges = [_overlies("intrusion", "host")]
    graph = EntityGraph(nodes=nodes, edges=edges)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    with pytest.raises(InsufficientUnitDataError) as exc_info:
        build_voxel_field(graph, grid)

    assert exc_info.value.unit_id == "intrusion"


# ── §1.3 — stratigraphic_constrain must not reject embedded bodies ────────────


def _make_nested_embedded_graph() -> EntityGraph:
    """
    halo (embedded in intrusion) + intrusion (embedded in basement) — all contacts
    centred at (5,5,5) with small Gaussian spread.  stratigraphic_constrain sees
    halo→intrusion mean_z ≈ intrusion→basement mean_z and must NOT reject.
    """
    anchor = (5.0, 5.0, 5.0)
    nodes: list = [
        _unit("halo", anchor=anchor),
        _unit("intrusion", anchor=anchor),
        _unit("basement"),
        # intrusion-basement contacts: sphere around (5,5,5) radius 2
        *[
            Contact(
                id=f"ci_{i}",
                position=(_pt(cx), _pt(cy), _gauss(cz, 0.1)),
                between=("intrusion", "basement"),
                provenance=_prov(),
            )
            for i, (cx, cy, cz) in enumerate([
                (7.0, 5.0, 5.0), (3.0, 5.0, 5.0),
                (5.0, 7.0, 5.0), (5.0, 3.0, 5.0),
                (5.0, 5.0, 7.0), (5.0, 5.0, 3.0),
                (6.4, 6.4, 5.0), (3.6, 3.6, 5.0),
            ])
        ],
        # halo-intrusion contacts: smaller sphere around (5,5,5) radius 1
        *[
            Contact(
                id=f"ch_{i}",
                position=(_pt(cx), _pt(cy), _gauss(cz, 0.1)),
                between=("halo", "intrusion"),
                provenance=_prov(),
            )
            for i, (cx, cy, cz) in enumerate([
                (6.0, 5.0, 5.0), (4.0, 5.0, 5.0),
                (5.0, 6.0, 5.0), (5.0, 4.0, 5.0),
                (5.0, 5.0, 6.0), (5.0, 5.0, 4.0),
                (5.7, 5.7, 5.0), (4.3, 4.3, 5.0),
            ])
        ],
    ]
    edges = [_overlies("halo", "intrusion"), _overlies("intrusion", "basement")]
    return EntityGraph(nodes=nodes, edges=edges)


def test_run_ensemble_with_embedded_body_under_uncertainty() -> None:
    """§1.3: nested embedded bodies must not be rejected by stratigraphic_constrain."""
    graph = _make_nested_embedded_graph()
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(5, 5, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble = run_ensemble(
            graph,
            lambda g: build_voxel_field(g, grid, bandwidth=0.5),
            n=4,
            seed=11,
            oversample_budget=20,
        )

    assert len(ensemble.realisations) == 4, (
        f"expected 4 realisations, got {len(ensemble.realisations)}; "
        f"n_rejected={ensemble.n_rejected}"
    )
    assert ensemble.n_rejected <= 1, (
        f"expected <=1 rejections but got {ensemble.n_rejected}; "
        "stratigraphic_constrain is incorrectly rejecting embedded bodies"
    )


# ── §1.4 — Fault nodes trigger FaultsIgnoredWarning and attrs entry ──────────


def test_fault_emits_ignored_warning_and_attr() -> None:
    """§1.4: presence of a Fault node must emit FaultsIgnoredWarning and populate attrs."""
    prov = _prov()
    # Minimal two-unit graph + a Fault node + OFFSET_BY edge.
    nodes: list = [
        _unit("above"),
        _unit("below"),
        _contact("cb0", (2.0, 2.0, 5.0), ("above", "below")),
        _contact("cb1", (8.0, 2.0, 5.0), ("above", "below")),
        _contact("cb2", (5.0, 8.0, 5.0), ("above", "below")),
        Fault(
            id="f_main",
            surface_points=["cb0", "cb1", "cb2"],
            provenance=prov,
        ),
    ]
    edges = [
        _overlies("above", "below"),
        GraphEdge(kind=EdgeKind.OFFSET_BY, source="cb0", target="f_main", provenance=prov),
    ]
    graph = EntityGraph(nodes=nodes, edges=edges)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(5, 5, 5))

    with pytest.warns(FaultsIgnoredWarning):
        field = build_voxel_field(graph, grid)

    assert "faults_ignored" in field.attrs
    assert field.attrs["faults_ignored"] == ["f_main"]
    assert field.attrs["offset_by_edges_ignored"] == [("cb0", "f_main")]


# ── §1.5 — Contact.polarity is layered-only ──────────────────────────────────


def test_polarity_on_embedded_warns_and_is_ignored() -> None:
    """§1.5: embedded bodies ignore polarity and keep anchor-derived support."""
    centre = (5.0, 5.0, 5.0)
    radius = 2.0

    def _make_graph(with_polarity: bool) -> EntityGraph:
        polarity = 1 if with_polarity else None
        nodes: list = [
            _unit("host"),
            _unit("intrusion", anchor=centre),
            *[
                _contact(f"ci_{i}", (cx, cy, cz), ("intrusion", "host"), polarity)
                for i, (cx, cy, cz) in enumerate([
                    (centre[0] + radius, centre[1], centre[2]),
                    (centre[0] - radius, centre[1], centre[2]),
                    (centre[0], centre[1] + radius, centre[2]),
                    (centre[0], centre[1] - radius, centre[2]),
                    (centre[0], centre[1], centre[2] + radius),
                    (centre[0], centre[1], centre[2] - radius),
                    (centre[0] + radius * 0.7, centre[1] + radius * 0.7, centre[2]),
                    (centre[0] - radius * 0.7, centre[1] - radius * 0.7, centre[2]),
                ])
            ],
        ]
        return EntityGraph(nodes=nodes, edges=[_overlies("intrusion", "host")])

    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(11, 11, 11))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field_no_polarity = build_voxel_field(_make_graph(with_polarity=False), grid, bandwidth=0.3)
    with pytest.warns(PolarityIgnoredOnEmbeddedWarning):
        field_polarity = build_voxel_field(_make_graph(with_polarity=True), grid, bandwidth=0.3)

    intr_idx_a = field_no_polarity.unit_ids.index("intrusion")
    intr_idx_p = field_polarity.unit_ids.index("intrusion")
    np.testing.assert_allclose(
        field_polarity.support_membership[intr_idx_p],
        field_no_polarity.support_membership[intr_idx_a],
        atol=1e-6,
    )


def test_polarity_flips_layered_sign() -> None:
    """§1.5: layered polarity overrides an inverted OVERLIES direction."""
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    def _make_graph(source: str, target: str, polarity: int | None) -> EntityGraph:
        return EntityGraph(
            nodes=[
                _unit("above"),
                _unit("below"),
                _contact("c0", (2.0, 2.0, 5.0), ("above", "below"), polarity),
                _contact("c1", (8.0, 2.0, 5.0), ("above", "below"), polarity),
                _contact("c2", (5.0, 8.0, 5.0), ("above", "below"), polarity),
            ],
            edges=[_overlies(source, target)],
        )

    baseline = build_voxel_field(_make_graph("above", "below", None), grid, bandwidth=0.3)
    corrected = build_voxel_field(_make_graph("below", "above", 1), grid, bandwidth=0.3)
    above_baseline = baseline.unit_ids.index("above")
    above_corrected = corrected.unit_ids.index("above")

    np.testing.assert_allclose(
        corrected.support_membership[above_corrected],
        baseline.support_membership[above_baseline],
        atol=0.05,
    )


# ── §1.6 — Sibling intrusions share overlap symmetrically ────────────────────


def _make_sibling_graph(
    a_first: bool = True,
    *,
    rank_a: int | None = None,
    rank_b: int | None = None,
) -> EntityGraph:
    """Two intrusions A and B, both OVERLIES host. Symmetric geometry with overlap at x=5."""
    centre_a = (4.0, 5.0, 5.0)
    centre_b = (6.0, 5.0, 5.0)
    radius = 2.0

    nodes: list = [
        _unit("host"),
        _unit("A", anchor=centre_a, chronology_rank=rank_a),
        _unit("B", anchor=centre_b, chronology_rank=rank_b),
        *_sphere_contacts(centre_a, radius, ("A", "host"), "ca"),
        *_sphere_contacts(centre_b, radius, ("B", "host"), "cb"),
    ]
    edges = (
        [_overlies("A", "host"), _overlies("B", "host")]
        if a_first
        else [_overlies("B", "host"), _overlies("A", "host")]
    )
    return EntityGraph(nodes=nodes, edges=edges)


def test_sibling_intrusions_share_overlap_symmetrically() -> None:
    """§1.6: two siblings overlying the same host must split their overlap symmetrically."""
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field_ab = build_voxel_field(_make_sibling_graph(a_first=True), grid, bandwidth=0.4)
        field_ba = build_voxel_field(_make_sibling_graph(a_first=False), grid, bandwidth=0.4)

    a_idx = field_ab.unit_ids.index("A")
    b_idx = field_ab.unit_ids.index("B")

    # Deep overlap: cells where BOTH intrusions have support > 0.45.
    # These are cells inside both ellipsoids simultaneously (near the intersection
    # plane), where the two anchors are roughly equidistant.
    support_a = field_ab.support_membership[a_idx]
    support_b = field_ab.support_membership[b_idx]
    overlap = (support_a > 0.3) & (support_b > 0.3)
    deep_overlap = (support_a > 0.45) & (support_b > 0.45)

    assert overlap.any(), "no overlap voxels found between the two intrusions"

    # Order-independence: results with A-first and B-first must match exactly
    # (simultaneous sibling erosion is deterministic and order-agnostic).
    a_idx_ba = field_ba.unit_ids.index("A")
    b_idx_ba = field_ba.unit_ids.index("B")
    max_order_diff_a = float(np.abs(field_ab.unit_probs[a_idx][overlap] - field_ba.unit_probs[a_idx_ba][overlap]).max())
    max_order_diff_b = float(np.abs(field_ab.unit_probs[b_idx][overlap] - field_ba.unit_probs[b_idx_ba][overlap]).max())
    assert max_order_diff_a < 0.05, f"A probs differ by {max_order_diff_a:.3f} with insertion order swap"
    assert max_order_diff_b < 0.05, f"B probs differ by {max_order_diff_b:.3f} with insertion order swap"

    # Spread in the deep-overlap zone: cells near the intersection plane are
    # roughly equidistant from both anchors; spread should be < 0.15.
    if deep_overlap.any():
        probs_a_deep = field_ab.unit_probs[a_idx][deep_overlap]
        probs_b_deep = field_ab.unit_probs[b_idx][deep_overlap]
        spread = float(np.abs(probs_a_deep - probs_b_deep).max())
        assert spread < 0.15, (
            f"max |p_A - p_B| = {spread:.3f} in deep overlap; siblings are not sharing equally "
            "(cascade order leak suspected)"
        )
        assert float(field_ab.entropy[deep_overlap].mean()) > 0.5


def test_sibling_rank_tiebreak_deterministic() -> None:
    """§1.6: unequal chronology_rank resolves sibling overlap deterministically."""
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field_ab = build_voxel_field(_make_sibling_graph(a_first=True, rank_a=1, rank_b=2), grid, bandwidth=0.4)
        field_ba = build_voxel_field(_make_sibling_graph(a_first=False, rank_a=1, rank_b=2), grid, bandwidth=0.4)

    a_idx = field_ab.unit_ids.index("A")
    b_idx = field_ab.unit_ids.index("B")
    support_a = field_ab.support_membership[a_idx]
    support_b = field_ab.support_membership[b_idx]
    overlap = (support_a > 0.3) & (support_b > 0.3)
    assert overlap.any(), "no overlap voxels found between the two intrusions"

    assert float(field_ab.unit_probs[b_idx][overlap].mean()) > float(field_ab.unit_probs[a_idx][overlap].mean())
    np.testing.assert_allclose(field_ab.unit_probs[a_idx], field_ba.unit_probs[field_ba.unit_ids.index("A")], atol=1e-6)
    np.testing.assert_allclose(field_ab.unit_probs[b_idx], field_ba.unit_probs[field_ba.unit_ids.index("B")], atol=1e-6)


def test_check_voxel_stratigraphic_order_ignores_embedded() -> None:
    """§1.9: embedded bodies can interrupt a vertical host sequence without failing the check."""
    graph = _make_cap_basement_graph(with_intrusion=True)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = build_voxel_field(graph, grid, bandwidth=0.3)

    assert check_voxel_stratigraphic_order(graph, field).severity == "pass"


def test_sibling_intrusions_with_chronology_a_wins() -> None:
    """§1.6c: when A OVERLIES B (explicit chronology), A dominates in the overlap region."""
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    centre_a = (4.0, 5.0, 5.0)
    centre_b = (6.0, 5.0, 5.0)
    radius = 2.0

    nodes: list = [
        _unit("host"),
        _unit("A", anchor=centre_a),
        _unit("B", anchor=centre_b),
        *_sphere_contacts(centre_a, radius, ("A", "host"), "ca"),
        *_sphere_contacts(centre_b, radius, ("B", "host"), "cb"),
        # A-B contacts in overlap zone (needed for A OVERLIES B pair)
        _contact("cab0", (5.0, 4.0, 5.0), ("A", "B")),
        _contact("cab1", (5.0, 6.0, 5.0), ("A", "B")),
        _contact("cab2", (5.0, 5.0, 4.0), ("A", "B")),
    ]
    edges = [
        _overlies("A", "host"),
        _overlies("B", "host"),
        _overlies("A", "B"),  # A is younger than B → A wins in overlap
    ]
    graph = EntityGraph(nodes=nodes, edges=edges)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = build_voxel_field(graph, grid, bandwidth=0.4)

    a_idx = field.unit_ids.index("A")
    b_idx = field.unit_ids.index("B")
    support_a = field.support_membership[a_idx]
    support_b = field.support_membership[b_idx]
    overlap = (support_a > 0.3) & (support_b > 0.3)

    if not overlap.any():
        pytest.skip("no overlap voxels found")

    probs_a = field.unit_probs[a_idx][overlap]
    probs_b = field.unit_probs[b_idx][overlap]
    # A must win clearly in the overlap: p_A > p_B on average
    assert float(probs_a.mean()) > float(probs_b.mean()), (
        f"A (mean={probs_a.mean():.3f}) should dominate B (mean={probs_b.mean():.3f}) "
        "since A OVERLIES B"
    )
