from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from graph_to_voxel.engine.loopstructural import (
    BandwidthMismatchWarning,
    GridSpec,
    InsufficientUnitDataError,
    MemoryBudgetWarning,
    build_voxel_field,
)
from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.graph import EntityGraph, GraphValidationError
from graph_to_voxel.schema import GraphDocument
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Contact, Location, Sample, StratigraphicUnit
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import GaussianUncertainty, PointUncertainty
from graph_to_voxel.voxel.ensemble import Ensemble
from tests.fixtures.toy_graphs import two_unit_horizontal


def _prov() -> Provenance:
    return Provenance(
        source="v17-test",
        confidence=1.0,
        timestamp=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )


def _pt(value: float) -> PointUncertainty:
    return PointUncertainty(value=value)


def _sample(sample_id: str, analyte: str) -> Sample:
    return Sample(
        id=sample_id,
        analyte=analyte,
        unit_of_measure="wt_pct",
        value=GaussianUncertainty(mean=0.8, std=0.1),
        provenance=_prov(),
    )


def _unit(unit_id: str, anchor: tuple[float, float, float] | None = None) -> StratigraphicUnit:
    return StratigraphicUnit(
        id=unit_id,
        unit_id=unit_id,
        series_id="s1",
        topology="embedded" if anchor is not None else "layer",
        anchor_inside=anchor,
        provenance=_prov(),
    )


def _contact(contact_id: str, position: tuple[float, float, float], between: tuple[str, str]) -> Contact:
    return Contact(
        id=contact_id,
        position=(_pt(position[0]), _pt(position[1]), _pt(position[2])),
        between=between,
        provenance=_prov(),
    )


def test_build_voxel_field_emits_mixing_support_membership_and_plane_probabilities() -> None:
    graph = two_unit_horizontal(z_interface=5.5)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))

    field = build_voxel_field(graph, grid, bandwidth=0.5)

    upper_idx = field.unit_ids.index("above")
    assert field.attrs["unit_probs_kind"] == "mixing"
    assert field.support_membership.shape == field.unit_probs.shape
    assert field.attrs["background_prob_max"] <= 1e-6
    assert np.isclose(field.unit_probs[upper_idx, 5, 5, 5], 0.5, atol=0.02)
    assert np.isclose(field.unit_probs[upper_idx, 5, 5, 6], 1.0 / (1.0 + np.exp(-2.0)), atol=0.02)
    assert np.isclose(field.unit_probs[upper_idx, 5, 5, 4], 1.0 / (1.0 + np.exp(2.0)), atol=0.02)


def test_grid_sample_points_uses_gauss_legendre_weights() -> None:
    grid = GridSpec(bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)), shape=(2, 2, 2))

    points, weights = grid.sample_points(subgrid_factor=3)

    assert points.shape == (8 * 27, 3)
    assert weights.shape == (27,)
    assert np.isclose(float(weights.sum()), 1.0)
    assert not np.allclose(weights, np.full_like(weights, 1.0 / len(weights)))


def test_support_membership_preserves_embedded_body_containment() -> None:
    centre = np.array([5.0, 5.0, 5.0])
    axes = np.eye(3)
    graph = EntityGraph(
        nodes=[
            _unit("host"),
            _unit("intrusion", tuple(centre)),
            _unit("halo", tuple(centre)),
            *[
                _contact(f"c_intr_{idx}_{sign}", tuple(centre + sign * 3.0 * axis), ("intrusion", "host"))
                for idx, axis in enumerate(axes)
                for sign in (-1.0, 1.0)
            ],
            *[
                _contact(f"c_halo_{idx}_{sign}", tuple(centre + sign * 1.5 * axis), ("halo", "intrusion"))
                for idx, axis in enumerate(axes)
                for sign in (-1.0, 1.0)
            ],
        ],
        edges=[
            GraphEdge(kind=EdgeKind.OVERLIES, source="halo", target="intrusion", provenance=_prov()),
            GraphEdge(kind=EdgeKind.OVERLIES, source="intrusion", target="host", provenance=_prov()),
        ],
    )

    field = build_voxel_field(
        graph,
        GridSpec(bounds=((-0.5, 10.5), (-0.5, 10.5), (-0.5, 10.5)), shape=(11, 11, 11)),
        bandwidth=0.5,
    )
    host_idx = field.unit_ids.index("host")
    intrusion_idx = field.unit_ids.index("intrusion")
    halo_idx = field.unit_ids.index("halo")

    assert field.attrs["composition_mode"]["halo__intrusion"] == "erode"
    assert field.support_membership[host_idx, 5, 5, 5] > 0.5
    assert field.support_membership[intrusion_idx, 5, 5, 5] > 0.5
    assert field.support_membership[halo_idx, 5, 5, 5] > 0.5
    assert field.unit_probs[halo_idx, 5, 5, 5] > 0.85
    assert field.unit_probs[host_idx, 5, 5, 5] < 0.1
    assert field.unit_probs[intrusion_idx, 5, 5, 5] < 0.1


def test_voxel_field_support_membership_round_trips_through_xarray() -> None:
    graph = two_unit_horizontal(z_interface=5.0)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(4, 4, 4)),
    )

    dataset = field.to_xarray()
    loaded = VoxelField.from_xarray(dataset)

    assert dataset["unit_probs"].attrs["unit_probs_kind"] == "mixing"
    assert "support_membership" in dataset
    np.testing.assert_array_equal(loaded.support_membership, field.support_membership)
    assert loaded.attrs["unit_probs_kind"] == "mixing"


def test_ensemble_reduce_promotes_probs_kind_and_drops_scalar_aggregation() -> None:
    graph = two_unit_horizontal(z_interface=5.0)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(4, 4, 4))
    field_a = build_voxel_field(graph, grid)
    field_b = build_voxel_field(graph, grid)
    shifted = field_b.unit_probs.copy()
    shifted[:] = shifted[::-1]
    field_b = VoxelField(
        most_likely_unit=shifted.argmax(axis=0).astype(np.int16),
        unit_probs=shifted,
        entropy=field_b.entropy,
        domain_mask=field_b.domain_mask,
        scalar_field=np.ones_like(field_b.scalar_field),
        x=field_b.x,
        y=field_b.y,
        z=field_b.z,
        unit_ids=field_b.unit_ids,
        feature_names=field_b.feature_names,
        support_membership=shifted,
        attrs=field_b.attrs,
    )

    reduced = Ensemble([field_a, field_b], unit_ids=field_a.unit_ids).reduce()

    assert reduced.attrs["unit_probs_kind"] == "ensemble"
    assert reduced.attrs["scalar_field_aggregated"] is False
    assert np.all(reduced.scalar_field == 0.0)
    np.testing.assert_allclose(reduced.support_membership, (field_a.support_membership + field_b.support_membership) / 2.0)


def test_location_at_edges_make_sample_positions_lockstep_in_realisations() -> None:
    location = Location(
        id="loc_bh01_250",
        position=(
            GaussianUncertainty(mean=0.0, std=2.0),
            GaussianUncertainty(mean=0.0, std=2.0),
            GaussianUncertainty(mean=250.0, std=5.0),
        ),
        provenance=_prov(),
    )
    unit = StratigraphicUnit(id="unit_host", unit_id="host", series_id="s1", topology="layer", provenance=_prov())
    samples = [_sample("s_cu", "Cu"), _sample("s_s", "S")]
    graph = EntityGraph(
        nodes=[unit, location, *samples],
        edges=[
            GraphEdge(kind=EdgeKind.AT, source=sample.id, target=location.id, provenance=_prov())
            for sample in samples
        ],
    )

    realised = graph.realise(np.random.default_rng(7))

    assert realised.location_for(realised.get_node("s_cu")).position == realised.location_for(realised.get_node("s_s")).position
    assert realised.position_array(realised.get_node("s_cu")).tolist() == realised.position_array(realised.get_node("s_s")).tolist()


def test_sample_without_at_edge_is_rejected() -> None:
    with pytest.raises(GraphValidationError, match="AT"):
        EntityGraph(nodes=[_sample("s_cu", "Cu")])


def test_v16_sample_position_document_is_auto_lifted_to_location_at() -> None:
    prov = _prov().model_dump(mode="json")
    document = GraphDocument.model_validate(
        {
            "nodes": [
                {
                    "kind": "Sample",
                    "id": "s_cu",
                    "position": [_pt(1.0).model_dump(), _pt(2.0).model_dump(), _pt(3.0).model_dump()],
                    "analyte": "Cu",
                    "unit_of_measure": "wt_pct",
                    "value": _pt(0.8).model_dump(),
                    "provenance": prov,
                }
            ],
            "edges": [],
            "metadata": {},
        }
    )
    graph = EntityGraph.from_document(document)

    assert "loc__s_cu" in graph.node_ids()
    assert graph.location_for(graph.get_node("s_cu")).id == "loc__s_cu"
    assert graph.metadata["v17_migration"]["samples_migrated"] == 1
    assert "position" not in next(node for node in graph.to_dict()["nodes"] if node["id"] == "s_cu")


# ── §8.2 Porphyry embedded-body fixture ──────────────────────────────────────


def test_porphyry_embedded_body_renders_correctly() -> None:
    """§8.2: porphyry fixture — erode cascade active, containment preserved."""
    fixture = Path(__file__).parent / "golden" / "loopstructural" / "porphyry_embedded.json"
    graph = EntityGraph.from_file(str(fixture))

    grid = GridSpec(bounds=((50.0, 250.0), (50.0, 250.0), (50.0, 250.0)), shape=(20, 20, 20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = build_voxel_field(graph, grid, bandwidth=5.0)

    host_idx = field.unit_ids.index("host_rock")
    intr_idx = field.unit_ids.index("mineralised_intrusion")
    halo_idx = field.unit_ids.index("ore_halo_05cutoff")

    assert field.attrs["composition_mode"]["ore_halo_05cutoff__mineralised_intrusion"] == "erode"
    assert field.attrs["composition_mode"]["mineralised_intrusion__host_rock"] == "erode"

    # Where halo wins (> 0.5), host must be near zero: host is eroded by intrusion which fills the region.
    halo_dominant = field.unit_probs[halo_idx] > 0.5
    assert halo_dominant.any(), "no halo-dominant voxels found"
    assert field.unit_probs[host_idx][halo_dominant].max() < 0.05

    # In the strong halo core (> 0.9), intrusion must be clearly suppressed by the ERODE cascade.
    halo_core = field.unit_probs[halo_idx] > 0.9
    if halo_core.any():
        assert field.unit_probs[intr_idx][halo_core].max() < field.unit_probs[halo_idx][halo_core].min()

    # Halo must be near zero at the exterior corner, far from the intrusion centre.
    # Grid bounds=(50,250), shape=(20,20,20): cell (0,0,0) centre is at (55,55,55),
    # >130 units from the intrusion centre at (150,150,150).
    assert field.unit_probs[halo_idx, 0, 0, 0] < 0.01
    assert field.unit_probs[intr_idx, 0, 0, 0] < 0.1


# ── §8.4 Gauss-Legendre sub-cell convergence ─────────────────────────────────


def test_subgrid_integration_converges_for_offcentre_plane() -> None:
    """§8.4: GL quadrature gives monotone convergence for a tight off-centre sigmoid."""
    # Grid with integer cell centres: bounds=(-0.5, 9.5) → centres at 0,1,...,9; spacing=1.0
    grid = GridSpec(bounds=((-0.5, 9.5), (-0.5, 9.5), (-0.5, 9.5)), shape=(10, 10, 10))
    h = 0.1  # tight bandwidth (0.1 × dz) → near-discontinuous → large integration errors
    z_interface = 5.3  # 0.3 dz above cell 5's centre at z=5.0

    # Analytical average of sigma((z - z_interface)/h) over cell [4.5, 5.5] (width = 1.0)
    u_low = (4.5 - z_interface) / h
    u_high = (5.5 - z_interface) / h
    analytic = h * (np.log(1.0 + np.exp(u_high)) - np.log(1.0 + np.exp(u_low)))

    # Flat index of cell (5,5,5) in a (10,10,10) grid with ij-indexing
    flat_idx = 5 * 10 * 10 + 5 * 10 + 5

    errors: dict[int, float] = {}
    for factor in [1, 2, 4, 8]:
        pts, wts = grid.sample_points(subgrid_factor=factor)
        n_sub = factor**3
        pts_cell = pts.reshape(10 * 10 * 10, n_sub, 3)[flat_idx]
        sigs = 1.0 / (1.0 + np.exp(-(pts_cell[:, 2] - z_interface) / h))
        errors[factor] = abs(float(np.dot(sigs, wts)) - analytic)

    assert errors[1] > errors[2], f"GL must beat cell-centre: {errors}"
    assert errors[2] >= errors[4], f"Must converge monotonically: {errors}"
    assert errors[4] <= errors[2] / 2.0, f"At least halving with doubled order: {errors}"
    assert errors[8] <= 0.01, f"8-pt GL must be near-exact for smooth sigmoid: {errors}"


# ── §8.7 Insufficient-data error path ────────────────────────────────────────


def test_insufficient_unit_data_raises_error() -> None:
    """§8.7: unit with fewer than 3 contacts between it and an adjacent unit raises InsufficientUnitDataError."""
    prov = _prov()
    graph = EntityGraph(
        nodes=[
            StratigraphicUnit(id="u_sparse", unit_id="u_sparse", series_id="s1", topology="layer", provenance=prov),
            StratigraphicUnit(id="u_host", unit_id="u_host", series_id="s1", topology="layer", provenance=prov),
            _contact("c0", (2.0, 5.0, 5.0), ("u_sparse", "u_host")),
            _contact("c1", (8.0, 5.0, 5.0), ("u_sparse", "u_host")),
        ],
        edges=[GraphEdge(kind=EdgeKind.OVERLIES, source="u_sparse", target="u_host", provenance=prov)],
    )
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(5, 5, 5))

    with pytest.raises(InsufficientUnitDataError) as exc_info:
        build_voxel_field(graph, grid)

    assert "u_sparse" in str(exc_info.value)


# ── §8.10 Conformable-boundary ERODE bias ────────────────────────────────────


def test_erode_composition_bias_reported_in_attrs() -> None:
    """§8.10: erode_bias_cells is ≈ 0.48 × h/dz when ERODE mode is forced."""
    prov = _prov()
    graph = EntityGraph(
        nodes=[
            StratigraphicUnit(id="above", unit_id="above", series_id="s1", topology="layer", provenance=prov),
            StratigraphicUnit(id="below", unit_id="below", series_id="s1", topology="layer", provenance=prov),
            *[
                _contact(f"c{i}", (x, y, 5.0), ("above", "below"))
                for i, (x, y) in enumerate([(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (5.0, 5.0)])
            ],
        ],
        edges=[
            GraphEdge(
                kind=EdgeKind.OVERLIES,
                source="above",
                target="below",
                provenance=prov,
                metadata={"composition": "erode"},
            )
        ],
    )
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))
    dz = grid.spacing[2]
    h = 0.5 * min(grid.spacing)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BandwidthMismatchWarning)
        field = build_voxel_field(graph, grid)

    assert field.attrs["composition_mode"]["above__below"] == "erode"
    bias = field.attrs["erode_bias_cells"]["above__below"]
    expected = 0.48 * h / dz
    assert abs(bias - expected) < 0.05, f"bias {bias:.4f} != expected {expected:.4f}"


# ── §8.12 Memory budget warning ──────────────────────────────────────────────


def test_memory_budget_warning_is_emitted() -> None:
    """§8.12: MemoryBudgetWarning fires when the projected field size exceeds 4 GiB."""
    from graph_to_voxel.engine.loopstructural.adapter import _warn_if_memory_budget_large

    grid = GridSpec(bounds=((0.0, 100.0), (0.0, 100.0), (0.0, 100.0)), shape=(100, 100, 100))

    # 100³ × 6³ × 5 units × 16 bytes ≈ 6.9 GiB → must warn
    with pytest.warns(MemoryBudgetWarning, match="GiB"):
        _warn_if_memory_budget_large(grid, n_units=5, subgrid_factor=6)

    # 100³ × 1³ × 2 units × 16 bytes ≈ 26 MB → must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", MemoryBudgetWarning)
        _warn_if_memory_budget_large(grid, n_units=2, subgrid_factor=1)
