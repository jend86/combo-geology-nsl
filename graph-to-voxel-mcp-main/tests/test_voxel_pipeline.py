from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import BaseModel

from graph_to_voxel.analyses.checks import check_bulk_volume
from graph_to_voxel.engine.voxel_field import DerivedChannelMetadata
from graph_to_voxel.engine.loopstructural import GridSpec, build_voxel_field
from graph_to_voxel.graph import EntityGraph
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Sample
from graph_to_voxel.schema.provenance import DerivationSpec, Provenance
from graph_to_voxel.schema.uncertainty import GaussianUncertainty, IntervalUncertainty, PointUncertainty
from graph_to_voxel.voxel import load_zarr, run_ensemble, save_zarr


def test_horizontal_two_unit_graph_builds_exportable_voxel_field(two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10)),
    )

    upper_idx = field.unit_ids.index("upper")
    lower_idx = field.unit_ids.index("lower")
    above = field.z > 5.0
    below = field.z < 5.0

    assert field.most_likely_unit.shape == (10, 10, 10)
    assert field.scalar_field.shape[1:] == (10, 10, 10)
    assert np.all(field.domain_mask)
    assert np.all(field.most_likely_unit[:, :, above] == upper_idx)
    assert np.all(field.most_likely_unit[:, :, below] == lower_idx)
    assert np.allclose(field.unit_probs.sum(axis=0), 1.0)


def test_voxel_field_round_trips_through_zarr(tmp_path, two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(6, 5, 4)),
        epsg=28350,
    )
    out = tmp_path / "field.zarr"

    save_zarr(field, out)
    loaded = load_zarr(out)

    assert loaded.unit_ids == field.unit_ids
    assert loaded.epsg == 28350
    assert np.array_equal(loaded.most_likely_unit, field.most_likely_unit)
    assert np.array_equal(loaded.domain_mask, field.domain_mask)


def test_voxel_field_derived_scalars_round_trip(tmp_path, two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(6, 5, 4)),
    )
    grade = np.full(field.shape, 0.85, dtype=np.float32)

    updated = field.with_derived_scalars({"Cu_pct_kriged_mean": grade})

    assert updated.scalar_field.shape == field.scalar_field.shape
    assert np.array_equal(updated.scalar_field, field.scalar_field)
    assert np.array_equal(updated.derived_scalars["Cu_pct_kriged_mean"], grade)

    out = tmp_path / "field-derived.zarr"
    save_zarr(updated, out)
    loaded = load_zarr(out)

    assert np.array_equal(loaded.derived_scalars["Cu_pct_kriged_mean"], grade)


def test_voxel_field_derived_scalar_provenance_round_trip(tmp_path, two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(6, 5, 4)),
    )
    grade = np.full(field.shape, 0.85, dtype=np.float32)
    spec = DerivationSpec(
        pipeline="kriging",
        input_node_ids=["s_BH01", "s_BH02", "s_BH03"],
        params={"interpolant": "rbf"},
        run_id="kriging-001",
    )
    metadata = DerivedChannelMetadata(derivation=spec, units="wt_pct")

    updated = field.with_derived_scalars(
        channels={"Cu_pct_kriged_mean": grade},
        provenance={"Cu_pct_kriged_mean": metadata},
    )
    out = tmp_path / "field-derived-provenance.zarr"
    save_zarr(updated, out)
    loaded = load_zarr(out)

    assert isinstance(DerivedChannelMetadata, type(BaseModel))
    assert loaded.derived_scalar_provenance["Cu_pct_kriged_mean"] == metadata
    assert loaded.derived_scalar_provenance["Cu_pct_kriged_mean"].derivation == spec
    assert "DerivedChannelMetadata" not in str(graph.to_dict())


def test_voxel_field_derived_scalar_last_writer_wins_for_values_and_provenance(two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(3, 3, 3)),
    )
    first = np.ones(field.shape, dtype=np.float32)
    second = np.full(field.shape, 2.0, dtype=np.float32)
    first_metadata = DerivedChannelMetadata(
        derivation=DerivationSpec(pipeline="kriging", input_node_ids=["s1"], params={}, run_id="run-1"),
        units="wt_pct",
    )
    second_metadata = DerivedChannelMetadata(
        derivation=DerivationSpec(pipeline="kriging", input_node_ids=["s2"], params={}, run_id="run-2"),
        units="wt_pct",
    )

    updated = field.with_derived_scalars(
        {"Cu_pct_kriged_mean": first},
        provenance={"Cu_pct_kriged_mean": first_metadata},
    ).with_derived_scalars(
        {"Cu_pct_kriged_mean": second},
        provenance={"Cu_pct_kriged_mean": second_metadata},
    )

    assert np.array_equal(updated.derived_scalars["Cu_pct_kriged_mean"], second)
    assert updated.derived_scalar_provenance["Cu_pct_kriged_mean"] == second_metadata


def test_voxel_field_derived_scalar_feature_name_collision_raises(two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(3, 3, 3)),
    )
    colliding_name = field.feature_names[0]

    with pytest.raises(ValueError, match=colliding_name):
        field.with_derived_scalars({colliding_name: np.zeros(field.shape, dtype=np.float32)})


def test_voxel_field_without_derived_scalars_round_trips_identically(tmp_path, two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(6, 5, 4)),
    )
    out = tmp_path / "field-no-derived.zarr"

    save_zarr(field, out)
    loaded = load_zarr(out)

    assert loaded.derived_scalars == {}
    assert np.array_equal(loaded.scalar_field, field.scalar_field)


def test_engine_re_exports_build_voxel_field():
    from graph_to_voxel.engine import build_voxel_field as exported

    assert exported is build_voxel_field


def test_build_voxel_field_warns_on_large_prior_voxel_drop(two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(10, 10, 10))
    prior = build_voxel_field(graph, grid)

    graph_v2_dict = {
        **two_unit_graph_dict,
        "nodes": [
            {
                **node,
                "position": [node["position"][0], node["position"][1], {"kind": "Point", "value": 8.0}],
            }
            if node["kind"] == "Contact"
            else node
            for node in two_unit_graph_dict["nodes"]
        ],
    }
    graph_v2 = EntityGraph.from_dict(graph_v2_dict)

    with pytest.warns(UserWarning, match="voxel.*drop"):
        build_voxel_field(graph_v2, grid, prior=prior)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_voxel_field(graph_v2, grid, prior=prior, drop_threshold=0.99)
    assert not caught


def test_geochem_kriging_iteration_loop_closes(tmp_path, two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    prov = Provenance.model_validate(two_unit_graph_dict["nodes"][0]["provenance"])
    samples = [
        Sample(
            id="sample_1",
            position=(PointUncertainty(value=2.0), PointUncertainty(value=2.0), PointUncertainty(value=7.0)),
            analyte="Cu",
            unit_of_measure="wt_pct",
            value=GaussianUncertainty(mean=0.75, std=0.04),
            provenance=prov,
        ),
        Sample(
            id="sample_2",
            position=(PointUncertainty(value=8.0), PointUncertainty(value=2.0), PointUncertainty(value=8.0)),
            analyte="Cu",
            unit_of_measure="wt_pct",
            value=GaussianUncertainty(mean=0.95, std=0.05),
            provenance=prov,
        ),
        Sample(
            id="sample_3",
            position=(PointUncertainty(value=5.0), PointUncertainty(value=8.0), PointUncertainty(value=6.0)),
            analyte="Cu",
            unit_of_measure="wt_pct",
            value=GaussianUncertainty(mean=0.65, std=0.03),
            provenance=prov,
        ),
    ]
    for sample in samples:
        graph.add_node(sample)
        graph.add_edge(GraphEdge(kind=EdgeKind.WITHIN, source=sample.id, target="unit_upper", provenance=prov))

    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(8, 8, 8)),
    )
    sample_positions = np.array([sample.position_array() for sample in graph.samples_for_unit("upper")])
    sample_values = np.array([sample.value.nominal() for sample in graph.samples_for_unit("upper")], dtype=float)
    xx, yy, zz = np.meshgrid(field.x, field.y, field.z, indexing="ij")
    targets = np.stack([xx, yy, zz], axis=-1)
    distances = np.linalg.norm(targets[..., None, :] - sample_positions, axis=-1)
    weights = 1.0 / np.maximum(distances, 1e-6) ** 2
    grade = np.sum(weights * sample_values, axis=-1) / np.sum(weights, axis=-1)
    upper_idx = field.unit_ids.index("upper")
    grade = np.where(field.most_likely_unit == upper_idx, grade, np.nan).astype(np.float32)

    field = field.with_derived_scalars({"Cu_pct_kriged_mean": grade})
    out = tmp_path / "kriged.zarr"
    save_zarr(field, out)
    loaded = load_zarr(out)

    assert np.array_equal(np.isnan(loaded.derived_scalars["Cu_pct_kriged_mean"]), np.isnan(grade))

    upper = graph.unit_node_by_unit_id()["upper"]
    volume = float(np.count_nonzero(loaded.most_likely_unit == upper_idx) * loaded.cell_volume)
    updated_provenance = upper.provenance.model_copy(
        update={
            "derivation": DerivationSpec(
                pipeline="kriging",
                input_node_ids=[sample.id for sample in samples],
                params={"interpolant": "idw", "power": 2},
                run_id="pytest-kriging-loop",
            )
        }
    )
    graph.add_node(
        upper.model_copy(
            update={
                "bulk_volume_bounds": IntervalUncertainty(lo=volume * 0.9, hi=volume * 1.1),
                "provenance": updated_provenance,
            }
        )
    )

    rebuilt = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(8, 8, 8)),
    )
    reloaded_graph = EntityGraph.from_dict(graph.to_dict())

    assert check_bulk_volume(graph, rebuilt).severity == "pass"
    assert reloaded_graph.unit_node_by_unit_id()["upper"].provenance.derivation.pipeline == "kriging"


def test_ensemble_reduction_places_entropy_near_uncertain_interface(gaussian_interface_graph_dict):
    graph = EntityGraph.from_dict(gaussian_interface_graph_dict)
    grid = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(5, 5, 31))
    ensemble = run_ensemble(graph, lambda realised: build_voxel_field(realised, grid), n=48, seed=11)
    field = ensemble.reduce()

    entropy_by_z = field.entropy.mean(axis=(0, 1))
    z_at_max_entropy = field.z[int(entropy_by_z.argmax())]

    assert len(ensemble.realisations) == 48
    assert 4.0 <= z_at_max_entropy <= 6.0
    assert entropy_by_z[0] < 0.05
    assert entropy_by_z[-1] < 0.05
