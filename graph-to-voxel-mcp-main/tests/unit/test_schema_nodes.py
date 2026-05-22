"""Tests for node schema models — must FAIL before implementation."""
import pytest
import numpy as np
from pydantic import ValidationError
from datetime import datetime, timezone

from graph_to_voxel.schema.nodes import (
    AnyNode,
    StratigraphicUnit,
    Contact,
    Orientation as OrientationNode,
    Fault,
    ObservationPoint,
    PositionWithUncertainty,
    Sample,
)
from graph_to_voxel.schema.provenance import DerivationSpec, Provenance
from graph_to_voxel.schema.uncertainty import (
    CategoricalUncertainty,
    DistributionUncertainty,
    GaussianUncertainty,
    IntervalUncertainty,
    PointUncertainty,
)


def _prov():
    return Provenance(
        source="test",
        confidence=0.9,
        timestamp=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def _pos():
    return (
        GaussianUncertainty(mean=100.0, std=5.0),
        GaussianUncertainty(mean=200.0, std=5.0),
        GaussianUncertainty(mean=-50.0, std=2.0),
    )


# ── Round-trips ───────────────────────────────────────────────────────────────

def test_stratigraphic_unit_round_trip():
    n = StratigraphicUnit(
        id="u1",
        unit_id="granodiorite",
        series_id="s1",
        topology="layer",
        p_exists=1.0,
        provenance=_prov(),
    )
    data = n.model_dump()
    n2 = StratigraphicUnit.model_validate(data)
    assert n2 == n


def test_stratigraphic_unit_requires_topology():
    with pytest.raises(ValidationError):
        StratigraphicUnit(id="u1", unit_id="granodiorite", series_id="s1", p_exists=1.0, provenance=_prov())


def test_layer_unit_rejects_anchor_inside():
    with pytest.raises(ValidationError, match="anchor_inside"):
        StratigraphicUnit(
            id="u1",
            unit_id="granodiorite",
            series_id="s1",
            topology="layer",
            anchor_inside=(1.0, 2.0, 3.0),
            p_exists=1.0,
            provenance=_prov(),
        )


def test_embedded_unit_accepts_typed_anchor_inside():
    n = StratigraphicUnit(
        id="u1",
        unit_id="intrusion",
        series_id="s1",
        topology="embedded",
        anchor_inside=[1.0, 2.0, 3.0],
        p_exists=1.0,
        provenance=_prov(),
    )

    assert n.anchor_inside == (1.0, 2.0, 3.0)


def test_contact_round_trip():
    n = Contact(
        id="c1",
        position=_pos(),
        between=("u1", "u2"),
        p_exists=0.8,
        provenance=_prov(),
    )
    data = n.model_dump()
    n2 = Contact.model_validate(data)
    assert n2.between == ("u1", "u2")


def test_orientation_node_round_trip():
    from graph_to_voxel.schema.uncertainty import OrientationUncertainty
    n = OrientationNode(
        id="o1",
        position=_pos(),
        dip=OrientationUncertainty(dip_mean=30.0, dip_kappa=5.0, azimuth_mean=90.0, azimuth_kappa=3.0),
        for_unit="u1",
        p_exists=1.0,
        provenance=_prov(),
    )
    assert n.for_unit == "u1"


def test_fault_round_trip():
    n = Fault(
        id="f1",
        surface_points=["c1", "c2", "c3"],
        p_exists=0.6,
        provenance=_prov(),
    )
    data = n.model_dump()
    n2 = Fault.model_validate(data)
    assert n2 == n


def test_observation_point_round_trip():
    n = ObservationPoint(
        id="op1",
        position=_pos(),
        notes="borehole at 200m",
        p_exists=1.0,
        provenance=_prov(),
    )
    data = n.model_dump()
    ObservationPoint.model_validate(data)


@pytest.mark.parametrize(
    "value",
    [
        PointUncertainty(value=0.85),
        GaussianUncertainty(mean=0.85, std=0.04),
        IntervalUncertainty(lo=0.75, hi=0.95),
        CategoricalUncertainty(probs={"below_cutoff": 0.2, "above_cutoff": 0.8}),
        DistributionUncertainty(name="uniform", params={"lo": 0.75, "hi": 0.95}),
    ],
)
def test_sample_round_trips_with_typed_uncertainty_values(value):
    n = Sample(
        id="s1",
        position=_pos(),
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=value,
        p_exists=1.0,
        provenance=_prov(),
    )

    node = AnyNode.model_validate(n.model_dump(mode="json")).root

    assert isinstance(node, Sample)
    assert type(node.value) is type(value)
    assert node.analyte == "Cu"


def test_metadata_measurement_value_remains_plain_dict():
    n = ObservationPoint(
        id="op_assay",
        position=_pos(),
        notes="Cu assay from BH01",
        p_exists=1.0,
        provenance=_prov(),
        metadata={
            "measurement": {
                "analyte": "Cu",
                "unit": "wt_pct",
                "value": {"kind": "Gaussian", "mean": 0.85, "std": 0.04},
            }
        },
    )

    node = ObservationPoint.model_validate(n.model_dump(mode="json"))

    assert isinstance(node.metadata["measurement"]["value"], dict)


def test_sample_position_helpers_expose_nominal_std_and_reserved_covariance():
    sample = Sample(
        id="s_pos",
        position=(
            PointUncertainty(value=1.0),
            PointUncertainty(value=2.0),
            GaussianUncertainty(mean=3.0, std=0.5),
        ),
        analyte="Cu",
        unit_of_measure="wt_pct",
        value=PointUncertainty(value=0.8),
        provenance=_prov(),
    )

    position = sample.position_with_std()

    assert np.array_equal(sample.position_array(), np.array([1.0, 2.0, 3.0]))
    assert isinstance(position, PositionWithUncertainty)
    assert np.array_equal(position.nominal, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(position.std, np.array([0.0, 0.0, 0.5]))
    assert position.covariance is None
    assert "per-axis 1-sigma (AABB)" in (PositionWithUncertainty.__doc__ or "")


def test_contact_and_orientation_position_helpers_match_sample_contract():
    from graph_to_voxel.schema.uncertainty import OrientationUncertainty

    contact = Contact(
        id="c_pos",
        position=(PointUncertainty(value=1.0), GaussianUncertainty(mean=2.0, std=0.25), PointUncertainty(value=3.0)),
        between=("u1", "u2"),
        provenance=_prov(),
    )
    orientation = OrientationNode(
        id="o_pos",
        position=contact.position,
        dip=OrientationUncertainty(dip_mean=30.0, dip_kappa=5.0, azimuth_mean=90.0, azimuth_kappa=3.0),
        for_unit="u1",
        provenance=_prov(),
    )

    for node in (contact, orientation):
        assert np.array_equal(node.position_array(), np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(node.position_with_std().std, np.array([0.0, 0.25, 0.0]))


def test_provenance_derivation_round_trips_as_typed_spec():
    provenance = Provenance(
        source="kriging run",
        confidence=0.8,
        timestamp=datetime(2026, 5, 6, tzinfo=timezone.utc),
        derivation=DerivationSpec(
            pipeline="kriging",
            input_node_ids=["s1", "s2", "s3"],
            params={"variogram_model": "exponential"},
            run_id="kriging-001",
        ),
    )

    round_tripped = Provenance.model_validate(provenance.model_dump(mode="json"))

    assert isinstance(round_tripped.derivation, DerivationSpec)
    assert round_tripped.derivation.pipeline == "kriging"


# ── p_exists bounds ───────────────────────────────────────────────────────────

def test_p_exists_below_zero_rejected():
    with pytest.raises(ValidationError):
        StratigraphicUnit(id="u1", unit_id="x", series_id="s1", topology="layer", p_exists=-0.1, provenance=_prov())


def test_p_exists_above_one_rejected():
    with pytest.raises(ValidationError):
        StratigraphicUnit(id="u1", unit_id="x", series_id="s1", topology="layer", p_exists=1.1, provenance=_prov())


# ── Discriminated union dispatch ──────────────────────────────────────────────

def test_any_node_from_dict():
    from graph_to_voxel.schema.nodes import AnyNode
    d = {
        "kind": "stratigraphic_unit",
        "id": "u99",
        "unit_id": "shale",
        "series_id": "s1",
        "topology": "layer",
        "p_exists": 1.0,
        "provenance": {
            "source": "test",
            "reference": None,
            "confidence": 1.0,
            "timestamp": "2026-05-06T00:00:00Z",
            "agent": None,
        },
    }
    node = AnyNode.model_validate(d)
    assert isinstance(node.root, StratigraphicUnit)
