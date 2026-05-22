from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from graph_to_voxel.graph import EntityGraph, GraphValidationError
from graph_to_voxel.schema import EdgeKind, GraphDocument
from graph_to_voxel.schema.uncertainty import (
    CategoricalUncertainty,
    DistributionUncertainty,
    GaussianUncertainty,
    IntervalUncertainty,
    OrientationUncertainty,
    PointUncertainty,
)


def test_graph_document_round_trips(two_unit_graph_dict):
    document = GraphDocument.model_validate(two_unit_graph_dict)
    dumped = document.model_dump(mode="json")
    reloaded = GraphDocument.model_validate(dumped)

    assert reloaded.model_dump(mode="json") == dumped
    assert EdgeKind.OVERLIES.value == "overlies"
    assert EdgeKind.OVERLIES.geosciml_uri.startswith("http://resource.geosciml.org/")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaussianUncertainty(kind="Gaussian", mean=0.0, std=0.0),
        lambda: GaussianUncertainty(kind="Gaussian", mean=0.0, std=-1.0),
        lambda: IntervalUncertainty(kind="Interval", lo=5.0, hi=3.0),
        lambda: CategoricalUncertainty(kind="Categorical", probs={"a": 0.4, "b": 0.5}),
        lambda: DistributionUncertainty(kind="Distribution", name="__import__", params={}),
        lambda: OrientationUncertainty(
            kind="Orientation",
            dip_mean=95.0,
            dip_kappa=1.0,
            azimuth_mean=0.0,
            azimuth_kappa=1.0,
        ),
    ],
)
def test_uncertainty_validators_reject_invalid(factory):
    with pytest.raises(ValidationError):
        factory()


def test_uncertainty_sampling_statistics_are_plausible():
    rng = np.random.default_rng(42)

    assert PointUncertainty(kind="Point", value=7.0).sample(rng) == 7.0

    gaussian = GaussianUncertainty(kind="Gaussian", mean=10.0, std=2.0)
    draws = np.array([gaussian.sample(rng) for _ in range(10_000)])
    assert abs(draws.mean() - 10.0) < 0.08
    assert abs(draws.std(ddof=1) - 2.0) < 0.08

    interval = IntervalUncertainty(kind="Interval", lo=-1.0, hi=3.0)
    draws = np.array([interval.sample(rng) for _ in range(10_000)])
    assert abs(draws.mean() - 1.0) < 0.08
    assert draws.min() >= -1.0
    assert draws.max() <= 3.0


def test_graph_validation_and_realisation_are_deterministic(two_unit_graph_dict):
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    graph.validate()

    realised_a = graph.realise(np.random.default_rng(123))
    realised_b = graph.realise(np.random.default_rng(123))

    assert realised_a.to_dict() == realised_b.to_dict()
    assert realised_a.node("c0").position[2].value == 5.0


def test_graph_validation_rejects_overlies_cycle(two_unit_graph_dict):
    bad = deepcopy(two_unit_graph_dict)
    bad["edges"].append(
        {
            **bad["edges"][0],
            "source": "unit_lower",
            "target": "unit_upper",
        }
    )

    with pytest.raises(GraphValidationError, match="OVERLIES cycle"):
        EntityGraph.from_dict(bad)


def test_graph_validation_rejects_dangling_contact(two_unit_graph_dict):
    bad = deepcopy(two_unit_graph_dict)
    contact = next(node for node in bad["nodes"] if node["kind"] == "Contact")
    contact["between"] = ["upper", "missing"]

    with pytest.raises(GraphValidationError, match="missing"):
        EntityGraph.from_dict(bad)
