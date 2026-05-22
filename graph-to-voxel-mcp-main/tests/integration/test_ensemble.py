"""Phase 3 ensemble tests — stratigraphic_constrain + MCUE rejection budget."""
from __future__ import annotations

import pytest

from graph_to_voxel.engine.loopstructural import GridSpec, build_voxel_field
from graph_to_voxel.graph import EntityGraph, RealisationInfeasible
from graph_to_voxel.voxel import run_ensemble
from graph_to_voxel.voxel.ensemble import stratigraphic_constrain
from tests.fixtures.toy_graphs import crossing_uncertainty_graph, three_unit_valid_realised, three_unit_crossing_realised


def test_stratigraphic_constrain_passes_valid_order():
    """A realised graph with correct boundary depths must pass without raising."""
    g = three_unit_valid_realised()
    result = stratigraphic_constrain(g)
    assert result.unit_catalog() == g.unit_catalog()


def test_stratigraphic_constrain_raises_on_layer_crossing():
    """A realised graph where A-B contact is below B-C contact must raise."""
    g = three_unit_crossing_realised()
    with pytest.raises(RealisationInfeasible, match="layer crossing"):
        stratigraphic_constrain(g)


def test_ensemble_n_rejected_nonzero_for_crossing_fixture():
    """Fixture with overlapping uncertainty ranges must produce n_rejected > 0."""
    graph = crossing_uncertainty_graph()
    grid = GridSpec(bounds=((0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0)), shape=(5, 5, 10))
    ensemble = run_ensemble(
        graph,
        lambda realised: build_voxel_field(realised, grid),
        n=20,
        seed=42,
        oversample_budget=200,
    )
    assert ensemble.n_rejected > 0
    assert len(ensemble.realisations) == 20
