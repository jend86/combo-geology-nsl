"""Bridging adapter integration tests — must FAIL before implementation."""
import pytest
import numpy as np

pytest.importorskip("LoopStructural", reason="LoopStructural not installed")

from graph_to_voxel.engine.voxel_field import VoxelField, GridSpec
from graph_to_voxel.engine.loopstructural.adapter import build_loopstructural
from tests.fixtures.toy_graphs import two_unit_horizontal


@pytest.fixture
def horizontal_field():
    g = two_unit_horizontal(z_interface=500.0)
    rng = np.random.default_rng(42)
    g = g.realise(rng)
    spec = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1000.0, 1000.0, 1000.0), nx=10, ny=12, nz=8)
    return build_loopstructural(g, spec), spec


def test_asymmetric_shape_preserved(horizontal_field):
    """Asymmetric grid (10×12×8) means axis-flipping is detectable."""
    field, spec = horizontal_field
    assert field.most_likely_unit.shape == (spec.nx, spec.ny, spec.nz)


def test_zarr_round_trip(horizontal_field, tmp_path):
    """VoxelField → zarr → reload must produce identical most_likely_unit."""
    from graph_to_voxel.voxel.persistence import save_zarr, load_zarr

    field, _ = horizontal_field
    path = tmp_path / "test.zarr"
    save_zarr(field, path)
    field2 = load_zarr(path)
    np.testing.assert_array_equal(field.most_likely_unit, field2.most_likely_unit)


def test_unit_catalog_stable_across_runs():
    """Two independent builds must use the same unit catalog ordering."""
    g = two_unit_horizontal(z_interface=500.0)
    spec = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1000.0, 1000.0, 1000.0), nx=8, ny=8, nz=8)

    f1 = build_loopstructural(g.realise(np.random.default_rng(10)), spec)
    f2 = build_loopstructural(g.realise(np.random.default_rng(20)), spec)

    assert f1.unit_catalog == f2.unit_catalog
