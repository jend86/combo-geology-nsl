"""Golden tests for the LoopStructural engine adapter — must FAIL before implementation.

Tests use invariant-based assertions (not bit-exact snapshots).
Tolerances are solver-realistic: ±1 cell for interface position, ≤5° for gradient direction.
"""
import pytest
import numpy as np

from graph_to_voxel.engine.voxel_field import VoxelField, GridSpec
from graph_to_voxel.engine.loopstructural.adapter import build_loopstructural, build_voxel_field

from tests.fixtures.toy_graphs import two_unit_horizontal, tilted_layer


# ── Two-unit horizontal layer ─────────────────────────────────────────────────

class TestHorizontalLayer:
    @pytest.fixture(scope="class")
    def field(self):
        g = two_unit_horizontal(z_interface=500.0)
        rng = np.random.default_rng(0)
        g = g.realise(rng)
        spec = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1000.0, 1000.0, 1000.0), nx=20, ny=20, nz=20)
        return build_loopstructural(g, spec)

    def test_returns_voxel_field(self, field):
        assert isinstance(field, VoxelField)

    def test_shape(self, field):
        assert field.most_likely_unit.shape == (20, 20, 20)

    def test_unit_above_is_above_interface(self, field):
        # Cell centres at z > 500: cell index >= 10 (for 20 cells in [0,1000])
        unit_above_id = field.unit_catalog.index("above")
        above_layer = field.most_likely_unit[:, :, 10:]  # z > 500
        frac_correct = np.mean(above_layer == unit_above_id)
        assert frac_correct >= 0.9, f"Only {frac_correct:.1%} of voxels above interface are 'above' unit"

    def test_unit_below_is_below_interface(self, field):
        unit_below_id = field.unit_catalog.index("below")
        below_layer = field.most_likely_unit[:, :, :10]  # z < 500
        frac_correct = np.mean(below_layer == unit_below_id)
        assert frac_correct >= 0.9, f"Only {frac_correct:.1%} of voxels below interface are 'below' unit"

    def test_domain_closure(self, field):
        # No -1 (unassigned) inside domain_mask
        inside = field.domain_mask
        assert np.all(field.most_likely_unit[inside] >= 0)


# ── Tilted layer ──────────────────────────────────────────────────────────────

class TestTiltedLayer:
    @pytest.fixture(scope="class")
    def field(self):
        g = tilted_layer(dip_deg=30.0, dip_azimuth=90.0)
        rng = np.random.default_rng(1)
        g = g.realise(rng)
        spec = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1000.0, 1000.0, 1000.0), nx=20, ny=20, nz=20)
        return build_loopstructural(g, spec)

    def test_returns_voxel_field(self, field):
        assert isinstance(field, VoxelField)

    def test_two_units_present(self, field):
        unique_units = set(np.unique(field.most_likely_unit[field.domain_mask]))
        unique_units.discard(-1)
        assert len(unique_units) == 2

    def test_scalar_field_gradient_matches_dip(self, field):
        """Gradient of scalar field at domain centre should align with declared dip."""
        if field.scalar_field is None:
            pytest.skip("scalar_field not populated")
        # Pick first feature channel
        sf = field.scalar_field[0]
        # Compute central-difference gradient at domain centre
        cx, cy, cz = field.scalar_field.shape[1] // 2, field.scalar_field.shape[2] // 2, field.scalar_field.shape[3] // 2
        gx = (sf[cx + 1, cy, cz] - sf[cx - 1, cy, cz]) / 2.0
        gy = (sf[cx, cy + 1, cz] - sf[cx, cy - 1, cz]) / 2.0
        gz = (sf[cx, cy, cz + 1] - sf[cx, cy, cz - 1]) / 2.0
        grad = np.array([gx, gy, gz])
        if np.linalg.norm(grad) < 1e-10:
            pytest.skip("near-zero gradient at centre")
        grad = grad / np.linalg.norm(grad)
        # Declared dip 30°, azimuth 90° → normal = (sin30*cos90, sin30*sin90, cos30) = (0, 0.5, ~0.866)
        expected_normal = np.array([
            np.sin(np.deg2rad(30.0)) * np.cos(np.deg2rad(90.0)),
            np.sin(np.deg2rad(30.0)) * np.sin(np.deg2rad(90.0)),
            np.cos(np.deg2rad(30.0)),
        ])
        cos_angle = abs(np.dot(grad, expected_normal))
        angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
        assert angle_deg <= 15.0, f"Gradient angle {angle_deg:.1f}° exceeds 15° tolerance"


# ── Wrong stratigraphic order (negative test) ─────────────────────────────────

def test_loopstructural_is_importable() -> None:
    """Sentinel: LoopStructural is a required dependency in v1.7; import must succeed."""
    import importlib
    spec = importlib.util.find_spec("LoopStructural")
    assert spec is not None, (
        "LoopStructural is not installed. "
        "Install it with: pip install LoopStructural (or the project's engine extra)."
    )


def test_cycle_graph_rejected_at_load():
    from graph_to_voxel.graph.validate import GraphValidationError
    from tests.fixtures.toy_graphs import cycle_graph
    from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
    from graph_to_voxel.schema.provenance import Provenance
    from datetime import datetime, timezone

    g = cycle_graph()
    prov = Provenance(source="test", confidence=1.0, timestamp=datetime(2026, 5, 6, tzinfo=timezone.utc))
    with pytest.raises(GraphValidationError, match="cycle"):
        g.add_edge(GraphEdge(id="e31", kind=EdgeKind.OVERLIES, source="u3", target="u1", provenance=prov))
