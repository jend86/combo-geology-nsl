"""Regression: concurrent spatial_add_point calls (each a fresh SpatialVoxelStore instance for
the same scratch dir, as the capability bridge dispatches them via asyncio.to_thread) must
ACCUMULATE into the layer, not clobber each other.

Before the fix, the non-atomic read -> remove_layer -> add_layer on a per-instance (stale)
in-memory index lost most concurrent updates and left the layer transiently empty for a racing
reader -- so 40 successful adds collapsed to an empty / few-voxel layer. The per-store lock +
disk-truthful read in _accumulate_voxels fixes it.
"""
from __future__ import annotations

import threading

import numpy as np

from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import GridSpec

GRID = {
    "origin": [66.5, 49.5, 0.0],
    "maximum": [71.5, 52.5, 80.0],
    "shape": [200, 200, 8],
    "crs": "EPSG:4326",
}


def _distinct_columns(values: np.ndarray) -> int:
    nz = np.argwhere(values != 0)
    return len({(int(x), int(y)) for x, y, _ in nz})


def test_concurrent_point_adds_accumulate(tmp_path):
    grid = GridSpec.from_dict(GRID)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    # 24 well-separated points (each lands in a distinct voxel column)
    pts = [(67.0 + 0.18 * i, 50.0 + 0.10 * i, 40.0) for i in range(24)]

    def worker(p):
        # fresh instance per call -- mimics one capability-bridge thread
        s = SpatialVoxelStore(scratch, grid)
        s.add_point_feature("L", p[0], p[1], p[2], value=1.0, radius_m=3000)

    threads = [threading.Thread(target=worker, args=(p,)) for p in pts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = SpatialVoxelStore(scratch, grid).get_layer_values("L")
    cols = _distinct_columns(final)
    # With the fix all 24 accumulate; without it, lost updates collapse to a handful.
    assert cols >= 20, f"only {cols} distinct columns survived (expected ~24) -> lost-update race"


def test_sequential_fresh_instance_adds_accumulate(tmp_path):
    # deterministic: each add uses a fresh instance (the per-capability-call pattern)
    grid = GridSpec.from_dict(GRID)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for i in range(10):
        s = SpatialVoxelStore(scratch, grid)
        s.add_point_feature("L", 67.0 + 0.18 * i, 50.0 + 0.10 * i, 40.0, value=1.0, radius_m=3000)
    final = SpatialVoxelStore(scratch, grid).get_layer_values("L")
    # each radius_m=3000 point covers ~9 columns; 10 well-separated points accumulate to ~90.
    # a lost-update bug would collapse this toward a single point's ~9 columns.
    assert _distinct_columns(final) >= 50
