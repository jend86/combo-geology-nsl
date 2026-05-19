"""Fixtures for MCP unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.voxel import entropy_from_probs


@pytest.fixture
def minimal_voxel_field() -> VoxelField:
    """A tiny 2x2x2 two-unit voxel field for testing storage/cache."""
    unit_ids = ["a", "b"]
    shape = (2, 2, 2)
    probs = np.zeros((2, *shape), dtype=np.float32)
    probs[0] = 0.8
    probs[1] = 0.2
    mask = np.ones(shape, dtype=bool)
    most_likely = probs.argmax(axis=0).astype(np.int16)
    return VoxelField(
        most_likely_unit=most_likely,
        unit_probs=probs,
        entropy=entropy_from_probs(probs, mask),
        domain_mask=mask,
        scalar_field=np.zeros((0, *shape), dtype=np.float32),
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        z=np.array([0.0, 1.0]),
        unit_ids=unit_ids,
        feature_names=[],
    )
