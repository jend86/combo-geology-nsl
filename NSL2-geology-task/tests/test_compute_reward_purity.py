"""Verify that TaskSpec.compute_reward() is pure — no container I/O.

The contract: compute_reward receives (initial: StateT, final: StateT,
artifacts: EpisodeArtifacts) and returns TaskReward. It has no access
to containers. This test validates the contract holds for the reference
implementation.
"""

import pytest

from src.task.types import EpisodeArtifacts, TaskReward
from tasks.memory_cleanup import MemoryCleanupState, MemoryCleanupTask


class TestComputeRewardPurity:
    """compute_reward must be callable with synthetic state — no Docker."""

    @pytest.fixture
    def task(self):
        return MemoryCleanupTask({})

    def test_reference_impl_with_synthetic_state(self, task):
        """MemoryCleanupTask.compute_reward works with plain dataclasses."""
        initial = MemoryCleanupState(
            used_kb={"c1": 600.0, "c2": 600.0},
            filesystem_groups={"device:8:0": ["c1", "c2"]},
        )
        final = MemoryCleanupState(
            used_kb={"c1": 500.0, "c2": 500.0},
            filesystem_groups={"device:8:0": ["c1", "c2"]},
        )
        artifacts = EpisodeArtifacts()
        reward = task.compute_reward(initial, final, artifacts)

        assert isinstance(reward, TaskReward)
        assert reward.value == 100.0  # deduplicated: one filesystem group
        assert reward.success is True

    def test_compute_reward_does_not_raise_on_zero_delta(self, task):
        """Zero change is not an error — it returns success=False."""
        state = MemoryCleanupState(
            used_kb={"c1": 500.0},
            filesystem_groups={"device:8:0": ["c1"]},
        )
        artifacts = EpisodeArtifacts()
        reward = task.compute_reward(state, state, artifacts)

        assert isinstance(reward, TaskReward)
        assert reward.value == 0.0
        assert reward.success is False

    def test_compute_reward_returns_breakdown(self, task):
        """Breakdown dict provides debugging data without container access."""
        initial = MemoryCleanupState(
            used_kb={"c1": 200.0},
            filesystem_groups={"device:8:0": ["c1"]},
        )
        final = MemoryCleanupState(
            used_kb={"c1": 100.0},
            filesystem_groups={"device:8:0": ["c1"]},
        )
        artifacts = EpisodeArtifacts()
        reward = task.compute_reward(initial, final, artifacts)

        assert "space_freed_kb" in reward.breakdown
        assert "initial_used_kb" in reward.breakdown
        assert "final_used_kb" in reward.breakdown
