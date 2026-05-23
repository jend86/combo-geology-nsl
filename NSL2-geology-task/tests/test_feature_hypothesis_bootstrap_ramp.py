"""Bootstrap concurrency ramp tests for FeatureHypothesisTask.

The framework owns ``parallel_episodes``; the task can only block inside
``populate()`` to soft-cap concurrency. We expose three knobs:

  - configured ``parallel_episodes`` slots (e.g. 4)
  - ``bootstrap_window_size`` over which to ramp (e.g. 8 episodes)
  - ``bootstrap_min_concurrency_fraction`` (default 0.5 — start at N/2)

Target active slots = ceil(configured * (min + (1-min) * progress)).

These tests cover the pure math first (no IO) and then the file-based
permit semaphore (acquire / release / stale-reap).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from tasks.feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation


def _task(tmp_path: Path) -> FeatureHypothesisTask:
    return FeatureHypothesisTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
        }
    )


def _variation(tmp_path: Path) -> FeatureHypothesisVariation:
    return FeatureHypothesisVariation(
        name="coe_fairbairn",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "coe_fairbairn"),
        kg_dir=str(tmp_path / "kg" / "coe_fairbairn"),
    )


class TestBootstrapTargetActive:
    """Pure-function ramp math — no IO."""

    def test_progress_zero_yields_min_fraction(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        assert task._bootstrap_target_active(
            bootstrap_episodes_seen=0,
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
        ) == 2

    def test_progress_full_yields_all_slots(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        assert task._bootstrap_target_active(
            bootstrap_episodes_seen=8,
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
        ) == 4

    def test_progress_beyond_window_clipped(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        assert task._bootstrap_target_active(
            bootstrap_episodes_seen=100,
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
        ) == 4

    def test_midway_progress_ceils(self, tmp_path: Path) -> None:
        # configured=4, min=0.5, window=8, seen=4 → 0.5 + 0.5*(4/8) = 0.75
        # ceil(4 * 0.75) = 3
        task = _task(tmp_path)
        assert task._bootstrap_target_active(
            bootstrap_episodes_seen=4,
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
        ) == 3

    def test_zero_window_uses_full_concurrency(self, tmp_path: Path) -> None:
        # Defensive: a misconfigured window of 0 should not divide-by-zero.
        task = _task(tmp_path)
        assert task._bootstrap_target_active(
            bootstrap_episodes_seen=0,
            configured_slots=4,
            window_size=0,
            min_fraction=0.5,
        ) == 4

    def test_min_fraction_floor(self, tmp_path: Path) -> None:
        # min_fraction=0.25 with 4 slots → at progress=0, target = ceil(4*0.25)=1
        task = _task(tmp_path)
        assert task._bootstrap_target_active(
            bootstrap_episodes_seen=0,
            configured_slots=4,
            window_size=8,
            min_fraction=0.25,
        ) == 1


class TestBootstrapPermit:
    """File-locked permit semaphore."""

    def test_acquire_then_release_is_clean(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        ok = task._acquire_bootstrap_permit(
            kg_dir,
            slot_id="slot-1",
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
            timeout_s=2.0,
        )
        assert ok is True
        task._release_bootstrap_permit(kg_dir, "slot-1")

        state = json.loads((kg_dir / "bootstrap_state.json").read_text())
        in_flight = [entry["slot_id"] for entry in state.get("in_flight", [])]
        assert "slot-1" not in in_flight

    def test_target_active_blocks_extra_slot(self, tmp_path: Path) -> None:
        # configured=4, min=0.5, seen=0 → target=2. The 3rd concurrent
        # acquirer must time out because only 2 permits are free.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)

        assert task._acquire_bootstrap_permit(
            kg_dir, "s1", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=2.0,
        ) is True
        assert task._acquire_bootstrap_permit(
            kg_dir, "s2", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=2.0,
        ) is True
        # 3rd is over the cap; with no release coming, it must time out.
        assert task._acquire_bootstrap_permit(
            kg_dir, "s3", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=0.5,
        ) is False

        task._release_bootstrap_permit(kg_dir, "s1")
        task._release_bootstrap_permit(kg_dir, "s2")

    def test_release_unblocks_waiter(self, tmp_path: Path) -> None:
        # A second thread waits while target is full; the main thread releases
        # and the waiter acquires.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)

        assert task._acquire_bootstrap_permit(
            kg_dir, "s1", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=2.0,
        ) is True
        assert task._acquire_bootstrap_permit(
            kg_dir, "s2", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=2.0,
        ) is True

        result: dict[str, bool] = {}

        def waiter() -> None:
            result["acquired"] = task._acquire_bootstrap_permit(
                kg_dir, "s3", configured_slots=4, window_size=8,
                min_fraction=0.5, timeout_s=5.0,
            )

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.4)  # let waiter start polling
        task._release_bootstrap_permit(kg_dir, "s1")
        thread.join(timeout=5.0)
        assert thread.is_alive() is False
        assert result["acquired"] is True

        task._release_bootstrap_permit(kg_dir, "s2")
        task._release_bootstrap_permit(kg_dir, "s3")

    def test_stale_permit_reaped(self, tmp_path: Path) -> None:
        # If a slot acquires but never releases (crash), the stale entry
        # should be reaped on a later acquire so the run does not deadlock.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)

        # Backdoor: inject a long-stale permit by writing the state file
        # directly with an "acquired_at" in the past.
        kg_dir.mkdir(parents=True, exist_ok=True)
        (kg_dir / "bootstrap_state.json").write_text(json.dumps({
            "bootstrap_episodes_seen": 0,
            "in_flight": [
                {"slot_id": "ghost-1", "acquired_at": 0.0},
                {"slot_id": "ghost-2", "acquired_at": 0.0},
            ],
        }))

        # configured=4, target=2 → both slots are full with ghosts. Without
        # reaping, this would time out. With reaping (stale_after_s small),
        # acquire should succeed quickly.
        ok = task._acquire_bootstrap_permit(
            kg_dir,
            "real-1",
            configured_slots=4,
            window_size=8,
            min_fraction=0.5,
            timeout_s=3.0,
            stale_after_s=0.1,
        )
        assert ok is True

        state = json.loads((kg_dir / "bootstrap_state.json").read_text())
        in_flight_ids = {entry["slot_id"] for entry in state["in_flight"]}
        assert "ghost-1" not in in_flight_ids
        assert "ghost-2" not in in_flight_ids
        assert "real-1" in in_flight_ids

    def test_episodes_seen_increments_on_release(self, tmp_path: Path) -> None:
        # bootstrap_episodes_seen drives the ramp. It increments on release —
        # not acquire — so the ramp tracks *completed* episodes. This keeps
        # raw parallelism from inflating the active-slot target before any
        # work has actually finished.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)

        # Acquire first; before any release, seen must still be 0.
        task._acquire_bootstrap_permit(
            kg_dir, "a", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=2.0,
        )
        task._acquire_bootstrap_permit(
            kg_dir, "b", configured_slots=4, window_size=8,
            min_fraction=0.5, timeout_s=2.0,
        )
        state_mid = json.loads((kg_dir / "bootstrap_state.json").read_text())
        assert state_mid["bootstrap_episodes_seen"] == 0

        task._release_bootstrap_permit(kg_dir, "a")
        task._release_bootstrap_permit(kg_dir, "b")

        state_end = json.loads((kg_dir / "bootstrap_state.json").read_text())
        assert state_end["bootstrap_episodes_seen"] == 2
