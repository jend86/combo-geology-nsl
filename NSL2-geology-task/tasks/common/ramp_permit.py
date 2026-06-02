"""Reusable file-backed permit ramp for task-side concurrency pacing."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from tasks.common.file_coordination import atomic_write_json, locked_dir, read_json_or


def ramp_target_active(
    episodes_seen: int,
    configured_slots: int,
    window_size: int,
    min_fraction: float,
) -> int:
    if window_size <= 0 or configured_slots <= 0:
        return max(0, configured_slots)
    progress = min(max(episodes_seen / window_size, 0.0), 1.0)
    fraction = min_fraction + (1.0 - min_fraction) * progress
    return max(1, math.ceil(configured_slots * fraction))


class SlotRampPermit:
    """File-backed semaphore whose active-slot target ramps over time."""

    def __init__(
        self,
        directory: Path | str,
        *,
        state_filename: str,
        lock_filename: str,
    ) -> None:
        self.directory = Path(directory)
        self.state_filename = state_filename
        self.lock_filename = lock_filename

    @property
    def state_path(self) -> Path:
        return self.directory / self.state_filename

    def read_state(self) -> dict[str, Any]:
        data = read_json_or(
            self.state_path,
            {"bootstrap_episodes_seen": 0, "in_flight": []},
        )
        data.setdefault("bootstrap_episodes_seen", 0)
        data.setdefault("in_flight", [])
        return data

    def acquire(
        self,
        slot_id: str,
        *,
        configured_slots: int,
        window_size: int,
        min_fraction: float,
        timeout_s: float = 600.0,
        stale_after_s: float = 1800.0,
        poll_interval_s: float = 0.5,
    ) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)

        while True:
            with locked_dir(self.directory, self.lock_filename):
                state = self.read_state()
                now = time.time()
                raw_in_flight = list(state.get("in_flight", []))
                in_flight = [
                    entry
                    for entry in raw_in_flight
                    if isinstance(entry, dict)
                    and isinstance(entry.get("acquired_at"), (int, float))
                    and (now - float(entry["acquired_at"])) < stale_after_s
                ]
                target = ramp_target_active(
                    episodes_seen=int(state.get("bootstrap_episodes_seen", 0)),
                    configured_slots=configured_slots,
                    window_size=window_size,
                    min_fraction=min_fraction,
                )
                if len(in_flight) < target:
                    in_flight.append({"slot_id": slot_id, "acquired_at": now})
                    state["in_flight"] = in_flight
                    atomic_write_json(self.state_path, state)
                    return True
                if in_flight != raw_in_flight:
                    state["in_flight"] = in_flight
                    atomic_write_json(self.state_path, state)

            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_s)

    def release(self, slot_id: str) -> None:
        if not self.state_path.exists():
            return
        with locked_dir(self.directory, self.lock_filename):
            state = self.read_state()
            before = state.get("in_flight", [])
            after = [
                entry
                for entry in before
                if not (isinstance(entry, dict) and entry.get("slot_id") == slot_id)
            ]
            state["in_flight"] = after
            if len(after) != len(before):
                state["bootstrap_episodes_seen"] = (
                    int(state.get("bootstrap_episodes_seen", 0)) + 1
                )
            atomic_write_json(self.state_path, state)
