"""Reusable file-backed ordered-pair queue mechanics."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tasks.common.file_coordination import (
    atomic_write_jsonl,
    locked_dir,
    read_jsonl_records,
)


class OrderedPairQueue:
    """Claim ordered pair entries from a JSONL queue under a file lock.

    Domain code supplies pair enumeration and scoring. This helper only owns
    the persistent queue, atomic refresh, and attempt-count/popped-at mutation.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        queue_filename: str,
        lock_filename: str,
    ) -> None:
        self.directory = Path(directory)
        self.queue_filename = queue_filename
        self.lock_filename = lock_filename

    @property
    def queue_path(self) -> Path:
        return self.directory / self.queue_filename

    def read(self) -> list[dict[str, Any]]:
        return read_jsonl_records(self.queue_path)

    def write(self, entries: list[dict[str, Any]]) -> None:
        atomic_write_jsonl(self.queue_path, entries)

    def refill(
        self,
        merge_entries: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        with locked_dir(self.directory, self.lock_filename):
            self.write(merge_entries(self.read()))

    def pop_pair(
        self,
        *,
        can_pop: Callable[[], bool],
        should_refresh: Callable[[list[dict[str, Any]]], bool],
        merge_entries: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        choose_entry: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
    ) -> tuple[str, str] | None:
        if not can_pop():
            return None

        with locked_dir(self.directory, self.lock_filename):
            entries = self.read()
            if not entries or should_refresh(entries):
                entries = merge_entries(entries)

            if not entries:
                return None

            chosen = choose_entry(entries)
            if chosen is None:
                return None

            chosen["attempt_count"] = int(chosen.get("attempt_count", 0)) + 1
            chosen["popped_at"] = time.time()
            self.write(entries)

            parents = chosen.get("parents") or []
            if len(parents) != 2:
                return None
            return str(parents[0]), str(parents[1])
