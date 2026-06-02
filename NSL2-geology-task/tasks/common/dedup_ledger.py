"""Reusable append-with-dedup ledger for task admission pools."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tasks.common.file_coordination import atomic_write_json, locked_dir, read_json_or


class JsonDedupLedger:
    """File-locked fingerprint ledger plus JSONL append.

    The helper is intentionally domain-neutral: callers supply the fingerprint
    and an optional admission callback for task-specific side effects.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        ledger_filename: str,
        records_filename: str,
        lock_filename: str,
        fingerprint_key: str = "fingerprints",
    ) -> None:
        self.directory = Path(directory)
        self.ledger_filename = ledger_filename
        self.records_filename = records_filename
        self.lock_filename = lock_filename
        self.fingerprint_key = fingerprint_key

    def admit(
        self,
        record: dict[str, Any],
        *,
        fingerprint: str,
        pre_admit: Callable[[], bool] | None = None,
        on_admit: Callable[[], None] | None = None,
    ) -> bool:
        """Append ``record`` iff ``fingerprint`` has not been seen."""
        with locked_dir(self.directory, self.lock_filename):
            ledger_path = self.directory / self.ledger_filename
            ledger = read_json_or(ledger_path, {self.fingerprint_key: []})
            seen = list(ledger.get(self.fingerprint_key, []))
            if fingerprint in seen:
                return False

            if pre_admit is not None and not pre_admit():
                return False

            if on_admit is not None:
                on_admit()

            records_path = self.directory / self.records_filename
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

            seen.append(fingerprint)
            atomic_write_json(ledger_path, {self.fingerprint_key: seen})
            return True
