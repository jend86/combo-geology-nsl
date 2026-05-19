"""Workspace GC / eviction policy.

Default v0 policy (design doc §5.6):
  - Immutable graph snapshots, submitted specs, action logs, reviewed results → durable.
  - Fields, preview fields, derived data, figures, temporary artefacts → evictable.
  - Experiments pin their input snapshots + result artefacts until completion + review.
  - workspace.gc(policy) evicts unpinned derived artefacts by LRU, size quota, or TTL.
  - Never evicts: submitted experiments, result records, action logs, score records,
    candidates, pinned resources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore

_EVICTABLE_KINDS = {"field", "data", "job"}
_DURABLE_KINDS = {"graph", "experiment", "result", "review", "score", "candidate", "hypothesis"}


@dataclass
class GCPolicy:
    max_bytes: int | None = None          # evict LRU when total exceeds this
    max_age_hours: float | None = None    # evict resources older than this
    dry_run: bool = False                 # report but don't delete
    kinds: list[str] = field(default_factory=lambda: list(_EVICTABLE_KINDS))


@dataclass
class GCReport:
    evicted: list[str]
    retained: list[str]
    bytes_freed: int
    dry_run: bool


class WorkspaceGC:
    def __init__(self, store: WorkspaceStore) -> None:
        self._store = store

    def run(self, policy: GCPolicy) -> GCReport:
        """Apply eviction policy and return a report."""
        all_records = list(self._store._latest_records().values())
        candidates = [
            r for r in all_records
            if r.kind in policy.kinds and not r.pinned_by
        ]

        to_evict: list[str] = []
        retained: list[str] = []
        bytes_freed = 0

        # TTL filter
        if policy.max_age_hours is not None:
            cutoff = datetime.now(UTC) - timedelta(hours=policy.max_age_hours)
            candidates = [r for r in candidates if r.created_at < cutoff]

        # Size quota: sort by last-accessed (use created_at as proxy), evict LRU
        if policy.max_bytes is not None:
            total = sum(r.size_bytes for r in all_records)
            if total > policy.max_bytes:
                # sort oldest first
                candidates.sort(key=lambda r: r.created_at)
                budget = total - policy.max_bytes
                for r in candidates:
                    if budget <= 0:
                        break
                    to_evict.append(r.uri)
                    budget -= r.size_bytes
                    bytes_freed += r.size_bytes
            else:
                candidates = []

        if not policy.max_bytes:
            # TTL-only pass: evict all TTL candidates
            for r in candidates:
                to_evict.append(r.uri)
                bytes_freed += r.size_bytes

        retained = [r.uri for r in all_records if r.uri not in to_evict]

        if not policy.dry_run:
            self._evict(to_evict)

        return GCReport(
            evicted=to_evict,
            retained=retained,
            bytes_freed=bytes_freed,
            dry_run=policy.dry_run,
        )

    def _evict(self, uris: list[str]) -> None:
        import shutil

        for uri in uris:
            try:
                rec = self._store.get_resource(uri)
            except Exception:
                continue
            # Remove physical storage
            if rec.kind == "field":
                from graph_to_voxel.mcp.workspace.store import _id_from_uri
                short = _id_from_uri(uri)
                field_dir = self._store._field_path(short)
                if field_dir.exists():
                    shutil.rmtree(field_dir, ignore_errors=True)
            # Mark as evicted in registry by updating with a tombstone tag
            updated = rec.model_copy(update={"tags": {**rec.tags, "_evicted": "true"}})
            self._store._update_record(updated)
