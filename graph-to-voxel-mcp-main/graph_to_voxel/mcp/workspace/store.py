"""WorkspaceStore: file-backed resource registry, action log, job registry, pins, and field cache."""
from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from graph_to_voxel.mcp.workspace.models import (
    ActionRecord,
    CandidateRecord,
    DataRecord,
    ExperimentRecord,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    FieldCacheEntry,
    FieldRecord,
    FieldRunSpec,
    GraphRecord,
    HypothesisRecord,
    JobRecord,
    JobStatus,
    ResourceRecord,
    ResultRecord,
    ReviewRecord,
    ScoreRecord,
    ScratchRecord,
    ScratchRevRecord,
    ExperimentReview,
)

if TYPE_CHECKING:
    from graph_to_voxel.graph.core import Graph
    from graph_to_voxel.engine.voxel_field import VoxelField


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_REGISTRY_FILE = "registry.jsonl"
_ACTIONS_FILE = "actions.jsonl"
_CACHE_INDEX_FILE = "cache_index.jsonl"

_SUBDIRS = [
    "graphs", "scratch", "scratch_revs", "fields", "data",
    "experiments", "results", "reviews", "hypotheses", "jobs",
    "candidates", "scores", "locks",
]


class WorkspaceStoreError(Exception):
    pass


class ResourceNotFound(WorkspaceStoreError):
    pass


class ResourceTypeMismatch(WorkspaceStoreError):
    pass


class ClaimError(WorkspaceStoreError):
    pass


# ---------------------------------------------------------------------------
# WorkspaceStore
# ---------------------------------------------------------------------------

class WorkspaceStore:
    """Central file-backed workspace substrate."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for d in _SUBDIRS:
            (self.root / d).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _registry_path(self) -> Path:
        return self.root / _REGISTRY_FILE

    def _actions_path(self) -> Path:
        return self.root / _ACTIONS_FILE

    def _cache_index_path(self) -> Path:
        return self.root / _CACHE_INDEX_FILE

    def _lock(self, name: str) -> FileLock:
        return FileLock(str(self.root / "locks" / f"{name}.lock"))

    def _append_jsonl(self, path: Path, record: Any) -> None:
        if hasattr(record, "model_dump_json"):
            line = record.model_dump_json() + "\n"
        else:
            line = json.dumps(record, default=str) + "\n"
        with self._lock("registry_append"):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _graph_path(self, content_hash: str) -> Path:
        return self.root / "graphs" / f"{content_hash}.json"

    def _scratch_path(self, scratch_id: str) -> Path:
        return self.root / "scratch" / f"{scratch_id}.json"

    def _scratch_rev_path(self, scratch_id: str, rev: int) -> Path:
        return self.root / "scratch_revs" / f"{scratch_id}_{rev}.json"

    def _field_path(self, field_id: str) -> Path:
        return self.root / "fields" / field_id

    def _data_path(self, data_id: str) -> Path:
        return self.root / "data" / data_id

    def _experiment_path(self, exp_id: str) -> Path:
        return self.root / "experiments" / f"{exp_id}.json"

    def _result_path(self, result_id: str) -> Path:
        return self.root / "results" / f"{result_id}.json"

    def _review_path(self, review_id: str) -> Path:
        return self.root / "reviews" / f"{review_id}.json"

    def _hypothesis_path(self, hyp_id: str) -> Path:
        return self.root / "hypotheses" / f"{hyp_id}.json"

    def _job_path(self, job_id: str) -> Path:
        return self.root / "jobs" / f"{job_id}.json"

    def _candidate_path(self, candidate_id: str) -> Path:
        return self.root / "candidates" / f"{candidate_id}.json"

    def _score_path(self, score_id: str) -> Path:
        return self.root / "scores" / f"{score_id}.json"

    @staticmethod
    def _content_hash(data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _short(h: str, n: int = 16) -> str:
        return h

    def _new_id(self) -> str:
        return uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Resource registry
    # ------------------------------------------------------------------

    def _register(self, record: ResourceRecord) -> None:
        self._append_jsonl(self._registry_path(), record)

    def _all_records(self) -> list[ResourceRecord]:
        raw = self._read_jsonl(self._registry_path())
        records: list[ResourceRecord] = []
        for r in raw:
            kind = r.get("kind") or ""
            cls = _KIND_MAP.get(kind, ResourceRecord)
            records.append(cls.model_validate(r))
        return records

    def _latest_records(self) -> dict[str, ResourceRecord]:
        """Most recent record per URI (later entries override earlier)."""
        by_uri: dict[str, ResourceRecord] = {}
        for rec in self._all_records():
            by_uri[rec.uri] = rec
        return by_uri

    def get_resource(self, uri: str) -> ResourceRecord:
        latest = self._latest_records()
        if uri not in latest:
            raise ResourceNotFound(uri)
        return latest[uri]

    def list_resources(self, kind: str | None = None) -> list[ResourceRecord]:
        latest = self._latest_records()
        records = list(latest.values())
        if kind:
            records = [r for r in records if r.kind == kind]
        return records

    def _update_record(self, record: ResourceRecord) -> None:
        """Append an updated version of a record (later wins in _latest_records)."""
        self._append_jsonl(self._registry_path(), record)

    # ------------------------------------------------------------------
    # Graph registration
    # ------------------------------------------------------------------

    def register_graph(
        self,
        graph: Graph,
        *,
        message: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Persist an immutable graph snapshot and return its URI."""
        graph_dict = graph.to_dict()
        raw = json.dumps(graph_dict, sort_keys=True)
        ch = self._content_hash(raw)
        uri = f"g2v://graph/{ch}"

        # idempotent: if the content already exists, return the same URI
        try:
            existing = self.get_resource(uri)
            return existing.uri
        except ResourceNotFound:
            pass

        path = self._graph_path(ch)
        if not path.exists():
            self._write_json(path, graph_dict)

        record = GraphRecord(
            uri=uri,
            content_hash=ch,
            size_bytes=len(raw.encode()),
            message=message,
            node_count=len(graph.node_ids()),
            edge_count=len(graph.get_edges()),
            unit_catalog=graph.unit_catalog(),
            tags=tags or {},
        )
        self._register(record)
        return uri

    def load_graph(self, uri: str) -> Graph:
        from graph_to_voxel.graph.core import Graph

        rec = self.get_resource(uri)
        if isinstance(rec, GraphRecord):
            path = self._graph_path(rec.content_hash)
            return Graph.from_dict(self._read_json(path))
        if isinstance(rec, ScratchRevRecord):
            path = self._scratch_rev_path(
                _scratch_id_from_uri(rec.scratch_uri), rec.rev
            )
            return Graph.from_dict(self._read_json(path))
        if isinstance(rec, ScratchRecord):
            # Convenience: load current head revision
            return self.load_graph(rec.head_rev_uri)
        raise ResourceTypeMismatch(f"{uri!r} is not a graph or scratch rev resource")

    # ------------------------------------------------------------------
    # Scratch branches
    # ------------------------------------------------------------------

    def create_scratch(self, base_graph_uri: str) -> str:
        """Create a mutable scratch branch from a graph snapshot."""
        base_graph = self.load_graph(base_graph_uri)
        scratch_id = self._new_id()
        scratch_uri = f"g2v://scratch/{scratch_id}"

        # write initial rev 0
        rev0_data = base_graph.to_dict()
        self._write_json(self._scratch_rev_path(scratch_id, 0), rev0_data)

        rev0_hash = self._content_hash(json.dumps(rev0_data, sort_keys=True))
        rev0_uri = f"g2v://scratch/{scratch_id}@rev/0"

        rev0_record = ScratchRevRecord(
            uri=rev0_uri,
            content_hash=rev0_hash,
            size_bytes=len(json.dumps(rev0_data).encode()),
            scratch_uri=scratch_uri,
            rev=0,
        )
        self._register(rev0_record)

        record = ScratchRecord(
            uri=scratch_uri,
            content_hash=rev0_hash,
            base_graph_uri=base_graph_uri,
            head_rev=0,
            head_rev_uri=rev0_uri,
        )
        self._register(record)
        return scratch_uri

    def get_scratch_record(self, scratch_uri: str) -> ScratchRecord:
        rec = self.get_resource(scratch_uri)
        if not isinstance(rec, ScratchRecord):
            raise ResourceTypeMismatch(f"{scratch_uri!r} is not a scratch resource")
        return rec

    def apply_scratch_patch(
        self,
        scratch_uri: str,
        operations: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Apply a list of graph operations transactionally.

        Returns (new_head_rev_uri, validation_report).
        Raises GraphValidationError on invalid final state (no state change).
        """
        rec = self.get_scratch_record(scratch_uri)
        if rec.committed:
            raise WorkspaceStoreError(f"Scratch {scratch_uri!r} has already been committed")

        # load current head
        graph = self.load_graph(rec.head_rev_uri)
        working = deepcopy(graph)

        report = _apply_operations(working, operations)

        # Validate the complete post-patch graph before moving the scratch head.
        type(graph)(working.nodes(), working.get_edges(), metadata=working.metadata)

        new_rev = rec.head_rev + 1
        scratch_id = _scratch_id_from_uri(scratch_uri)
        rev_path = self._scratch_rev_path(scratch_id, new_rev)
        rev_data = working.to_dict()
        self._write_json(rev_path, rev_data)

        rev_hash = self._content_hash(json.dumps(rev_data, sort_keys=True))
        new_rev_uri = f"g2v://scratch/{scratch_id}@rev/{new_rev}"

        rev_record = ScratchRevRecord(
            uri=new_rev_uri,
            content_hash=rev_hash,
            size_bytes=len(json.dumps(rev_data).encode()),
            scratch_uri=scratch_uri,
            rev=new_rev,
        )
        self._register(rev_record)

        updated_rec = rec.model_copy(
            update={"head_rev": new_rev, "head_rev_uri": new_rev_uri, "content_hash": rev_hash}
        )
        self._update_record(updated_rec)

        return new_rev_uri, report

    def commit_scratch(self, scratch_uri: str, message: str | None = None) -> str:
        """Commit scratch head to an immutable graph snapshot."""
        rec = self.get_scratch_record(scratch_uri)
        if rec.committed:
            raise WorkspaceStoreError(f"Scratch {scratch_uri!r} already committed")

        graph = self.load_graph(rec.head_rev_uri)
        graph_uri = self.register_graph(graph, message=message)

        updated_rec = rec.model_copy(update={"committed": True})
        self._update_record(updated_rec)
        return graph_uri

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def register_field(self, field: VoxelField, spec: FieldRunSpec) -> str:
        """Persist a voxel field and its run spec; return URI."""
        from graph_to_voxel.voxel.persistence import save_zarr

        spec_hash = spec.content_hash()
        short = self._short(spec_hash)
        uri = f"g2v://field/{short}"

        try:
            existing = self.get_resource(uri)
            return existing.uri
        except ResourceNotFound:
            pass

        field_dir = self._field_path(short)
        field_dir.mkdir(parents=True, exist_ok=True)
        save_zarr(field, field_dir / "field.zarr")

        spec_json = spec.model_dump_json()
        (field_dir / "spec.json").write_text(spec_json, encoding="utf-8")

        # Derive grid info from spec or from field arrays
        grid_origin = spec.grid_origin or (float(field.x[0]), float(field.y[0]), float(field.z[0]))
        grid_maximum = spec.grid_maximum or (float(field.x[-1]), float(field.y[-1]), float(field.z[-1]))
        grid_shape = spec.grid_shape or (len(field.x), len(field.y), len(field.z))

        record = FieldRecord(
            uri=uri,
            content_hash=spec_hash,
            size_bytes=sum(f.stat().st_size for f in field_dir.rglob("*") if f.is_file()),
            field_run_spec_hash=spec_hash,
            graph_uri=f"g2v://graph/{self._short(spec.graph_content_hash)}",
            grid_origin=grid_origin,
            grid_maximum=grid_maximum,
            grid_shape=grid_shape,
            unit_catalog=list(field.unit_ids),
            epsg=spec.epsg,
        )
        self._register(record)
        self._register_cache_entry(FieldCacheEntry(spec_hash=spec_hash, field_uri=uri))
        return uri

    def load_field(self, uri: str) -> VoxelField:
        from graph_to_voxel.voxel.persistence import load_zarr

        rec = self.get_resource(uri)
        if not isinstance(rec, FieldRecord):
            raise ResourceTypeMismatch(f"{uri!r} is not a field resource")
        short = _id_from_uri(uri)
        field_dir = self._field_path(short)
        return load_zarr(str(field_dir / "field.zarr"))

    # ------------------------------------------------------------------
    # Field cache
    # ------------------------------------------------------------------

    def _register_cache_entry(self, entry: FieldCacheEntry) -> None:
        self._append_jsonl(self._cache_index_path(), entry)

    def get_cached_field(self, spec: FieldRunSpec) -> str | None:
        """Return field URI if a cached field exists for this spec, else None."""
        target = spec.content_hash()
        entries = self._read_jsonl(self._cache_index_path())
        for raw in reversed(entries):
            if raw.get("spec_hash") == target:
                field_uri = raw["field_uri"]
                try:
                    self.get_resource(field_uri)
                    return field_uri
                except ResourceNotFound:
                    pass
        return None

    # ------------------------------------------------------------------
    # Data resources
    # ------------------------------------------------------------------

    def register_data(
        self,
        source_path_rel: str,
        *,
        media_type: str = "application/octet-stream",
        detected_schema: dict | None = None,
        crs: str | None = None,
        preview_text: str = "",
        parent_uri: str | None = None,
        raw_bytes: bytes | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        ch = self._content_hash(raw_bytes or source_path_rel.encode())
        short = self._short(ch)
        uri = f"g2v://data/{short}"

        try:
            existing = self.get_resource(uri)
            return existing.uri
        except ResourceNotFound:
            pass

        record = DataRecord(
            uri=uri,
            content_hash=ch,
            size_bytes=len(raw_bytes) if raw_bytes else 0,
            source_path_rel=source_path_rel,
            media_type=media_type,
            detected_schema=detected_schema or {},
            crs=crs,
            preview_text=preview_text,
            parent_uri=parent_uri,
            tags=tags or {},
        )
        self._register(record)
        return uri

    def get_data_record(self, uri: str) -> DataRecord:
        rec = self.get_resource(uri)
        if not isinstance(rec, DataRecord):
            raise ResourceTypeMismatch(f"{uri!r} is not a data resource")
        return rec

    def export_field_to_data(self, field_uri: str, field: VoxelField, *, format: str = "zarr") -> str:
        """Save a VoxelField as a data resource (zarr) and return its URI."""
        from graph_to_voxel.voxel.persistence import save_zarr

        short = _id_from_uri(field_uri)
        export_dir = self._data_path(f"{short}_export")
        export_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = export_dir / "field.zarr"
        save_zarr(field, str(zarr_path))

        rel_path = str(zarr_path.relative_to(self.root))
        raw_bytes = rel_path.encode()
        return self.register_data(
            source_path_rel=rel_path,
            media_type="application/zarr",
            parent_uri=field_uri,
            raw_bytes=raw_bytes,
        )

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------

    def register_hypothesis(
        self,
        statement: str,
        *,
        graph_refs: list[str] | None = None,
        data_refs: list[str] | None = None,
        rationale: str | None = None,
    ) -> str:
        hyp_id = self._new_id()
        uri = f"g2v://hypothesis/{hyp_id}"
        data = {
            "id": hyp_id,
            "statement": statement,
            "graph_refs": graph_refs or [],
            "data_refs": data_refs or [],
            "rationale": rationale,
        }
        self._write_json(self._hypothesis_path(hyp_id), data)

        raw = json.dumps(data, sort_keys=True)
        record = HypothesisRecord(
            uri=uri,
            content_hash=self._content_hash(raw),
            statement=statement,
            graph_refs=graph_refs or [],
            data_refs=data_refs or [],
            rationale=rationale,
        )
        self._register(record)
        return uri

    def get_hypothesis_record(self, uri: str) -> HypothesisRecord:
        rec = self.get_resource(uri)
        if not isinstance(rec, HypothesisRecord):
            raise ResourceTypeMismatch(f"{uri!r} is not a hypothesis resource")
        hyp_id = _id_from_uri(uri)
        path = self._hypothesis_path(hyp_id)
        if path.exists():
            data = self._read_json(path)
            return HypothesisRecord(
                uri=uri,
                content_hash=rec.content_hash,
                statement=data.get("statement", rec.statement),
                graph_refs=data.get("graph_refs", rec.graph_refs),
                data_refs=data.get("data_refs", rec.data_refs),
                rationale=data.get("rationale", rec.rationale),
            )
        return rec

    def update_hypothesis(
        self,
        uri: str,
        *,
        statement: str | None = None,
        graph_refs: list[str] | None = None,
        data_refs: list[str] | None = None,
        rationale: str | None = None,
    ) -> HypothesisRecord:
        rec = self.get_hypothesis_record(uri)
        hyp_id = _id_from_uri(uri)
        updates: dict[str, Any] = {}
        if statement is not None:
            updates["statement"] = statement
        if graph_refs is not None:
            updates["graph_refs"] = graph_refs
        if data_refs is not None:
            updates["data_refs"] = data_refs
        if rationale is not None:
            updates["rationale"] = rationale
        updated = rec.model_copy(update=updates)
        data = {
            "id": hyp_id,
            "statement": updated.statement,
            "graph_refs": updated.graph_refs,
            "data_refs": updated.data_refs,
            "rationale": updated.rationale,
        }
        self._write_json(self._hypothesis_path(hyp_id), data)
        self._update_record(updated)
        return updated

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def register_experiment(
        self,
        spec: ExperimentSpec,
        *,
        graph_ref_resolution: list[dict[str, Any]] | None = None,
    ) -> str:
        exp_id = spec.id
        uri = f"g2v://experiment/{exp_id}"
        raw = spec.model_dump_json()
        record = ExperimentRecord(
            uri=uri,
            content_hash=self._content_hash(raw),
            spec=spec,
            status="queued",
            graph_ref_resolution=graph_ref_resolution or [],
        )
        self._write_json(self._experiment_path(exp_id), record.model_dump(mode="json"))
        self._register(record)
        return uri

    def get_experiment_record(self, uri: str) -> ExperimentRecord:
        rec = self.get_resource(uri)
        if not isinstance(rec, ExperimentRecord):
            raise ResourceTypeMismatch(f"{uri!r} is not an experiment resource")
        # Read from authoritative file for latest state
        exp_id = _id_from_uri(uri)
        path = self._experiment_path(exp_id)
        if path.exists():
            return ExperimentRecord.model_validate(self._read_json(path))
        return rec

    def update_experiment_status(
        self,
        uri: str,
        status: ExperimentStatus,
        *,
        claim_id: str | None = None,
        lease_expires_at: datetime | None = None,
        result_uri: str | None = None,
    ) -> ExperimentRecord:
        rec = self.get_experiment_record(uri)
        updates: dict = {"status": status}
        if claim_id is not None:
            updates["claim_id"] = claim_id
        if lease_expires_at is not None:
            updates["lease_expires_at"] = lease_expires_at
        if result_uri is not None:
            updates["result_uri"] = result_uri
        updated = rec.model_copy(update=updates)
        self._write_json(self._experiment_path(rec.spec.id), updated.model_dump(mode="json"))
        self._update_record(updated)
        return updated

    def claim_experiment(self, uri: str, lease_s: int = 300) -> tuple[str, ExperimentRecord]:
        """Atomically claim an experiment. Returns (claim_id, updated_record)."""
        exp_id = _id_from_uri(uri)
        with self._lock(f"exp_{exp_id}"):
            rec = self.get_experiment_record(uri)
            # check if expired claim should be released
            if rec.status == "claimed" and rec.lease_expires_at:
                if datetime.now(UTC) > rec.lease_expires_at:
                    if rec.retry_count >= rec.max_retries:
                        rec = self.update_experiment_status(uri, "failed")
                    else:
                        updated = rec.model_copy(
                            update={"status": "queued", "claim_id": None,
                                    "lease_expires_at": None,
                                    "retry_count": rec.retry_count + 1}
                        )
                        self._write_json(self._experiment_path(exp_id), updated.model_dump(mode="json"))
                        self._update_record(updated)
                        rec = updated

            if rec.status != "queued":
                raise ClaimError(f"Experiment {uri!r} is not claimable (status={rec.status!r})")

            claim_id = self._new_id()
            expires = datetime.now(UTC) + timedelta(seconds=lease_s)
            updated = self.update_experiment_status(
                uri, "claimed",
                claim_id=claim_id,
                lease_expires_at=expires,
            )
            return claim_id, updated

    # ------------------------------------------------------------------
    # Experiment results
    # ------------------------------------------------------------------

    def register_result(self, experiment_uri: str, result: ExperimentResult) -> str:
        result_id = result.id
        uri = f"g2v://experiment/{_id_from_uri(experiment_uri)}/result"
        raw = result.model_dump_json()
        record = ResultRecord(
            uri=uri,
            content_hash=self._content_hash(raw),
            experiment_uri=experiment_uri,
            result=result,
        )
        self._write_json(self._result_path(result_id), record.model_dump(mode="json"))
        self._register(record)
        return uri

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def register_review(self, experiment_uri: str, review: ExperimentReview) -> str:
        review_id = self._new_id()
        uri = f"g2v://review/{review_id}"
        raw = review.model_dump_json()
        record = ReviewRecord(
            uri=uri,
            content_hash=self._content_hash(raw),
            experiment_uri=experiment_uri,
            review=review,
        )
        self._write_json(self._review_path(review_id), record.model_dump(mode="json"))
        self._register(record)
        return uri

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def register_job(self, job_type: str, input_uris: list[str] | None = None) -> str:
        job_id = self._new_id()
        uri = f"g2v://job/{job_id}"
        record = JobRecord(
            uri=uri,
            content_hash=job_id,
            job_type=job_type,
            status="queued",
            input_uris=input_uris or [],
        )
        self._write_json(self._job_path(job_id), record.model_dump(mode="json"))
        self._register(record)
        return uri

    def get_job_record(self, uri: str) -> JobRecord:
        job_id = _id_from_uri(uri)
        path = self._job_path(job_id)
        if path.exists():
            return JobRecord.model_validate(self._read_json(path))
        rec = self.get_resource(uri)
        if not isinstance(rec, JobRecord):
            raise ResourceTypeMismatch(f"{uri!r} is not a job resource")
        return rec

    def update_job(
        self,
        uri: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        current_step: str | None = None,
        result_refs: list[str] | None = None,
        error: str | None = None,
    ) -> JobRecord:
        job_id = _id_from_uri(uri)
        rec = self.get_job_record(uri)
        updates: dict = {}
        if status is not None:
            updates["status"] = status
        if progress is not None:
            updates["progress"] = progress
        if current_step is not None:
            updates["current_step"] = current_step
        if result_refs is not None:
            updates["result_refs"] = result_refs
        if error is not None:
            updates["error"] = error
        updated = rec.model_copy(update=updates)
        self._write_json(self._job_path(job_id), updated.model_dump(mode="json"))
        self._update_record(updated)
        return updated

    # ------------------------------------------------------------------
    # Scores + Candidates
    # ------------------------------------------------------------------

    def register_score(
        self,
        *,
        candidate_graph_uri: str,
        reference_a_graph_uri: str,
        reference_b_graph_uri: str,
        candidate_field_uri: str,
        reference_a_field_uri: str,
        reference_b_field_uri: str,
        score_value: float | None = None,
        breakdown: dict | None = None,
    ) -> str:
        score_id = self._new_id()
        uri = f"g2v://score/{score_id}"
        record = ScoreRecord(
            uri=uri,
            content_hash=score_id,
            candidate_graph_uri=candidate_graph_uri,
            reference_a_graph_uri=reference_a_graph_uri,
            reference_b_graph_uri=reference_b_graph_uri,
            candidate_field_uri=candidate_field_uri,
            reference_a_field_uri=reference_a_field_uri,
            reference_b_field_uri=reference_b_field_uri,
            score_value=score_value,
            breakdown=breakdown or {},
        )
        self._write_json(self._score_path(score_id), record.model_dump(mode="json"))
        self._register(record)
        return uri

    def register_candidate(
        self,
        graph_uri: str,
        reference_pair: tuple[str, str],
        evidence_refs: list[str],
        score_refs: list[str],
    ) -> str:
        cand_id = self._new_id()
        uri = f"g2v://candidate/{cand_id}"
        record = CandidateRecord(
            uri=uri,
            content_hash=cand_id,
            graph_uri=graph_uri,
            reference_pair=reference_pair,
            evidence_refs=evidence_refs,
            score_refs=score_refs,
        )
        self._write_json(self._candidate_path(cand_id), record.model_dump(mode="json"))
        self._register(record)
        return uri

    # ------------------------------------------------------------------
    # Pins
    # ------------------------------------------------------------------

    def pin(self, resource_uri: str, *, held_by: str) -> None:
        try:
            rec = self.get_resource(resource_uri)
        except ResourceNotFound:
            return
        if held_by not in rec.pinned_by:
            updated = rec.model_copy(update={"pinned_by": [*rec.pinned_by, held_by]})
            self._update_record(updated)

    def unpin(self, resource_uri: str, *, held_by: str) -> None:
        try:
            rec = self.get_resource(resource_uri)
        except ResourceNotFound:
            return
        updated = rec.model_copy(
            update={"pinned_by": [x for x in rec.pinned_by if x != held_by]}
        )
        self._update_record(updated)

    def is_pinned(self, resource_uri: str) -> bool:
        try:
            rec = self.get_resource(resource_uri)
            return bool(rec.pinned_by)
        except ResourceNotFound:
            return False

    # ------------------------------------------------------------------
    # Action log
    # ------------------------------------------------------------------

    def log_action(
        self,
        *,
        agent_id: str,
        role: str,
        capability: str,
        tool: str,
        input_uris: list[str] | None = None,
        output_uris: list[str] | None = None,
        job_uri: str | None = None,
        experiment_uri: str | None = None,
        candidate_uri: str | None = None,
        cache_hit: bool | None = None,
        budget_used: dict | None = None,
        error: str | None = None,
    ) -> ActionRecord:
        action_id = self._new_id()
        record = ActionRecord(
            id=action_id,
            agent_id=agent_id,
            role=role,
            capability=capability,
            tool=tool,
            input_uris=input_uris or [],
            output_uris=output_uris or [],
            job_uri=job_uri,
            experiment_uri=experiment_uri,
            candidate_uri=candidate_uri,
            cache_hit=cache_hit,
            budget_used=budget_used,
            error=error,
        )
        self._append_jsonl(self._actions_path(), record)
        return record

    def query_actions(
        self,
        agent_id: str | None = None,
        tool: str | None = None,
        capability: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ActionRecord]:
        raw_records = self._read_jsonl(self._actions_path())
        actions = [ActionRecord.model_validate(r) for r in raw_records]
        if agent_id:
            actions = [a for a in actions if a.agent_id == agent_id]
        if tool:
            actions = [a for a in actions if a.tool == tool]
        if capability:
            actions = [a for a in actions if a.capability == capability]
        return actions[offset : offset + limit]

    # ------------------------------------------------------------------
    # Snapshot-on-submit helper
    # ------------------------------------------------------------------

    def snapshot_graph_ref(self, uri: str) -> str:
        """If uri points to a scratch/scratch-rev, return an immutable graph URI.
        If already immutable, return unchanged."""
        if uri.startswith("g2v://graph/"):
            return uri
        # scratch or scratch_rev
        graph = self.load_graph(uri)
        return self.register_graph(graph)


# ---------------------------------------------------------------------------
# Patch application helper
# ---------------------------------------------------------------------------

def _apply_operations(graph: Graph, operations: list[dict]) -> dict:
    """Apply a list of graph operations to *graph* in place.

    Returns a report dict. Raises on unknown op type.
    """
    from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
    from graph_to_voxel.schema.nodes import AnyNode

    applied: list[str] = []
    for op in operations:
        op_type = op.get("op")

        if op_type == "add_node":
            node = AnyNode.model_validate(op.get("node", op.get("params"))).root
            if node.id in graph._nodes:
                raise ValueError(f"Node {node.id!r} already exists")
            graph._nodes[node.id] = node
            applied.append(f"add_node:{node.id}")

        elif op_type == "replace_node":
            node_id = op["id"]
            if node_id not in graph._nodes:
                raise ValueError(f"Node {node_id!r} not found for replace_node")
            node = AnyNode.model_validate(op["node"]).root
            if node.id != node_id:
                raise ValueError("replace_node payload id must match operation id")
            graph._nodes[node_id] = node
            applied.append(f"replace_node:{node_id}")

        elif op_type == "remove_node":
            node_id = op.get("id", op.get("node_id"))
            if node_id not in graph._nodes:
                raise ValueError(f"Node {node_id!r} not found for remove_node")
            incident = [e for e in graph._edges if e.source == node_id or e.target == node_id]
            cascade = bool(op.get("cascade_edges", True))
            if incident and not cascade:
                raise ValueError(f"Node {node_id!r} still has incident edges")
            graph._nodes.pop(node_id, None)
            graph._edges = [e for e in graph._edges
                            if e.source != node_id and e.target != node_id]
            applied.append(f"remove_node:{node_id}")

        elif op_type == "update_node":
            node_id = op["node_id"]
            existing = graph._nodes.get(node_id)
            if existing is None:
                raise WorkspaceStoreError(f"Node {node_id!r} not found for update_node")
            updated_node = existing.model_copy(update=op["patch"])
            graph._nodes[node_id] = updated_node
            applied.append(f"update_node:{node_id}")

        elif op_type == "add_edge":
            if "edge" in op:
                edge = GraphEdge.model_validate(op["edge"])
            else:
                edge = GraphEdge(
                    kind=EdgeKind[op["edge_type"].upper()],
                    source=op["src"],
                    target=op["dst"],
                    **{k: v for k, v in op.get("params", {}).items()},
                )
            if edge.id is not None and any(existing.id == edge.id for existing in graph._edges):
                raise ValueError(f"Edge {edge.id!r} already exists")
            edge = graph._edge_with_id(edge)
            graph._edges.append(edge)
            applied.append(f"add_edge:{edge.id}")

        elif op_type == "replace_edge":
            edge_id = op["id"]
            idx = _edge_index(graph._edges, edge_id)
            edge = GraphEdge.model_validate(op["edge"])
            if edge.id != edge_id:
                raise ValueError("replace_edge payload id must match operation id")
            graph._edges[idx] = edge
            applied.append(f"replace_edge:{edge_id}")

        elif op_type == "remove_edge":
            edge_id = op.get("id", op.get("edge_id"))
            _edge_index(graph._edges, edge_id)
            graph._edges = [e for e in graph._edges if e.id != edge_id]
            applied.append(f"remove_edge:{edge_id}")

        elif op_type == "set_metadata":
            metadata = op.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError("set_metadata requires a metadata object")
            graph.metadata = dict(metadata)
            applied.append("set_metadata")

        elif op_type == "add_sample":
            from graph_to_voxel.schema.uncertainty import Uncertainty

            params = {k: v for k, v in op.items() if k != "op"}
            if isinstance(params.get("value"), dict):
                params["value"] = Uncertainty.model_validate(params["value"]).root
            sample = graph.add_sample(**params)
            applied.append(f"add_sample:{sample.id}")

        else:
            raise WorkspaceStoreError(f"Unknown patch operation type: {op_type!r}")

    _ensure_unique_edge_ids(graph._edges)
    return {"valid": True, "applied": applied, "count": len(applied)}


def _edge_index(edges: list[GraphEdge], edge_id: str | None) -> int:
    if edge_id is None:
        raise ValueError("edge id is required")
    for idx, edge in enumerate(edges):
        if edge.id == edge_id:
            return idx
    raise ValueError(f"Edge {edge_id!r} not found")


def _ensure_unique_edge_ids(edges: list[GraphEdge]) -> None:
    seen: set[str] = set()
    for edge in edges:
        if edge.id is None:
            continue
        if edge.id in seen:
            raise ValueError(f"Duplicate edge id {edge.id!r}")
        seen.add(edge.id)


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def _id_from_uri(uri: str) -> str:
    """Extract the last path component from a g2v:// URI."""
    return uri.rstrip("/").split("/")[-1]


def _scratch_id_from_uri(uri: str) -> str:
    """Extract scratch ID from g2v://scratch/<id> or g2v://scratch/<id>@rev/<n>."""
    path = uri.replace("g2v://scratch/", "")
    return path.split("@")[0]


# ---------------------------------------------------------------------------
# Kind → model class mapping
# ---------------------------------------------------------------------------

_KIND_MAP: dict[str, type] = {
    "graph": GraphRecord,
    "scratch": ScratchRecord,
    "scratch_rev": ScratchRevRecord,
    "field": FieldRecord,
    "data": DataRecord,
    "hypothesis": HypothesisRecord,
    "experiment": ExperimentRecord,
    "result": ResultRecord,
    "review": ReviewRecord,
    "job": JobRecord,
    "candidate": CandidateRecord,
    "score": ScoreRecord,
}
