from __future__ import annotations

import csv
import mimetypes
from io import StringIO
from pathlib import Path
from typing import Any

from graph_to_voxel.graph.core import Graph
from graph_to_voxel.mcp.workspace.models import (
    DataRecord,
    ExperimentRecord,
    ExperimentSpec,
    FieldRecord,
    FieldRunSpec,
    GraphRecord,
    JobRecord,
    ResourceRecord,
    ScratchRecord,
    ScratchRevRecord,
)
from graph_to_voxel.mcp.workspace.store import ResourceNotFound, WorkspaceStore


PROCEDURES: dict[str, dict[str, Any]] = {
    "g2v://procedure/build_field": {
        "name": "build_field",
        "schema": {
            "required": ["graph_refs", "procedure_params", "budget"],
            "procedure_params": {"grid": "Grid bounds/shape or equivalent field spec"},
        },
        "input_types": ["graph"],
        "output_types": ["field"],
        "budget_hints": {"time_s": 30, "memory_mb": 512},
    },
    "g2v://procedure/sample_field": {
        "name": "sample_field",
        "schema": {"required": ["field_refs", "procedure_params"]},
        "input_types": ["field"],
        "output_types": ["result"],
        "budget_hints": {"time_s": 5, "memory_mb": 128},
    },
}


class FileWorkspace:
    """High-level file-backed workspace facade used by the v0 MCP monolith.

    The underlying store is JSONL plus resource files. This facade owns the
    agent-facing contracts: typed URIs, action stamping, snapshot-on-submit, and
    bounded local-volume ingest.
    """

    def __init__(self, root: str | Path, *, input_roots: list[str | Path] | None = None) -> None:
        self.root = Path(root)
        self.store = WorkspaceStore(self.root)
        self.input_roots = [Path(path).resolve() for path in input_roots or []]

    def put_graph(self, graph: Graph, *, agent_id: str = "local", message: str | None = None) -> str:
        uri = self.store.register_graph(graph, message=message)
        self.store.log_action(
            agent_id=agent_id,
            role="proposer",
            capability="graph:commit",
            tool="graph.put",
            output_uris=[uri],
        )
        return uri

    def load_graph(self, uri: str) -> Graph:
        return self.store.load_graph(uri)

    def branch_graph(self, graph_uri: str, *, agent_id: str = "local") -> dict[str, str]:
        scratch_uri = self.store.create_scratch(graph_uri)
        scratch = self.store.get_scratch_record(scratch_uri)
        self.store.log_action(
            agent_id=agent_id,
            role="proposer",
            capability="graph:edit",
            tool="graph.branch",
            input_uris=[graph_uri],
            output_uris=[scratch_uri, scratch.head_rev_uri],
        )
        return {
            "scratch_uri": scratch_uri,
            "base_graph_uri": graph_uri,
            "head_rev_uri": scratch.head_rev_uri,
        }

    def apply_graph_patch(
        self,
        scratch_uri: str,
        *,
        operations: list[dict[str, Any]],
        agent_id: str = "local",
    ) -> dict[str, Any]:
        head_rev_uri, report = self.store.apply_scratch_patch(scratch_uri, operations)
        self.store.log_action(
            agent_id=agent_id,
            role="proposer",
            capability="graph:edit",
            tool="graph.apply_patch",
            input_uris=[scratch_uri],
            output_uris=[head_rev_uri],
        )
        return {"head_rev_uri": head_rev_uri, "validation_report": report}

    def scratch_head(self, scratch_uri: str) -> str:
        return self.store.get_scratch_record(scratch_uri).head_rev_uri

    def commit_graph(
        self,
        scratch_uri: str,
        *,
        message: str | None = None,
        agent_id: str = "local",
    ) -> str:
        graph_uri = self.store.commit_scratch(scratch_uri, message=message)
        self.store.log_action(
            agent_id=agent_id,
            role="proposer",
            capability="graph:commit",
            tool="graph.commit",
            input_uris=[scratch_uri],
            output_uris=[graph_uri],
        )
        return graph_uri

    def ingest_data(
        self,
        path: str | Path,
        *,
        media_type_hint: str | None = None,
        agent_id: str = "local",
    ) -> str:
        resolved, relative = self._resolve_ingest_path(path)
        raw = resolved.read_bytes()
        media_type = media_type_hint or mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        preview = raw[:64_000].decode("utf-8", errors="replace")
        uri = self.store.register_data(
            relative.as_posix(),
            media_type=media_type,
            preview_text=preview,
            raw_bytes=raw,
        )
        data_dir = self.store._data_path(uri.rsplit("/", 1)[-1])
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / resolved.name
        if not data_file.exists():
            data_file.write_bytes(raw)
        self.store.log_action(
            agent_id=agent_id,
            role="proposer",
            capability="data:ingest",
            tool="data.ingest",
            output_uris=[uri],
        )
        return uri

    def preview_data(self, data_uri: str, *, rows: int = 10) -> dict[str, Any]:
        record = self.store.get_data_record(data_uri)
        text = record.preview_text
        if record.media_type in {"text/csv", "application/csv"} or record.source_path_rel.endswith(".csv"):
            parsed = list(csv.reader(StringIO(text)))[:rows]
            return {"uri": data_uri, "media_type": record.media_type, "rows": parsed}
        return {"uri": data_uri, "media_type": record.media_type, "text": text[: rows * 1000]}

    def submit_experiment(self, spec: dict[str, Any], *, agent_id: str = "local") -> str:
        procedure_uri = str(spec.get("procedure_uri", ""))
        if procedure_uri not in PROCEDURES:
            raise ValueError(f"Unknown procedure URI: {procedure_uri!r}")

        frozen_graph_refs: list[str] = []
        resolution: list[dict[str, Any]] = []
        for raw_ref in spec.get("graph_refs", []):
            original = _ref_uri(raw_ref)
            frozen = self.store.snapshot_graph_ref(original)
            record = self.store.get_resource(frozen)
            frozen_graph_refs.append(frozen)
            resolution.append(
                {
                    "original_ref": original,
                    "snapshot_ref": frozen,
                    "content_hash": record.content_hash,
                    "agent_id": agent_id,
                }
            )

        for uri in [_ref_uri(item) for item in spec.get("data_refs", [])]:
            self._require_kind(uri, "data")
        for uri in [_ref_uri(item) for item in spec.get("field_refs", [])]:
            self._require_kind(uri, "field")

        frozen_spec = dict(spec)
        frozen_spec["graph_refs"] = frozen_graph_refs
        frozen_spec["data_refs"] = [_ref_uri(item) for item in spec.get("data_refs", [])]
        frozen_spec["field_refs"] = [_ref_uri(item) for item in spec.get("field_refs", [])]
        experiment_spec = ExperimentSpec.model_validate(frozen_spec)
        experiment_uri = self.store.register_experiment(
            experiment_spec,
            graph_ref_resolution=resolution,
        )
        for graph_uri in frozen_graph_refs:
            self.store.pin(graph_uri, held_by=experiment_uri)
        self.store.log_action(
            agent_id=agent_id,
            role="proposer",
            capability="experiment:submit",
            tool="experiment.submit",
            input_uris=[item["original_ref"] for item in resolution],
            output_uris=[experiment_uri],
            experiment_uri=experiment_uri,
        )
        return experiment_uri

    def create_job(
        self,
        *,
        kind: str,
        input_uris: list[str] | None = None,
        budget: dict[str, Any] | None = None,
        agent_id: str = "local",
    ) -> str:
        job_uri = self.store.register_job(kind, input_uris=input_uris)
        self.store.log_action(
            agent_id=agent_id,
            role="executor",
            capability=_capability_for_job(kind),
            tool="job.create",
            input_uris=input_uris or [],
            output_uris=[job_uri],
            job_uri=job_uri,
            budget_used=budget,
        )
        return job_uri

    def job_status(self, job_uri: str) -> dict[str, Any]:
        record = self.store.get_job_record(job_uri)
        body = record.model_dump(mode="json")
        body["state"] = record.status
        return body

    def cancel_job(
        self,
        job_uri: str,
        *,
        reason: str | None = None,
        agent_id: str = "local",
    ) -> dict[str, Any]:
        record = self.store.update_job(job_uri, status="cancelled", current_step=reason or "cancelled")
        self.store.log_action(
            agent_id=agent_id,
            role="executor",
            capability="engine:run",
            tool="job.cancel",
            input_uris=[job_uri],
            output_uris=[job_uri],
            job_uri=job_uri,
        )
        body = record.model_dump(mode="json")
        body["state"] = record.status
        return body

    def describe(self, uri: str) -> dict[str, Any]:
        if uri in PROCEDURES:
            procedure = PROCEDURES[uri]
            return {
                "uri": uri,
                "type": "procedure",
                "media_type": "application/json",
                "size": 0,
                "size_bytes": 0,
                "state": "available",
                "pins": [],
                "summary": {"name": procedure["name"]},
                "metadata": procedure,
            }
        record = self.store.get_resource(uri)
        return _describe_record(record)

    def read_resource(self, uri: str) -> dict[str, Any]:
        if uri in PROCEDURES:
            return PROCEDURES[uri]
        record = self.store.get_resource(uri)
        if isinstance(record, GraphRecord | ScratchRecord | ScratchRevRecord):
            return self.load_graph(uri).to_dict()
        if isinstance(record, DataRecord):
            return {
                "uri": uri,
                "media_type": record.media_type,
                "source_path": record.source_path_rel,
                "preview": record.preview_text,
            }
        if isinstance(record, ExperimentRecord):
            record = self.store.get_experiment_record(uri)
            body = record.model_dump(mode="json")
            body["state"] = record.status
            return body
        if isinstance(record, JobRecord):
            return self.job_status(uri)
        return record.model_dump(mode="json")

    def list_procedures(self) -> list[str]:
        return list(PROCEDURES)

    def describe_procedure(self, procedure_uri: str) -> dict[str, Any]:
        if procedure_uri not in PROCEDURES:
            raise ResourceNotFound(procedure_uri)
        return PROCEDURES[procedure_uri]

    def query_actions(self, *, tool_name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return [
            action.model_dump(mode="json")
            for action in self.store.query_actions(tool=tool_name, limit=limit)
        ]

    def gc(self, policy: dict[str, Any] | None = None) -> dict[str, Any]:
        dry_run = bool((policy or {}).get("dry_run", True))
        return {"evicted": [], "retained": [record.uri for record in self.store.list_resources()], "bytes_freed": 0, "dry_run": dry_run}

    def _resolve_ingest_path(self, path: str | Path) -> tuple[Path, Path]:
        if not self.input_roots:
            raise PermissionError("No input roots are configured for data ingest")
        raw = Path(path)
        candidates = [raw.resolve()] if raw.is_absolute() else [(root / raw).resolve() for root in self.input_roots]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            for root in self.input_roots:
                try:
                    return candidate, candidate.relative_to(root)
                except ValueError:
                    continue
        raise PermissionError(f"Path {str(path)!r} is outside configured input roots")

    def _require_kind(self, uri: str, kind: str) -> None:
        record = self.store.get_resource(uri)
        if record.kind != kind:
            raise ValueError(f"Expected {kind} resource, got {record.kind}: {uri}")


def _describe_record(record: ResourceRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    state = "durable"
    media_type = "application/json"

    if isinstance(record, GraphRecord):
        summary = {
            "node_count": record.node_count,
            "edge_count": record.edge_count,
            "unit_catalog": record.unit_catalog,
        }
    elif isinstance(record, ScratchRecord):
        state = "committed" if record.committed else "open"
        summary = {"base_graph_uri": record.base_graph_uri, "head_rev_uri": record.head_rev_uri}
    elif isinstance(record, ScratchRevRecord):
        summary = {"scratch_uri": record.scratch_uri, "rev": record.rev}
    elif isinstance(record, DataRecord):
        media_type = record.media_type
        metadata = {
            "source_path": record.source_path_rel,
            "detected_schema": record.detected_schema,
            "crs": record.crs,
            "parent_uri": record.parent_uri,
        }
        summary = {"preview": record.preview_text[:1000]}
    elif isinstance(record, FieldRecord):
        summary = {"graph_uri": record.graph_uri, "grid_shape": record.grid_shape, "unit_catalog": record.unit_catalog}
    elif isinstance(record, ExperimentRecord):
        state = record.status
        summary = {"procedure_uri": record.spec.procedure_uri, "graph_refs": record.spec.graph_refs}
    elif isinstance(record, JobRecord):
        state = record.status
        summary = {"job_type": record.job_type, "progress": record.progress, "current_step": record.current_step}

    return {
        "uri": record.uri,
        "type": record.kind,
        "media_type": media_type,
        "size": record.size_bytes,
        "size_bytes": record.size_bytes,
        "state": state,
        "pins": list(record.pinned_by),
        "content_hash": record.content_hash,
        "summary": summary,
        "metadata": metadata,
    }


def _ref_uri(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "uri" in value:
        return str(value["uri"])
    if hasattr(value, "uri"):
        return str(value.uri)
    raise ValueError(f"Invalid resource reference: {value!r}")


def _capability_for_job(kind: str) -> str:
    if kind.startswith("engine."):
        return "engine:run"
    if kind.startswith("lab."):
        return "lab:run"
    return "experiment:execute"


__all__ = ["FieldRunSpec", "FileWorkspace", "PROCEDURES"]
