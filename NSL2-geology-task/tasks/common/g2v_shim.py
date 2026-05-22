"""Task-owned library-mode adapter for graph-to-voxel tools.

The geology task does not expose the upstream FastMCP server. It keeps a
small, explicit dispatch table here and runs this code either in-process for
tests or inside the g2v representation container via ``g2v_worker``.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ROUTED_TOOLS: frozenset[str] = frozenset(
    {
        "graph_query",
        "graph_subgraph",
        "graph_diff",
        "graph_provenance",
        "graph_ingest",
        "seed_graph_submit",
        "graph_branch",
        "graph_apply_patch",
        "graph_commit",
        "refine_commit",
        "hypothesis_create",
        "hypothesis_list",
        "hypothesis_get",
        "engine_run",
        "engine_run_preview",
        "job_status",
        "voxel_sample",
        "voxel_stats",
        "data_ingest",
        "candidate_submit",
        "workspace_get",
    }
)


def _is_engine_exception(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(
        needle in name or needle in text
        for needle in (
            "loopstructural",
            "outofmemory",
            "memoryerror",
            "geometry",
            "singular",
            "realisationinfeasible",
        )
    )


def _error_payload(exc: BaseException) -> dict[str, Any]:
    kind = "engine_failed" if _is_engine_exception(exc) else "tool_error"
    return {
        "error": kind,
        "type": type(exc).__name__,
        "detail": str(exc),
    }


def _call_tool(fn: Callable[..., dict[str, Any]], store: Any, args: dict[str, Any]) -> dict[str, Any]:
    return fn(store, **dict(args or {}))


@dataclass
class G2VShim:
    """Explicit dispatch table over upstream g2v tool functions."""

    workspace: Path
    imports_root: Path | None = None
    max_workers: int = 2
    _store: Any = field(init=False, repr=False)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _futures: dict[str, Future[dict[str, Any]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        from graph_to_voxel.mcp.workspace.store import WorkspaceStore

        self.workspace = Path(self.workspace)
        self.imports_root = Path(self.imports_root) if self.imports_root else None
        self._store = WorkspaceStore(self.workspace)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.max_workers)))

    @property
    def store(self) -> Any:
        return self._store

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def dispatch(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if tool not in ROUTED_TOOLS:
            return {"error": "tool_not_routed", "tool": tool}
        args = dict(args or {})
        try:
            if tool == "data_ingest":
                return self._data_ingest(args)
            if tool == "workspace_get":
                return self._workspace_get(args)
            if tool == "engine_run":
                return self._engine_run(args)
            if tool == "job_status":
                return self._job_status(args)
            return _call_tool(self._dispatch_table()[tool], self._store, args)
        except BaseException as exc:  # noqa: BLE001 - serialized for the agent
            return _error_payload(exc)

    def _dispatch_table(self) -> dict[str, Callable[..., dict[str, Any]]]:
        from graph_to_voxel.mcp.tools.candidate_tools import candidate_submit
        from graph_to_voxel.mcp.tools.engine_tools import (
            engine_run_preview,
            voxel_sample,
            voxel_stats,
        )
        from graph_to_voxel.mcp.tools.graph_tools import (
            graph_apply_patch,
            graph_branch,
            graph_commit,
            graph_diff,
            graph_ingest,
            graph_provenance,
            graph_query,
            refine_commit,
            seed_graph_submit,
            graph_subgraph,
        )
        from graph_to_voxel.mcp.tools.hypothesis_tools import (
            hypothesis_create,
            hypothesis_get,
            hypothesis_list,
        )

        return {
            "candidate_submit": candidate_submit,
            "engine_run_preview": engine_run_preview,
            "graph_apply_patch": graph_apply_patch,
            "graph_branch": graph_branch,
            "graph_commit": graph_commit,
            "graph_diff": graph_diff,
            "graph_ingest": graph_ingest,
            "graph_provenance": graph_provenance,
            "graph_query": graph_query,
            "refine_commit": refine_commit,
            "seed_graph_submit": seed_graph_submit,
            "graph_subgraph": graph_subgraph,
            "hypothesis_create": hypothesis_create,
            "hypothesis_get": hypothesis_get,
            "hypothesis_list": hypothesis_list,
            "voxel_sample": voxel_sample,
            "voxel_stats": voxel_stats,
        }

    def _data_ingest(self, args: dict[str, Any]) -> dict[str, Any]:
        from graph_to_voxel.mcp.tools.data_tools import data_ingest

        args = dict(args)
        promoted_path = args.pop("path", None)
        if promoted_path is not None:
            file_path = self._validated_import_path(str(promoted_path))
            args["file_path"] = str(file_path)
            args.setdefault("filename", file_path.name)
        return data_ingest(self._store, **args)

    def _validated_import_path(self, path: str) -> Path:
        if self.imports_root is None:
            raise ValueError("path-based data_ingest is disabled without imports_root")
        candidate = Path(path).resolve()
        root = self.imports_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("data_ingest path must be under the episode imports directory")
        if not candidate.is_file():
            raise ValueError(f"data_ingest path is not a file: {path}")
        return candidate

    def _workspace_get(self, args: dict[str, Any]) -> dict[str, Any]:
        uri = args.get("uri") or args.get("resource_uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError("workspace_get requires uri")
        rec = self._store.get_resource(uri)
        base = {
            "uri": rec.uri,
            "kind": rec.kind,
            "size_bytes": rec.size_bytes,
            "content_hash": rec.content_hash,
            "tags": dict(rec.tags),
            "pinned_by": list(rec.pinned_by),
        }
        if rec.kind == "graph":
            base["graph"] = self._store.load_graph(uri).to_dict()
        elif rec.kind == "hypothesis":
            hyp = self._store.get_hypothesis_record(uri)
            base.update(
                {
                    "statement": hyp.statement,
                    "graph_refs": list(hyp.graph_refs),
                    "data_refs": list(hyp.data_refs),
                    "rationale": hyp.rationale,
                }
            )
        elif rec.kind == "data":
            data = self._store.get_data_record(uri)
            base.update(
                {
                    "media_type": data.media_type,
                    "preview_text": data.preview_text,
                    "crs": data.crs,
                    "detected_schema": dict(data.detected_schema),
                }
            )
        return base

    def _engine_run(self, args: dict[str, Any]) -> dict[str, Any]:
        graph_ref = args.get("graph_ref") or args.get("graph_uri")
        field_spec = args.get("field_spec")
        if not isinstance(graph_ref, str) or not graph_ref:
            raise ValueError("engine_run requires graph_ref")
        if not isinstance(field_spec, dict):
            raise ValueError("engine_run requires field_spec object")
        job_uri = self._store.register_job("engine.run", input_uris=[graph_ref])
        future = self._executor.submit(self._build_field_job, job_uri, graph_ref, field_spec)
        self._futures[job_uri] = future
        return {"job_uri": job_uri, "graph_uri": graph_ref}

    def _build_field_job(
        self,
        job_uri: str,
        graph_ref: str,
        field_spec: dict[str, Any],
    ) -> dict[str, Any]:
        from graph_to_voxel.mcp.tools.engine_tools import engine_run_preview

        self._store.update_job(job_uri, status="running", progress=0.1, current_step="build_voxel_field")
        try:
            result = engine_run_preview(
                self._store,
                graph_ref=graph_ref,
                field_spec=field_spec,
                preview_budget={"max_voxels": int(1e12)},
            )
            field_uri = result.get("field_uri")
            if not isinstance(field_uri, str):
                raise RuntimeError(f"engine_run did not produce a field: {result}")
            self._store.update_job(
                job_uri,
                status="completed",
                progress=1.0,
                current_step="completed",
                result_refs=[field_uri],
            )
            return {"field_uri": field_uri, "graph_uri": result.get("graph_uri", graph_ref)}
        except BaseException as exc:  # noqa: BLE001 - serialized in job record
            self._store.update_job(
                job_uri,
                status="failed",
                progress=1.0,
                current_step="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        from graph_to_voxel.mcp.tools.job_tools import job_status

        job_uri = args.get("job_uri")
        if isinstance(job_uri, str):
            future = self._futures.get(job_uri)
            if future is not None and future.done():
                # Observe exceptions so they are not left dangling. The job
                # record already carries the serialized error.
                try:
                    future.result()
                except BaseException:
                    pass
        return job_status(self._store, **args)


__all__ = ["G2VShim", "ROUTED_TOOLS"]
