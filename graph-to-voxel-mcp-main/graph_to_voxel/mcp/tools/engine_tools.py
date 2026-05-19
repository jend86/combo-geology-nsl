from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from graph_to_voxel.engine import GridSpec, build_voxel_field
from graph_to_voxel.mcp.workspace.models import FieldRunSpec
from graph_to_voxel.mcp.workspace.store import WorkspaceStore


def engine_run(
    store: WorkspaceStore,
    graph_ref: str,
    field_spec: dict[str, Any],
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_uri, spec = _resolve_field_spec(store, graph_ref, field_spec)
    cached = store.get_cached_field(spec)
    if cached is not None:
        return {"field_uri": cached, "from_cache": True, "graph_uri": graph_uri}
    job_uri = store.register_job("engine.run", input_uris=[graph_uri])
    return {"job_uri": job_uri, "graph_uri": graph_uri, "field_run_spec_hash": spec.content_hash()}


def engine_run_preview(
    store: WorkspaceStore,
    graph_ref: str,
    field_spec: dict[str, Any],
    preview_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_uri, spec = _resolve_field_spec(store, graph_ref, field_spec)
    cached = store.get_cached_field(spec)
    if cached is not None:
        return {"field_uri": cached, "from_cache": True, "graph_uri": graph_uri}

    max_voxels = int(
        (preview_budget or {}).get("max_voxels", os.environ.get("G2V_PREVIEW_MAX_VOXELS", 500_000))
    )
    voxel_count = math.prod(spec.grid_shape or (0, 0, 0))
    if voxel_count > max_voxels:
        job_uri = store.register_job("engine.run_preview", input_uris=[graph_uri])
        return {"job_uri": job_uri, "graph_uri": graph_uri, "estimated_voxels": voxel_count}

    graph = store.load_graph(graph_uri)
    grid = GridSpec(
        origin=spec.grid_origin,
        maximum=spec.grid_maximum,
        nx=spec.grid_shape[0],
        ny=spec.grid_shape[1],
        nz=spec.grid_shape[2],
    )
    field = build_voxel_field(
        graph,
        grid,
        bandwidth=spec.bandwidth,
        subgrid_factor=spec.subgrid_factor,
        min_membership=spec.min_membership,
        epsg=spec.epsg,
    )
    field_uri = store.register_field(field, spec)
    return {"field_uri": field_uri, "from_cache": False, "graph_uri": graph_uri}


def voxel_sample(
    store: WorkspaceStore,
    field_uri: str,
    points: list[tuple[float, float, float]],
    limit: int | None = None,
) -> dict[str, Any]:
    field = store.load_field(field_uri)
    max_points = len(points) if limit is None else max(0, limit)
    samples = []
    for point in points[:max_points]:
        x_idx = _nearest_index(field.x, point[0])
        y_idx = _nearest_index(field.y, point[1])
        z_idx = _nearest_index(field.z, point[2])
        probs = field.unit_probs[:, x_idx, y_idx, z_idx].astype(float)
        total = float(probs.sum())
        if total > 0.0:
            probs = probs / total
        sample: dict[str, Any] = {
            "point": [float(point[0]), float(point[1]), float(point[2])],
            "unit_probs": {unit_id: float(probs[idx]) for idx, unit_id in enumerate(field.unit_ids)},
            "entropy": float(field.entropy[x_idx, y_idx, z_idx]),
        }
        if field.support_membership is not None:
            sample["support_membership"] = {
                unit_id: float(field.support_membership[idx, x_idx, y_idx, z_idx])
                for idx, unit_id in enumerate(field.unit_ids)
            }
        samples.append(sample)
    return {"field_uri": field_uri, "samples": samples, "truncated": len(points) > max_points}


def voxel_stats(store: WorkspaceStore, field_uri: str, region: dict[str, Any] | None = None) -> dict[str, Any]:
    field = store.load_field(field_uri)
    mask = np.asarray(field.domain_mask, dtype=bool)
    if not mask.any():
        coverage = {unit_id: 0.0 for unit_id in field.unit_ids}
        mean_entropy = 0.0
    else:
        probs = field.unit_probs[:, mask].astype(float)
        means = probs.mean(axis=1)
        total = float(means.sum())
        if total > 0.0:
            means = means / total
        coverage = {unit_id: float(means[idx]) for idx, unit_id in enumerate(field.unit_ids)}
        mean_entropy = float(np.asarray(field.entropy)[mask].mean())
    return {
        "field_uri": field_uri,
        "shape": list(field.shape),
        "unit_coverage": coverage,
        "mean_entropy": mean_entropy,
        "domain_fraction": float(mask.mean()),
    }


def voxel_export(
    store: WorkspaceStore,
    field_uri: str,
    format: str,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if format != "zarr":
        raise ValueError(f"Unsupported voxel export format: {format!r}")
    store.get_resource(field_uri)
    raw = json.dumps({"field_uri": field_uri, "format": format}).encode()
    data_uri = store.register_data(
        f"exports/{field_uri.rsplit('/', 1)[-1]}.zarr",
        media_type="application/vnd+zarr",
        preview_text=f"Exported {field_uri} as zarr",
        parent_uri=field_uri,
        raw_bytes=raw,
    )
    return {"data_uri": data_uri, "field_uri": field_uri, "format": format}


def _resolve_field_spec(
    store: WorkspaceStore,
    graph_ref: str,
    field_spec: dict[str, Any],
) -> tuple[str, FieldRunSpec]:
    graph_uri = store.snapshot_graph_ref(graph_ref) if graph_ref.startswith("g2v://scratch/") else graph_ref
    graph_record = store.get_resource(graph_uri)
    store.load_graph(graph_uri)
    spec = FieldRunSpec(
        graph_content_hash=graph_record.content_hash,
        grid_origin=tuple(field_spec["grid_origin"]),
        grid_maximum=tuple(field_spec["grid_maximum"]),
        grid_shape=tuple(field_spec["grid_shape"]),
        bandwidth=field_spec.get("bandwidth"),
        subgrid_factor=int(field_spec.get("subgrid_factor", 1)),
        min_membership=float(field_spec.get("min_membership", 0.05)),
        epsg=field_spec.get("epsg"),
        engine_name=field_spec.get("engine_name", "loopstructural"),
        engine_version=field_spec.get("engine_version", "unknown"),
        options=dict(field_spec.get("options", {})),
        prior_field_hash=field_spec.get("prior_field_hash"),
        drop_threshold=field_spec.get("drop_threshold"),
    )
    return graph_uri, spec


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(np.asarray(values, dtype=float) - float(target)).argmin())


__all__ = ["engine_run", "engine_run_preview", "voxel_export", "voxel_sample", "voxel_stats"]
