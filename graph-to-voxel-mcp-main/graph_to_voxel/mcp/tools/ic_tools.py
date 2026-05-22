from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from graph_to_voxel.mcp.workspace.store import WorkspaceStore
from graph_to_voxel.refinement import RefinementCriterionConfig, score_refinement


def ic_score(
    store: WorkspaceStore,
    *,
    candidate_graph_uri: str,
    candidate_field_uri: str,
    reference_a_graph_uri: str,
    reference_a_field_uri: str,
    reference_b_graph_uri: str,
    reference_b_field_uri: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_graph = store.load_graph(candidate_graph_uri)
    candidate_field = store.load_field(candidate_field_uri)
    reference_a_graph = store.load_graph(reference_a_graph_uri)
    reference_a_field = store.load_field(reference_a_field_uri)
    reference_b_graph = store.load_graph(reference_b_graph_uri)
    reference_b_field = store.load_field(reference_b_field_uri)
    cfg = RefinementCriterionConfig(**(config or {}))

    score = score_refinement(
        candidate_graph=candidate_graph,
        candidate_field=candidate_field,
        reference_a_graph=reference_a_graph,
        reference_a_field=reference_a_field,
        reference_b_graph=reference_b_graph,
        reference_b_field=reference_b_field,
        config=cfg,
    )
    breakdown = {
        "structural_bits": float(score.structural_bits),
        "fit_bits": float(score.fit_bits),
        "physics_bits": float(score.physics_bits),
        "gate_failures": [_to_dict(failure) for failure in score.gate_failures],
        "diagnostics": dict(score.diagnostics),
        "flags": list(score.flags),
    }
    score_uri = store.register_score(
        candidate_graph_uri=candidate_graph_uri,
        candidate_field_uri=candidate_field_uri,
        reference_a_graph_uri=reference_a_graph_uri,
        reference_a_field_uri=reference_a_field_uri,
        reference_b_graph_uri=reference_b_graph_uri,
        reference_b_field_uri=reference_b_field_uri,
        score_value=float(score.score_bits),
        breakdown=breakdown,
    )
    return {
        "score_uri": score_uri,
        "score_bits": float(score.score_bits),
        "structural_bits": float(score.structural_bits),
        "fit_bits": float(score.fit_bits),
        "passed_gates": score.passed_gates,
        "breakdown": breakdown,
    }


def ic_score_from_graphs(
    store: WorkspaceStore,
    *,
    candidate_graph_uri: str,
    reference_a_graph_uri: str,
    reference_b_graph_uri: str,
    field_spec: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store.load_graph(candidate_graph_uri)
    store.load_graph(reference_a_graph_uri)
    store.load_graph(reference_b_graph_uri)
    job_uri = store.register_job(
        "ic.score_from_graphs",
        input_uris=[candidate_graph_uri, reference_a_graph_uri, reference_b_graph_uri],
    )
    return {"job_uri": job_uri, "field_spec": field_spec, "config": config or {}}


def _to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


__all__ = ["ic_score", "ic_score_from_graphs"]
