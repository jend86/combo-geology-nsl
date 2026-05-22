"""One-shot authoritative scorer for geology graph refinement episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_FIELD_SPEC: dict[str, Any] = {
    "grid_origin": [0.0, 0.0, 0.0],
    "grid_maximum": [10.0, 10.0, 10.0],
    "grid_shape": [8, 8, 8],
    "min_membership": 0.01,
}


def _criterion_config(config: dict[str, Any]) -> Any:
    from graph_to_voxel.refinement import RefinementCriterionConfig
    from graph_to_voxel.refinement.criterion import StructuralCostConfig

    cfg = dict(config or {})
    structural = cfg.get("structural")
    if isinstance(structural, dict):
        cfg["structural"] = StructuralCostConfig(**structural)
    return RefinementCriterionConfig(**cfg)


def _load_or_build_field(
    store: Any,
    graph_uri: str,
    field_uri: str | None,
    field_spec: dict[str, Any],
) -> tuple[Any, str, bool]:
    from graph_to_voxel.mcp.tools.engine_tools import engine_run_preview

    if field_uri:
        try:
            return store.load_field(field_uri), field_uri, False
        except Exception:
            pass
    result = engine_run_preview(
        store,
        graph_ref=graph_uri,
        field_spec=field_spec,
        preview_budget={"max_voxels": int(1e12)},
    )
    rebuilt_uri = result.get("field_uri")
    if not isinstance(rebuilt_uri, str):
        raise RuntimeError(f"field rebuild did not produce field_uri: {result}")
    return store.load_field(rebuilt_uri), rebuilt_uri, True


def _score(args: argparse.Namespace) -> dict[str, Any]:
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore
    from graph_to_voxel.refinement import score_refinement

    store = WorkspaceStore(Path(args.workspace))
    config = _criterion_config(json.loads(args.config or "{}"))
    field_spec = json.loads(args.field_spec or json.dumps(DEFAULT_FIELD_SPEC))

    candidate_graph = store.load_graph(args.candidate_graph_uri)
    reference_a_graph = store.load_graph(args.ref_a_uri)
    reference_b_graph = store.load_graph(args.ref_b_uri)

    candidate_field, candidate_field_uri, candidate_rebuilt = _load_or_build_field(
        store,
        args.candidate_graph_uri,
        args.candidate_field_uri or None,
        field_spec,
    )
    reference_a_field, ref_a_field_uri, ref_a_rebuilt = _load_or_build_field(
        store,
        args.ref_a_uri,
        args.ref_a_field_uri or None,
        field_spec,
    )
    reference_b_field, ref_b_field_uri, ref_b_rebuilt = _load_or_build_field(
        store,
        args.ref_b_uri,
        args.ref_b_field_uri or None,
        field_spec,
    )
    pool = [store.load_graph(uri) for uri in args.pool_graph_uri]

    score = score_refinement(
        candidate_graph=candidate_graph,
        candidate_field=candidate_field,
        reference_a_graph=reference_a_graph,
        reference_a_field=reference_a_field,
        reference_b_graph=reference_b_graph,
        reference_b_field=reference_b_field,
        config=config,
        pool=pool or None,
    )
    candidate_rec = store.get_resource(args.candidate_graph_uri)
    return {
        "passed_gates": score.passed_gates,
        "score_bits": float(score.score_bits),
        "structural_bits": float(score.structural_bits),
        "fit_bits": float(score.fit_bits),
        "physics_bits": float(score.physics_bits),
        "gate_failures": [gf.name for gf in score.gate_failures],
        "gate_failure_details": [gf.details for gf in score.gate_failures],
        "diagnostics": dict(score.diagnostics),
        "candidate_graph_hash": candidate_rec.content_hash,
        "candidate_field_uri": candidate_field_uri,
        "reference_a_field_uri": ref_a_field_uri,
        "reference_b_field_uri": ref_b_field_uri,
        "rebuilt_fields": {
            "candidate": candidate_rebuilt,
            "reference_a": ref_a_rebuilt,
            "reference_b": ref_b_rebuilt,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a candidate graph refinement")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--candidate-graph-uri", required=True)
    parser.add_argument("--candidate-field-uri", default="")
    parser.add_argument("--ref-a-uri", required=True)
    parser.add_argument("--ref-a-field-uri", default="")
    parser.add_argument("--ref-b-uri", required=True)
    parser.add_argument("--ref-b-field-uri", default="")
    parser.add_argument("--pool-graph-uri", action="append", default=[])
    parser.add_argument("--config", default="{}")
    parser.add_argument("--field-spec", default="")
    parser.add_argument("--output-format", choices=["json"], default="json")
    args = parser.parse_args(argv)
    try:
        payload = _score(args)
    except BaseException as exc:  # noqa: BLE001 - stdout stays machine-readable
        payload = {
            "passed_gates": False,
            "score_bits": math.inf,
            "structural_bits": math.inf,
            "fit_bits": math.inf,
            "physics_bits": math.inf,
            "gate_failures": ["field_build_failed"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
