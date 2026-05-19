"""Bootstrap hard-gate validator for seed graphs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tasks.common.g2v_scorer import DEFAULT_FIELD_SPEC, _criterion_config, _load_or_build_field


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    from graph_to_voxel.analyses.checks import check_domain_closure, check_voxel_stratigraphic_order
    from graph_to_voxel.graph import GraphValidationError, RealisationInfeasible
    from graph_to_voxel.mcp.workspace.store import WorkspaceStore
    from graph_to_voxel.refinement.criterion import GateFailure, reject_dedup
    from graph_to_voxel.voxel import stratigraphic_constrain

    store = WorkspaceStore(Path(args.workspace))
    config = _criterion_config(json.loads(args.config or "{}"))
    field_spec = json.loads(args.field_spec or json.dumps(DEFAULT_FIELD_SPEC))
    graph = store.load_graph(args.seed_graph_uri)
    field, field_uri, rebuilt = _load_or_build_field(
        store,
        args.seed_graph_uri,
        args.seed_field_uri or None,
        field_spec,
    )

    failures: list[GateFailure] = []
    try:
        graph.validate()
    except GraphValidationError as exc:
        failures.append(GateFailure("schema_validity", str(exc)))
    try:
        stratigraphic_constrain(graph)
    except RealisationInfeasible as exc:
        failures.append(GateFailure("stratigraphic_consistency", str(exc)))
    domain = check_domain_closure(field)
    if domain.severity == "fail":
        failures.append(GateFailure(domain.name, domain.details))
    try:
        order = check_voxel_stratigraphic_order(graph, field)
    except Exception as exc:
        failures.append(GateFailure("voxel_stratigraphic_order", str(exc)))
    else:
        if order.severity == "fail":
            failures.append(GateFailure(order.name, order.details))

    pool = [store.load_graph(uri) for uri in args.pool_graph_uri]
    if pool and reject_dedup(graph, pool, config=config):
        failures.append(GateFailure("dedup", "seed is structurally within dedup_epsilon of a pool member"))

    graph_rec = store.get_resource(args.seed_graph_uri)
    return {
        "passed_gates": not failures,
        "gate_failures": [failure.name for failure in failures],
        "gate_failure_details": [failure.details for failure in failures],
        "seed_graph_hash": graph_rec.content_hash,
        "seed_field_uri": field_uri,
        "rebuilt_field": rebuilt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a bootstrap seed graph")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seed-graph-uri", required=True)
    parser.add_argument("--seed-field-uri", default="")
    parser.add_argument("--pool-graph-uri", action="append", default=[])
    parser.add_argument("--config", default="{}")
    parser.add_argument("--field-spec", default="")
    parser.add_argument("--output-format", choices=["json"], default="json")
    args = parser.parse_args(argv)
    try:
        payload = _validate(args)
    except BaseException as exc:  # noqa: BLE001
        payload = {
            "passed_gates": False,
            "gate_failures": ["seed_validator_error"],
            "score_bits": math.inf,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
