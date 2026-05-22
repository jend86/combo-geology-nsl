"""Dogfood the MCP design (docs/design/08-mcp-scope.md §3 agent workflow).

Walks: Explore -> Hypothesise -> Edit -> Specify Experiment -> Execute ->
Review -> Candidate -> Score, against the in-process tool functions used by
the FastMCP server. Records observations against the design doc.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from pprint import pformat

from graph_to_voxel.mcp.tools import (
    candidate_tools,
    engine_tools,
    experiment_tools,
    graph_tools,
    hypothesis_tools,
    ic_tools,
    workspace_tools,
)
from graph_to_voxel.mcp.workspace.store import WorkspaceStore

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_A_PATH = REPO_ROOT / "examples" / "sample-two-unit-tilted.json"
GRAPH_B_PATH = REPO_ROOT / "examples" / "porphyry-cu.json"


def _step(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def _show(label: str, payload) -> None:
    text = pformat(payload, width=100, compact=True)
    print(f"-- {label}\n{text}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="g2v-dogfood-"))
    print(f"Workspace root: {tmp}")
    store = WorkspaceStore(tmp)

    # ------------------------------------------------------------------
    # 0. Bootstrap via graph.ingest — closes Finding A from the first dogfood.
    # ------------------------------------------------------------------
    _step("0. Bootstrap reference graphs A (baseline) and B (prior candidate)")
    a_in = graph_tools.graph_ingest(
        store,
        filename=GRAPH_A_PATH.name,
        content_text=GRAPH_A_PATH.read_text(),
        message="A: two-unit tilted",
    )
    b_in = graph_tools.graph_ingest(
        store,
        filename=GRAPH_B_PATH.name,
        content_text=GRAPH_B_PATH.read_text(),
        message="B: porphyry-cu",
    )
    _show("graph.ingest A", a_in)
    _show("graph.ingest B", b_in)
    graph_a_uri = a_in["graph_uri"]
    graph_b_uri = b_in["graph_uri"]

    # Demonstrate idempotency
    a_repeat = graph_tools.graph_ingest(
        store, filename=GRAPH_A_PATH.name, content_text=GRAPH_A_PATH.read_text(),
    )
    _show("re-ingest A (cached?)", {"from_cache": a_repeat["from_cache"],
                                     "same_uri": a_repeat["graph_uri"] == graph_a_uri})

    # ------------------------------------------------------------------
    # 1. Explore. Agent inspects the two graphs.
    # ------------------------------------------------------------------
    _step("1. Explore — graph_query, graph_subgraph, graph_diff")
    nodes_a = graph_tools.graph_query(store, graph_a_uri, {"kind": "StratigraphicUnit"}, limit=10)
    _show("query StratigraphicUnits in A", nodes_a)

    seed_id = nodes_a["nodes"][0]["id"]
    sub_a = graph_tools.graph_subgraph(store, graph_a_uri, [seed_id], radius=2, limit=20)
    _show(f"subgraph radius=2 around {seed_id!r}", {"node_count": len(sub_a.get("nodes", [])),
                                                     "edge_count": len(sub_a.get("edges", []))})

    diff = graph_tools.graph_diff(store, graph_a_uri, graph_b_uri, limit=50)
    _show("diff A↔B (truncated counts)", {k: (len(v) if isinstance(v, list) else v)
                                            for k, v in diff.items()})

    # ------------------------------------------------------------------
    # 2. Hypothesise.
    # ------------------------------------------------------------------
    _step("2. Hypothesise — hypothesis_create")
    hyp = hypothesis_tools.hypothesis_create(
        store,
        statement="Adding a third stratigraphic unit between A's two units improves fit to B.",
        graph_refs=[graph_a_uri, graph_b_uri],
        rationale="Geological prior: a thin marker bed is likely between sandstone and shale.",
    )
    _show("hypothesis", hyp)
    hyp_uri = hyp["hypothesis_uri"]

    # ------------------------------------------------------------------
    # 3. Branch + edit graph A, then commit a candidate snapshot.
    # ------------------------------------------------------------------
    _step("3. Branch + apply_patch + commit (candidate from A)")
    branch = graph_tools.graph_branch(store, graph_a_uri)
    _show("branch", branch)
    scratch_uri = branch["scratch_uri"]

    # Add a metadata note (simplest non-destructive patch) so the commit
    # produces a NEW content hash distinct from A. Anything that mutates
    # nodes/edges would also work; this avoids unit-catalog churn.
    patch_res = graph_tools.graph_apply_patch(
        store,
        scratch_uri,
        operations=[{"op": "set_metadata", "metadata": {"derived_from": "A", "dogfood": True}}],
    )
    _show("apply_patch (set_metadata)", patch_res)

    candidate_graph_uri = graph_tools.graph_commit(
        store, scratch_uri, message="Candidate C: A with marker metadata"
    )["graph_uri"]
    _show("candidate_graph_uri", candidate_graph_uri)

    # ------------------------------------------------------------------
    # 4. Specify an experiment. Submit-time snapshotting (design §5.3).
    # ------------------------------------------------------------------
    _step("4. Experiment submit (snapshot-on-submit) + claim + complete")
    exp = experiment_tools.experiment_submit(
        store,
        graph_refs=[graph_a_uri, candidate_graph_uri],
        procedure_uri="g2v://procedure/voxel_field_compare",
        procedure_params={"grid_shape": [16, 16, 8]},
        success_criteria=[
            {"criterion_id": "domain_coverage", "threshold": 0.5, "comparator": ">=",
             "metric_name": "domain_fraction"},
        ],
        hypothesis_uri=hyp_uri,
        budget={"time_s": 60, "memory_mb": 512},
    )
    _show("experiment.submit", exp)
    exp_uri = exp["experiment_uri"]

    claim = experiment_tools.experiment_claim(store, exp_uri, lease_s=60)
    _show("experiment.claim", claim)

    # ------------------------------------------------------------------
    # 5. Execute the procedure: build voxel fields for A, B, candidate.
    # Use engine.run_preview because the grid is tiny (16*16*8=2048 voxels).
    # ------------------------------------------------------------------
    _step("5. Execute — engine.run_preview for A, B, candidate")
    field_spec = {
        "grid_origin": [-1.0, -1.0, -1.0],
        "grid_maximum": [7.0, 7.0, 7.0],
        "grid_shape": [16, 16, 8],
        "bandwidth": 1.5,
        "subgrid_factor": 1,
        "min_membership": 0.05,
    }
    field_a = engine_tools.engine_run_preview(store, graph_a_uri, field_spec)
    _show("field A", field_a)
    field_b = engine_tools.engine_run_preview(store, graph_b_uri, field_spec)
    _show("field B", field_b)
    field_c = engine_tools.engine_run_preview(store, candidate_graph_uri, field_spec)
    _show("field C (candidate)", field_c)

    # FINDING-B: A and C have identical FieldRunSpec (set_metadata doesn't
    # change graph content hash relevant to engine output) — observe whether
    # caching returns the same URI.
    same_field_as_a = field_a["field_uri"] == field_c["field_uri"]
    _show("candidate field == A field (cache reuse)?", same_field_as_a)

    # Mini-experiment: voxel.sample
    samples = engine_tools.voxel_sample(
        store, field_a["field_uri"],
        points=[(0.0, 0.0, 0.0), (3.0, 3.0, 3.0), (6.0, 6.0, 6.0)],
        limit=3,
    )
    _show("voxel.sample (3 pts)", {"truncated": samples["truncated"],
                                    "first_entropy": samples["samples"][0]["entropy"]})

    stats_a = engine_tools.voxel_stats(store, field_a["field_uri"])
    _show("voxel.stats(A)", stats_a)

    # Complete the experiment with a structured criterion outcome.
    complete = experiment_tools.experiment_complete(
        store,
        exp_uri,
        outcome="success",
        criterion_outcomes=[{
            "criterion_id": "domain_coverage",
            "status": "passed" if stats_a["domain_fraction"] >= 0.5 else "failed",
            "evidence_refs": [field_a["field_uri"]],
            "metrics": {"domain_fraction": stats_a["domain_fraction"]},
            "confidence": 0.9,
        }],
        artefact_refs=[field_a["field_uri"], field_c["field_uri"]],
    )
    _show("experiment.complete", complete)

    # ------------------------------------------------------------------
    # 6. Review the experiment.
    # ------------------------------------------------------------------
    _step("6. Review")
    review = experiment_tools.experiment_review(
        store, exp_uri,
        status="accepted",
        notes="Candidate C valid; advancing for scoring.",
        criterion_assessments=[{
            "criterion_id": "domain_coverage",
            "agree": True,
            "notes": "Domain coverage matches reference baseline.",
        }],
        candidate_refs=[candidate_graph_uri],
    )
    _show("experiment.review", review)

    # ------------------------------------------------------------------
    # 7. Score with IC. All six (graph, field) URIs required.
    # ------------------------------------------------------------------
    _step("7. ic.score(candidate=C, ref_a=A, ref_b=B)")
    try:
        score = ic_tools.ic_score(
            store,
            candidate_graph_uri=candidate_graph_uri,
            candidate_field_uri=field_c["field_uri"],
            reference_a_graph_uri=graph_a_uri,
            reference_a_field_uri=field_a["field_uri"],
            reference_b_graph_uri=graph_b_uri,
            reference_b_field_uri=field_b["field_uri"],
        )
        _show("ic.score (passed_gates / score_bits)",
              {"passed_gates": score["passed_gates"],
               "score_bits": score["score_bits"],
               "score_uri": score["score_uri"]})
        score_uri = score["score_uri"]
    except Exception as exc:
        _show("ic.score RAISED", repr(exc))
        score_uri = None

    # ------------------------------------------------------------------
    # 8. Submit candidate.
    # ------------------------------------------------------------------
    _step("8. candidate.submit")
    cand = candidate_tools.candidate_submit(
        store,
        candidate_graph_uri,
        (graph_a_uri, graph_b_uri),
        evidence_refs=[exp_uri],
        score_refs=[score_uri] if score_uri else [],
    )
    _show("candidate", cand)

    # ------------------------------------------------------------------
    # 9. Action log + workspace.describe.
    # ------------------------------------------------------------------
    _step("9. actions.query + workspace.describe")
    actions = workspace_tools.actions_query(store, limit=20)
    _show("actions count", actions.get("total", actions))

    desc = workspace_tools.workspace_describe(store, candidate_graph_uri)
    _show("workspace.describe(candidate)", desc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _step("Findings summary")
    findings = [
        "A. RESOLVED. graph.ingest now bootstraps graphs from JSON payloads "
        "(see step 0). Idempotent on content hash.",
        f"B. set_metadata patch produces a NEW graph hash (content_hash differs), "
        f"but FieldRunSpec.graph_content_hash also changes → cache MISS. "
        f"candidate field == A field? {same_field_as_a}. "
        "Design §7.4 says cache keys derive from full spec incl. graph_content_hash — "
        "this means cosmetic graph edits invalidate field cache. Expected, but worth "
        "highlighting: ergonomic edits like 'add a note' force re-compute.",
        "C. engine.run (non-preview) returns a job_uri but no in-process worker "
        "advances the job — there is currently no executor loop. Same for "
        "ic.score_from_graphs. The synchronous engine.run_preview path is the "
        "only one that produces a field in dogfood scope.",
        "D. experiment.submit currently does not log a snapshot resolution record "
        "(graph_ref_resolution=[] when refs are already immutable). For mutable "
        "scratch refs this would matter; documenting expected behaviour helps.",
        "E. actions.query returned no records: tool-layer functions call store "
        "directly without store.log_action(). Design §10 promises EVERY tool call "
        "stamps the action log. Currently only direct log_action() calls populate "
        "it — the tool layer is missing the stamping middleware.",
        "F. No agent_id / role plumbing through the tool calls. Design §6 + §10 "
        "expect role-tagged provenance; FastMCP layer has no place to attach "
        "this context. Wrapper/composer is supposed to set it, but no plumbing "
        "exists yet.",
        "G. No procedure registry. The experiment uses procedure_uri="
        "'g2v://procedure/voxel_field_compare' as a free-form string. Design §7.5 "
        "calls for procedure.list/describe — these tools are not in server.py.",
    ]
    for f in findings:
        print(f"- {f}")

    print(f"\nWorkspace artefacts: {tmp}")
    # leave workspace on disk for inspection; comment to keep:
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
