"""Integration tests: queue-driven parent IDs survive into the kg_record.

The TODO at ``tasks/feature_hypothesis.py:1510-1511`` ("parent_node_1 = None")
gets closed by this work. Two surfaces must agree:

  1. ``populate()`` in crossbreed mode pops an ordered pair from the queue and
     stamps it into ``episode_context["crossbreed_context"]["parent_ids"]``
     and into ``phase_records["hypothesise"]["parent_experiments"]``.
  2. ``_exec_submit_rewrite`` reads those parents and writes them as
     ``parent_node_1`` / ``parent_node_2`` on the kg record persisted to
     ``experiments.jsonl``.

These tests drive both surfaces through the public capability API instead of
inspecting private state, so they will keep working if internals are
refactored.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.task.types import (
    CapabilityExecutionContext,
    CapabilityInvocation,
)
from tasks.feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation


def _task(tmp_path: Path) -> FeatureHypothesisTask:
    return FeatureHypothesisTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
        }
    )


def _variation(tmp_path: Path) -> FeatureHypothesisVariation:
    return FeatureHypothesisVariation(
        name="coe_fairbairn",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "coe_fairbairn"),
        kg_dir=str(tmp_path / "kg" / "coe_fairbairn"),
        min_features=0,
        crossbreed_enabled=True,
    )


def _seed_experiments(kg_dir: Path, node_ids: list[str]) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("w") as fh:
        for i, node_id in enumerate(node_ids):
            fh.write(json.dumps({
                "node_id": node_id,
                "hypothesis": f"hyp_{node_id}",
                "response": f"response_{node_id}",
                "bic_delta": -(10.0 + i),
                "layer_name": f"layer_{node_id}",
                "mutual_info": {},
                "masking_test_passed": True,
                "stage_completed": "stage_2_completed",
                "crossbreed_parent_eligible": True,
            }) + "\n")


def _seed_features_index(store_dir: Path, n: int) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    payload = {"layers": {f"layer_{i}": {} for i in range(n)}}
    (store_dir / "index.json").write_text(json.dumps(payload))


def _seed_pairwise_distance(
    kg_dir: Path,
    distances: dict[tuple[str, str], float],
) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "pairwise_distance.jsonl").open("w") as fh:
        for (a, b), dist in distances.items():
            fh.write(json.dumps({
                "pair_id": f"{min(a, b)}_{max(a, b)}",
                "node_1": a,
                "node_2": b,
                "pairwise_distance": dist,
            }) + "\n")


def test_populate_stamps_parents_from_queue(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg_dir = Path(variation.kg_dir)
    store_dir = Path(variation.store_dir)
    # Need >= 5 admitted experiments to clear the crossbreed floor.
    seeded = ["alpha", "beta", "gamma", "delta", "epsilon"]
    _seed_experiments(kg_dir, seeded)
    _seed_features_index(store_dir, n=len(seeded))

    outcome = task.populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "crossbreed"
    parents = outcome.episode_context["crossbreed_context"]["parent_ids"]
    assert isinstance(parents, list)
    assert len(parents) == 2
    assert len(set(parents)) == 2  # an ordered pair of distinct parents
    assert set(parents) <= set(seeded)


def test_concurrent_populates_get_distinct_parent_pairs(tmp_path: Path) -> None:
    # Two consecutive populates against the same kg_dir should NOT receive
    # the same `(parent_a, parent_b)` pair — the queue serves diversity.
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg_dir = Path(variation.kg_dir)
    store_dir = Path(variation.store_dir)
    # Need >= 5 admitted experiments to clear the crossbreed floor.
    seeded = ["alpha", "beta", "gamma", "delta", "epsilon"]
    _seed_experiments(kg_dir, seeded)
    _seed_features_index(store_dir, n=len(seeded))

    first = task.populate([], variation).episode_context["crossbreed_context"]["parent_ids"]
    second = task.populate([], variation).episode_context["crossbreed_context"]["parent_ids"]
    assert tuple(first) != tuple(second), (
        "queue must serve distinct ordered pairs to consecutive episodes"
    )


def test_populate_stays_survey_until_five_viable_parents(tmp_path: Path) -> None:
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg_dir = Path(variation.kg_dir)
    store_dir = Path(variation.store_dir)
    # Five raw parents, but alpha/beta are a practical duplicate cluster.
    seeded = ["alpha", "beta", "gamma", "delta", "epsilon"]
    _seed_experiments(kg_dir, seeded)
    _seed_features_index(store_dir, n=len(seeded))
    _seed_pairwise_distance(kg_dir, {("alpha", "beta"): 0.01})

    outcome = task.populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "survey"


def test_kg_record_persists_parent_node_ids(tmp_path: Path) -> None:
    # End-to-end-ish: drive submit_rewrite with phase_records containing
    # parent_experiments, and check experiments.jsonl gets parent_node_1/2.
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg_dir = Path(variation.kg_dir)
    store_dir = Path(variation.store_dir)
    kg_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    episode_id = "ep_test_parents"
    episode_context: dict = {
        "episode_id": episode_id,
        "store_dir": str(store_dir),
        "kg_dir": str(kg_dir),
        "grid_spec": variation.grid_spec,
        "workflow_kind": "crossbreed",
        "phase_records": {
            "hypothesise": {
                "hypothesis": "child hypothesis built from alpha and beta",
                "data_spec": {},
                "parent_experiments": ["alpha", "beta"],
            },
            "code": {"result_summary": "ok"},
            "translate": {"feature_layer_name": "child_layer"},
            "evaluate": {
                "bic_delta": -42.0,
                "admitted": True,
                "mutual_info": {},
                "masking_test_passed": True,
                "masking_test_improvement": 0.3,
                "masking_test_direction": "improvement",
                "stage_completed": "stage_2_completed",
            },
        },
    }
    ctx = CapabilityExecutionContext(episode_id, "rewrite", episode_context)

    task.execute_capability(
        CapabilityInvocation(
            "submit_rewrite",
            {"prompt": "p", "response": "r"},
        ),
        [],
        variation,
        ctx,
    )

    experiments_file = kg_dir / "experiments.jsonl"
    assert experiments_file.exists(), "submit_rewrite must persist to kg_dir/experiments.jsonl"
    record = json.loads(experiments_file.read_text().strip().splitlines()[-1])
    assert record["parent_node_1"] == "alpha"
    assert record["parent_node_2"] == "beta"


def test_duplicate_submit_kept_as_success_but_not_in_jsonl(tmp_path: Path) -> None:
    # Two episodes with the same (parents, hypothesis) fingerprint should
    # both succeed (TaskReward.success stays True via capability return) but
    # only one entry ends up in experiments.jsonl.
    task = _task(tmp_path)
    variation = _variation(tmp_path)
    kg_dir = Path(variation.kg_dir)
    store_dir = Path(variation.store_dir)
    kg_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    def _drive_submit(ep_id: str) -> None:
        episode_context: dict = {
            "episode_id": ep_id,
            "store_dir": str(store_dir),
            "kg_dir": str(kg_dir),
            "grid_spec": variation.grid_spec,
            "workflow_kind": "crossbreed",
            "phase_records": {
                "hypothesise": {
                    "hypothesis": "same exact hypothesis text",
                    "data_spec": {},
                    "parent_experiments": ["alpha", "beta"],
                },
                "code": {"result_summary": "ok"},
                "translate": {"feature_layer_name": "child_layer"},
                "evaluate": {
                    "bic_delta": -42.0,
                    "admitted": True,
                    "mutual_info": {},
                    "masking_test_passed": True,
                    "masking_test_improvement": 0.3,
                    "masking_test_direction": "improvement",
                    "stage_completed": "stage_2_completed",
                },
            },
        }
        ctx = CapabilityExecutionContext(ep_id, "rewrite", episode_context)
        result = task.execute_capability(
            CapabilityInvocation("submit_rewrite", {"prompt": "p", "response": "r"}),
            [],
            variation,
            ctx,
        )
        assert result.success is True, "duplicate must still return capability success"

    _drive_submit("ep_first")
    _drive_submit("ep_second")

    lines = (kg_dir / "experiments.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1, (
        f"only one entry should reach experiments.jsonl; saw {len(lines)}"
    )
    record = json.loads(lines[0])
    assert record["node_id"].endswith("ep_first")
