"""Novelty-nudge tests for FeatureHypothesisKazakhstanTask.

Spec: Approach B of
``docs/design/kazakhstan-variance-and-throughput-2026-05-24.md``.

The agent was observed converging on 3 unique fingerprints across 245
episodes. The nudge surfaces the last K admitted hypotheses in the
proposer prompt as a "do not propose variants of these" block, attacking
the diversity collapse at the prompt layer (orthogonal to the dedup gate,
which only catches lexical duplicates after the fact).

These tests pin the contract:

  - ``_recent_admitted_hypotheses`` reads ``experiments.jsonl`` and
    returns the last K admitted rows in append order, gracefully handling
    missing files and rows missing the ``hypothesis`` field.
  - ``_render_novelty_block`` produces an empty string on first episodes
    (no admits) and otherwise an enumerated, length-capped list with an
    explicit "avoid" instruction.
  - Both ``_survey_workflow`` and ``_crossbreed_workflow`` route the
    block into the hypothesise step prompt when admissions exist, and
    skip it cleanly when the variation knob is off.
"""

from __future__ import annotations

import json
from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import (
    FeatureHypothesisKazakhstanTask,
    FeatureHypothesisKazakhstanVariation,
)


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask({
        "store_dir": str(tmp_path / "store_root"),
        "kg_dir": str(tmp_path / "kg_root"),
    })


def _variation(
    tmp_path: Path,
    *,
    novelty_nudge_enabled: bool = True,
    novelty_recent_k: int = 8,
) -> FeatureHypothesisKazakhstanVariation:
    kg_dir = tmp_path / "kg" / "teniz_basin"
    store_dir = tmp_path / "store" / "teniz_basin"
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(store_dir),
        kg_dir=str(kg_dir),
        novelty_nudge_enabled=novelty_nudge_enabled,
        novelty_recent_k=novelty_recent_k,
    )


def _seed_experiments(kg_dir: Path, rows: list[dict]) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _hypothesise_prompt(workflow) -> str:
    for step in workflow.steps:
        if step.name == "hypothesise":
            return step.prompt
    raise AssertionError(
        "workflow missing hypothesise step: " + repr([s.name for s in workflow.steps])
    )


class TestRecentAdmittedHypotheses:
    def test_empty_when_no_experiments_file(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = tmp_path / "kg" / "teniz_basin"
        # File never written; nothing to read.
        assert task._recent_admitted_hypotheses(kg_dir, k=8) == []

    def test_returns_last_k_in_admit_order(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = tmp_path / "kg" / "teniz_basin"
        rows = [
            {"node_id": f"exp_{i}", "hypothesis": f"H{i}", "layer_name": f"L{i}"}
            for i in range(10)
        ]
        _seed_experiments(kg_dir, rows)

        # Last 3 entries, file order = admit order.
        recent = task._recent_admitted_hypotheses(kg_dir, k=3)
        assert [r["hypothesis"] for r in recent] == ["H7", "H8", "H9"]
        assert [r["layer_name"] for r in recent] == ["L7", "L8", "L9"]

    def test_skips_rows_missing_hypothesis(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = tmp_path / "kg" / "teniz_basin"
        rows = [
            {"node_id": "a", "hypothesis": "alpha", "layer_name": "La"},
            {"node_id": "b", "layer_name": "Lb"},  # no hypothesis
            {"node_id": "c", "hypothesis": "   ", "layer_name": "Lc"},  # blank
            {"node_id": "d", "hypothesis": "delta", "layer_name": "Ld"},
        ]
        _seed_experiments(kg_dir, rows)

        recent = task._recent_admitted_hypotheses(kg_dir, k=4)
        assert [r["hypothesis"] for r in recent] == ["alpha", "delta"]

    def test_k_zero_returns_empty(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = tmp_path / "kg" / "teniz_basin"
        _seed_experiments(kg_dir, [{"node_id": "x", "hypothesis": "h"}])
        assert task._recent_admitted_hypotheses(kg_dir, k=0) == []


class TestRenderNoveltyBlock:
    def test_empty_for_empty_input(self) -> None:
        assert FeatureHypothesisKazakhstanTask._render_novelty_block([]) == ""

    def test_includes_layer_name_and_hypothesis(self) -> None:
        block = FeatureHypothesisKazakhstanTask._render_novelty_block([
            {"layer_name": "fold_proximity", "hypothesis": "h1"},
            {"layer_name": "fault_density", "hypothesis": "h2"},
        ])
        assert "fold_proximity" in block
        assert "h1" in block
        assert "fault_density" in block
        assert "h2" in block
        # Numbered list.
        assert "1." in block
        assert "2." in block

    def test_truncates_long_hypothesis(self) -> None:
        long_hyp = "X" * 1000
        block = FeatureHypothesisKazakhstanTask._render_novelty_block(
            [{"layer_name": "L", "hypothesis": long_hyp}],
            max_chars=120,
        )
        # Truncated form ends in ellipsis; total truncated body well below 1000.
        assert "..." in block
        assert "X" * 500 not in block

    def test_contains_avoid_instruction(self) -> None:
        block = FeatureHypothesisKazakhstanTask._render_novelty_block([
            {"layer_name": "L", "hypothesis": "h"},
        ])
        # Some explicit "avoid" / "different" language so the agent
        # understands the intent (and isn't tempted to mirror).
        lowered = block.lower()
        assert "avoid" in lowered or "different" in lowered or "new" in lowered


class TestNoveltyInjectionInWorkflow:
    def test_survey_hypothesise_includes_block_when_admissions_exist(
        self, tmp_path: Path
    ) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        _seed_experiments(kg_dir, [
            {
                "node_id": "exp1",
                "hypothesis": "Copper prospects cluster near fold axes.",
                "layer_name": "fold_proximity",
            },
        ])
        workflow = task._survey_workflow(variation, {"workflow_kind": "survey"})
        prompt = _hypothesise_prompt(workflow)
        assert "fold_proximity" in prompt
        assert "Copper prospects cluster near fold axes." in prompt

    def test_survey_hypothesise_omits_block_when_no_admissions(
        self, tmp_path: Path
    ) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        # No experiments.jsonl seeded.
        workflow = task._survey_workflow(variation, {"workflow_kind": "survey"})
        prompt = _hypothesise_prompt(workflow)
        # The block header marker is the canonical novelty-block signal.
        assert "Already discovered" not in prompt

    def test_crossbreed_hypothesise_includes_block(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        _seed_experiments(kg_dir, [
            {
                "node_id": "exp1",
                "hypothesis": "Fold-axis proximity correlates with copper.",
                "layer_name": "fold_proximity",
            },
        ])
        ctx = {
            "workflow_kind": "crossbreed",
            "crossbreed_context": {
                "parent_ids": ["pA", "pB"],
                "prompt": "Combine parent insights.",
            },
        }
        workflow = task._crossbreed_workflow(variation, ctx)
        prompt = _hypothesise_prompt(workflow)
        assert "fold_proximity" in prompt
        assert "Fold-axis proximity correlates with copper." in prompt

    def test_disabled_via_knob(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path, novelty_nudge_enabled=False)
        kg_dir = Path(variation.kg_dir)
        _seed_experiments(kg_dir, [
            {"node_id": "x", "hypothesis": "should not appear", "layer_name": "L"},
        ])
        workflow = task._survey_workflow(variation, {"workflow_kind": "survey"})
        prompt = _hypothesise_prompt(workflow)
        assert "should not appear" not in prompt
        assert "Already discovered" not in prompt

    def test_recent_k_zero_omits_block(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path, novelty_recent_k=0)
        kg_dir = Path(variation.kg_dir)
        _seed_experiments(kg_dir, [
            {"node_id": "x", "hypothesis": "h", "layer_name": "L"},
        ])
        workflow = task._survey_workflow(variation, {"workflow_kind": "survey"})
        prompt = _hypothesise_prompt(workflow)
        assert "Already discovered" not in prompt
