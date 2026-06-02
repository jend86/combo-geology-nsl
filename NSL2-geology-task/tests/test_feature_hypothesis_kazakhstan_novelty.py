"""Novelty-nudge tests for FeatureHypothesisKazakhstanTask.

The novelty-nudge machinery reads recent admitted hypotheses and can render a
length-capped "avoid repeating" block plus a mechanism-family summary. It is
retained for offline analysis only; workflow prompts deliberately do not inject
it because direct negative examples previously caused agents to mirror saturated
families instead of diversifying.

These tests pin the contract:

  - ``_recent_admitted_hypotheses`` reads ``experiments.jsonl`` and
    returns the last K admitted rows in append order, gracefully handling
    missing files and rows missing the ``hypothesis`` field.
  - ``_render_novelty_block`` produces an empty string on first episodes
    (no admits) and otherwise an enumerated, length-capped list with an
    explicit "avoid" instruction.
  - ``_classify_mechanism`` + ``_render_mechanism_summary`` produce a
    one-line summary of the dominant mechanism family in the recent
    admit pool, with sensible degenerate-case handling.
  - ``_survey_workflow`` and ``_crossbreed_workflow`` keep the block out of
    prompts so the retained helper cannot accidentally steer live episodes.
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


def _step_prompt(workflow, name: str) -> str:
    for step in workflow.steps:
        if step.name == name:
            return step.prompt
    raise AssertionError(
        f"workflow missing {name!r} step: " + repr([s.name for s in workflow.steps])
    )


def _explore_prompt(workflow) -> str:
    # Survey + hypothesise are merged into a single `explore` step; the novelty
    # block is prepended there.
    return _step_prompt(workflow, "explore")


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


class TestClassifyMechanism:
    def test_fold_terms_route_to_structural(self) -> None:
        for txt in [
            "Distance to fold axis predicts copper",
            "Anticline limbs concentrate mineralization",
            "Strike-slip faults control fluid flow",
        ]:
            assert (
                FeatureHypothesisKazakhstanTask._classify_mechanism(txt)
                == "structural"
            )

    def test_redox_terms_route_to_geochemical(self) -> None:
        for txt in [
            "Redox boundary between red beds and reduced strata",
            "Pyrite-to-hematite zoning at the host contact",
            "Spectral analysis assays at depth",
        ]:
            assert (
                FeatureHypothesisKazakhstanTask._classify_mechanism(txt)
                == "geochemical"
            )

    def test_lithological_terms(self) -> None:
        assert (
            FeatureHypothesisKazakhstanTask._classify_mechanism(
                "Vladimirov suite sandstone facies host most prospects"
            )
            == "lithological"
        )

    def test_basin_geometry_terms(self) -> None:
        assert (
            FeatureHypothesisKazakhstanTask._classify_mechanism(
                "Distance to basin margin correlates with prospect density"
            )
            == "basin_geometry"
        )

    def test_drillhole_terms(self) -> None:
        assert (
            FeatureHypothesisKazakhstanTask._classify_mechanism(
                "Per-borehole SP curve polarity flips at lithology contacts"
            )
            == "drillhole"
        )

    def test_unknown_returns_other(self) -> None:
        assert (
            FeatureHypothesisKazakhstanTask._classify_mechanism(
                "Generic text with no domain keywords"
            )
            == "other"
        )

    def test_empty_returns_other(self) -> None:
        assert (
            FeatureHypothesisKazakhstanTask._classify_mechanism("") == "other"
        )

    def test_first_match_wins_when_multiple_buckets_overlap(self) -> None:
        # "fold" is in structural; the rest of the sentence has basin terms.
        # First-match-wins per _MECHANISM_BUCKETS ordering: structural.
        assert (
            FeatureHypothesisKazakhstanTask._classify_mechanism(
                "Fold proximity at basin margin"
            )
            == "structural"
        )


class TestRenderMechanismSummary:
    def test_empty_recent_returns_empty(self) -> None:
        assert FeatureHypothesisKazakhstanTask._render_mechanism_summary([]) == ""

    def test_all_other_returns_empty(self) -> None:
        recent = [
            {"layer_name": "L", "hypothesis": "Generic content with no terms"}
        ]
        assert (
            FeatureHypothesisKazakhstanTask._render_mechanism_summary(recent) == ""
        )

    def test_includes_dominant_family(self) -> None:
        recent = [
            {"layer_name": "L1", "hypothesis": "Anticline distance"},
            {"layer_name": "L2", "hypothesis": "Fold-axis proximity"},
            {"layer_name": "L3", "hypothesis": "Distance to basin margin"},
        ]
        summary = FeatureHypothesisKazakhstanTask._render_mechanism_summary(recent)
        assert summary
        assert "structural" in summary
        # Cardinality (2/3) surfaced so the agent can gauge severity.
        assert "(2/3)" in summary
        # Fact-only summary (no directive examples) — softened after
        # 2026-05-29 fresh run showed agents parroting the "redox/lithology"
        # examples verbatim. Keep only the family-share fact.
        assert summary.startswith("Recent admissions concentrate on:")


class TestNoveltyNudgeNotInjected:
    """The novelty nudge was deprecated 2026-05-31. Its machinery (tested above)
    is retained but no longer injected into any workflow prompt. These guards
    ensure it stays dormant; remove them only when the unused-code decision is
    made (see _novelty_block_for's docstring).
    """

    def test_explore_prompt_has_no_novelty_block(self, tmp_path: Path) -> None:
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
        explore = _explore_prompt(workflow)
        assert "Already discovered" not in explore
        assert "Copper prospects cluster near fold axes." not in explore
        assert "Recent admissions concentrate on:" not in explore

    def test_crossbreed_prompt_has_no_novelty_block(self, tmp_path: Path) -> None:
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
        explore = _explore_prompt(workflow)
        assert "Already discovered" not in explore
        assert "Fold-axis proximity correlates with copper." not in explore
        # The crossbreed context itself is still present on the explore entry.
        assert "Crossbreed Mode" in explore
