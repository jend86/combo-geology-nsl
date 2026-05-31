"""Approach C: restore diversity steering on crossbreed episodes.

Pre-pulldown, crossbreed ran the survey step (with the novelty/family nudge) as
its entry, so crossbreed proposals were steered toward diversity. The merge
deprecated the novelty nudge AND scoped file rotation to survey-only, leaving
crossbreed — the dominant episode type — with NO diversity steering. Measured
result: crossbreed hypotheses collapsed to a single family (see
docs/design/sft-explore-boundary-resplit-2026-05-31.md follow-up).

C restores both levers for crossbreed:
  1. file rotation also assigns a least-explored source (+ pre-read sample) to
     crossbreed episodes, and the crossbreed prompt grounds in it; and
  2. the family-balance (novelty) block is injected into the crossbreed prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import (
    FeatureHypothesisKazakhstanTask,
    FeatureHypothesisKazakhstanVariation,
)


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {"store_dir": str(tmp_path / "store_root"), "kg_dir": str(tmp_path / "kg_root")}
    )


def _variation(tmp_path: Path, kg_dir: str | None = None) -> FeatureHypothesisKazakhstanVariation:
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=kg_dir or str(tmp_path / "kg" / "teniz_basin"),
    )


def _seed_admits(kg_dir: Path, hypotheses: list[str]) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("w") as fh:
        for i, hyp in enumerate(hypotheses):
            fh.write(
                json.dumps(
                    {
                        "node_id": f"n{i}",
                        "hypothesis": hyp,
                        "layer_name": f"layer_{i}",
                        "bic_delta": -(5.0 + i),
                    }
                )
                + "\n"
            )


def _explore_step(workflow):
    for step in workflow.steps:
        if step.name == "explore":
            return step
    raise AssertionError(f"no explore step: {[s.name for s in workflow.steps]}")


def _crossbreed_ctx(**extra) -> dict:
    ctx = {
        "workflow_kind": "crossbreed",
        "crossbreed_context": {"parent_ids": ["pA", "pB"], "prompt": "Combine parent insights."},
    }
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# (1) file rotation now reaches crossbreed
# ---------------------------------------------------------------------------


def test_populate_assigns_rotation_source_for_crossbreed(tmp_path: Path) -> None:
    variation = _variation(tmp_path)
    # >= 5 admitted experiments clears the crossbreed floor.
    _seed_admits(Path(variation.kg_dir), [f"distinct hypothesis number {i}" for i in range(5)])

    outcome = _task(tmp_path).populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "crossbreed"
    assert outcome.episode_context.get("assigned_source"), (
        "crossbreed episodes must receive a rotated least-explored source "
        "(C: diversity steering) — currently survey-only"
    )


# ---------------------------------------------------------------------------
# (2) crossbreed prompt grounds in the assigned source
# ---------------------------------------------------------------------------


def test_crossbreed_prompt_includes_assigned_source(tmp_path: Path) -> None:
    ctx = _crossbreed_ctx(
        assigned_source={
            "key": "smolianova_devonian",
            "path": "36572_Smolianova_1984/chunks/",
            "description": "Devonian stratigraphy",
            "glob_pattern": "*Devonian*.md",
        },
        source_sample="XSAMPLE_CROSS_77 light/purple-brown devonian redbed alternation",
    )
    wf = _task(tmp_path)._crossbreed_workflow(_variation(tmp_path), ctx)
    prompt = _explore_step(wf).prompt
    assert "smolianova_devonian" in prompt
    assert "XSAMPLE_CROSS_77" in prompt


# ---------------------------------------------------------------------------
# (3) crossbreed prompt carries the family-balance (novelty) signal
# ---------------------------------------------------------------------------


def test_crossbreed_prompt_includes_family_balance_block(tmp_path: Path) -> None:
    kg = tmp_path / "kg" / "teniz_basin"
    _seed_admits(kg, [f"copper redox reduced facies variant {i}" for i in range(6)])
    variation = _variation(tmp_path, kg_dir=str(kg))

    wf = _task(tmp_path)._crossbreed_workflow(variation, _crossbreed_ctx())
    prompt = _explore_step(wf).prompt
    assert "DO NOT propose variants" in prompt, (
        "crossbreed prompt must carry the family-balance signal (novelty block)"
    )


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


def test_crossbreed_prompt_preserves_grounding_and_mode(tmp_path: Path) -> None:
    wf = _task(tmp_path)._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
    prompt = _explore_step(wf).prompt
    assert "Crossbreed Mode" in prompt
    assert "ground yourself in the dataset" in prompt
    assert "pA" in prompt and "pB" in prompt


def test_survey_explore_prompt_keeps_headers(tmp_path: Path) -> None:
    # Guards the _assigned_source_blocks refactor: survey prompt headers (which
    # the SFT transform regex-extracts) must be byte-stable.
    ec = {
        "assigned_source": {
            "key": "dev",
            "path": "36572_Smolianova_1984/chunks/x.md",
            "description": "d",
        },
        "source_sample": "SURVEY_SAMPLE_55 body text",
    }
    prompt = _task(tmp_path)._generate_explore_prompt(ec)
    assert "ASSIGNED SOURCE FOR THIS EPISODE" in prompt
    assert "SAMPLE CONTENT FROM YOUR ASSIGNED SOURCE" in prompt
    assert "SURVEY_SAMPLE_55" in prompt
