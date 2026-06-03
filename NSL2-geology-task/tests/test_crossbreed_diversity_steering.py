"""Crossbreed prompt parity with numpy-slices.

Crossbreed episodes should be grounded by parent experiment context plus a
generic instruction to inspect relevant sources. They must not receive the
survey-only assigned-source/sample anchoring block.
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
                        "crossbreed_parent_eligible": True,
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
# (1) file rotation remains survey-only
# ---------------------------------------------------------------------------


def test_populate_does_not_assign_rotation_source_for_crossbreed(tmp_path: Path) -> None:
    from tasks.feature_hypothesis_kazakhstan import _KAZAKHSTAN_SOURCE_FILES

    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)
    # >= 5 admits clears the crossbreed floor; the rabbit-hole-bias gate also
    # requires every source visited + greedy init complete before crossbreed.
    _seed_admits(kg, [f"distinct hypothesis number {i}" for i in range(5)])
    kg.mkdir(parents=True, exist_ok=True)
    (kg / "file_rotation_state.json").write_text(
        json.dumps({"counts": {s["key"]: 2 for s in _KAZAKHSTAN_SOURCE_FILES}})
    )
    (kg / "greedy_init_complete.json").write_text(json.dumps({"status": "complete"}))

    outcome = _task(tmp_path).populate([], variation)

    assert outcome.episode_context["workflow_kind"] == "crossbreed"
    assert "assigned_source" not in outcome.episode_context
    assert "source_sample" not in outcome.episode_context


# ---------------------------------------------------------------------------
# (2) crossbreed prompt uses generic dataset grounding, not assigned-source grounding
# ---------------------------------------------------------------------------


def test_crossbreed_prompt_excludes_assigned_source(tmp_path: Path) -> None:
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
    assert "smolianova_devonian" not in prompt
    assert "XSAMPLE_CROSS_77" not in prompt
    assert "ASSIGNED SOURCE FOR THIS EPISODE" not in prompt
    assert "SAMPLE CONTENT FROM YOUR ASSIGNED SOURCE" not in prompt
    assert "open a\nfew relevant sources" in prompt


# ---------------------------------------------------------------------------
# (3) novelty nudge is REVERTED — no explicit "be a different family" steering
#     (it backfired via negation-priming; diversity must emerge organically).
# ---------------------------------------------------------------------------


def test_crossbreed_prompt_has_no_explicit_novelty_nudge(tmp_path: Path) -> None:
    kg = tmp_path / "kg" / "teniz_basin"
    _seed_admits(kg, [f"copper redox reduced facies variant {i}" for i in range(6)])
    variation = _variation(tmp_path, kg_dir=str(kg))

    wf = _task(tmp_path)._crossbreed_workflow(variation, _crossbreed_ctx())
    prompt = _explore_step(wf).prompt
    assert "DO NOT propose variants" not in prompt, (
        "novelty nudge was reverted — it must not be injected into the crossbreed prompt"
    )
    assert "different angle" not in prompt and "saturated" not in prompt, (
        "no explicit diversity instruction — diversity must emerge organically"
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
