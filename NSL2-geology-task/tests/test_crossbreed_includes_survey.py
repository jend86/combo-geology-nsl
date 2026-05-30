"""Crossbreed episodes must still run the Phase-1 survey step.

Regression guard for the bug where ``_crossbreed_workflow`` dropped the
``survey`` step entirely (filtered it out and promoted ``hypothesise`` to the
entry). That silently disabled the data-grounding survey for every crossbreed
episode — i.e. once the feature pool had crossbreed pairs, NO episode read the
source files during a survey phase. Survey is supposed to happen even in
crossbreed mode.
"""

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


def _variation(tmp_path: Path) -> FeatureHypothesisKazakhstanVariation:
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=str(tmp_path / "kg" / "teniz_basin"),
    )


def _crossbreed_ctx() -> dict:
    return {
        "workflow_kind": "crossbreed",
        "crossbreed_context": {
            "parent_ids": ["pA", "pB"],
            "prompt": "Combine parent insights.",
        },
    }


def _step(workflow, name):
    for step in workflow.steps:
        if step.name == name:
            return step
    raise AssertionError(
        f"workflow missing {name!r} step: {[s.name for s in workflow.steps]}"
    )


class TestCrossbreedIncludesSurvey:
    def test_survey_step_present(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        assert "survey" in {s.name for s in wf.steps}

    def test_survey_is_the_entry_step(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        # entry_step raises if there are multiple is_entry steps.
        assert wf.entry_step is not None
        assert wf.entry_step.name == "survey"

    def test_survey_precedes_crossbreed_hypothesise(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        survey = _step(wf, "survey")
        hyp = _step(wf, "hypothesise")
        assert survey.next_steps == ("hypothesise",)
        assert hyp is not None
        assert hyp.is_entry is False

    def test_survey_keeps_corpora_sampling_mandate(self, tmp_path: Path) -> None:
        # The whole point of survey: force the agent to open the data files
        # across all three corpus classes before hypothesising.
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        survey = _step(wf, "survey")
        assert "analysis_shell" in survey.capabilities
        assert "sample at least one source" in survey.prompt
        for corpus in ("vector", "tabular", "text"):
            assert corpus in survey.prompt

    def test_hypothesise_remains_crossbreed(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        hyp = _step(wf, "hypothesise")
        assert "Crossbreed Mode" in hyp.prompt
        assert "pA" in hyp.prompt and "pB" in hyp.prompt

    def test_workflow_validates(self, tmp_path: Path) -> None:
        # Exactly one entry, no fan-in, all steps reachable.
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        ctx = _crossbreed_ctx()
        wf = task._crossbreed_workflow(variation, ctx)
        cap_names = {c.name for c in task.list_capabilities(variation, ctx)}
        wf.validate(cap_names)  # raises on any structural problem

    def test_full_step_chain(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        names = [s.name for s in wf.topological_order()]
        assert names == ["survey", "hypothesise", "code", "translate", "rewrite"]
