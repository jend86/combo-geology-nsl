"""Crossbreed episodes must still ground in data before hypothesising.

Regression guard for the bug where ``_crossbreed_workflow`` dropped the
data-grounding entry step (promoting a bare ``hypothesise`` prompt to the
entry). That silently disabled data grounding for every crossbreed episode —
i.e. once the feature pool had crossbreed pairs, NO episode read the source
files before hypothesising.

After the file-rotation pulldown the survey and hypothesise steps are merged
into a single ``explore`` step. Crossbreed mode keeps an ``explore``-named
entry (with ``analysis_shell``) so grounding still happens; only the prompt is
swapped for the crossbreed variant.
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


class TestCrossbreedGroundsViaExplore:
    def test_explore_step_present(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        assert "explore" in {s.name for s in wf.steps}
        # The merge means there is no longer a separate hypothesise step.
        assert "hypothesise" not in {s.name for s in wf.steps}

    def test_explore_is_the_entry_step(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        # entry_step raises if there are multiple is_entry steps.
        assert wf.entry_step is not None
        assert wf.entry_step.name == "explore"

    def test_explore_precedes_code(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        explore = _step(wf, "explore")
        assert explore.next_steps == ("code",)

    def test_explore_keeps_grounding_capability(self, tmp_path: Path) -> None:
        # The whole point: force the agent to open the data files (analysis_shell)
        # before hypothesising, even in crossbreed mode.
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        explore = _step(wf, "explore")
        assert "analysis_shell" in explore.capabilities
        assert "ground yourself in the dataset" in explore.prompt

    def test_explore_remains_crossbreed(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        wf = task._crossbreed_workflow(_variation(tmp_path), _crossbreed_ctx())
        explore = _step(wf, "explore")
        assert "Crossbreed Mode" in explore.prompt
        assert "pA" in explore.prompt and "pB" in explore.prompt

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
        assert names == ["explore", "code", "translate", "rewrite"]
