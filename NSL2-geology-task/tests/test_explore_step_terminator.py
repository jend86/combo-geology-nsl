"""Kazakhstan survey workflow: survey + hypothesise merged into one explore step.

The explore step is the entry, terminates on record_phase(phase='hypothesise'),
flows to code, and (when a source is assigned) injects the assignment + pre-read
sample into its prompt. The translate step gains the spatial search tools.
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
        description="t",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=str(tmp_path / "kg" / "teniz_basin"),
    )


def _ctx(assigned: dict | None = None) -> dict:
    ctx: dict = {"workflow_kind": "survey", "episode_id": "ep-x"}
    if assigned is not None:
        ctx["assigned_source"] = assigned
        ctx["source_sample"] = "SAMPLE DATA HERE"
    return ctx


def _step(wf, name):
    for s in wf.steps:
        if s.name == name:
            return s
    raise AssertionError(f"missing {name}: {[s.name for s in wf.steps]}")


class TestExploreStep:
    def test_single_explore_entry_no_separate_survey_or_hypothesise(self, tmp_path: Path) -> None:
        wf = _task(tmp_path)._survey_workflow(_variation(tmp_path), _ctx())
        names = {s.name for s in wf.steps}
        assert "explore" in names
        assert "survey" not in names and "hypothesise" not in names

    def test_explore_terminates_on_record_phase_and_flows_to_code(self, tmp_path: Path) -> None:
        wf = _task(tmp_path)._survey_workflow(_variation(tmp_path), _ctx())
        explore = _step(wf, "explore")
        assert explore.is_entry is True
        assert explore.terminator_capabilities == ("record_phase",)
        assert explore.next_steps == ("code",)
        assert "analysis_shell" in explore.capabilities

    def test_assigned_source_and_sample_injected_into_prompt(self, tmp_path: Path) -> None:
        assigned = {
            "key": "smolianova_tectonics",
            "path": "36572_Smolianova_1984/chunks/",
            "glob_pattern": "*TECTONICS*.md",
            "description": "Tectonics chunks.",
        }
        wf = _task(tmp_path)._survey_workflow(_variation(tmp_path), _ctx(assigned))
        explore = _step(wf, "explore")
        assert "ASSIGNED SOURCE FOR THIS EPISODE" in explore.prompt
        assert "smolianova_tectonics" in explore.prompt
        assert "SAMPLE DATA HERE" in explore.prompt
        assert "record_phase" in explore.prompt

    def test_full_step_chain(self, tmp_path: Path) -> None:
        wf = _task(tmp_path)._survey_workflow(_variation(tmp_path), _ctx())
        names = [s.name for s in wf.topological_order()]
        assert names == ["explore", "code", "translate", "rewrite"]

    def test_translate_has_search_capabilities(self, tmp_path: Path) -> None:
        wf = _task(tmp_path)._survey_workflow(_variation(tmp_path), _ctx())
        translate = _step(wf, "translate")
        assert "search_web_geological" in translate.capabilities
        assert "search_geonames_lookup" in translate.capabilities

    def test_workflow_validates(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        ctx = _ctx()
        wf = task._survey_workflow(variation, ctx)
        cap_names = {c.name for c in task.list_capabilities(variation, ctx)}
        wf.validate(cap_names)  # raises on any structural problem
