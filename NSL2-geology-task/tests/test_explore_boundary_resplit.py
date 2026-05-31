"""Tests for the explore-phase query/response boundary re-split.

Two coupled changes are exercised here:

  #1  analysis_plan target must understand the new explore data_spec schema
      ``{analysis, files, output}`` (output -> target feature) instead of
      silently collapsing to a bare ``Required files:`` line.

  #2  The source material the agent read (ASSIGNED SOURCE + SAMPLE CONTENT,
      and the data_spec ``analysis`` observation) belongs on the QUERY side:
        - dataset_hypothesis (T2): query carries a clean source excerpt and
          NOT the record_phase / task-constraint scaffolding.
        - analysis_plan (T3): query carries the observation + source excerpt;
          the target is the planned feature/files only (observation moved out).

  Plus a best-effort enrichment that reads the actual dataset file from a
  configured ``dataset_dir`` when available.

All of these fail on current ``main`` and pass after the boundary re-split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training_data.transforms import EpisodeTrainingRows
from tasks.feature_hypothesis_kazakhstan import (
    PAIR_KIND_ANALYSIS_PLAN,
    PAIR_KIND_DATASET_HYPOTHESIS,
    ExperimentReasoningRows,
)

# A unique token planted inside the SAMPLE CONTENT block so we can assert it
# travels to the query side intact.
SAMPLE_MARKER = "ZZQ_DEVONIAN_REDBED_MARKER_42"
OBSERVATION = (
    "Examination of the Devonian stratigraphy section reveals an alternation "
    "between light-colored quartz sandstones and purple-brown siltstones."
)
OUTPUT_FEATURE = "devonian_redox_alternation_layer"
ASSIGNED_FILE = (
    "/workspace/input/36572_Smolianova_1984/chunks/"
    "36572_020_V_STRATIGRAPHY_Devonian_Gray-brown_sandstone.md"
)


def _kz_explore_prompt(*, sample_marker: str = SAMPLE_MARKER) -> str:
    """A realistic Kazakhstan *survey* explore prompt with the two parseable
    blocks (assignment + pre-read sample) plus the tool-call scaffolding that
    must NOT leak onto the query side."""
    return (
        "[system] You are analyzing Kazakhstan mineral prospects.\n\n"
        "Phase 1: Explore and Hypothesise\n\n"
        "ASSIGNED SOURCE FOR THIS EPISODE\n"
        "  Section: smolianova_stratigraphy_devonian\n"
        "  Details: Smolianova 1984 Ch. V - Devonian stratigraphy (~16 chunks)\n\n"
        "SAMPLE CONTENT FROM YOUR ASSIGNED SOURCE\n"
        "-----------------------------------------\n"
        "--- 36572_017_V_STRATIGRAPHY_Devonian_-152-.md ---\n"
        "# V. STRATIGRAPHY - Devonian\n"
        f"Light-colored sandstones interbedded with purple-brown siltstones. {sample_marker}\n"
        "\n"
        "Use analysis_shell to read and explore your assigned source.\n"
        "When you have identified a promising geological pattern, record ONE "
        "falsifiable hypothesis:\n\n"
        "  record_phase(\n"
        "      phase='hypothesise',\n"
        "      hypothesis='...',\n"
        "      data_spec={ 'files': [...], 'analysis': '...', 'output': '...' }\n"
        "  )\n\n"
        "Your hypothesis MUST be grounded in what you found in the assigned source above.\n"
        "Task constraints:\n"
        "- task tool calls: at most 100\n"
    )


def _base_row(
    *,
    episode_id: str,
    row_index: int,
    workflow_step: str,
    prompt: str,
    raw_response: str,
    actor_role: str | None = None,
) -> dict[str, Any]:
    return {
        "row_id": f"{episode_id}:{row_index}",
        "parent_row_id": None,
        "prompt": prompt,
        "raw_response": raw_response,
        "interaction_type": "orchestrator",
        "source_interaction_type": "orchestrator",
        "timestamp": "2026-05-31T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": 0,
        "generation_id": 1,
        "episode_score": 1.0,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": actor_role,
        "record_meta": {},
    }


def _survey_group(
    *,
    episode_id: str = "ep-kz",
    explore_prompt: str | None = None,
    data_spec: dict[str, Any] | None = None,
    hypothesis: str = (
        "Devonian terrigenous redbed alternation in the Teniz Basin marks a "
        "fluctuating redox front favourable to sandstone-hosted copper."
    ),
) -> EpisodeTrainingRows:
    """A de-novo (survey / no-parent) episode whose explore step carries the
    assigned-source sample and a new-schema data_spec."""
    if explore_prompt is None:
        explore_prompt = _kz_explore_prompt()
    if data_spec is None:
        data_spec = {
            "analysis": OBSERVATION,
            "files": [ASSIGNED_FILE],
            "output": OUTPUT_FEATURE,
        }
    rows = [
        _base_row(
            episode_id=episode_id,
            row_index=0,
            workflow_step="explore",
            prompt=explore_prompt,
            raw_response="",
        ),
        _base_row(
            episode_id=episode_id,
            row_index=1,
            workflow_step="code",
            prompt="phase_get(phase='hypothesise') returned files",
            raw_response="execution finalized with artifacts",
        ),
        _base_row(
            episode_id=episode_id,
            row_index=2,
            workflow_step="translate",
            prompt="hypothesis + available files",
            raw_response="feature layer built",
        ),
        _base_row(
            episode_id=episode_id,
            row_index=3,
            workflow_step="rewrite",
            actor_role="rewriter_output",
            prompt="hypothesis + built feature",
            raw_response=(
                "The redox-alternation feature captures a non-structural "
                "control on mineralization.\n"
                "Result: -1.5000 BIC delta. Admitted."
            ),
        ),
    ]
    group = EpisodeTrainingRows(
        episode_id=episode_id,
        episode_index=0,
        generation_id=1,
        episode_score=1.0,
        rows=rows,
    )
    group.episode_context = {  # type: ignore[attr-defined]
        "success": True,
        "duplicate_rejected": False,
        "workflow_kind": "survey",
        "phase_records": {
            "hypothesise": {
                "hypothesis": hypothesis,
                "data_spec": data_spec,
                "parent_experiments": [],
            },
            "translate": {"feature_layer_name": OUTPUT_FEATURE},
            "evaluate": {"bic_delta": -1.5},
        },
    }
    return group


def _rows_of_kind(result: list[EpisodeTrainingRows], kind: str) -> list[dict[str, Any]]:
    return [
        row
        for group in result
        for row in group.rows
        if row.get("record_meta", {}).get("pair_kind") == kind
    ]


def _run(transform: ExperimentReasoningRows, groups: list[EpisodeTrainingRows]):
    return transform.transform_export_rows(context=None, episodes=groups)


# ---------------------------------------------------------------------------
# #1 - analysis_plan formatter understands the new {analysis, files, output} schema
# ---------------------------------------------------------------------------


class TestAnalysisPlanFormatter:
    def test_target_includes_output_feature(self) -> None:
        result = _run(ExperimentReasoningRows(), [_survey_group()])
        plan_rows = _rows_of_kind(result, PAIR_KIND_ANALYSIS_PLAN)
        assert plan_rows, "expected an analysis_plan row"
        target = plan_rows[0]["raw_response"]
        assert OUTPUT_FEATURE in target, (
            f"analysis_plan target dropped the data_spec 'output' feature: {target!r}"
        )

    def test_target_is_not_required_files_only(self) -> None:
        result = _run(ExperimentReasoningRows(), [_survey_group()])
        target = _rows_of_kind(result, PAIR_KIND_ANALYSIS_PLAN)[0]["raw_response"]
        stripped = target.strip()
        assert not stripped.startswith("Required files:") or "\n" in stripped, (
            f"analysis_plan target degenerated to a bare files line: {target!r}"
        )

    def test_old_schema_still_renders_steps(self) -> None:
        old_spec = {
            "target_feature": "co_ag_overlap",
            "required_files": ["/workspace/input/USGS/TZ_ssCu_Prospects.csv"],
            "analysis_steps": ["filter high Co", "filter high Ag", "intersect"],
        }
        result = _run(ExperimentReasoningRows(), [_survey_group(data_spec=old_spec)])
        target = _rows_of_kind(result, PAIR_KIND_ANALYSIS_PLAN)[0]["raw_response"]
        assert "co_ag_overlap" in target
        assert "filter high Co" in target


# ---------------------------------------------------------------------------
# #2 - source-read evidence belongs on the query side
# ---------------------------------------------------------------------------


class TestDatasetHypothesisBoundary:
    def test_query_contains_source_excerpt(self) -> None:
        result = _run(ExperimentReasoningRows(), [_survey_group()])
        rows = _rows_of_kind(result, PAIR_KIND_DATASET_HYPOTHESIS)
        assert rows, "expected a dataset_hypothesis row for a de-novo episode"
        assert SAMPLE_MARKER in rows[0]["prompt"], (
            "dataset_hypothesis query must carry the assigned-source excerpt"
        )

    def test_query_excludes_tool_scaffolding(self) -> None:
        result = _run(ExperimentReasoningRows(), [_survey_group()])
        prompt = _rows_of_kind(result, PAIR_KIND_DATASET_HYPOTHESIS)[0]["prompt"]
        assert "record_phase(" not in prompt, (
            f"dataset_hypothesis query leaked record_phase scaffolding: {prompt!r}"
        )
        assert "Task constraints" not in prompt, (
            f"dataset_hypothesis query leaked task-constraint scaffolding: {prompt!r}"
        )

    def test_evidence_is_bic_sanitized(self) -> None:
        leaky_prompt = _kz_explore_prompt().replace(
            f"{SAMPLE_MARKER}\n",
            f"{SAMPLE_MARKER}\nResult: -3.2100 BIC delta. Admitted.\n",
        )
        result = _run(
            ExperimentReasoningRows(), [_survey_group(explore_prompt=leaky_prompt)]
        )
        prompt = _rows_of_kind(result, PAIR_KIND_DATASET_HYPOTHESIS)[0]["prompt"]
        assert "BIC delta" not in prompt
        assert "Admitted" not in prompt


class TestAnalysisPlanBoundary:
    def test_query_contains_observation_and_excerpt(self) -> None:
        result = _run(ExperimentReasoningRows(), [_survey_group()])
        prompt = _rows_of_kind(result, PAIR_KIND_ANALYSIS_PLAN)[0]["prompt"]
        assert OBSERVATION in prompt, (
            "analysis_plan query must carry the data_spec 'analysis' observation"
        )
        assert SAMPLE_MARKER in prompt, (
            "analysis_plan query must carry the assigned-source excerpt"
        )

    def test_target_excludes_observation(self) -> None:
        result = _run(ExperimentReasoningRows(), [_survey_group()])
        target = _rows_of_kind(result, PAIR_KIND_ANALYSIS_PLAN)[0]["raw_response"]
        assert OBSERVATION not in target, (
            "analysis_plan target must not repeat the observation moved to the query"
        )


# ---------------------------------------------------------------------------
# best-effort dataset_dir disk backfill
# ---------------------------------------------------------------------------


class TestSourceSideCapture:
    """Part 2 - future runs persist the evidence at record_phase so the export
    no longer depends on transcript regex-parsing or on-disk dataset reads."""

    def test_transform_prefers_captured_source_excerpt(self) -> None:
        # An explore prompt with NO parseable ASSIGNED/SAMPLE blocks: the only
        # way evidence reaches the query is the persisted source_excerpt.
        plain_prompt = (
            "Phase 1: Explore and Hypothesise\n"
            "Use analysis_shell to read your source.\n"
            "record_phase(phase='hypothesise', ...)\n"
        )
        captured = "CAPTURED_EXCERPT_MARKER_77 light/purple-brown redbed alternation"
        group = _survey_group(explore_prompt=plain_prompt)
        group.episode_context["phase_records"]["hypothesise"]["source_excerpt"] = captured  # type: ignore[attr-defined]
        group.episode_context["phase_records"]["hypothesise"]["assigned_section"] = "smolianova_devonian"  # type: ignore[attr-defined]

        result = _run(ExperimentReasoningRows(), [group])
        prompt = _rows_of_kind(result, PAIR_KIND_DATASET_HYPOTHESIS)[0]["prompt"]
        assert "CAPTURED_EXCERPT_MARKER_77" in prompt, (
            "transform should use the persisted source_excerpt when present"
        )
        assert "smolianova_devonian" in prompt

    def test_record_phase_captures_source_evidence(self) -> None:
        from types import SimpleNamespace

        from src.task.loader import load_task

        task = load_task(
            "tasks.feature_hypothesis_kazakhstan.FeatureHypothesisKazakhstanTask"
        )
        ctx = SimpleNamespace(
            episode_context={
                "assigned_source": {
                    "key": "smolianova_stratigraphy_devonian",
                    "path": "36572_Smolianova_1984/chunks",
                },
                "source_sample": "SAMPLE_TEXT_MARKER_55 pre-read assigned-source content",
            }
        )
        task._exec_record_phase(
            {
                "phase": "hypothesise",
                "hypothesis": "h",
                "data_spec": {"files": [], "analysis": "a", "output": "o"},
            },
            ctx,
        )
        rec = ctx.episode_context["phase_records"]["hypothesise"]
        assert rec.get("source_excerpt") == (
            "SAMPLE_TEXT_MARKER_55 pre-read assigned-source content"
        )
        assert rec.get("assigned_section") == "smolianova_stratigraphy_devonian"


class TestDatasetDirBackfill:
    def test_evidence_enriched_from_dataset_dir(self, tmp_path: Path) -> None:
        rel = ASSIGNED_FILE.replace("/workspace/input/", "")
        disk_file = tmp_path / rel
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_marker = "ZZQ_ON_DISK_FULLTEXT_99"
        disk_file.write_text(
            "# Full chunk text pulled from the dataset on disk.\n"
            f"Detailed Devonian redbed description. {disk_marker}\n",
            encoding="utf-8",
        )
        transform = ExperimentReasoningRows(dataset_dir=str(tmp_path))
        result = _run(transform, [_survey_group()])
        prompt = _rows_of_kind(result, PAIR_KIND_DATASET_HYPOTHESIS)[0]["prompt"]
        assert disk_marker in prompt, (
            "with dataset_dir set, evidence should be enriched from the on-disk file"
        )
