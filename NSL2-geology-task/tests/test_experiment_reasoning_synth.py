"""Tests for synthesized geology reasoning prompt-completion rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.training_data.transforms import EpisodeTrainingRows, validate_training_row_groups
from tasks.feature_hypothesis_kazakhstan import (
    PAIR_KIND_ANALYSIS_PLAN,
    PAIR_KIND_DATASET_HYPOTHESIS,
    PAIR_KIND_OUTCOME_NARRATIVE,
    PAIR_KIND_PARENT_HYPOTHESIS,
    ExperimentReasoningRows,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BIC_PATTERN = re.compile(r"-?\d+\.\d+")
_RESULT_APPENDIX = re.compile(r"Result:\s*-?\d+\.\d+\s+BIC delta\.")


def _base_row(
    *,
    episode_id: str = "ep-0",
    row_index: int = 0,
    workflow_step: str = "explore",
    actor_role: str | None = None,
    prompt: str | None = None,
    raw_response: str | None = None,
    success: bool = True,
    episode_score: float = 1.0,
) -> dict[str, Any]:
    return {
        "row_id": f"{episode_id}:{row_index}",
        "parent_row_id": None,
        "prompt": prompt if prompt is not None else f"prompt-{row_index}",
        "raw_response": raw_response if raw_response is not None else f"response-{row_index}",
        "interaction_type": "orchestrator",
        "source_interaction_type": "orchestrator",
        "timestamp": "2026-05-24T00:00:00",
        "success": success,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": 0,
        "generation_id": 1,
        "episode_score": episode_score,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": actor_role,
        "record_meta": {},
    }


def _episode(
    *,
    episode_id: str = "ep-0",
    episode_score: float = 1.0,
    success: bool = True,
    phase_records: dict[str, Any] | None = None,
    duplicate_rejected: bool = False,
) -> EpisodeTrainingRows:
    """Build a minimal EpisodeTrainingRows that resembles a real geology episode.

    The episode carries phase_records and success state in episode_context;
    ExperimentReasoningRows must read these to synthesize pairs.
    """
    if phase_records is None:
        phase_records = _default_phase_records()

    parent_ids = phase_records.get("hypothesise", {}).get("parent_experiments", [])
    has_parents = bool(parent_ids)
    explore_prompt = "dataset facts: depth columns, lithology columns ..."
    crossbreed_context: dict[str, Any] = {}
    if has_parents:
        parent_prompt = (
            'Experiment 1: "porosity correlates with depth"\n'
            'Experiment 2: "seismic boundaries mark reservoir quality"\n'
            "Given these parent findings, ground yourself in the data and "
            "propose a combined hypothesis."
        )
        explore_prompt = (
            "Phase 1: Explore + Hypothesise (Crossbreed Mode)\n\n"
            f"{parent_prompt}"
        )
        crossbreed_context = {
            "parent_ids": parent_ids,
            "prompt": parent_prompt,
        }

    rows = [
        _base_row(
            episode_id=episode_id,
            row_index=0,
            workflow_step="explore",
            prompt=explore_prompt,
            raw_response=(
                "Hypothesis: acoustic impedance contrast predicts reservoir quality.\n"
                "Reasoning: impedance contrasts at formation boundaries indicate ...\n"
                "DataSpec: {files: ['seismic.csv'], transform: 'log_ratio'}"
            ),
            success=success,
            episode_score=episode_score,
        ),
        _base_row(
            episode_id=episode_id,
            row_index=1,
            workflow_step="code",
            prompt="phase_get(phase='hypothesise') returned seismic.csv and well_logs.csv",
            raw_response="execution finalized with artifacts",
            success=success,
            episode_score=episode_score,
        ),
        _base_row(
            episode_id=episode_id,
            row_index=2,
            workflow_step="translate",
            prompt="hypothesis + available files: seismic.csv, well_logs.csv",
            raw_response=(
                "DataSpec plan: use seismic.csv columns AI_top, AI_base; "
                "compute log ratio; no code yet."
            ),
            success=success,
            episode_score=episode_score,
        ),
        _base_row(
            episode_id=episode_id,
            row_index=3,
            workflow_step="rewrite",
            prompt=(
                "hypothesis + context: acoustic impedance contrast predicts reservoir quality"
            ),
            raw_response=(
                "Narrative: the feature captures formation boundary sharpness.\n"
                "Verdict: the hypothesis is supported by the improvement in BIC.\n"
                "Result: -1.5000 BIC delta. Admitted."
            ),
            success=success,
            episode_score=episode_score,
        ),
    ]

    # Inject phase_records and control flags via record_meta on the episode
    # representation.  ExperimentReasoningRows receives an EpisodeTrainingRows
    # but also needs episode-level context.  We attach it as extra attributes
    # in a way that the transform can discover (via episode_context dict or a
    # dedicated field).  We use a separate attribute "episode_context" on the
    # EpisodeTrainingRows for now; the implementation must define the contract.
    group = EpisodeTrainingRows(
        episode_id=episode_id,
        episode_index=0,
        generation_id=1,
        episode_score=episode_score,
        rows=rows,
    )
    # Attach episode-level context that the transform needs.
    # The implementation will read these from the episode object.
    group.episode_context = {  # type: ignore[attr-defined]
        "success": success,
        "duplicate_rejected": duplicate_rejected,
        "phase_records": phase_records,
        "workflow_kind": "crossbreed" if has_parents else "survey",
    }
    if crossbreed_context:
        group.episode_context["crossbreed_context"] = crossbreed_context  # type: ignore[attr-defined]
    return group


def _default_phase_records(
    *,
    admitted: bool = True,
    bic_delta: float = -1.5,
) -> dict[str, Any]:
    return {
        "hypothesise": {
            "hypothesis": "acoustic impedance contrast predicts reservoir quality",
            "data_spec": {"files": ["seismic.csv"], "transform": "log_ratio"},
            "parent_experiments": ["ep-parent-0"],
        },
        "code": {"result_summary": "feature computed, 1200 rows"},
        "translate": {"feature_layer_name": "ai_contrast_log_ratio"},
        "evaluate": {
            "bic_delta": bic_delta,
            "admitted": admitted,
            "mutual_info": {"target": 0.12},
            "masking_test_passed": True,
            "masking_test_improvement": 0.31,
            "masking_test_direction": "improvement",
            "stage_completed": "stage_2_completed",
        },
    }


def _run_transform(episodes: list[EpisodeTrainingRows]) -> list[EpisodeTrainingRows]:
    transform = ExperimentReasoningRows()
    return transform.transform_export_rows(context=None, episodes=episodes)


def test_source_episode_payload_loader_tail_reads_only_appended_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tasks.feature_hypothesis_kazakhstan as kz

    path = tmp_path / "all_episodes.jsonl"
    path.write_text(
        json.dumps({"episode_id": "ep-0", "value": 0}) + "\n",
        encoding="utf-8",
    )
    transform = ExperimentReasoningRows()
    context = SimpleNamespace(source_all_episodes_path=path)

    first = transform._load_source_episode_payloads(context)
    assert set(first) == {"ep-0"}

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"episode_id": "ep-1", "value": 1}) + "\n")

    parsed_ids: list[str] = []
    real_loads = kz.json.loads

    def tracking_loads(text: str) -> Any:
        payload = real_loads(text)
        parsed_ids.append(payload["episode_id"])
        return payload

    monkeypatch.setattr(kz.json, "loads", tracking_loads)

    second = transform._load_source_episode_payloads(context)

    assert set(second) == {"ep-0", "ep-1"}
    assert parsed_ids == ["ep-1"]


# ---------------------------------------------------------------------------
# Masking / leakage tests
# ---------------------------------------------------------------------------


class TestMaskingLeakage:
    def test_no_query_contains_bic_or_admitted(self) -> None:
        """No synthesized row's prompt field may contain a BIC delta, the word
        'admitted', or 'not admitted' (case-insensitive)."""
        episode = _episode()
        result = _run_transform([episode])

        for group in result:
            for row in group.rows:
                prompt: str = row["prompt"]
                assert _BIC_PATTERN.search(prompt) is None, (
                    f"row {row['row_id']} prompt contains a numeric BIC-like value: {prompt!r}"
                )
                lower = prompt.lower()
                assert "admitted" not in lower, (
                    f"row {row['row_id']} prompt contains 'admitted': {prompt!r}"
                )

    def test_outcome_narrative_strips_exact_bic_result_appendix(self) -> None:
        """Outcome narrative completions must NOT contain the
        'Result: <bic> BIC delta.' appendix that _exec_submit_rewrite injects."""
        episode = _episode()
        result = _run_transform([episode])

        narrative_rows = [
            row
            for group in result
            for row in group.rows
            if row.get("record_meta", {}).get("pair_kind")
            == PAIR_KIND_OUTCOME_NARRATIVE
        ]
        assert narrative_rows, "expected at least one outcome narrative row"
        for row in narrative_rows:
            raw_response: str = row["raw_response"]
            assert _RESULT_APPENDIX.search(raw_response) is None, (
                f"row {row['row_id']} still contains BIC result appendix: {raw_response!r}"
            )

    def test_compose_child_not_in_parent_query(self) -> None:
        """Parent-hypothesis prompts must not contain the child hypothesis.

        They represent parent context and should not leak the target answer.
        """
        episode = _episode()
        result = _run_transform([episode])

        parent_rows = [
            row
            for group in result
            for row in group.rows
            if row.get("record_meta", {}).get("pair_kind")
            == PAIR_KIND_PARENT_HYPOTHESIS
        ]
        assert parent_rows, "expected at least one parent-hypothesis row"
        child_hypothesis = (
            "acoustic impedance contrast predicts reservoir quality"
        )
        for row in parent_rows:
            assert child_hypothesis not in row["prompt"], (
                f"parent-hypothesis prompt contains the child hypothesis verbatim; "
                f"it should only contain parent context. row_id={row['row_id']!r}"
            )


# ---------------------------------------------------------------------------
# Scope tests
# ---------------------------------------------------------------------------


class TestScope:
    def test_only_training_successful_episodes_yield_rows(self) -> None:
        """Episodes where success=False must produce zero output rows."""
        failed_episode = _episode(episode_id="ep-fail", success=False, episode_score=0.0)
        result = _run_transform([failed_episode])

        total_rows = sum(len(g.rows) for g in result)
        assert total_rows == 0, (
            f"Expected 0 rows for failed episode, got {total_rows}"
        )

    def test_duplicate_kg_rejection_still_yields_rows(self) -> None:
        """An episode that was rejected by the KG duplicate check but is
        training-successful (masking+BIC both passed) must still produce rows."""
        dup_episode = _episode(
            episode_id="ep-dup",
            success=True,
            duplicate_rejected=True,
            phase_records=_default_phase_records(admitted=True, bic_delta=-1.2),
        )
        result = _run_transform([dup_episode])

        total_rows = sum(len(g.rows) for g in result)
        assert total_rows > 0, (
            "KG-duplicate-rejected but training-successful episode should still "
            f"yield rows, got {total_rows}"
        )


# ---------------------------------------------------------------------------
# Provenance / metadata tests
# ---------------------------------------------------------------------------


class TestProvenanceMetadata:
    def test_collapsed_explore_step_backfills_hypothesise_material(self) -> None:
        """Kazakhstan now has one explore row, not separate survey/hypothesise rows."""
        episode = _episode(episode_id="ep-explore-only")
        assert {row["workflow_step"] for row in episode.rows} >= {
            "explore",
            "code",
            "translate",
            "rewrite",
        }
        assert "survey" not in {row["workflow_step"] for row in episode.rows}
        assert "hypothesise" not in {row["workflow_step"] for row in episode.rows}

        result = _run_transform([episode])

        pair_kinds = {row["record_meta"]["pair_kind"] for row in result[0].rows}
        assert PAIR_KIND_PARENT_HYPOTHESIS in pair_kinds
        assert PAIR_KIND_ANALYSIS_PLAN in pair_kinds
        assert PAIR_KIND_OUTCOME_NARRATIVE in pair_kinds
        for row in result[0].rows:
            if row["record_meta"]["pair_kind"] in {
                PAIR_KIND_PARENT_HYPOTHESIS,
                PAIR_KIND_ANALYSIS_PLAN,
                PAIR_KIND_DATASET_HYPOTHESIS,
            }:
                assert row["parent_row_id"] == "ep-explore-only:0"
            if row["record_meta"]["pair_kind"] == PAIR_KIND_DATASET_HYPOTHESIS:
                assert "Experiment 1" not in row["prompt"]
                assert "porosity correlates with depth" not in row["prompt"]

    def test_outcome_narrative_tagged_post_hoc(self) -> None:
        """Outcome narratives are marked as reconstructed after evaluation."""
        episode = _episode()
        result = _run_transform([episode])

        narrative_rows = [
            row
            for group in result
            for row in group.rows
            if row.get("record_meta", {}).get("pair_kind")
            == PAIR_KIND_OUTCOME_NARRATIVE
        ]
        assert narrative_rows, "expected at least one outcome narrative row"
        for row in narrative_rows:
            assert row["record_meta"].get("faithfulness") == "post_hoc", (
                f"outcome narrative row {row['row_id']} missing faithfulness=post_hoc: "
                f"{row['record_meta']!r}"
            )

    def test_synthesized_rows_carry_pair_kind(self) -> None:
        """Every synthesized row must have a known descriptive pair_kind."""
        episode = _episode()
        result = _run_transform([episode])

        valid_kinds = {
            PAIR_KIND_PARENT_HYPOTHESIS,
            PAIR_KIND_DATASET_HYPOTHESIS,
            PAIR_KIND_ANALYSIS_PLAN,
            PAIR_KIND_OUTCOME_NARRATIVE,
        }
        for group in result:
            for row in group.rows:
                kind = row.get("record_meta", {}).get("pair_kind")
                assert kind in valid_kinds, (
                    f"row {row['row_id']} has invalid pair_kind={kind!r}; "
                    f"expected one of {valid_kinds}"
                )

    def test_missing_parent_skips_parent_hypothesis_row(self) -> None:
        """If the episode has no recoverable parent experiment context, the
        transform must skip the parent-hypothesis row."""
        no_parent_phase = _default_phase_records()
        no_parent_phase["hypothesise"]["parent_experiments"] = []  # empty parent list

        episode = _episode(
            episode_id="ep-no-parent",
            phase_records=no_parent_phase,
        )
        # Must not raise; parent-hypothesis rows are simply absent.
        result = _run_transform([episode])

        parent_rows = [
            row
            for group in result
            for row in group.rows
            if row.get("record_meta", {}).get("pair_kind")
            == PAIR_KIND_PARENT_HYPOTHESIS
        ]
        assert len(parent_rows) == 0, (
            "Expected no parent-hypothesis rows when parent context is missing, "
            f"got {len(parent_rows)}"
        )

    def test_historical_tool_output_prompts_backfill_phase_data(self) -> None:
        """Older exports store phase data in embedded tool-output JSON."""
        hypothesis = "silver overprint follows cobalt-rich reduced facies"
        data_spec = {
            "analysis_steps": ["filter high Co", "filter high Ag", "intersect prospects"],
            "required_files": ["/workspace/input/USGS/TZ_ssCu_Prospects.csv"],
            "target_feature": "co_ag_overlap",
        }
        phase_output = {
            "output": {
                "hypothesis": hypothesis,
                "data_spec": data_spec,
                "parent_experiments": ["parent-a", "parent-b"],
            },
            "success": True,
        }
        result_output = {
            "output": {
                "hypothesis": hypothesis,
                "data_spec": data_spec,
                "code_executed": "print('analysis')",
                "result_summary": "identified overlapping Co and Ag prospects",
                "feature_layer_name": "co_ag_overlap",
                "bic_delta": -1.5,
                "admitted": True,
                "mutual_info": {},
            },
            "success": True,
        }
        group = EpisodeTrainingRows(
            episode_id="ep-historical",
            episode_index=0,
            generation_id=1,
            episode_score=1.0,
            rows=[
                _base_row(
                    episode_id="ep-historical",
                    row_index=0,
                    workflow_step="hypothesise",
                    prompt=(
                        'Experiment 1: "cobalt tracks reduced facies"\n'
                        'Experiment 2: "silver follows late fluid pathways"\n'
                        "Given these parent findings, propose a combined hypothesis."
                    ),
                    raw_response="",
                ),
                _base_row(
                    episode_id="ep-historical",
                    row_index=1,
                    workflow_step="code",
                    prompt=f"[tool]\n{json.dumps(phase_output)}",
                    raw_response="",
                ),
                _base_row(
                    episode_id="ep-historical",
                    row_index=2,
                    workflow_step="translate",
                    prompt=f"[tool]\n{json.dumps(result_output)}",
                    raw_response="",
                ),
                _base_row(
                    episode_id="ep-historical",
                    row_index=3,
                    workflow_step="rewrite",
                    actor_role="rewriter_output",
                    prompt="[user]\nDataset context and tested hypothesis.",
                    raw_response=(
                        "Narrative: the overlap feature captures a plausible "
                        "multi-stage mineralization signal.\n"
                        "Result: -1.5000 BIC delta. Admitted."
                    ),
                ),
            ],
        )

        result = _run_transform([group])

        pair_kinds = {row["record_meta"]["pair_kind"] for row in result[0].rows}
        assert PAIR_KIND_PARENT_HYPOTHESIS in pair_kinds
        assert PAIR_KIND_ANALYSIS_PLAN in pair_kinds
        assert PAIR_KIND_OUTCOME_NARRATIVE in pair_kinds


# ---------------------------------------------------------------------------
# Yield tests
# ---------------------------------------------------------------------------


class TestYield:
    def test_rows_per_episode_in_range(self) -> None:
        """Default config must yield 3–4 rows per training-successful episode."""
        # Parent context plus standard phase records should enable the full set
        # of synthesized row kinds.
        episode = _episode(episode_id="ep-yield")
        result = _run_transform([episode])

        assert len(result) == 1
        count = len(result[0].rows)
        assert 3 <= count <= 4, (
            f"Expected 3-4 rows per episode, got {count}"
        )


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


class TestPipelineContract:
    def test_output_passes_validate_training_row_groups(self) -> None:
        """The output of transform_export_rows must pass the existing
        validate_training_row_groups without raising."""
        episode = _episode(episode_id="ep-contract")
        result = _run_transform([episode])

        # Should not raise
        validate_training_row_groups(
            result,
            source_row_ids={row["row_id"] for row in episode.rows},
        )


# ---------------------------------------------------------------------------
# Curation tests
# ---------------------------------------------------------------------------


class TestCuration:
    def test_duplicate_hypotheses_with_distinct_reasoning_survive_until_family_cap(self) -> None:
        """Duplicate hypotheses are still useful if the reasoning path differs.

        The curation layer should collapse exact prompt/completion duplicates,
        not every episode that reaches the same hypothesis string.
        """
        ep_a = _episode(episode_id="ep-dup-a")
        ep_b = _episode(episode_id="ep-dup-b")
        ep_b.rows[3]["raw_response"] = (
            "Narrative: a different analysis path reaches the same hypothesis "
            "through well-log contrasts rather than seismic boundaries."
        )

        result = ExperimentReasoningRows(max_per_family=5).transform_export_rows(
            context=None,
            episodes=[ep_a, ep_b],
        )

        non_empty_episode_ids = {group.episode_id for group in result if group.rows}
        assert non_empty_episode_ids == {"ep-dup-a", "ep-dup-b"}

    def test_exact_duplicate_full_pairs_collapse_keeps_strongest_signal(self) -> None:
        weak_phase = _default_phase_records(bic_delta=-0.2)
        strong_phase = _default_phase_records(bic_delta=-2.0)
        weak = _episode(episode_id="ep-weak", phase_records=weak_phase)
        strong = _episode(episode_id="ep-strong", phase_records=strong_phase)

        result = ExperimentReasoningRows(max_per_family=5).transform_export_rows(
            context=None,
            episodes=[weak, strong],
        )

        non_empty_episode_ids = [group.episode_id for group in result if group.rows]
        assert non_empty_episode_ids == ["ep-strong"]

    def test_family_balance_caps_dominant_cluster_but_keeps_t2_seed(self) -> None:
        ep_a = _episode(episode_id="ep-family-a")
        ep_b = _episode(episode_id="ep-family-b")
        ep_a.episode_context["phase_records"]["hypothesise"]["hypothesis"] = (  # type: ignore[attr-defined]
            "acoustic impedance contrast predicts reservoir quality at basin center"
        )
        ep_b.episode_context["phase_records"]["hypothesise"]["hypothesis"] = (  # type: ignore[attr-defined]
            "acoustic impedance contrast predicts reservoir quality at basin margins"
        )

        result = ExperimentReasoningRows(max_per_family=1).transform_export_rows(
            context=None,
            episodes=[ep_a, ep_b],
        )

        capped = [group for group in result if group.episode_id == "ep-family-b"][0]
        assert capped.rows, "dataset-hypothesis seed should survive the family cap"
        assert {row["record_meta"]["pair_kind"] for row in capped.rows} == {
            PAIR_KIND_DATASET_HYPOTHESIS
        }
