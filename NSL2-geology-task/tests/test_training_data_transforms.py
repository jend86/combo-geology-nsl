"""TDD tests for the training data export transform pipeline.

All imports from new/changed modules will fail until implementation is done.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from src.training_data.transforms import (
    EpisodeTrainingRows,
    TrainingDataExportContext,
    build_training_export,
    count_training_rows,
    validate_training_row_groups,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData

REQUIRED_ROW_FIELDS = {
    "row_id",
    "parent_row_id",
    "prompt",
    "raw_response",
    "interaction_type",
    "source_interaction_type",
    "timestamp",
    "success",
    "error_message",
    "episode_id",
    "episode_index",
    "generation_id",
    "episode_score",
    "episode_score_scope",
    "source_episode_id",
    "source_row_index",
    "workflow_step",
    "actor_role",
    "record_meta",
}


# ---------------------------------------------------------------------------
# Row / group helpers
# ---------------------------------------------------------------------------


def _make_row(
    *,
    episode_id: str = "ep-0",
    episode_index: int = 0,
    generation_id: int = 1,
    row_index: int = 0,
    workflow_step: str | None = "explore",
    actor_role: str | None = None,
) -> dict[str, Any]:
    row_id = f"{episode_id}:{row_index}"
    return {
        "row_id": row_id,
        "parent_row_id": None,
        "prompt": f"prompt-{row_index}",
        "raw_response": f"response-{row_index}",
        "interaction_type": "orchestrator",
        "source_interaction_type": "orchestrator",
        "timestamp": "2026-05-16T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "generation_id": generation_id,
        "episode_score": 1.0,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": actor_role,
        "record_meta": {},
    }


def _make_episode_rows(
    episode_id: str = "ep-0",
    *,
    num_rows: int = 2,
    episode_index: int = 0,
    generation_id: int = 1,
    episode_score: float = 1.0,
    workflow_step: str | None = "explore",
) -> EpisodeTrainingRows:
    rows = [
        _make_row(
            episode_id=episode_id,
            episode_index=episode_index,
            generation_id=generation_id,
            row_index=i,
            workflow_step=workflow_step,
        )
        for i in range(num_rows)
    ]
    return EpisodeTrainingRows(
        episode_id=episode_id,
        episode_index=episode_index,
        generation_id=generation_id,
        episode_score=episode_score,
        rows=rows,
    )


def _make_context(generation_id: int = 1) -> TrainingDataExportContext:
    return TrainingDataExportContext(
        generation_id=generation_id,
        run_id="run-test",
        task_name="test_task",
        source_generation_dir=Path("/tmp/gen"),
        source_all_episodes_path=Path("/tmp/gen/all_episodes.jsonl"),
        export_recipe_hash="abc123",
    )


def _make_trajectory(
    episode_id: str,
    *,
    episode_index: int = 0,
    generation_id: int = 1,
    score: float = 1.0,
    success: bool = True,
    raw_training_rows: list[dict[str, Any]] | None = None,
) -> EpisodeTrajectory:
    if raw_training_rows is None:
        raw_training_rows = [
            _make_row(episode_id=episode_id, episode_index=episode_index, row_index=i)
            for i in range(2)
        ]
    return EpisodeTrajectory(
        episode_id=episode_id,
        generation_id=generation_id,
        episode_index=episode_index,
        prompt_responses=[],
        trajectory={},
        score=score,
        episode_runtime_success=success,
        success=success,
        llm_turns_count=2,
        container_variation="v1",
        started_at="2026-05-16T00:00:00",
        completed_at="2026-05-16T00:01:00",
        duration_seconds=60.0,
        raw_training_rows=raw_training_rows,
    )


# ---------------------------------------------------------------------------
# Stub transforms used across tests
# ---------------------------------------------------------------------------


class _IdentityTransform:
    @property
    def name(self) -> str:
        return "Identity[v1]"

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        return episodes


class _DropAllTransform:
    @property
    def name(self) -> str:
        return "DropAll[v1]"

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        return [
            EpisodeTrainingRows(
                episode_id=ep.episode_id,
                episode_index=ep.episode_index,
                generation_id=ep.generation_id,
                episode_score=ep.episode_score,
                rows=[],
            )
            for ep in episodes
        ]


class _KeepByWorkflowStep:
    def __init__(self, allowed: set[str]) -> None:
        self._allowed = allowed

    @property
    def name(self) -> str:
        return "KeepByStep[v1]"

    def config(self) -> Mapping[str, Any]:
        return {"allowed": sorted(self._allowed)}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        return [
            EpisodeTrainingRows(
                episode_id=ep.episode_id,
                episode_index=ep.episode_index,
                generation_id=ep.generation_id,
                episode_score=ep.episode_score,
                rows=[r for r in ep.rows if r.get("workflow_step") in self._allowed],
            )
            for ep in episodes
        ]


class _RaisingTransform:
    @property
    def name(self) -> str:
        return "Raiser[v1]"

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        raise ValueError("intentional failure from Raiser transform")


class _TaggingTransform:
    """Appends a tag suffix to raw_response so chaining order is observable."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    @property
    def name(self) -> str:
        return f"Tagger[{self._tag}]"

    def config(self) -> Mapping[str, Any]:
        return {"tag": self._tag}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        result = []
        for ep in episodes:
            rows = [
                {**r, "raw_response": r["raw_response"] + f"|{self._tag}"}
                for r in ep.rows
            ]
            result.append(
                EpisodeTrainingRows(
                    episode_id=ep.episode_id,
                    episode_index=ep.episode_index,
                    generation_id=ep.generation_id,
                    episode_score=ep.episode_score,
                    rows=rows,
                )
            )
        return result


# ---------------------------------------------------------------------------
# EpisodeTrainingRows
# ---------------------------------------------------------------------------


class TestEpisodeTrainingRows(unittest.TestCase):
    def test_basic_construction(self) -> None:
        rows = [_make_row(row_index=0), _make_row(row_index=1)]
        group = EpisodeTrainingRows(
            episode_id="ep-0",
            episode_index=0,
            generation_id=1,
            episode_score=0.75,
            rows=rows,
        )
        self.assertEqual(group.episode_id, "ep-0")
        self.assertEqual(group.episode_index, 0)
        self.assertEqual(group.generation_id, 1)
        self.assertEqual(group.episode_score, 0.75)
        self.assertEqual(len(group.rows), 2)

    def test_empty_rows_is_valid(self) -> None:
        group = EpisodeTrainingRows(
            episode_id="ep-empty",
            episode_index=0,
            generation_id=1,
            episode_score=None,
            rows=[],
        )
        self.assertEqual(group.rows, [])
        self.assertIsNone(group.episode_score)

    def test_nullable_episode_index(self) -> None:
        group = EpisodeTrainingRows(
            episode_id="ep-0",
            episode_index=None,
            generation_id=1,
            episode_score=0.5,
            rows=[],
        )
        self.assertIsNone(group.episode_index)


# ---------------------------------------------------------------------------
# validate_training_row_groups
# ---------------------------------------------------------------------------


class TestValidateTrainingRowGroups(unittest.TestCase):
    def test_valid_groups_pass(self) -> None:
        groups = [
            _make_episode_rows("ep-0"),
            _make_episode_rows("ep-1", episode_index=1),
        ]
        validate_training_row_groups(groups)  # must not raise

    def test_empty_groups_list_passes(self) -> None:
        validate_training_row_groups([])

    def test_empty_rows_in_group_passes(self) -> None:
        group = EpisodeTrainingRows(
            episode_id="ep-0",
            episode_index=0,
            generation_id=1,
            episode_score=1.0,
            rows=[],
        )
        validate_training_row_groups([group])

    def test_missing_required_field_fails(self) -> None:
        row = _make_row()
        del row["workflow_step"]
        group = EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row])
        with self.assertRaises(ValueError):
            validate_training_row_groups([group])

    def test_missing_row_id_fails(self) -> None:
        row = _make_row()
        del row["row_id"]
        group = EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row])
        with self.assertRaises(ValueError):
            validate_training_row_groups([group])

    def test_duplicate_row_id_across_groups_fails(self) -> None:
        row_a = _make_row(episode_id="ep-0", row_index=0)
        row_b = _make_row(episode_id="ep-0", row_index=0)  # same row_id
        group_a = EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row_a])
        group_b = EpisodeTrainingRows("ep-1", 1, 1, 1.0, rows=[row_b])
        with self.assertRaises(ValueError):
            validate_training_row_groups([group_a, group_b])

    def test_invalid_episode_score_scope_fails(self) -> None:
        row = _make_row()
        row["episode_score_scope"] = "per_row"
        group = EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row])
        with self.assertRaises(ValueError):
            validate_training_row_groups([group])

    def test_non_json_serializable_field_fails(self) -> None:
        row = _make_row()
        row["record_meta"] = {"obj": object()}
        group = EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row])
        with self.assertRaises((ValueError, TypeError)):
            validate_training_row_groups([group])

    def test_duplicate_row_id_within_single_group_fails(self) -> None:
        row_a = _make_row(row_index=0)
        row_b = _make_row(row_index=0)  # same row_id as row_a
        group = EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row_a, row_b])
        with self.assertRaises(ValueError):
            validate_training_row_groups([group])


# ---------------------------------------------------------------------------
# build_training_export
# ---------------------------------------------------------------------------


def _make_generation_data(
    *, num_episodes: int = 2, rows_per_episode: int = 2
) -> GenerationData:
    gd = GenerationData(generation_id=1)
    for i in range(num_episodes):
        raw_rows = [
            _make_row(episode_id=f"ep-{i}", episode_index=i, row_index=j)
            for j in range(rows_per_episode)
        ]
        gd.add_episode(
            _make_trajectory(f"ep-{i}", episode_index=i, raw_training_rows=raw_rows)
        )
    return gd


class TestBuildSftTrainingExport(unittest.TestCase):
    def test_no_transforms_produces_all_raw_rows(self) -> None:
        gd = _make_generation_data(num_episodes=2, rows_per_episode=3)
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(result.report["training_row_count"], 6)
        self.assertEqual(result.report["raw_successful_row_count"], 6)
        self.assertEqual(result.report["rows_removed"], 0)
        self.assertEqual(result.report["rows_added"], 0)

    def test_drop_all_transform_empties_output(self) -> None:
        gd = _make_generation_data(num_episodes=2, rows_per_episode=3)
        ctx = _make_context()

        result = build_training_export(
            gd, transforms=[_DropAllTransform()], context=ctx
        )

        self.assertEqual(result.report["training_row_count"], 0)
        self.assertEqual(result.report["raw_successful_row_count"], 6)
        self.assertEqual(result.report["rows_removed"], 6)
        self.assertEqual(result.report["rows_added"], 0)

    def test_transform_order_is_load_bearing(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=1)
        ctx = _make_context()

        result_ab = build_training_export(
            gd,
            transforms=[_TaggingTransform("A"), _TaggingTransform("B")],
            context=ctx,
        )
        result_ba = build_training_export(
            gd,
            transforms=[_TaggingTransform("B"), _TaggingTransform("A")],
            context=ctx,
        )

        self.assertEqual(len(result_ab.rows), 1)
        self.assertIn("|A|B", result_ab.rows[0]["raw_response"])
        self.assertIn("|B|A", result_ba.rows[0]["raw_response"])

    def test_raising_transform_fails_strictly(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=1)
        ctx = _make_context()

        with self.assertRaises(Exception):
            build_training_export(
                gd, transforms=[_RaisingTransform()], context=ctx
            )

    def test_episode_score_preserved_after_row_filtering(self) -> None:
        raw_rows = [
            _make_row(episode_id="ep-0", row_index=0, workflow_step="explore"),
            _make_row(episode_id="ep-0", row_index=1, workflow_step="execute"),
        ]
        gd = GenerationData(generation_id=1)
        gd.add_episode(
            _make_trajectory("ep-0", score=0.9, raw_training_rows=raw_rows)
        )
        ctx = _make_context()

        result = build_training_export(
            gd,
            transforms=[_KeepByWorkflowStep({"explore"})],
            context=ctx,
        )

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["episode_score"], 0.9)
        self.assertEqual(result.rows[0]["episode_score_scope"], "whole_episode")
        self.assertEqual(result.rows[0]["workflow_step"], "explore")

    def test_per_transform_deltas_in_report(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=4)
        ctx = _make_context()

        result = build_training_export(
            gd,
            transforms=[
                _IdentityTransform(),   # keeps 4
                _DropAllTransform(),    # drops 4
            ],
            context=ctx,
        )

        per_transform = result.report["per_transform"]
        self.assertEqual(len(per_transform), 2)
        self.assertEqual(per_transform[0]["rows_removed"], 0)
        self.assertEqual(per_transform[1]["rows_removed"], 4)

    def test_failed_episodes_excluded_from_export(self) -> None:
        gd = GenerationData(generation_id=1)
        raw_rows = [_make_row(episode_id="ep-fail")]
        gd.add_episode(
            _make_trajectory(
                "ep-fail", score=0.0, success=False, raw_training_rows=raw_rows
            )
        )
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(result.report["training_row_count"], 0)
        self.assertTrue(result.report["successful_episodes_only"])

    def test_report_includes_required_fields(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=2)
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        required_report_keys = {
            "training_row_count",
            "raw_successful_row_count",
            "rows_removed",
            "rows_added",
            "per_transform",
            "successful_episodes_only",
            "target_count_basis",
        }
        for key in required_report_keys:
            self.assertIn(key, result.report, f"report missing key: {key}")

    def test_target_count_basis_is_exported_sft_rows(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=2)
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(result.report["target_count_basis"], "training_rows")

    def test_empty_generation_data_produces_zero_rows(self) -> None:
        gd = GenerationData(generation_id=1)
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(result.rows, [])
        self.assertEqual(result.report["training_row_count"], 0)

    def test_rows_are_ordered_by_episode_then_source_row_index(self) -> None:
        raw_rows_ep0 = [
            _make_row(episode_id="ep-0", episode_index=0, row_index=0),
            _make_row(episode_id="ep-0", episode_index=0, row_index=1),
        ]
        raw_rows_ep1 = [
            _make_row(episode_id="ep-1", episode_index=1, row_index=0),
        ]
        gd = GenerationData(generation_id=1)
        gd.add_episode(
            _make_trajectory("ep-0", episode_index=0, raw_training_rows=raw_rows_ep0)
        )
        gd.add_episode(
            _make_trajectory("ep-1", episode_index=1, raw_training_rows=raw_rows_ep1)
        )
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        episode_ids = [r["episode_id"] for r in result.rows]
        self.assertEqual(episode_ids, ["ep-0", "ep-0", "ep-1"])


# ---------------------------------------------------------------------------
# count_training_rows
# ---------------------------------------------------------------------------


class TestCountExportedSftRows(unittest.TestCase):
    def _make_gd_with_steps(self, workflow_steps: list[str]) -> GenerationData:
        gd = GenerationData(generation_id=1)
        raw_rows = [
            _make_row(episode_id="ep-0", row_index=i, workflow_step=step)
            for i, step in enumerate(workflow_steps)
        ]
        gd.add_episode(_make_trajectory("ep-0", raw_training_rows=raw_rows))
        return gd

    def test_no_transforms_equals_raw_count(self) -> None:
        gd = self._make_gd_with_steps(["explore", "hypothesise", "execute"])
        ctx = _make_context()

        self.assertEqual(count_training_rows(gd, transforms=[], context=ctx), 3)

    def test_filter_transform_reduces_count(self) -> None:
        gd = self._make_gd_with_steps(
            ["explore", "hypothesise", "execute", "explore"]
        )
        ctx = _make_context()

        count = count_training_rows(
            gd,
            transforms=[_KeepByWorkflowStep({"explore", "hypothesise"})],
            context=ctx,
        )
        self.assertEqual(count, 3)

    def test_drop_all_returns_zero(self) -> None:
        gd = self._make_gd_with_steps(["explore", "execute"])
        ctx = _make_context()

        count = count_training_rows(
            gd, transforms=[_DropAllTransform()], context=ctx
        )
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# record_to_row enrichment
# ---------------------------------------------------------------------------


class TestRecordToRowEnrichment(unittest.TestCase):
    """Fails until training_row_adapter.py is updated."""

    def _make_record(
        self,
        *,
        episode_id: str = "ep-0",
        phase: str = "orchestrator",
        workflow_step: str | None = "explore",
        actor_role: str | None = None,
    ):
        from src.harness.recorder import TrajectoryRecord

        return TrajectoryRecord(
            episode_id=episode_id,
            phase=phase,
            messages=[{"role": "user", "content": "hello"}],
            response="response text",
            usage=None,
            timestamp="2026-05-16T00:00:00",
            success=True,
            error_message=None,
            meta={"workflow_step": workflow_step, "actor_role": actor_role},
        )

    def test_includes_workflow_step(self) -> None:
        from src.harness.training_row_adapter import record_to_row

        record = self._make_record(workflow_step="hypothesise")
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=0)

        self.assertEqual(row["workflow_step"], "hypothesise")

    def test_workflow_step_nullable(self) -> None:
        from src.harness.training_row_adapter import record_to_row

        record = self._make_record(workflow_step=None)
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=0)

        self.assertIn("workflow_step", row)
        self.assertIsNone(row["workflow_step"])

    def test_includes_actor_role(self) -> None:
        from src.harness.training_row_adapter import record_to_row

        record = self._make_record(actor_role="proposer")
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=0)

        self.assertEqual(row["actor_role"], "proposer")

    def test_includes_source_fields(self) -> None:
        from src.harness.training_row_adapter import record_to_row

        record = self._make_record(episode_id="ep-42", phase="mcp_explore_call")
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=3)

        self.assertEqual(row["source_episode_id"], "ep-42")
        self.assertEqual(row["source_row_index"], 3)
        self.assertEqual(row["source_interaction_type"], "mcp_explore_call")
        self.assertIsNone(row["parent_row_id"])

    def test_generates_stable_row_id(self) -> None:
        from src.harness.training_row_adapter import record_to_row

        record = self._make_record(episode_id="ep-5")
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=2)

        self.assertEqual(row["row_id"], "ep-5:2")

    def test_includes_record_meta(self) -> None:
        from src.harness.training_row_adapter import record_to_row

        record = self._make_record()
        record.meta["extra_key"] = "extra_val"
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=0)

        self.assertIn("record_meta", row)
        self.assertIsInstance(row["record_meta"], dict)

    def test_records_to_rows_enumerates_source_row_index(self) -> None:
        from src.harness.training_row_adapter import records_to_rows

        records = [self._make_record() for _ in range(3)]
        rows = records_to_rows(records, run_id="run-1", version="v1")

        self.assertEqual(rows[0]["source_row_index"], 0)
        self.assertEqual(rows[1]["source_row_index"], 1)
        self.assertEqual(rows[2]["source_row_index"], 2)
        self.assertEqual(rows[0]["row_id"], "ep-0:0")

    def test_all_required_fields_present_after_enrichment(self) -> None:
        from src.harness.training_row_adapter import (
            enrich_training_rows_for_episode,
            record_to_row,
        )

        record = self._make_record()
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=0)
        enriched = enrich_training_rows_for_episode(
            [row],
            episode_id="ep-0",
            episode_index=0,
            generation_id=1,
            episode_score=0.9,
        )

        missing = REQUIRED_ROW_FIELDS - set(enriched[0])
        self.assertEqual(missing, set(), f"enriched row missing fields: {missing}")


# ---------------------------------------------------------------------------
# TaskSpec.training_data_transforms default
# ---------------------------------------------------------------------------


class TestTaskSpecTransformsDefault(unittest.TestCase):
    def _make_minimal_task(self):
        from src.task.base import TaskSpec

        class _MinimalTask(TaskSpec):
            name = "test"
            description = "test"
            metric_name = "score"
            metric_unit = "pts"
            higher_is_better = True
            docker_compose_dir = "."
            agent_service_name = "agent"

            def list_variations(self):
                return []

            def populate(self, containers, variation):
                pass

            def prompt_spec(self, variation, episode_context):
                pass

            def measure_initial_state(self, containers, episode_context, *, private_context=None):
                pass

            def compute_reward(self, initial, final, artifacts):
                pass

        return _MinimalTask.__new__(_MinimalTask)

    def test_default_returns_empty_sequence(self) -> None:
        task = self._make_minimal_task()
        result = task.training_data_transforms()
        self.assertEqual(tuple(result), ())

    def test_method_exists_on_class(self) -> None:
        from src.task.base import TaskSpec

        self.assertTrue(
            hasattr(TaskSpec, "training_data_transforms"),
            "TaskSpec must define training_data_transforms()",
        )


# ---------------------------------------------------------------------------
# GeologyProposerRows
# ---------------------------------------------------------------------------


class TestGeologyProposerRows(unittest.TestCase):
    """Fails until tasks/geology_graph.py is updated."""

    def _make_geology_episode(
        self,
        workflow_steps: list[str | None],
        episode_id: str = "ep-0",
    ) -> EpisodeTrainingRows:
        rows = [
            _make_row(episode_id=episode_id, row_index=i, workflow_step=step)
            for i, step in enumerate(workflow_steps)
        ]
        return EpisodeTrainingRows(
            episode_id=episode_id,
            episode_index=0,
            generation_id=1,
            episode_score=1.0,
            rows=rows,
        )

    def test_keeps_proposer_steps(self) -> None:
        from tasks.geology_graph import GeologyProposerRows

        transform = GeologyProposerRows()
        episodes = [
            self._make_geology_episode(
                ["explore", "hypothesise", "execute", "submit"]
            )
        ]
        ctx = _make_context()

        result = transform.transform_export_rows(ctx, episodes)

        kept_steps = [r["workflow_step"] for r in result[0].rows]
        self.assertIn("explore", kept_steps)
        self.assertIn("hypothesise", kept_steps)
        self.assertIn("submit", kept_steps)
        self.assertNotIn("execute", kept_steps)

    def test_raises_on_null_workflow_step(self) -> None:
        from tasks.geology_graph import GeologyProposerRows

        transform = GeologyProposerRows()
        row = _make_row()
        row["workflow_step"] = None
        episodes = [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[row])]
        ctx = _make_context()

        with self.assertRaises(ValueError):
            transform.transform_export_rows(ctx, episodes)

    def test_config_includes_allowed_steps(self) -> None:
        from tasks.geology_graph import GeologyProposerRows

        config = GeologyProposerRows().config()
        self.assertIn("included_workflow_steps", config)
        self.assertIn("explore", config["included_workflow_steps"])
        self.assertIn("hypothesise", config["included_workflow_steps"])
        self.assertNotIn("execute", config["included_workflow_steps"])

    def test_preserves_episode_score_after_filtering(self) -> None:
        from tasks.geology_graph import GeologyProposerRows

        transform = GeologyProposerRows()
        episodes = [
            EpisodeTrainingRows(
                episode_id="ep-0",
                episode_index=0,
                generation_id=1,
                episode_score=0.85,
                rows=[
                    _make_row(row_index=0, workflow_step="explore"),
                    _make_row(row_index=1, workflow_step="execute"),
                ],
            )
        ]
        ctx = _make_context()

        result = transform.transform_export_rows(ctx, episodes)

        self.assertEqual(result[0].episode_score, 0.85)
        self.assertEqual(len(result[0].rows), 1)

    def test_name_is_stable_versioned_string(self) -> None:
        from tasks.geology_graph import GeologyProposerRows

        self.assertEqual(GeologyProposerRows().name, "GeologyProposerRows[v1]")


# ---------------------------------------------------------------------------
# EventRecorder workflow_step stamping
# ---------------------------------------------------------------------------


class TestRecorderWorkflowStepStamping(unittest.TestCase):
    """Fails until recorder.py stamps last_workflow_step into record.meta."""

    def _make_recorder(self, episode_id: str = "ep-test") -> tuple[Any, str]:
        from src.harness.recorder import EventRecorder

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        return EventRecorder(episode_id, Path(tmp.name)), tmp.name

    def test_workflow_step_stamped_into_meta(self) -> None:
        from src.harness.recorder import TrajectoryRecord

        recorder, _ = self._make_recorder()
        recorder.set_label("last_workflow_step", "hypothesise")

        record = TrajectoryRecord(
            episode_id="ep-test",
            phase="orchestrator",
            messages=[],
            response="test",
            usage=None,
            timestamp="2026-05-16T00:00:00",
            success=True,
        )
        recorder.record_inference(record)

        self.assertEqual(
            recorder.inference_records[0].meta.get("workflow_step"), "hypothesise"
        )

    def test_no_label_leaves_workflow_step_none_or_absent(self) -> None:
        from src.harness.recorder import TrajectoryRecord

        recorder, _ = self._make_recorder()
        # No set_label call

        record = TrajectoryRecord(
            episode_id="ep-test",
            phase="orchestrator",
            messages=[],
            response="test",
            usage=None,
            timestamp="2026-05-16T00:00:00",
            success=True,
        )
        recorder.record_inference(record)

        meta = recorder.inference_records[0].meta
        self.assertIsNone(meta.get("workflow_step"))

    def test_label_changes_between_records(self) -> None:
        from src.harness.recorder import TrajectoryRecord

        recorder, _ = self._make_recorder()

        def _record(phase: str) -> TrajectoryRecord:
            return TrajectoryRecord(
                episode_id="ep-test",
                phase=phase,
                messages=[],
                response="r",
                usage=None,
                timestamp="2026-05-16T00:00:00",
                success=True,
            )

        recorder.set_label("last_workflow_step", "explore")
        recorder.record_inference(_record("p1"))
        recorder.set_label("last_workflow_step", "hypothesise")
        recorder.record_inference(_record("p2"))

        records = recorder.inference_records
        self.assertEqual(records[0].meta.get("workflow_step"), "explore")
        self.assertEqual(records[1].meta.get("workflow_step"), "hypothesise")


if __name__ == "__main__":
    unittest.main()
