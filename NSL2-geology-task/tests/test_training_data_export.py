from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.harness.recorder import TrajectoryRecord
from src.harness.training_row_adapter import (
    enrich_training_rows_for_episode,
    records_to_rows,
)
from src.training_data.transforms import (
    EpisodeTrainingRows,
    TrainingDataExportContext,
    build_export_recipe,
    build_training_export,
    regenerate_sft_export,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData


def _row(episode_id: str, row_index: int, *, workflow_step: str | None) -> dict[str, Any]:
    return {
        "row_id": f"{episode_id}:{row_index}",
        "parent_row_id": None,
        "prompt": f"prompt-{episode_id}-{row_index}",
        "raw_response": f"response-{episode_id}-{row_index}",
        "interaction_type": "phase",
        "source_interaction_type": "phase",
        "timestamp": "2026-05-16T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": int(episode_id.rsplit("-", 1)[1]),
        "generation_id": 3,
        "episode_score": 1.0,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": None,
        "record_meta": {"workflow_step": workflow_step},
    }


def _episode(
    episode_index: int,
    *,
    workflow_steps: list[str | None],
    success: bool = True,
) -> EpisodeTrajectory:
    episode_id = f"ep-{episode_index}"
    raw_rows = [
        _row(episode_id, row_index, workflow_step=workflow_step)
        for row_index, workflow_step in enumerate(workflow_steps)
    ]
    return EpisodeTrajectory(
        episode_id=episode_id,
        generation_id=3,
        episode_index=episode_index,
        prompt_responses=[
            {
                "prompt": row["prompt"],
                "raw_response": row["raw_response"],
                "interaction_type": row["interaction_type"],
                "timestamp": row["timestamp"],
                "success": row["success"],
                "error_message": row["error_message"],
            }
            for row in raw_rows
        ],
        trajectory={},
        score=1.0 if success else 0.0,
        episode_runtime_success=success,
        success=success,
        llm_turns_count=len(raw_rows),
        container_variation="v1",
        started_at="2026-05-16T00:00:00",
        completed_at="2026-05-16T00:00:01",
        duration_seconds=1.0,
        raw_training_rows=raw_rows,
    )


def _generation_data() -> GenerationData:
    generation_data = GenerationData(generation_id=3)
    generation_data.add_episode(_episode(0, workflow_steps=["keep", "drop"]))
    generation_data.add_episode(_episode(1, workflow_steps=["keep"]))
    generation_data.add_episode(_episode(2, workflow_steps=["keep"], success=False))
    return generation_data


def _context(
    tmp_path: Path,
    transforms: tuple[Any, ...],
) -> TrainingDataExportContext:
    recipe = build_export_recipe(transforms)
    return TrainingDataExportContext(
        generation_id=3,
        run_id="run-123",
        task_name="dummy-task",
        source_generation_dir=tmp_path,
        source_all_episodes_path=tmp_path / "all_episodes.jsonl",
        export_recipe_hash=recipe.recipe_hash,
    )


class KeepWorkflowStep:
    def __init__(self, workflow_step: str) -> None:
        self.workflow_step = workflow_step

    @property
    def name(self) -> str:
        return "KeepWorkflowStep[v1]"

    def config(self) -> dict[str, Any]:
        return {"workflow_step": self.workflow_step}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        return [
            EpisodeTrainingRows(
                episode_id=episode.episode_id,
                episode_index=episode.episode_index,
                generation_id=episode.generation_id,
                episode_score=episode.episode_score,
                rows=[
                    row
                    for row in episode.rows
                    if row.get("workflow_step") == self.workflow_step
                ],
            )
            for episode in episodes
        ]


class RewritePrompt:
    @property
    def name(self) -> str:
        return "RewritePrompt[v1]"

    def config(self) -> dict[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        out: list[EpisodeTrainingRows] = []
        for episode in episodes:
            rows = []
            for row in episode.rows:
                rewritten = dict(row)
                rewritten["prompt"] = f"rewritten::{row['prompt']}"
                rows.append(rewritten)
            out.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=episode.episode_score,
                    rows=rows,
                )
            )
        return out


class DuplicateFirstRow:
    @property
    def name(self) -> str:
        return "DuplicateFirstRow[v1]"

    def config(self) -> dict[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        first = episodes[0].rows[0]
        return [
            EpisodeTrainingRows(
                episode_id=episodes[0].episode_id,
                episode_index=episodes[0].episode_index,
                generation_id=episodes[0].generation_id,
                episode_score=episodes[0].episode_score,
                rows=[first, dict(first)],
            )
        ]


class DummyTask:
    name = "dummy-task"

    def __init__(self, transforms: tuple[Any, ...]) -> None:
        self._transforms = transforms

    def training_data_transforms(self) -> tuple[Any, ...]:
        return self._transforms



def test_records_to_rows_preserves_generic_provenance() -> None:
    record = TrajectoryRecord(
        episode_id="ep-9",
        phase="orchestrator",
        messages=[{"role": "user", "content": "hello"}],
        response="world",
        usage=None,
        timestamp="2026-05-16T00:00:00",
        success=True,
        meta={
            "workflow_step": "explore",
            "actor_role": "proposer",
            "client": "unit-test",
            "not_jsonable": object(),
        },
    )

    rows = records_to_rows([record], run_id="run-123", version="abc123")
    enriched = enrich_training_rows_for_episode(
        rows,
        episode_id="ep-9",
        episode_index=9,
        generation_id=3,
        episode_score=0.75,
    )

    assert rows[0]["row_id"] == "ep-9:0"
    assert rows[0]["source_interaction_type"] == "orchestrator"
    assert rows[0]["workflow_step"] == "explore"
    assert rows[0]["actor_role"] == "proposer"
    assert rows[0]["record_meta"]["client"] == "unit-test"
    assert "not_jsonable" not in rows[0]["record_meta"]
    assert enriched[0]["episode_index"] == 9
    assert enriched[0]["generation_id"] == 3
    assert enriched[0]["episode_score"] == 0.75
    assert enriched[0]["episode_score_scope"] == "whole_episode"


def test_export_pipeline_filters_rows_and_reports_deltas(tmp_path: Path) -> None:
    transforms = (KeepWorkflowStep("keep"),)
    export = build_training_export(
        _generation_data(),
        transforms,
        _context(tmp_path, transforms),
    )

    assert len(export.rows) == 2
    assert {row["workflow_step"] for row in export.rows} == {"keep"}
    assert export.report["target_count_basis"] == "training_rows"
    assert export.report["raw_successful_row_count"] == 3
    assert export.report["training_row_count"] == 2
    assert export.report["rows_removed"] == 1
    assert export.report["rows_added"] == 0
    assert export.report["rows_modified"] == 0
    assert export.report["per_transform"][0]["name"] == "KeepWorkflowStep[v1]"
    assert export.report["per_transform"][0]["rows_in"] == 3
    assert export.report["per_transform"][0]["rows_out"] == 2


def test_export_pipeline_tracks_modified_rows(tmp_path: Path) -> None:
    transforms = (RewritePrompt(),)
    export = build_training_export(
        _generation_data(),
        transforms,
        _context(tmp_path, transforms),
    )

    assert len(export.rows) == 3
    assert export.report["rows_modified"] == 3
    assert export.report["rows_removed"] == 0
    assert export.report["rows_added"] == 0


def test_export_validation_rejects_duplicate_row_ids(tmp_path: Path) -> None:
    transforms = (DuplicateFirstRow(),)
    with pytest.raises(ValueError, match="duplicate row_id"):
        build_training_export(
            _generation_data(),
            transforms,
            _context(tmp_path, transforms),
        )


def test_save_and_regenerate_publish_sft_bundle(tmp_path: Path) -> None:
    from src.execution.generation import save_generation_data

    transforms = (KeepWorkflowStep("keep"),)
    generation_dir = save_generation_data(
        generation_data=_generation_data(),
        output_dir=tmp_path,
        run_id="run-123",
        task=DummyTask(transforms),
    )

    latest_path = generation_dir / "exports" / "sft" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    first_rows_path = generation_dir / latest["sft_training_rows_path"]
    first_report_path = generation_dir / latest["training_data_export_report_path"]

    assert first_rows_path.exists()
    assert first_report_path.exists()
    assert not (generation_dir / "sft_training_rows.jsonl").exists()
    assert len(first_rows_path.read_text(encoding="utf-8").splitlines()) == 2

    replay = regenerate_sft_export(
        generation_dir,
        DummyTask(transforms),
        export_id="replay",
    )
    latest_after_replay = json.loads(latest_path.read_text(encoding="utf-8"))

    assert replay.export_id == "replay"
    assert latest_after_replay["export_id"] == "replay"
    assert (generation_dir / latest_after_replay["sft_training_rows_path"]).exists()


def test_geology_proposer_rows_requires_workflow_step(tmp_path: Path) -> None:
    from tasks.geology_graph import GeologyProposerRows

    transform = GeologyProposerRows()
    context = _context(tmp_path, (transform,))
    episode = EpisodeTrainingRows(
        episode_id="ep-0",
        episode_index=0,
        generation_id=3,
        episode_score=1.0,
        rows=[_row("ep-0", 0, workflow_step=None)],
    )

    with pytest.raises(ValueError, match="missing workflow_step"):
        transform.transform_export_rows(context, [episode])

    episode.rows = [
        _row("ep-0", 0, workflow_step="explore"),
        _row("ep-0", 1, workflow_step="execute"),
    ]
    filtered = transform.transform_export_rows(context, [episode])

    assert [row["workflow_step"] for row in filtered[0].rows] == ["explore"]
