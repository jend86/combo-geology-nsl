from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import random
import time
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from src.container import (
    ContainerManager,
    compose_services,
    project_name_from_compose,
    resolve_compose_file,
)
from src.execution.backend_runtime import BackendRuntime
from src.execution.episode_runner import run_single_episode
from src.task.base import TaskEnvironmentError
from src.training_data.transforms import (
    TARGET_COUNT_BASIS,
    TrainingDataExportContext,
    build_export_recipe,
    build_training_export,
    count_training_rows,
    publish_training_export,
)
from src.typing.config import AppConfig
from src.typing.training import (
    append_episode_jsonl,
    load_generation_checkpoint,
    save_generation_checkpoint,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData


class _NullProgressBar:
    def __init__(self, total: Optional[int] = None, initial: int = 0) -> None:
        self.total = total
        self.n = initial

    def update(self, increment: int = 1) -> None:
        self.n += increment

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def update_harness_error_breaker(
    *,
    error_category: str | None,
    consecutive: int,
    limit: int,
) -> tuple[int, bool]:
    if error_category == "harness_error":
        new_count = consecutive + 1
        should_trip = limit > 0 and new_count >= limit
        return new_count, should_trip
    return 0, False


def select_variation_index(
    strategy: str,
    episode_index: int,
    variation_count: int,
    rng: Optional[random.Random] = None,
) -> int:
    if variation_count <= 0:
        raise ValueError("variation_count must be positive")
    if strategy == "round_robin":
        return episode_index % variation_count
    if strategy == "random":
        active_rng = rng or random.Random()
        return active_rng.randrange(variation_count)
    raise ValueError(f"Unsupported variation strategy: {strategy}")


def _make_progress_bar(
    enabled: bool,
    total: Optional[int],
    description: str,
    initial: int = 0,
    position: int = 0,
):
    if not enabled:
        return _NullProgressBar(total=total, initial=initial)
    return tqdm(total=total, desc=description, initial=initial, position=position)


def _generation_dir(output_dir: Path, generation_id: int) -> Path:
    return output_dir / f"generation_{generation_id}"


def _is_bootstrap_episode(episode: EpisodeTrajectory) -> bool:
    return bool((episode.task_breakdown or {}).get("bootstrap_active"))


def _count_bootstrap_episodes(episodes: list[EpisodeTrajectory]) -> int:
    return sum(1 for episode in episodes if _is_bootstrap_episode(episode))


def _has_any_admission(episodes: list[EpisodeTrajectory]) -> bool:
    return any(
        bool((episode.task_breakdown or {}).get("admitted")) for episode in episodes
    )


def _load_existing_generation_data(
    all_episodes_path: Path,
    generation_id: int,
) -> GenerationData:
    generation_data = GenerationData(generation_id=generation_id)
    if not all_episodes_path.exists():
        return generation_data
    with all_episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            generation_data.add_episode(EpisodeTrajectory.from_dict(json.loads(line)))
    return generation_data


def _task_training_data_transforms(task: Any) -> tuple[Any, ...]:
    transforms = task.training_data_transforms()
    return tuple(transforms)


def _export_context(
    *,
    generation_id: int,
    run_id: str | None,
    task: Any,
    generation_dir: Path,
    all_episodes_path: Path,
    export_recipe_hash: str,
) -> TrainingDataExportContext:
    return TrainingDataExportContext(
        generation_id=generation_id,
        run_id=run_id,
        task_name=str(getattr(task, "name", type(task).__name__)),
        source_generation_dir=generation_dir,
        source_all_episodes_path=all_episodes_path,
        export_recipe_hash=export_recipe_hash,
    )


def _refresh_training_row_count(
    generation_data: GenerationData,
    transforms: tuple[Any, ...],
    context: TrainingDataExportContext,
) -> int:
    count = count_training_rows(generation_data, transforms, context)
    generation_data.set_training_row_count(count)
    return count


def _build_checkpoint_payload(
    generation_data: GenerationData,
    next_episode_index: int,
    run_id: str,
    *,
    target_training_rows: int,
    export_recipe_hash: str,
) -> dict[str, Any]:
    payload = generation_data.to_metadata_dict(run_id=run_id)
    payload["next_episode_index"] = next_episode_index
    payload["target_count_basis"] = TARGET_COUNT_BASIS
    payload["target_training_rows"] = target_training_rows
    payload["export_recipe_hash"] = export_recipe_hash
    payload["training_row_count"] = generation_data.training_row_count
    return payload


def run_generation_sequential(
    rt: BackendRuntime,
    *,
    generation_id: int,
    target_rows: int | None = None,
    max_episodes: int | None = None,
) -> GenerationData:
    generation_config = rt.config.generation or AppConfig.GenerationConfig()
    target_training_rows = (
        target_rows
        if target_rows is not None
        else generation_config.target_training_rows
    )
    max_episode_count = (
        max_episodes if max_episodes is not None else generation_config.max_episodes
    )
    max_bootstrap_count = generation_config.max_bootstrap_episodes
    output_dir = Path(generation_config.generation_output_dir)
    generation_dir = _generation_dir(output_dir, generation_id)
    checkpoint_path = generation_dir / "checkpoint.json"
    all_episodes_path = generation_dir / "all_episodes.jsonl"
    transforms = _task_training_data_transforms(rt.task)
    recipe = build_export_recipe(transforms)
    export_context = _export_context(
        generation_id=generation_id,
        run_id=rt.run_id,
        task=rt.task,
        generation_dir=generation_dir,
        all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )

    start_episode_index = 0
    generation_data = GenerationData(generation_id=generation_id)
    if generation_config.resume_from_checkpoint:
        checkpoint = load_generation_checkpoint(checkpoint_path)
        if checkpoint is not None:
            checkpoint_recipe_hash = checkpoint.get("export_recipe_hash")
            if checkpoint_recipe_hash != recipe.recipe_hash:
                raise RuntimeError(
                    "checkpoint export recipe hash differs from current task "
                    f"recipe: {checkpoint_recipe_hash!r} != {recipe.recipe_hash!r}"
                )
            start_episode_index = int(checkpoint.get("next_episode_index", 0))
            generation_data = _load_existing_generation_data(
                all_episodes_path, generation_id
            )
            _refresh_training_row_count(
                generation_data,
                transforms,
                export_context,
            )
            generation_data.started_at = (
                checkpoint.get("started_at") or generation_data.started_at
            )

    if generation_data.started_at is None:
        generation_data.started_at = datetime.now().isoformat()

    compose_dir_str = rt.config.docker_compose_dir or rt.task.docker_compose_dir
    project_name_pattern = None
    expected_services = None
    if compose_dir_str and Path(compose_dir_str).exists():
        compose_file = resolve_compose_file(compose_dir_str)
        project_name_pattern = project_name_from_compose(compose_file)
        expected_services = compose_services(compose_file)

    container_manager = ContainerManager(
        docker_client=rt.docker_client,
        container_ids=rt.config.container_ids,
        docker_compose_dir=compose_dir_str,
        post_rebuild_wait_seconds=generation_config.post_rebuild_wait_seconds,
        project_name_pattern=project_name_pattern,
        expected_services=expected_services,
        task=rt.task,
    )
    try:
        ready = container_manager.verify_ready()
    except Exception:
        ready = False
    if not ready:
        if rt.config.dynamic_container and compose_dir_str:
            container_manager.rebuild()
        else:
            raise RuntimeError("Containers failed readiness check at startup")

    variations = rt.task.list_variations()
    variation_count = len(variations)
    rng = (
        random.Random(generation_config.variation_random_seed)
        if generation_config.variation_strategy == "random"
        else None
    )
    progress_episode_total = (
        max_episode_count + max_bootstrap_count
        if max_bootstrap_count is not None
        else max_episode_count
    )
    episode_progress = _make_progress_bar(
        generation_config.show_progress,
        progress_episode_total,
        "episodes",
        initial=generation_data.total_episodes_run,
    )
    rows_progress = _make_progress_bar(
        generation_config.show_progress,
        target_training_rows,
        "training rows",
        initial=generation_data.training_row_count,
    )
    generation_started_at = time.perf_counter()
    consecutive_verification_failures = 0
    consecutive_rebuild_failures = 0
    consecutive_harness_errors = 0
    harness_error_limit = rt.config.harness.consecutive_harness_error_limit
    episode_index = start_episode_index
    pending_variation_index: Optional[int] = None
    sequential_harness_session: dict[str, Any] = {}
    bootstrap_episodes_run = _count_bootstrap_episodes(generation_data.all_episodes)
    regular_episodes_run = max(
        0, generation_data.total_episodes_run - bootstrap_episodes_run
    )
    has_admission = _has_any_admission(generation_data.all_episodes)

    def _budget_exhausted() -> bool:
        # Absolute cap always wins regardless of bootstrap vs regular mode.
        if episode_index >= max_episode_count:
            return True
        if max_bootstrap_count is None:
            return False
        # Until the pool has any admitted graph, the next episode is bootstrap;
        # only the bootstrap cap applies. Once we've admitted at least one
        # graph, the pool is non-empty and future episodes run the regular
        # workflow, so the regular cap takes over.
        if not has_admission:
            return bootstrap_episodes_run >= max_bootstrap_count
        return regular_episodes_run >= max_episode_count

    try:
        while not _budget_exhausted():
            if generation_data.training_row_count >= target_training_rows:
                generation_data.termination_reason = "target_reached"
                break

            if pending_variation_index is None:
                variation_index = select_variation_index(
                    generation_config.variation_strategy,
                    episode_index,
                    variation_count,
                    rng=rng,
                )
            else:
                variation_index = pending_variation_index

            selected_variation = variations[variation_index % variation_count]
            try:
                population_outcome, verified = container_manager.populate_with_task(
                    container_manager.get_containers(),
                    selected_variation,
                )
            except TaskEnvironmentError as exc:
                pending_variation_index = variation_index
                logger.info(
                    f"Episode {episode_index}: environment error during "
                    f"population: {exc}. Rebuilding containers."
                )
                try:
                    if exc.container_ids:
                        container_manager.rebuild_containers(exc.container_ids)
                    else:
                        container_manager.rebuild()
                except Exception as rebuild_exc:
                    consecutive_rebuild_failures += 1
                    logger.warning(
                        f"Episode {episode_index}: container recovery failed "
                        f"({consecutive_rebuild_failures} consecutive): "
                        f"{rebuild_exc}"
                    )
                    if consecutive_rebuild_failures >= 3:
                        logger.error(
                            "Circuit breaker tripped: 3 consecutive container "
                            "rebuild failures. Aborting generation."
                        )
                        generation_data.termination_reason = "force_stop"
                        break
                else:
                    consecutive_rebuild_failures = 0
                continue

            pending_variation_index = None
            consecutive_rebuild_failures = 0
            previous_rows = generation_data.training_row_count
            episode = run_single_episode(
                rt,
                container_manager=container_manager,
                generation_id=generation_id,
                episode_index=episode_index,
                variation_index=variation_index,
                population_outcome=population_outcome,
                verified=verified,
                variation=selected_variation,
                harness_session=sequential_harness_session,
            )
            generation_data.add_episode(episode)
            _refresh_training_row_count(
                generation_data,
                transforms,
                export_context,
            )
            if _is_bootstrap_episode(episode):
                bootstrap_episodes_run += 1
            else:
                regular_episodes_run += 1
            if not has_admission and bool(
                (episode.task_breakdown or {}).get("admitted")
            ):
                has_admission = True

            if episode.error_message == "container population verification failed":
                consecutive_verification_failures += 1
                logger.warning(
                    f"Episode {episode_index}: container population verification "
                    f"failed ({consecutive_verification_failures} consecutive). "
                    f"Variation: {episode.container_variation}"
                )
                if (
                    generation_config.max_consecutive_verification_failures > 0
                    and consecutive_verification_failures
                    >= generation_config.max_consecutive_verification_failures
                ):
                    logger.error(
                        f"Circuit breaker tripped: "
                        f"{consecutive_verification_failures} consecutive "
                        f"container population verification failures. "
                        f"This indicates a systematic issue with the container setup. "
                        f"Aborting generation."
                    )
                    generation_data.termination_reason = "force_stop"
                    break
            else:
                consecutive_verification_failures = 0

            consecutive_harness_errors, should_trip = update_harness_error_breaker(
                error_category=episode.error_category,
                consecutive=consecutive_harness_errors,
                limit=harness_error_limit,
            )
            if episode.error_category == "harness_error":
                logger.warning(
                    f"Episode {episode_index}: harness error "
                    f"({consecutive_harness_errors} consecutive): "
                    f"{episode.error_message}"
                )
            if should_trip:
                logger.error(
                    f"HARNESS CIRCUIT BREAKER: {consecutive_harness_errors} "
                    f"consecutive HarnessError occurrences. Aborting generation - "
                    f"this is distinct from environment failure and indicates the "
                    f"harness implementation or its configuration is broken."
                )
                generation_data.termination_reason = "harness_error"
                break

            episode_progress.update(1)
            rows_progress.update(generation_data.training_row_count - previous_rows)
            elapsed_hours = (time.perf_counter() - generation_started_at) / 3600
            if elapsed_hours > 0:
                episode_postfix: dict[str, str] = {}
                if episode.average_output_tokens_per_second is not None:
                    episode_postfix["tok/s"] = (
                        f"{episode.average_output_tokens_per_second:.1f}"
                    )
                if episode.inference_duty_cycle is not None:
                    episode_postfix["duty"] = f"{episode.inference_duty_cycle:.0%}"
                if episode.peak_gpu_utilization_pct is not None:
                    episode_postfix["gpu"] = f"{episode.peak_gpu_utilization_pct:.0f}%"
                if episode.peak_cpu_utilization_pct is not None:
                    episode_postfix["cpu"] = f"{episode.peak_cpu_utilization_pct:.0f}%"
                if episode_postfix:
                    episode_progress.set_postfix(episode_postfix)
                rows_progress.set_postfix(
                    {
                        "rows/hr": (
                            f"{generation_data.training_row_count / elapsed_hours:.1f}"
                        )
                    }
                )

            if generation_config.checkpoint_every_episode:
                append_episode_jsonl(episode.to_dict(), all_episodes_path)
                save_generation_checkpoint(
                    _build_checkpoint_payload(
                        generation_data,
                        episode_index + 1,
                        rt.run_id,
                        target_training_rows=target_training_rows,
                        export_recipe_hash=recipe.recipe_hash,
                    ),
                    checkpoint_path,
                )

            completed_episodes = episode_index + 1
            if max_bootstrap_count is None:
                episodes_remaining = completed_episodes < max_episode_count
            elif not has_admission:
                episodes_remaining = bootstrap_episodes_run < max_bootstrap_count
            else:
                episodes_remaining = regular_episodes_run < max_episode_count
            if (
                generation_config.container_rebuild_interval > 0
                and episodes_remaining
                and completed_episodes % generation_config.container_rebuild_interval
                == 0
            ):
                container_manager.rebuild()
                consecutive_verification_failures = 0
                consecutive_rebuild_failures = 0
            elif (
                generation_config.container_restart_interval > 0
                and episodes_remaining
                and completed_episodes % generation_config.container_restart_interval
                == 0
            ):
                container_manager.restart()
                consecutive_verification_failures = 0
                consecutive_rebuild_failures = 0

            episode_index += 1
    finally:
        episode_progress.close()
        rows_progress.close()

    if generation_data.termination_reason is None:
        if generation_data.training_row_count >= target_training_rows:
            generation_data.termination_reason = "target_reached"
        elif max_bootstrap_count is None:
            if episode_index >= max_episode_count:
                generation_data.termination_reason = "max_episodes"
        elif not has_admission:
            if bootstrap_episodes_run >= max_bootstrap_count:
                generation_data.termination_reason = "max_bootstrap_episodes"
        elif regular_episodes_run >= max_episode_count:
            generation_data.termination_reason = "max_episodes"

    generation_data.completed_at = datetime.now().isoformat()
    return generation_data


def save_generation_data(
    generation_data: GenerationData,
    output_dir: Path | str,
    run_id: str,
    task: Any,
) -> Path:
    output_dir = Path(output_dir)
    generation_dir = _generation_dir(output_dir, generation_data.generation_id)
    generation_dir.mkdir(parents=True, exist_ok=True)
    successful_dir = generation_dir / "successful"
    failed_dir = generation_dir / "failed"
    successful_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    all_episodes_path = generation_dir / "all_episodes.jsonl"
    with all_episodes_path.open("w", encoding="utf-8") as handle:
        for episode in generation_data.all_episodes:
            handle.write(json.dumps(episode.to_dict(), default=str) + "\n")

    for episode in generation_data.successful_episodes:
        with (successful_dir / f"{episode.episode_id}.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(episode.to_dict(), handle, indent=2, default=str)

    for episode in generation_data.failed_episodes:
        with (failed_dir / f"{episode.episode_id}.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(episode.to_dict(), handle, indent=2, default=str)

    transforms = _task_training_data_transforms(task)
    recipe = build_export_recipe(transforms)
    export_context = _export_context(
        generation_id=generation_data.generation_id,
        run_id=run_id,
        task=task,
        generation_dir=generation_dir,
        all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )
    export = build_training_export(generation_data, transforms, export_context)
    published_export = publish_training_export(generation_dir, export)
    generation_data.set_training_row_count(len(published_export.rows))

    metadata_payload = generation_data.to_metadata_dict(run_id=run_id)
    metadata_payload["active_sft_export_id"] = published_export.export_id
    metadata_payload["target_count_basis"] = TARGET_COUNT_BASIS
    metadata_payload["export_recipe_hash"] = recipe.recipe_hash
    metadata_path = generation_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, default=str)

    save_generation_checkpoint(
        _build_checkpoint_payload(
            generation_data,
            generation_data.total_episodes_run,
            run_id,
            target_training_rows=generation_data.training_row_count,
            export_recipe_hash=recipe.recipe_hash,
        ),
        generation_dir / "checkpoint.json",
    )

    return generation_dir


__all__ = [
    "run_generation_sequential",
    "save_generation_data",
    "select_variation_index",
    "update_harness_error_breaker",
]
