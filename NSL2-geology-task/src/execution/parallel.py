from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import itertools
import json
from pathlib import Path
import random
import signal
import sys
import threading
import time
from typing import Any

from loguru import logger

import src.parallel as parallel_runtime
from src.backend.endpoint_pool import EndpointPoolUnavailable
from src.execution.backend_runtime import BackendRuntime, _coerce_runtime
from src.execution.episode_runner import run_single_episode
from src.execution.telemetry import (
    initial_framework_telemetry,
    telemetry_columns_for_harness,
)
from src.harness.loader import construct_harness
from src.parallel import (
    StopReason,
    ThreadSafeGenerationCollector,
    WorkerSlot,
)
from src.training_data.transforms import (
    TARGET_COUNT_BASIS,
    TrainingDataExportContext,
    build_export_recipe,
    count_training_rows,
)
from src.typing.config import AppConfig
from src.typing.training import (
    append_episode_jsonl,
    load_generation_checkpoint,
    save_generation_checkpoint,
)
from src.typing.trajectory import GenerationData


_ROW_COUNT_REFRESH_EVERY_EPISODES = 10
_ROW_COUNT_NEAR_TARGET_MARGIN = 2


def _episode_triggers_endpoint_quarantine(error_category: str | None) -> bool:
    """Whether an episode's failure should quarantine its inference endpoint.

    Only a genuine outage (``endpoint_unavailable``) does. A request timeout
    (``inference_timeout``) is a benign, retryable failure — the endpoint is
    healthy, the model was just slow (e.g. decode starvation). Quarantining the
    (possibly sole) endpoint on a timeout would breach the capacity floor and
    abort the whole run, so it must not. Mirrors ``EndpointAwareGenner``, which
    keys quarantine on the ``inference_unavailable`` prefix, not on timeouts.
    """
    return error_category == "endpoint_unavailable"


@contextmanager
def _null_context():
    yield


@contextmanager
def _scoped_parallel_logging(log_path: Path):
    logger.remove()
    try:
        logger.add(sys.stderr, level="WARNING")
        logger.add(str(log_path), level="DEBUG", enqueue=True)
        yield
    finally:
        logger.remove()
        logger.add(sys.stderr)


def _save_parallel_checkpoint(
    collector: ThreadSafeGenerationCollector,
    checkpoint_path: Path,
    run_id: str,
    *,
    target_training_rows: int,
    export_recipe_hash: str,
) -> None:
    generation_data = collector.get_generation_data()
    row_count_state = collector.row_count_state()
    payload = generation_data.to_metadata_dict(run_id=run_id)
    payload["total_episodes_completed"] = generation_data.total_episodes_run
    payload["target_count_basis"] = TARGET_COUNT_BASIS
    payload["target_training_rows"] = target_training_rows
    payload["export_recipe_hash"] = export_recipe_hash
    payload["training_row_count"] = row_count_state.training_row_count
    payload["training_row_count_is_exact"] = (
        row_count_state.training_row_count_is_exact
    )
    payload["training_row_count_last_refreshed_episode"] = (
        row_count_state.training_row_count_last_refreshed_episode
    )
    save_generation_checkpoint(payload, checkpoint_path)


def _generation_dir(output_dir: Path, generation_id: int) -> Path:
    return output_dir / f"generation_{generation_id}"


def _load_existing_generation_data(
    all_episodes_path: Path,
    generation_id: int,
) -> GenerationData:
    from src.typing.trajectory import EpisodeTrajectory

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


def select_variation_index(
    strategy: str,
    episode_index: int,
    variation_count: int,
    rng: random.Random | None = None,
) -> int:
    if variation_count <= 0:
        raise ValueError("variation_count must be positive")
    if strategy == "round_robin":
        return episode_index % variation_count
    if strategy == "random":
        active_rng = rng or random.Random()
        return active_rng.randrange(variation_count)
    raise ValueError(f"Unsupported variation strategy: {strategy}")


def _run_generation_parallel(
    rt: BackendRuntime, *, generation_id: int
) -> GenerationData:
    generation_config = rt.config.generation or AppConfig.GenerationConfig()
    output_dir = Path(generation_config.generation_output_dir)
    generation_dir = _generation_dir(output_dir, generation_id)
    generation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = generation_dir / "checkpoint.json"
    all_episodes_path = generation_dir / "all_episodes.jsonl"

    n_workers = generation_config.parallel_episodes
    target_rows = generation_config.target_training_rows
    start_episode_index = 0
    generation_data = GenerationData(generation_id=generation_id)
    if generation_config.resume_from_checkpoint and all_episodes_path.exists():
        generation_data = _load_existing_generation_data(
            all_episodes_path, generation_id
        )
        start_episode_index = generation_data.total_episodes_run

    if generation_data.started_at is None:
        generation_data.started_at = datetime.now().isoformat()

    transforms = tuple(rt.task.training_data_transforms())
    recipe = build_export_recipe(transforms)
    export_context = TrainingDataExportContext(
        generation_id=generation_id,
        run_id=rt.run_id,
        task_name=str(getattr(rt.task, "name", type(rt.task).__name__)),
        source_generation_dir=generation_dir,
        source_all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )

    def _count_training_rows(data: GenerationData) -> int:
        return count_training_rows(data, transforms, export_context)

    if generation_config.resume_from_checkpoint and all_episodes_path.exists():
        checkpoint = load_generation_checkpoint(checkpoint_path)
        checkpoint_recipe_hash = checkpoint.get("export_recipe_hash") if checkpoint else None
        if checkpoint_recipe_hash != recipe.recipe_hash:
            raise RuntimeError(
                "checkpoint export recipe hash differs from current task recipe: "
                f"{checkpoint_recipe_hash!r} != {recipe.recipe_hash!r}"
            )
        generation_data.set_training_row_count(_count_training_rows(generation_data))

    base_compose_dir = (
        Path(rt.config.docker_compose_dir or rt.task.docker_compose_dir)
        if (rt.config.docker_compose_dir or rt.task.docker_compose_dir)
        else None
    )
    if base_compose_dir is None:
        raise ValueError(
            "docker_compose_dir is required for parallel episode execution"
        )

    slots = parallel_runtime.create_worker_slots(
        n_slots=n_workers,
        base_compose_dir=base_compose_dir,
        generation_dir=generation_dir,
        run_id=rt.run_id,
        code_host_cache_path=Path(rt.config.code_host_cache_path),
        post_rebuild_wait_seconds=generation_config.post_rebuild_wait_seconds,
        task=rt.task,
    )
    if len(slots) < n_workers:
        logger.warning(
            f"Generation running with {len(slots)} worker slots instead of "
            f"{n_workers} requested - Docker network pool capacity was "
            f"insufficient to create all slots."
        )

    collector = parallel_runtime.ThreadSafeGenerationCollector(
        generation_data,
        training_row_count_fn=_count_training_rows if transforms else None,
    )
    global_circuit = parallel_runtime.GlobalCircuitBreaker(
        [s.circuit_breaker for s in slots], threshold=0.5
    )
    endpoint_pool = rt.endpoint_pool
    file_lock = threading.Lock()
    stop_event = threading.Event()
    stop_reason = StopReason()
    row_count_refresh_lock = threading.Lock()
    episode_counter = itertools.count(start=start_episode_index)
    counter_lock = threading.Lock()
    variations = rt.task.list_variations()
    variation_count = len(variations)

    from rich.console import Console as RichConsole

    from src.display import ParallelProgressDisplay, scoped_loguru_to_rich

    display_console = RichConsole(stderr=True)
    display: ParallelProgressDisplay | None = None
    if generation_config.show_progress:
        display = ParallelProgressDisplay(
            n_slots=len(slots),
            target_rows=target_rows,
            max_episodes=generation_config.max_episodes,
            run_id=rt.run_id,
            generation_id=generation_id,
            console=display_console,
            metrics_collector=rt.metrics,
            max_context_tokens=(
                rt.config.vllm.max_model_len
                if rt.config.vllm
                else (rt.config.sglang.max_model_len if rt.config.sglang else None)
            ),
        )
        display.set_telemetry_columns(
            telemetry_columns_for_harness(construct_harness(rt.config.harness))
        )

    if rt.metrics is not None:
        rt.metrics.start_utilization_sampling(
            inference_metrics_url=rt.metrics.inference_metrics_url,
            inference_metrics_backend=rt.metrics.inference_metrics_backend,
            inference_metrics_api_key=getattr(rt.metrics, "inference_metrics_api_key", None),
        )

    def _display_update_slot(slot_id: int, **kwargs: Any) -> None:
        if display is not None:
            display.update_slot(slot_id, **kwargs)

    def _signal_stop(reason: str) -> None:
        stop_reason.set(reason)
        stop_event.set()

    def _endpoint_capacity_floor_tripped() -> bool:
        return bool(
            endpoint_pool is not None and endpoint_pool.below_capacity_floor()
        )

    def _mark_endpoint_unavailable(lease: Any, detail: str | None) -> None:
        if endpoint_pool is None or lease is None:
            return
        if endpoint_pool.is_healthy(lease.endpoint_id):
            endpoint_pool.mark_unhealthy(lease.endpoint_id, detail)
        if endpoint_pool.below_capacity_floor():
            _signal_stop("endpoint_capacity_floor")

    def _annotate_endpoint_metadata(episode: Any, lease: Any) -> None:
        if lease is None:
            return
        trajectory = getattr(episode, "trajectory", None)
        if not isinstance(trajectory, dict):
            return
        extra = trajectory.setdefault("extra", {})
        if isinstance(extra, dict):
            extra["inference_endpoint"] = dict(lease.metadata)

    def _make_telemetry_observer(slot_id: int):
        active_display = display
        if active_display is None:
            return None

        def _observer(
            prompt_tokens: int | None,
            telemetry: dict[str, str],
        ) -> None:
            active_display.update_slot(
                slot_id,
                last_prompt_tokens=prompt_tokens,
                telemetry=dict(telemetry),
            )

        return _observer

    def _display_update_progress() -> None:
        if display is not None:
            episodes, rows = collector.progress_snapshot()
            display.update_progress(rows=rows, episodes=episodes)

    def _should_refresh_training_row_count() -> bool:
        state = collector.row_count_state()
        if state.training_row_count_is_exact:
            return False
        episodes_since_refresh = (
            state.total_episodes_run - state.training_row_count_last_refreshed_episode
        )
        if episodes_since_refresh >= _ROW_COUNT_REFRESH_EVERY_EPISODES:
            return True
        return state.training_row_count >= max(
            0,
            target_rows - _ROW_COUNT_NEAR_TARGET_MARGIN,
        )

    def _maybe_refresh_training_row_count() -> None:
        if not _should_refresh_training_row_count():
            return
        if not row_count_refresh_lock.acquire(blocking=False):
            return
        try:
            if _should_refresh_training_row_count():
                collector.refresh_training_row_count()
        finally:
            row_count_refresh_lock.release()

    def _confirm_target_reached() -> bool:
        if not collector.should_stop(target_rows):
            return False
        state = collector.row_count_state()
        if not state.training_row_count_is_exact:
            with row_count_refresh_lock:
                collector.refresh_training_row_count()
        return collector.should_stop(target_rows)

    def worker_loop(slot: WorkerSlot) -> None:
        slot_config = rt.config.model_copy(
            update={"code_host_cache_path": str(slot.cache_dir)}
        )
        slot_runtime = BackendRuntime(
            config=slot_config,
            run_id=rt.run_id,
            task=rt.task,
            genner=rt.genner,
            docker_client=slot.docker_client,
            metrics=rt.metrics,
        )
        slot_rng = (
            random.Random((generation_config.variation_random_seed or 0) + slot.slot_id)
            if generation_config.variation_strategy == "random"
            else None
        )
        slot_episodes_since_rebuild = 0
        slot_successes = 0
        slot_failures = 0
        max_episodes = generation_config.max_episodes

        while (
            not stop_event.is_set()
            and not collector.should_stop(target_rows)
            and not global_circuit.is_tripped()
        ):
            if _endpoint_capacity_floor_tripped():
                _signal_stop("endpoint_capacity_floor")
                break
            with counter_lock:
                episode_index = next(episode_counter)
            if episode_index >= max_episodes:
                _signal_stop("max_episodes")
                break

            _display_update_slot(
                slot.slot_id,
                status="running",
                episode=episode_index,
                telemetry=initial_framework_telemetry(),
                last_prompt_tokens=None,
            )

            try:
                variation_index = select_variation_index(
                    generation_config.variation_strategy,
                    episode_index,
                    variation_count,
                    rng=slot_rng,
                )
                selected_variation = variations[variation_index % variation_count]
                try:
                    population_outcome, verified = (
                        slot.container_manager.populate_with_task(
                            slot.container_manager.get_containers(),
                            selected_variation,
                        )
                    )
                except Exception as exc:
                    if not hasattr(exc, "container_ids"):
                        raise
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"environment error during population: {exc}"
                    )
                    _display_update_slot(slot.slot_id, status="rebuilding")
                    try:
                        if exc.container_ids:
                            slot.container_manager.rebuild_containers(exc.container_ids)
                        else:
                            slot.container_manager.rebuild()
                    except Exception:
                        slot.circuit_breaker.record_failure()
                        slot_failures += 1
                        _display_update_slot(slot.slot_id, failures=slot_failures)
                        if slot.circuit_breaker.is_tripped():
                            logger.error(
                                f"Slot {slot.slot_id}: circuit breaker tripped "
                                f"after rebuild failures"
                            )
                            _display_update_slot(
                                slot.slot_id,
                                status="tripped",
                                cb_tripped=True,
                            )
                        continue
                    slot.circuit_breaker.reset()
                    _display_update_slot(slot.slot_id, status="running")
                    continue

                lease = None
                try:
                    if endpoint_pool is not None:
                        lease = endpoint_pool.lease(
                            home_key=slot.slot_id,
                            stop_event=stop_event,
                        )
                        episode_runtime = replace(slot_runtime, genner=lease.genner)
                    else:
                        episode_runtime = slot_runtime
                except EndpointPoolUnavailable as exc:
                    logger.warning(
                        f"Slot {slot.slot_id}: endpoint lease unavailable - {exc}"
                    )
                    _signal_stop(
                        "endpoint_capacity_floor"
                        if _endpoint_capacity_floor_tripped()
                        else "endpoint_pool_unavailable"
                    )
                    break

                try:
                    episode = run_single_episode(
                        episode_runtime,
                        container_manager=slot.container_manager,
                        generation_id=generation_id,
                        episode_index=episode_index,
                        variation_index=variation_index,
                        population_outcome=population_outcome,
                        verified=verified,
                        parallel_episodes=n_workers,
                        stop_event=stop_event,
                        stop_reason=stop_reason,
                        variation=selected_variation,
                        telemetry_observer=_make_telemetry_observer(slot.slot_id),
                        harness_session=slot.harness_session,
                    )
                    _annotate_endpoint_metadata(episode, lease)
                    if _episode_triggers_endpoint_quarantine(episode.error_category):
                        _mark_endpoint_unavailable(lease, episode.error_message)
                finally:
                    if endpoint_pool is not None:
                        endpoint_pool.release(lease)

                if episode.error_message == "container population verification failed":
                    slot.circuit_breaker.record_verification_failure()
                    if slot.circuit_breaker.is_verification_tripped():
                        logger.error(
                            f"Slot {slot.slot_id}: verification circuit breaker tripped"
                        )
                elif episode.error_category in {
                    "context_overflow",
                    "repetition_collapse",
                    "inference_timeout",
                }:
                    slot.circuit_breaker.record_benign_abort()
                elif episode.error_category == "harness_error":
                    slot.circuit_breaker.record_failure()
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"harness error - {episode.error_message}"
                    )
                elif episode.error_category == "endpoint_unavailable":
                    slot.circuit_breaker.record_benign_abort()
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"endpoint unavailable - {episode.error_message}"
                    )
                else:
                    slot.circuit_breaker.record_success()

                collector.add_episode(episode)
                if episode.success:
                    slot_successes += len(episode.prompt_responses)
                _display_update_slot(slot.slot_id, successes=slot_successes)
                _maybe_refresh_training_row_count()
                _display_update_progress()

                if _confirm_target_reached():
                    _signal_stop("target_reached")
                if generation_config.checkpoint_every_episode:
                    with file_lock:
                        append_episode_jsonl(episode.to_dict(), all_episodes_path)
                        _save_parallel_checkpoint(
                            collector,
                            checkpoint_path,
                            rt.run_id,
                            target_training_rows=target_rows,
                            export_recipe_hash=recipe.recipe_hash,
                        )

                slot_episodes_since_rebuild += 1
                if (
                    generation_config.container_rebuild_interval > 0
                    and slot_episodes_since_rebuild
                    >= generation_config.container_rebuild_interval
                ):
                    _display_update_slot(slot.slot_id, status="rebuilding")
                    try:
                        slot.container_manager.rebuild()
                        slot.circuit_breaker.reset()
                        slot_episodes_since_rebuild = 0
                    except Exception as exc:
                        logger.warning(
                            f"Slot {slot.slot_id}: randomization rebuild failed: {exc}"
                        )
                        slot.circuit_breaker.record_failure()
                    _display_update_slot(slot.slot_id, status="running")

                if slot.circuit_breaker.is_tripped():
                    logger.warning(
                        f"Slot {slot.slot_id}: circuit breaker tripped, attempting rebuild"
                    )
                    _display_update_slot(
                        slot.slot_id, status="rebuilding", cb_tripped=True
                    )
                    try:
                        slot.container_manager.rebuild()
                    except Exception:
                        logger.error(
                            f"Slot {slot.slot_id}: recovery rebuild failed, slot retiring"
                        )
                        _display_update_slot(slot.slot_id, status="done")
                        break
                    slot.circuit_breaker.reset()
                    _display_update_slot(
                        slot.slot_id, status="running", cb_tripped=False
                    )
            except Exception:
                logger.exception(
                    f"Slot {slot.slot_id} episode {episode_index}: unhandled exception"
                )
                slot.circuit_breaker.record_failure()
                slot_failures += 1
                _display_update_slot(slot.slot_id, failures=slot_failures)
                if slot.circuit_breaker.is_tripped():
                    logger.error(
                        f"Slot {slot.slot_id}: circuit breaker tripped after unhandled "
                        f"exceptions, attempting rebuild"
                    )
                    _display_update_slot(
                        slot.slot_id, status="rebuilding", cb_tripped=True
                    )
                    try:
                        slot.container_manager.rebuild()
                    except Exception:
                        logger.error(
                            f"Slot {slot.slot_id}: recovery rebuild failed, slot retiring"
                        )
                        _display_update_slot(slot.slot_id, status="done")
                        break
                    slot.circuit_breaker.reset()
                    _display_update_slot(
                        slot.slot_id, status="running", cb_tripped=False
                    )

        _display_update_slot(slot.slot_id, status="done")

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(slot,),
            name=f"worker-slot-{slot.slot_id}",
            daemon=False,
        )
        for slot in slots
    ]
    original_handler = signal.getsignal(signal.SIGINT)

    def _graceful_shutdown(signum, frame):
        del signum, frame
        logger.info("SIGINT received, signaling workers to stop...")
        _signal_stop("sigint")

    signal.signal(signal.SIGINT, _graceful_shutdown)

    deadline_timer: threading.Timer | None = None
    if generation_config.generation_timeout_s:

        def _on_deadline() -> None:
            logger.warning(
                f"Generation deadline ({generation_config.generation_timeout_s}s) "
                "reached; signalling workers to stop"
            )
            _signal_stop("deadline_exceeded")

        deadline_timer = threading.Timer(
            generation_config.generation_timeout_s, _on_deadline
        )
        deadline_timer.daemon = True
        deadline_timer.start()

    inference_timeout = rt.config.inference.timeout if rt.config.inference else 300
    worker_join_timeout_s = inference_timeout + 30
    log_path = generation_dir / "run.log"
    log_context = (
        scoped_loguru_to_rich(display_console, log_path)
        if display is not None
        else _scoped_parallel_logging(log_path)
    )

    result: GenerationData | None = None
    display_context = display if display is not None else _null_context()
    with log_context:
        try:
            with display_context:
                try:
                    for thread in threads:
                        thread.start()

                    while True:
                        alive = [thread for thread in threads if thread.is_alive()]
                        if not alive or stop_event.is_set():
                            break
                        for thread in alive:
                            thread.join(timeout=0.2)

                    if stop_event.is_set():
                        join_deadline = time.monotonic() + worker_join_timeout_s
                        for thread in threads:
                            remaining = max(0.0, join_deadline - time.monotonic())
                            thread.join(timeout=remaining)

                        timed_out = [thread for thread in threads if thread.is_alive()]
                        if timed_out:
                            logger.warning(
                                f"{len(timed_out)} worker threads did not exit within "
                                f"{worker_join_timeout_s}s after stop signal - forcing stop "
                                f"and proceeding to metric collection"
                            )
                            stop_reason.set("force_stop")
                            stop_event.set()
                            final_deadline = time.monotonic() + 30
                            for thread in timed_out:
                                thread.join(
                                    timeout=max(0.0, final_deadline - time.monotonic())
                                )
                finally:
                    if deadline_timer is not None:
                        deadline_timer.cancel()
                    signal.signal(signal.SIGINT, original_handler)
                    _display_update_progress()

            result = collector.get_generation_data()
            result.completed_at = datetime.now().isoformat()
            target_reached = _confirm_target_reached()
            result.termination_reason = stop_reason.get() or (
                "target_reached"
                if target_reached
                else "force_stop"
                if global_circuit.is_tripped()
                else "max_episodes"
            )
        finally:
            try:
                if rt.metrics is not None:
                    utilization_summary = rt.metrics.stop_utilization_sampling()
                    logger.info(
                        f"Parallel generation utilization: "
                        f"peak_gpu={utilization_summary.peak_gpu_utilization_pct}, "
                        f"peak_kv_cache={utilization_summary.peak_kv_cache_usage_pct}, "
                        f"peak_requests_running={utilization_summary.peak_num_requests_running}, "
                        f"peak_requests_waiting={utilization_summary.peak_num_requests_waiting}"
                    )
                    delta = utilization_summary.inference_metrics_delta
                    if delta is not None:
                        logger.info(
                            "Parallel generation vLLM deltas (window="
                            f"{delta.get('window_seconds')}s): "
                            f"preemptions={delta.get('preemptions')}, "
                            f"prefix_cache_hit_rate={delta.get('prefix_cache_hit_rate')}, "
                            f"prompt_tokens_cached_rate={delta.get('prompt_tokens_cached_rate')}, "
                            f"prompt_tok/s={delta.get('prompt_tokens_per_second')}, "
                            f"gen_tok/s={delta.get('generation_tokens_per_second')}"
                        )
                    rt.metrics.flush()
            finally:
                parallel_runtime.teardown_worker_slots(slots)

    assert result is not None
    logger.info(
        f"Parallel generation complete: rows={result.training_row_count}, "
        f"episodes={result.total_episodes_run}"
    )
    for slot in slots:
        logger.info(
            f"Slot {slot.slot_id} final state: "
            f"{'tripped' if slot.circuit_breaker.is_tripped() else 'ok'}"
        )
    return result


def run_generation_parallel(*args: Any, **kwargs: Any) -> GenerationData:
    if args and isinstance(args[0], BackendRuntime):
        rt = args[0]
        generation_id = kwargs.pop("generation_id", args[1] if len(args) > 1 else None)
        if generation_id is None:
            raise TypeError("run_generation_parallel requires generation_id")
        return _run_generation_parallel(rt, generation_id=int(generation_id))

    genner = kwargs.pop("genner", args[0] if len(args) > 0 else None)
    docker_client = kwargs.pop("docker_client", args[1] if len(args) > 1 else None)
    config = kwargs.pop("config", args[2] if len(args) > 2 else None)
    generation_id = kwargs.pop("generation_id", args[3] if len(args) > 3 else None)
    run_id = kwargs.pop("run_id", args[4] if len(args) > 4 else None)
    metrics_collector = kwargs.pop("metrics_collector", None)
    task = kwargs.pop("task", None)

    if None in (genner, docker_client, config, generation_id, run_id, task):
        raise TypeError("run_generation_parallel missing required legacy arguments")

    rt = _coerce_runtime(
        config=config,
        run_id=run_id,
        task=task,
        genner=genner,
        docker_client=docker_client,
        metrics=metrics_collector,
    )
    return _run_generation_parallel(rt, generation_id=int(generation_id))


__all__ = ["_scoped_parallel_logging", "run_generation_parallel"]
