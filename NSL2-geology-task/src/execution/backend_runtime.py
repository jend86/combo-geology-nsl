from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import docker
from loguru import logger

from src.backend import resolve_backend_context
from src.genner.Base import Genner
from src.harness.profiles import resolve_profile
from src.harness.tool_contract import (
    emit_static_tool_contract_warnings,
    run_tool_call_contract_probe,
)
from src.helper import generate_readable_run_id
from src.observability import MetricsCollector, MetricsGenner
from src.task import load_task
from src.task.base import TaskSpec
from src.typing.config import AppConfig


@dataclass
class BackendRuntime:
    config: AppConfig
    run_id: str
    task: TaskSpec
    genner: Genner
    docker_client: docker.DockerClient
    metrics: MetricsCollector | None = None


def _run_backend_smoke_test(backend_session: Any) -> None:
    smoke_test = getattr(backend_session, "smoke_test", None)
    if not callable(smoke_test):
        return
    logger.info("Running quick inference test...")
    logger.info(f"Inference test response: {smoke_test()}")


def _configure_metrics_url(
    metrics_collector: MetricsCollector | None,
    backend_session: Any,
) -> None:
    if metrics_collector is None:
        return

    metrics_url = getattr(backend_session, "metrics_url", None)
    backend = getattr(getattr(backend_session, "genner", None), "identifier", "vllm")
    if backend not in {"vllm", "sglang"}:
        backend = "vllm"
    metrics_collector.inference_metrics_url = metrics_url
    metrics_collector.inference_metrics_backend = backend
    metrics_collector.vllm_metrics_url = metrics_url
    if metrics_collector.inference_metrics_url:
        logger.info(
            f"{backend} metrics scraping enabled: {metrics_collector.inference_metrics_url}"
        )
        return

    logger.info("Inference metrics scraping disabled: backend exposed no metrics_url")


def _wrap_genner(
    backend_session: Any,
    metrics_collector: MetricsCollector | None,
) -> Genner:
    genner = backend_session.genner
    if metrics_collector is None:
        return genner
    return MetricsGenner(genner, metrics_collector)


def _resolve_tool_contract_profile(config: AppConfig) -> Any | None:
    harness = config.harness
    container = harness.container
    if harness.name != "container" or container is None:
        return None
    profile = resolve_profile(container.profile, container.profile_config)
    if profile.tool_call_contract_probe() is None:
        return None
    return profile


def _coerce_runtime(
    *,
    config: AppConfig,
    run_id: str,
    task: TaskSpec,
    genner: Genner,
    docker_client: docker.DockerClient,
    metrics: MetricsCollector | None,
) -> BackendRuntime:
    return BackendRuntime(
        config=config,
        run_id=run_id,
        task=task,
        genner=genner,
        docker_client=docker_client,
        metrics=metrics,
    )


@contextmanager
def open_backend_runtime(
    config: AppConfig,
    *,
    run_id: str | None = None,
    docker_client: docker.DockerClient | None = None,
    task: TaskSpec | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> Iterator[BackendRuntime]:
    active_run_id = run_id or generate_readable_run_id()
    active_docker_client = (
        docker_client if docker_client is not None else docker.from_env()
    )
    active_task = (
        task if task is not None else load_task(config.task.class_, config.task.config)
    )
    active_metrics = (
        metrics_collector
        if metrics_collector is not None
        else MetricsCollector.from_config(config, active_run_id)
    )
    emit_static_tool_contract_warnings(config)
    tool_contract_profile = _resolve_tool_contract_profile(config)

    backend_context = resolve_backend_context(config)
    if backend_context is None:
        raise RuntimeError(f"No backend context registered for {config.model_name}")

    with ExitStack() as backend_stack:
        backend_session = backend_stack.enter_context(backend_context)
        _run_backend_smoke_test(backend_session)
        _configure_metrics_url(active_metrics, backend_session)
        genner = _wrap_genner(backend_session, active_metrics)
        try:
            if tool_contract_profile is not None:
                run_tool_call_contract_probe(
                    config=config,
                    profile=tool_contract_profile,
                    genner=genner,
                    backend_session=backend_session,
                    run_id=active_run_id,
                )
            runtime = _coerce_runtime(
                config=config,
                run_id=active_run_id,
                task=active_task,
                genner=genner,
                docker_client=active_docker_client,
                metrics=active_metrics,
            )
            yield runtime
        finally:
            if active_metrics is not None:
                active_metrics.flush()


__all__ = ["BackendRuntime", "open_backend_runtime"]
