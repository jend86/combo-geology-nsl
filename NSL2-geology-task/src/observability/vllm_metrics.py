"""Scrape inference-server Prometheus ``/metrics`` endpoints.

Focuses on gauges (KV cache, queue depth) and counters (token totals).
Histogram extraction (TTFT, e2e latency) is deferred — those require
maintaining state between scrapes to compute per-request deltas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests
from loguru import logger

from src.observability.prometheus import parse_prometheus_text


@dataclass
class InferenceMetricsSnapshot:
    scraped_at: float  # time.time() — for time-series correlation
    kv_cache_usage_pct: Optional[float] = None
    num_requests_running: Optional[int] = None
    num_requests_waiting: Optional[int] = None
    # Counter-based (cumulative, useful for deltas between scrapes)
    total_prompt_tokens: Optional[int] = None
    total_generation_tokens: Optional[int] = None


VllmMetricsSnapshot = InferenceMetricsSnapshot


_METRIC_NAMES: dict[str, dict[str, str]] = {
    "vllm": {
        "kv_cache_usage": "vllm:kv_cache_usage_perc",
        "num_requests_running": "vllm:num_requests_running",
        "num_requests_waiting": "vllm:num_requests_waiting",
        "total_prompt_tokens": "vllm:prompt_tokens_total",
        "total_generation_tokens": "vllm:generation_tokens_total",
    },
    "sglang": {
        "kv_cache_usage": "sglang:token_usage",
        "num_requests_running": "sglang:num_running_reqs",
        "num_requests_waiting": "sglang:num_queue_reqs",
        "total_prompt_tokens": "sglang:prompt_tokens_total",
        "total_generation_tokens": "sglang:gen_throughput_total",
    },
}


def snapshot_inference_metrics(
    metrics_url: str,
    *,
    backend: Literal["vllm", "sglang"] = "vllm",
    timeout: float = 2.0,
    api_key: str | None = None,
) -> Optional[InferenceMetricsSnapshot]:
    """Scrape an inference ``/metrics`` endpoint, returning ``None`` on failure."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(metrics_url, timeout=timeout, headers=headers)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            logger.warning(
                f"{backend} /metrics returned {status} — check API key configuration"
            )
        else:
            logger.debug(f"{backend} /metrics scrape failed: {exc}")
        return None
    except Exception as exc:
        logger.debug(f"{backend} /metrics scrape failed: {exc}")
        return None

    parsed = parse_prometheus_text(response.text)
    scraped_at = time.time()
    names = _METRIC_NAMES[backend]

    return InferenceMetricsSnapshot(
        scraped_at=scraped_at,
        kv_cache_usage_pct=_first_gauge_pct(parsed, names["kv_cache_usage"]),
        num_requests_running=_first_gauge_int(parsed, names["num_requests_running"]),
        num_requests_waiting=_first_gauge_int(parsed, names["num_requests_waiting"]),
        total_prompt_tokens=_first_gauge_int(parsed, names["total_prompt_tokens"]),
        total_generation_tokens=_first_gauge_int(
            parsed,
            names["total_generation_tokens"],
        ),
    )


def snapshot_vllm_metrics(
    metrics_url: str,
    timeout: float = 2.0,
    api_key: str | None = None,
) -> Optional[VllmMetricsSnapshot]:
    """Backward-compatible vLLM metrics scraper wrapper."""

    return snapshot_inference_metrics(
        metrics_url,
        backend="vllm",
        timeout=timeout,
        api_key=api_key,
    )


def _first_gauge_pct(
    parsed: dict, metric_name: str
) -> Optional[float]:
    """Extract the first value for a metric and convert from fraction to percent."""
    entries = parsed.get(metric_name)
    if not entries:
        return None
    _, value = entries[0]
    return value * 100.0


def _first_gauge_int(
    parsed: dict, metric_name: str
) -> Optional[int]:
    """Extract the first value for a metric as an integer."""
    entries = parsed.get(metric_name)
    if not entries:
        return None
    _, value = entries[0]
    return int(value)
