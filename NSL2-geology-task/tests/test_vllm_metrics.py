from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.observability.vllm_metrics import snapshot_vllm_metrics

SAMPLE_METRICS_RESPONSE = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="test"} 0.42
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="test"} 2
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="test"} 1
# HELP vllm:prompt_tokens_total Prefill tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="test"} 12345
# HELP vllm:generation_tokens_total Generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="test"} 6789
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="test"} 21
# The _created sibling is a unix-timestamp gauge that must NOT be mistaken for the counter value.
# TYPE vllm:num_preemptions_created gauge
vllm:num_preemptions_created{model_name="test"} 1780140784.98
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="test"} 800
# TYPE vllm:prefix_cache_hits_created gauge
vllm:prefix_cache_hits_created{model_name="test"} 1780140784.98
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="test"} 1000
# TYPE vllm:external_prefix_cache_hits_total counter
vllm:external_prefix_cache_hits_total{model_name="test"} 10
# TYPE vllm:external_prefix_cache_queries_total counter
vllm:external_prefix_cache_queries_total{model_name="test"} 40
# TYPE vllm:prompt_tokens_cached_total counter
vllm:prompt_tokens_cached_total{model_name="test"} 5000
"""

# A build/version that exposes counters WITHOUT the ``_total`` suffix (bare form,
# per current vLLM docs). The scraper must fall back to the bare name.
BARE_COUNTER_RESPONSE = """\
# TYPE vllm:num_preemptions counter
vllm:num_preemptions{model_name="test"} 7
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{model_name="test"} 3
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{model_name="test"} 9
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens{model_name="test"} 100
# TYPE vllm:generation_tokens counter
vllm:generation_tokens{model_name="test"} 50
"""


def _scrape(text: str):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = text
    mock_response.raise_for_status = MagicMock()
    with patch(
        "src.observability.vllm_metrics.requests.get", return_value=mock_response
    ):
        return snapshot_vllm_metrics("http://localhost:8000/metrics")


class TestSnapshotVllmMetrics(unittest.TestCase):
    def test_extracts_gauges(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_METRICS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.observability.vllm_metrics.requests.get", return_value=mock_response):
            snapshot = snapshot_vllm_metrics("http://localhost:8000/metrics")

        self.assertIsNotNone(snapshot)
        self.assertAlmostEqual(snapshot.kv_cache_usage_pct, 42.0)
        self.assertEqual(snapshot.num_requests_running, 2)
        self.assertEqual(snapshot.num_requests_waiting, 1)

    def test_extracts_counters(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_METRICS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.observability.vllm_metrics.requests.get", return_value=mock_response):
            snapshot = snapshot_vllm_metrics("http://localhost:8000/metrics")

        self.assertEqual(snapshot.total_prompt_tokens, 12345)
        self.assertEqual(snapshot.total_generation_tokens, 6789)

    def test_returns_none_on_connection_error(self) -> None:
        with patch("src.observability.vllm_metrics.requests.get", side_effect=ConnectionError):
            snapshot = snapshot_vllm_metrics("http://localhost:8000/metrics")

        self.assertIsNone(snapshot)

    def test_returns_none_on_timeout(self) -> None:
        import requests

        with patch("src.observability.vllm_metrics.requests.get", side_effect=requests.Timeout):
            snapshot = snapshot_vllm_metrics("http://localhost:8000/metrics")

        self.assertIsNone(snapshot)

    def test_handles_missing_metrics_gracefully(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Only comments\n"
        mock_response.raise_for_status = MagicMock()

        with patch("src.observability.vllm_metrics.requests.get", return_value=mock_response):
            snapshot = snapshot_vllm_metrics("http://localhost:8000/metrics")

        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot.kv_cache_usage_pct)
        self.assertIsNone(snapshot.num_requests_running)

    def test_extracts_preemption_and_prefix_cache_counters(self) -> None:
        snapshot = _scrape(SAMPLE_METRICS_RESPONSE)
        self.assertEqual(snapshot.total_preemptions, 21)
        self.assertEqual(snapshot.prefix_cache_hits, 800)
        self.assertEqual(snapshot.prefix_cache_queries, 1000)
        self.assertEqual(snapshot.external_prefix_cache_hits, 10)
        self.assertEqual(snapshot.external_prefix_cache_queries, 40)
        self.assertEqual(snapshot.prompt_tokens_cached, 5000)

    def test_ignores_created_timestamp_siblings(self) -> None:
        # The ``_created`` lines hold a ~1.78e9 unix timestamp; picking them up
        # instead of the ``_total`` counter would be a silent, catastrophic bug.
        snapshot = _scrape(SAMPLE_METRICS_RESPONSE)
        self.assertEqual(snapshot.total_preemptions, 21)
        self.assertLess(snapshot.total_preemptions, 1000)
        self.assertEqual(snapshot.prefix_cache_hits, 800)

    def test_falls_back_to_bare_counter_names(self) -> None:
        # A build that omits the ``_total`` suffix must still be scraped.
        snapshot = _scrape(BARE_COUNTER_RESPONSE)
        self.assertEqual(snapshot.total_preemptions, 7)
        self.assertEqual(snapshot.prefix_cache_hits, 3)
        self.assertEqual(snapshot.prefix_cache_queries, 9)
        self.assertEqual(snapshot.total_prompt_tokens, 100)
        self.assertEqual(snapshot.total_generation_tokens, 50)

    def test_new_counters_default_to_none_when_absent(self) -> None:
        snapshot = _scrape(
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            'vllm:kv_cache_usage_perc{model_name="test"} 0.5\n'
        )
        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot.total_preemptions)
        self.assertIsNone(snapshot.prefix_cache_hits)
        self.assertIsNone(snapshot.external_prefix_cache_queries)
        self.assertIsNone(snapshot.prompt_tokens_cached)


if __name__ == "__main__":
    unittest.main()
