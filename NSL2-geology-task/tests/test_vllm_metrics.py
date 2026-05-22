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
"""


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



if __name__ == "__main__":
    unittest.main()
