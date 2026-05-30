from __future__ import annotations

import unittest

from src.observability.vllm_metrics import (
    InferenceMetricsSnapshot,
    inference_metrics_delta,
)


def _snap(scraped_at: float, **kw) -> InferenceMetricsSnapshot:
    return InferenceMetricsSnapshot(scraped_at=scraped_at, **kw)


class TestInferenceMetricsDelta(unittest.TestCase):
    def test_counter_deltas_over_window(self) -> None:
        prev = _snap(
            100.0,
            total_preemptions=21,
            total_prompt_tokens=1000,
            total_generation_tokens=200,
        )
        curr = _snap(
            160.0,
            total_preemptions=26,
            total_prompt_tokens=4000,
            total_generation_tokens=800,
        )
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.window_seconds, 60.0)
        self.assertEqual(delta.preemptions, 5)
        self.assertEqual(delta.prompt_tokens, 3000)
        self.assertEqual(delta.generation_tokens, 600)

    def test_token_throughput_rates(self) -> None:
        prev = _snap(0.0, total_prompt_tokens=0, total_generation_tokens=0)
        curr = _snap(10.0, total_prompt_tokens=5000, total_generation_tokens=500)
        delta = inference_metrics_delta(prev, curr)
        self.assertAlmostEqual(delta.prompt_tokens_per_second, 500.0)
        self.assertAlmostEqual(delta.generation_tokens_per_second, 50.0)

    def test_prefix_cache_hit_rate(self) -> None:
        prev = _snap(0.0, prefix_cache_hits=100, prefix_cache_queries=1000)
        curr = _snap(10.0, prefix_cache_hits=900, prefix_cache_queries=2000)
        delta = inference_metrics_delta(prev, curr)
        # window hits=800, queries=1000 -> 0.8 (NOT cumulative 900/2000=0.45)
        self.assertEqual(delta.prefix_cache_hits, 800)
        self.assertEqual(delta.prefix_cache_queries, 1000)
        self.assertAlmostEqual(delta.prefix_cache_hit_rate, 0.8)

    def test_external_prefix_cache_hit_rate(self) -> None:
        prev = _snap(0.0, external_prefix_cache_hits=0, external_prefix_cache_queries=0)
        curr = _snap(
            10.0, external_prefix_cache_hits=30, external_prefix_cache_queries=60
        )
        delta = inference_metrics_delta(prev, curr)
        self.assertAlmostEqual(delta.external_prefix_cache_hit_rate, 0.5)

    def test_prompt_tokens_cached_rate(self) -> None:
        prev = _snap(0.0, total_prompt_tokens=0, prompt_tokens_cached=0)
        curr = _snap(10.0, total_prompt_tokens=1000, prompt_tokens_cached=750)
        delta = inference_metrics_delta(prev, curr)
        self.assertAlmostEqual(delta.prompt_tokens_cached_rate, 0.75)

    def test_zero_window_yields_no_rates(self) -> None:
        prev = _snap(50.0, total_prompt_tokens=0)
        curr = _snap(50.0, total_prompt_tokens=1000)
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.window_seconds, 0.0)
        self.assertIsNone(delta.prompt_tokens_per_second)

    def test_zero_queries_yields_no_hit_rate(self) -> None:
        prev = _snap(0.0, prefix_cache_hits=10, prefix_cache_queries=100)
        curr = _snap(10.0, prefix_cache_hits=10, prefix_cache_queries=100)
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.prefix_cache_queries, 0)
        self.assertIsNone(delta.prefix_cache_hit_rate)

    def test_counter_reset_yields_none(self) -> None:
        # Server restart: counters reset to a smaller value -> negative raw delta.
        prev = _snap(0.0, total_preemptions=100, total_prompt_tokens=9999)
        curr = _snap(10.0, total_preemptions=3, total_prompt_tokens=5)
        delta = inference_metrics_delta(prev, curr)
        self.assertIsNone(delta.preemptions)
        self.assertIsNone(delta.prompt_tokens)

    def test_missing_fields_yield_none(self) -> None:
        prev = _snap(0.0)
        curr = _snap(10.0, total_preemptions=5)
        delta = inference_metrics_delta(prev, curr)
        self.assertIsNone(delta.preemptions)  # prev side missing
        self.assertIsNone(delta.prefix_cache_hit_rate)

    def test_gauges_taken_from_current_snapshot(self) -> None:
        prev = _snap(0.0, kv_cache_usage_pct=10.0, num_requests_running=1)
        curr = _snap(
            10.0,
            kv_cache_usage_pct=88.0,
            num_requests_running=8,
            num_requests_waiting=3,
        )
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.kv_cache_usage_pct, 88.0)
        self.assertEqual(delta.num_requests_running, 8)
        self.assertEqual(delta.num_requests_waiting, 3)


if __name__ == "__main__":
    unittest.main()
