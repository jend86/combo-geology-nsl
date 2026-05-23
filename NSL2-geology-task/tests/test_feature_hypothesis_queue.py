"""Crossbreed pair-queue tests for FeatureHypothesisTask.

Today `_get_crossbreed_context` picks the single best `(parent_a, parent_b)`
pair and hands it to *every* concurrent slot, which guarantees duplicates.
Approach A in the design doc replaces that with a file-locked ordered-pair
queue: every concurrent slot pops a distinct pair. `(A, B)` and `(B, A)` are
distinct entries (parent_episodes is not commutative).

These tests cover the queue primitive itself; integration with `populate()`
is covered in `test_feature_hypothesis_parent_persist.py`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from tasks.feature_hypothesis import FeatureHypothesisTask, FeatureHypothesisVariation


def _task(tmp_path: Path) -> FeatureHypothesisTask:
    return FeatureHypothesisTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
        }
    )


def _variation(tmp_path: Path) -> FeatureHypothesisVariation:
    return FeatureHypothesisVariation(
        name="coe_fairbairn",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "coe_fairbairn"),
        kg_dir=str(tmp_path / "kg" / "coe_fairbairn"),
    )


def _seed_experiments(kg_dir: Path, node_ids: list[str]) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("w") as fh:
        for i, node_id in enumerate(node_ids):
            fh.write(json.dumps({
                "node_id": node_id,
                "hypothesis": f"hyp_{node_id}",
                "bic_delta": -(10.0 + i),  # all successful (bic_delta < 0)
                "layer_name": f"layer_{node_id}",
                "mutual_info": {},
            }) + "\n")


class TestQueueEnumeration:
    def test_n_experiments_yield_n_times_n_minus_1_pairs(self, tmp_path: Path) -> None:
        # 4 experiments → 4*3 = 12 ordered pairs (no self-pairs).
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b", "c", "d"])

        task._queue_refill(kg_dir)

        entries = []
        with (kg_dir / "crossbreed_queue.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        assert len(entries) == 12
        for entry in entries:
            assert entry["parents"][0] != entry["parents"][1]

    def test_ab_and_ba_are_distinct_entries(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b"])

        task._queue_refill(kg_dir)

        entries = []
        with (kg_dir / "crossbreed_queue.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        parent_tuples = {tuple(entry["parents"]) for entry in entries}
        assert parent_tuples == {("a", "b"), ("b", "a")}


class TestQueuePop:
    def test_sequential_pops_yield_distinct_pairs(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b", "c"])

        # 3 experiments → 6 ordered pairs.
        popped: set[tuple[str, str]] = set()
        for _ in range(6):
            pair = task._queue_pop_pair(kg_dir)
            assert pair is not None
            assert pair not in popped, f"pair {pair} repeated before refill"
            popped.add(pair)

        assert len(popped) == 6

    def test_refill_after_exhaustion(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b"])

        # 2 experiments → 2 ordered pairs.
        first = task._queue_pop_pair(kg_dir)
        second = task._queue_pop_pair(kg_dir)
        assert {first, second} == {("a", "b"), ("b", "a")}

        # Third pop must refill and yield a pair again (round-robin).
        third = task._queue_pop_pair(kg_dir)
        assert third in {("a", "b"), ("b", "a")}

    def test_empty_experiments_returns_none(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        kg_dir.mkdir(parents=True, exist_ok=True)
        # No experiments.jsonl at all.
        assert task._queue_pop_pair(kg_dir) is None

    def test_single_experiment_returns_none(self, tmp_path: Path) -> None:
        # Need at least 2 admitted experiments to form a pair.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["solo"])

        assert task._queue_pop_pair(kg_dir) is None


class TestQueueConcurrency:
    def test_concurrent_pops_observe_distinct_pairs(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b", "c", "d"])

        # 4 experiments → 12 ordered pairs. Have 6 threads pop simultaneously.
        results: list[tuple[str, str] | None] = []
        results_lock = threading.Lock()

        def popper() -> None:
            pair = task._queue_pop_pair(kg_dir)
            with results_lock:
                results.append(pair)

        threads = [threading.Thread(target=popper) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()

        observed = [p for p in results if p is not None]
        assert len(observed) == 6, "all 6 threads should have popped a pair"
        assert len(set(observed)) == 6, "no two threads should have popped the same pair"


class TestQueueScoring:
    def test_higher_score_pops_first(self, tmp_path: Path) -> None:
        # The queue should be ordered by (combined_bic - mi_score) desc, same
        # as the current `_get_crossbreed_context` ranking.
        # exp_h has the strongest bic; the first pop should include it.
        kg_dir = Path(_variation(tmp_path).kg_dir)
        kg_dir.mkdir(parents=True, exist_ok=True)
        with (kg_dir / "experiments.jsonl").open("w") as fh:
            fh.write(json.dumps({
                "node_id": "exp_low",
                "hypothesis": "low",
                "bic_delta": -1.0,
                "layer_name": "l_low",
                "mutual_info": {},
            }) + "\n")
            fh.write(json.dumps({
                "node_id": "exp_h",
                "hypothesis": "high",
                "bic_delta": -100.0,
                "layer_name": "l_high",
                "mutual_info": {},
            }) + "\n")
            fh.write(json.dumps({
                "node_id": "exp_mid",
                "hypothesis": "mid",
                "bic_delta": -50.0,
                "layer_name": "l_mid",
                "mutual_info": {},
            }) + "\n")

        task = _task(tmp_path)
        pair = task._queue_pop_pair(kg_dir)
        assert pair is not None
        # The best-scoring pair pairs the two largest |bic_delta| values:
        # exp_h (100) + exp_mid (50).
        assert set(pair) == {"exp_h", "exp_mid"}
