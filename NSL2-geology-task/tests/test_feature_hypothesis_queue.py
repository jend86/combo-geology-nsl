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


def _seed_crossbreed_child(
    kg_dir: Path,
    *,
    child_id: str,
    parent_1: str,
    parent_2: str,
    bic_delta: float = -5.0,
) -> None:
    """Append an admitted crossbreed experiment whose parents form a pair.

    The "consummated" check should match an unordered {parent_1, parent_2}
    set against queue pair entries — i.e. (parent_1, parent_2) and
    (parent_2, parent_1) both become consummated.
    """
    with (kg_dir / "experiments.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "node_id": child_id,
            "hypothesis": f"crossbreed_{child_id}",
            "bic_delta": bic_delta,
            "layer_name": f"layer_{child_id}",
            "parent_node_1": parent_1,
            "parent_node_2": parent_2,
            "mutual_info": {},
        }) + "\n")


def _read_queue_entries(kg_dir: Path) -> list[dict]:
    entries: list[dict] = []
    with (kg_dir / "crossbreed_queue.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class TestQueueAttemptDecay:
    def test_attempt_count_increments_and_persists_per_pop(
        self, tmp_path: Path
    ) -> None:
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b"])

        # Each pop bumps exactly one entry's attempt_count. With two equal-
        # scoring entries the decay logic alternates which one fires, so
        # the invariant we check is the *total* attempt_count.
        for n_pops in range(1, 4):
            pair = task._queue_pop_pair(kg_dir)
            assert pair is not None
            entries = _read_queue_entries(kg_dir)
            total_attempts = sum(int(e.get("attempt_count", 0)) for e in entries)
            assert total_attempts == n_pops, (
                f"after {n_pops} pop(s), total attempt_count is {total_attempts}"
            )
            chosen = next(
                e for e in entries if e["pair_id"] == f"{pair[0]}->{pair[1]}"
            )
            assert chosen.get("attempt_count", 0) >= 1

    def test_decay_demotes_repeatedly_tried_pair(self, tmp_path: Path) -> None:
        # Two pairs with very different raw scores. After enough attempts on
        # the high-score pair, the low-score pair should eventually win the
        # next pop because score / (1 + alpha * attempts) decays.
        kg_dir = Path(_variation(tmp_path).kg_dir)
        kg_dir.mkdir(parents=True, exist_ok=True)
        # Three nodes so we have multiple unordered pairs, but rig bic_delta
        # so one combination dominates: {h, m} scores |100|+|50|=150;
        # {h, l} scores 101; {m, l} scores 51.
        with (kg_dir / "experiments.jsonl").open("w") as fh:
            for node_id, bic in (("h", -100.0), ("m", -50.0), ("l", -1.0)):
                fh.write(json.dumps({
                    "node_id": node_id,
                    "hypothesis": f"hyp_{node_id}",
                    "bic_delta": bic,
                    "layer_name": f"layer_{node_id}",
                    "mutual_info": {},
                }) + "\n")

        task = _task(tmp_path)
        observed: list[frozenset[str]] = []
        # Six ordered pairs; pop many times so decay has a chance to flip the
        # ordering. After every pair has been touched once, score-decay must
        # eventually cause the (h, l)=101 or (m, l)=51 pairs to overtake the
        # decayed (h, m)=150 pair.
        for _ in range(20):
            pair = task._queue_pop_pair(kg_dir)
            assert pair is not None
            observed.append(frozenset(pair))

        # The top combination shouldn't monopolise — across 20 pops we expect
        # to see every unordered combination at least once.
        assert frozenset({"h", "m"}) in observed
        assert frozenset({"h", "l"}) in observed
        assert frozenset({"m", "l"}) in observed

    def test_existing_queue_without_attempt_count_defaults_to_zero(
        self, tmp_path: Path
    ) -> None:
        # Simulate an existing on-disk queue written by the older version
        # that has no attempt_count field; pop should still work and treat
        # missing as 0.
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b"])
        # Write a queue file by hand without attempt_count.
        with (kg_dir / "crossbreed_queue.jsonl").open("w") as fh:
            fh.write(json.dumps({
                "pair_id": "a->b",
                "parents": ["a", "b"],
                "score": 21.0,
                "popped_at": None,
            }) + "\n")
            fh.write(json.dumps({
                "pair_id": "b->a",
                "parents": ["b", "a"],
                "score": 21.0,
                "popped_at": None,
            }) + "\n")
        # Touch experiments file so the queue is considered current.
        (kg_dir / "crossbreed_queue.jsonl").touch()

        task_inst = _task(tmp_path)
        result = task_inst._queue_pop_pair(kg_dir)
        assert result is not None
        entries = _read_queue_entries(kg_dir)
        chosen = next(
            e for e in entries if e["pair_id"] == f"{result[0]}->{result[1]}"
        )
        assert chosen.get("attempt_count") == 1


class TestQueueConsummatedTier:
    def test_consummated_pair_yields_to_unconsummated(
        self, tmp_path: Path
    ) -> None:
        # 3 experiments → 6 ordered pairs. Mark one combination consummated
        # via an admitted crossbreed child. Even though (a, b) has the highest
        # score, the first pop should land on a different (unconsummated) pair.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b", "c"])
        # a (bic 10), b (bic 11), c (bic 12) → top pair is (b, c)|(c, b) at 23.
        # Consummate the (b, c) combination via an admitted child.
        _seed_crossbreed_child(
            kg_dir, child_id="child_bc", parent_1="b", parent_2="c"
        )

        first = task._queue_pop_pair(kg_dir)
        assert first is not None
        assert set(first) != {"b", "c"}, (
            f"consummated pair popped first; got {first}"
        )

    def test_consummated_match_is_unordered(self, tmp_path: Path) -> None:
        # An admitted crossbreed whose parents were recorded as (b, a) must
        # consummate both (a, b) and (b, a) queue entries. To distinguish
        # the new logic from the existing round-robin, rig the scores so
        # that (a, b) is the highest-scoring pair — under the old code it
        # would fire on pop 1; under the new design it goes to the slow
        # lane and pop 1 picks a different (unconsummated) pair.
        kg_dir = Path(_variation(tmp_path).kg_dir)
        kg_dir.mkdir(parents=True, exist_ok=True)
        with (kg_dir / "experiments.jsonl").open("w") as fh:
            for node_id, bic in (("a", -100.0), ("b", -50.0), ("c", -10.0)):
                fh.write(json.dumps({
                    "node_id": node_id,
                    "hypothesis": f"hyp_{node_id}",
                    "bic_delta": bic,
                    "layer_name": f"layer_{node_id}",
                    "mutual_info": {},
                }) + "\n")
        # Consummate {a, b} via an admitted child whose parents were stored
        # in (b, a) order — verifies unordered matching.
        _seed_crossbreed_child(
            kg_dir, child_id="child_ba", parent_1="b", parent_2="a",
            bic_delta=-1.0,
        )

        task = _task(tmp_path)
        # The top two ordered pair scores are (a,b)/(b,a) at 150 (consummated).
        # The next-highest unconsummated pair is (a, c)/(c, a) at 110.
        # Pops 1 and 2 must avoid the {a, b} set.
        for _ in range(2):
            pair = task._queue_pop_pair(kg_dir)
            assert pair is not None
            assert set(pair) != {"a", "b"}, (
                f"consummated unordered pair fired in fast lane: {pair}"
            )

    def test_consummated_pair_eventually_pops(self, tmp_path: Path) -> None:
        # "Slow lane, not banned lane." Consummated pairs sit at a discounted
        # effective score (β multiplier). Unconsummated pairs decay with
        # attempts; once their decayed score falls below the consummated
        # tier's, the consummated pair must fire. This guards against the
        # bug where consummated == permanently masked.
        task = _task(tmp_path)
        kg_dir = Path(_variation(tmp_path).kg_dir)
        _seed_experiments(kg_dir, ["a", "b"])
        _seed_crossbreed_child(
            kg_dir, child_id="ab", parent_1="a", parent_2="b"
        )

        observed: set[frozenset[str]] = set()
        for _ in range(80):
            pair = task._queue_pop_pair(kg_dir)
            assert pair is not None
            observed.add(frozenset(pair))
            if frozenset({"a", "b"}) in observed:
                break

        assert frozenset({"a", "b"}) in observed, (
            "consummated pair never popped — slow lane is banned, not slow"
        )
