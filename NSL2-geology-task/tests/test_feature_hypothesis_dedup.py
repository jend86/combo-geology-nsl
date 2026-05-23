"""Duplicate-handling tests for FeatureHypothesisTask.

These tests target Approach A of
``docs/design/feature_hypothesis_duplicate_handling_and_bootstrap_ramp.md``:

  - ``_fingerprint`` is pure, order-sensitive, and handles missing parents.
  - ``_admit_with_dedup`` admits the first occurrence and silently rejects the
    second, leaving ``experiments.jsonl`` clean of duplicates while still
    returning a "success" to the caller (= the episode keeps its reward).
"""

from __future__ import annotations

import json
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


def _kg_record(*, hypothesis: str, parents: list[str], node_id: str) -> dict:
    return {
        "node_id": node_id,
        "hypothesis": hypothesis,
        "parent_node_1": parents[0] if len(parents) > 0 else None,
        "parent_node_2": parents[1] if len(parents) > 1 else None,
        "bic_delta": -10.0,
        "layer_name": "demo_layer",
    }


def _admit(
    task: FeatureHypothesisTask,
    kg_dir: Path,
    record: dict,
) -> bool:
    parents = [
        p for p in (record.get("parent_node_1"), record.get("parent_node_2"))
        if isinstance(p, str) and p
    ]
    return task._admit_with_dedup(
        kg_dir, record, parents=parents, hypothesis=record.get("hypothesis", "")
    )


class TestFingerprint:
    def test_same_inputs_same_hash(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        fp1 = task._fingerprint(["a", "b"], "permian overlies ordovician")
        fp2 = task._fingerprint(["a", "b"], "permian overlies ordovician")
        assert fp1 == fp2

    def test_parent_order_matters(self, tmp_path: Path) -> None:
        # (A, B) and (B, A) must hash differently — see open-question 1 in the
        # design doc and the user's explicit "parent_episodes are NOT
        # commutative" note.
        task = _task(tmp_path)
        ab = task._fingerprint(["a", "b"], "h")
        ba = task._fingerprint(["b", "a"], "h")
        assert ab != ba

    def test_hypothesis_text_matters(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        a = task._fingerprint(["x", "y"], "hypothesis one")
        b = task._fingerprint(["x", "y"], "hypothesis two")
        assert a != b

    def test_empty_parents_falls_back_to_hypothesis(self, tmp_path: Path) -> None:
        # Survey-mode episodes have no parents; the fingerprint must still be
        # deterministic and dedup on hypothesis alone.
        task = _task(tmp_path)
        a = task._fingerprint([], "survey hypothesis")
        b = task._fingerprint(None, "survey hypothesis")
        c = task._fingerprint([], "survey hypothesis")
        assert a == c
        assert a == b

    def test_whitespace_normalized(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        a = task._fingerprint([], "  hello world  ")
        b = task._fingerprint([], "hello world")
        c = task._fingerprint([], "hello  world")  # collapsed inner whitespace
        assert a == b
        assert a == c


class TestAdmitWithDedup:
    def test_first_admission_writes_jsonl(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        record = _kg_record(hypothesis="h1", parents=["pa", "pb"], node_id="exp_1")

        admitted = _admit(task, kg_dir, record)

        assert admitted is True
        lines = (kg_dir / "experiments.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["node_id"] == "exp_1"

    def test_second_duplicate_rejected(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        record1 = _kg_record(hypothesis="h1", parents=["pa", "pb"], node_id="exp_1")
        record2 = _kg_record(hypothesis="h1", parents=["pa", "pb"], node_id="exp_2")

        first = _admit(task, kg_dir, record1)
        second = _admit(task, kg_dir, record2)

        assert first is True
        assert second is False
        lines = (kg_dir / "experiments.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1, "duplicate must not be appended to experiments.jsonl"
        assert json.loads(lines[0])["node_id"] == "exp_1"

    def test_distinct_parents_both_admitted(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        record1 = _kg_record(hypothesis="h", parents=["pa", "pb"], node_id="exp_1")
        record2 = _kg_record(hypothesis="h", parents=["pb", "pa"], node_id="exp_2")

        assert _admit(task, kg_dir, record1) is True
        assert _admit(task, kg_dir, record2) is True
        lines = (kg_dir / "experiments.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_ledger_persists_fingerprints(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        record = _kg_record(hypothesis="h", parents=["pa", "pb"], node_id="exp_1")

        _admit(task, kg_dir, record)

        ledger_path = kg_dir / "admitted_index.json"
        assert ledger_path.exists()
        ledger = json.loads(ledger_path.read_text())
        assert isinstance(ledger.get("fingerprints"), list)
        assert len(ledger["fingerprints"]) == 1
        assert ledger["fingerprints"][0] == task._fingerprint(["pa", "pb"], "h")

    def test_dedup_survives_process_restart(self, tmp_path: Path) -> None:
        # A fresh task instance must still see the prior fingerprint via the
        # on-disk ledger — i.e. the dedup state is not in-memory.
        task1 = _task(tmp_path)
        variation = _variation(tmp_path)
        kg_dir = Path(variation.kg_dir)
        record1 = _kg_record(hypothesis="h", parents=["pa", "pb"], node_id="exp_1")
        record2 = _kg_record(hypothesis="h", parents=["pa", "pb"], node_id="exp_2")

        assert _admit(task1, kg_dir, record1) is True

        task2 = _task(tmp_path)
        assert _admit(task2, kg_dir, record2) is False
