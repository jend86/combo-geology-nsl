"""File rotation: least-explored-first source assignment (_pick_assigned_source).

Each survey episode is assigned the least-explored source from a rotation list;
the running counts are persisted to ``{kg_dir}/file_rotation_state.json`` so
coverage spreads evenly across the dataset over many episodes.
"""

import json
from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import (
    FeatureHypothesisKazakhstanTask,
    _KAZAKHSTAN_SOURCE_FILES,
)


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask({
        "store_dir": str(tmp_path / "store_root"),
        "kg_dir": str(tmp_path / "kg_root"),
    })


SOURCES = [{"key": "a", "path": "a"}, {"key": "b", "path": "b"}, {"key": "c", "path": "c"}]


class TestPickAssignedSource:
    def test_first_pick_is_first_entry_and_persists(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg = str(tmp_path / "kg")
        out = task._pick_assigned_source(kg, SOURCES)
        assert out["source"]["key"] == "a"
        assert out["all_counts"]["a"] == 1
        state = json.loads((Path(kg) / "file_rotation_state.json").read_text())
        assert state["counts"]["a"] == 1

    def test_rotates_to_least_explored_then_wraps(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg = str(tmp_path / "kg")
        keys = [task._pick_assigned_source(kg, SOURCES)["source"]["key"] for _ in range(3)]
        assert keys == ["a", "b", "c"]  # least-explored each round, ties by list order
        # fourth pick wraps back to "a" (all entries now at count 1)
        assert task._pick_assigned_source(kg, SOURCES)["source"]["key"] == "a"

    def test_ties_broken_by_list_order(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg = Path(tmp_path / "kg")
        kg.mkdir(parents=True, exist_ok=True)
        # b is least-explored → must be picked even though a/c appear around it
        (kg / "file_rotation_state.json").write_text(
            json.dumps({"counts": {"a": 2, "b": 1, "c": 2}})
        )
        assert task._pick_assigned_source(str(kg), SOURCES)["source"]["key"] == "b"

    def test_corrupt_state_treated_as_empty(self, tmp_path: Path) -> None:
        task = _task(tmp_path)
        kg = Path(tmp_path / "kg")
        kg.mkdir(parents=True, exist_ok=True)
        (kg / "file_rotation_state.json").write_text("{ not valid json")
        out = task._pick_assigned_source(str(kg), SOURCES)
        assert out["source"]["key"] == "a"  # falls back to all-zero counts

    def test_real_source_list_18_unique_entries(self) -> None:
        keys = [s["key"] for s in _KAZAKHSTAN_SOURCE_FILES]
        assert len(keys) == 18
        assert len(set(keys)) == 18
