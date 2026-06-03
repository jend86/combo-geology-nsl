"""min_features must be wired from task config into the variation so the
survey->crossbreed transition (and greedy BIC init) can be gated on a minimum
admitted-layer count, not only on source coverage.

Before this, `min_features` was hardcoded at the dataclass default (0), so the
pipeline flipped to crossbreed as soon as all sources were visited — with as few
as ~4 layers, and greedy initialized crossbreed's foundation from that tiny pool.
"""

from __future__ import annotations

from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _task(tmp_path: Path, **extra) -> FeatureHypothesisKazakhstanTask:
    cfg = {
        "store_dir": str(tmp_path / "store"),
        "kg_dir": str(tmp_path / "kg"),
        "dataset_dir": str(tmp_path / "data"),
    }
    cfg.update(extra)
    return FeatureHypothesisKazakhstanTask(cfg)


def test_min_features_defaults_to_zero(tmp_path: Path) -> None:
    task = _task(tmp_path)
    assert task.list_variations()[0].min_features == 0


def test_min_features_wired_from_config(tmp_path: Path) -> None:
    task = _task(tmp_path, min_features=9)
    assert task.list_variations()[0].min_features == 9
