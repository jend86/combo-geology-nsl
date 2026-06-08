"""TDD: crossbreed queue ordering must rank parents by the INTENSIVE per-sample
BIC (``bic_delta_per_sample_mean``), not the extensive raw-Σ ``bic_delta``.

Background: commit 629a2c4 made ``experiments.jsonl``'s ``bic_delta`` the raw Σ
(extensive — it scales with ``n_holdout_rows``). The queue score is

    log1p(|bic_A|) + log1p(|bic_B|) + λ·distance        (λ = _PAIR_DISTANCE_WEIGHT = 2.0)

``log1p`` and ``λ`` were tuned when ``bic_delta`` was the per-sample MEAN
(|bic|≈single digits → log1p≈1-2, comparable to the distance term). Under raw-Σ
the per-parent term jumped to ~8-11, which (a) drowns the diversity term and
(b) biases ordering toward big (many-voxel) layers over small high-quality ones
— size, not quality. These tests pin the queue to the per-voxel quality term;
the raw-Σ ``bic_delta`` remains the admission GATE (sign only), unaffected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "dataset_dir": str(tmp_path / "dataset"),
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "knowledge"),
            "artifact_dir": str(tmp_path / "artifacts"),
        }
    )


def _seed(kg_dir: Path, rows: list[dict]) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _normal_parent(node_id: str, *, bic_raw: float, bic_per_sample: float) -> dict:
    """A normal (predictor-lift) crossbreed-eligible admit. Admitted layers have
    bic_delta < 0; the queue uses |bic| so sign is irrelevant to ordering."""
    return {
        "node_id": node_id,
        "layer_name": f"layer_{node_id}",
        "admission_path": "normal",
        "crossbreed_parent_eligible": True,
        "bic_delta": bic_raw,
        "bic_delta_raw": bic_raw,
        "bic_delta_per_sample_mean": bic_per_sample,
        "mutual_info": {},
    }


def _scores(entries: list[dict]) -> dict[tuple[str, str], float]:
    return {tuple(e["parents"]): float(e["score"]) for e in entries}


def test_queue_ranks_by_per_sample_bic_not_raw_sigma(tmp_path: Path) -> None:
    """A small, high-per-sample-quality layer must outrank a big layer whose
    raw-Σ |bic| is huge ONLY because it has many holdout rows.

    big:   |raw|=80000  but |per_sample|=0.05  (large, mediocre per voxel)
    small: |raw|= 2000  but |per_sample|=3.0   (small,  excellent per voxel)
    base:  neutral third parent so the two candidates form comparable pairs.

    Under raw-Σ ordering (the bug) (big,base) wins; under per-sample it loses.
    """
    kg = tmp_path / "knowledge" / "teniz_basin"
    _seed(kg, [
        _normal_parent("big", bic_raw=-80000.0, bic_per_sample=-0.05),
        _normal_parent("small", bic_raw=-2000.0, bic_per_sample=-3.0),
        _normal_parent("base", bic_raw=-1000.0, bic_per_sample=-0.5),
    ])
    # No pairwise_distance.jsonl → all distances "unknown" (0 contribution),
    # so pairs differ ONLY by the per-parent quality term.

    entries = _task(tmp_path)._enumerate_pairs(kg)
    scores = _scores(entries)

    # The small-but-excellent parent's pairings beat the big-but-mediocre ones.
    assert scores[("small", "base")] > scores[("big", "base")]
    assert scores[("base", "small")] > scores[("base", "big")]

    # The single top-ranked pair (entries are sorted score-desc) involves 'small',
    # never a pure big-layer pairing.
    top = max(entries, key=lambda e: e["score"])
    assert "small" in top["parents"]


def test_founders_still_neutralized_to_distance_only(tmp_path: Path) -> None:
    """Seed founders (diverse_seed / first_layer_auto) keep bic neutralized to 0
    regardless of which BIC field is read — they rank by diversity only."""
    kg = tmp_path / "knowledge" / "teniz_basin"
    founder = {
        "node_id": "seed",
        "layer_name": "layer_seed",
        "admission_path": "diverse_seed",
        "crossbreed_parent_eligible": True,
        "bic_delta": -999999.0,            # inflated small-pool magnitude
        "bic_delta_per_sample_mean": -88.0,
        "mutual_info": {},
    }
    normal = _normal_parent("norm", bic_raw=-2000.0, bic_per_sample=-1.0)
    _seed(kg, [founder, normal])

    entries = _task(tmp_path)._enumerate_pairs(kg)
    # With no distance index the only term is the normal parent's log1p(|per_sample|);
    # the founder contributes 0 (neutralized), so the score is finite and small,
    # NOT dominated by the founder's huge raw |bic|.
    expected = math.log1p(1.0)  # log1p(|per_sample_norm|) + log1p(0)
    assert entries, "expected enumerated founder×normal pairs"
    for e in entries:
        assert e["score"] == pytest.approx(expected)
