"""TDD: ``_update_pairwise_distance_index`` must compute REAL pairwise distances
from the persisted admitted-layer arrays — not the ``0.0`` sentinel.

Root cause (2026-06-04): ``evaluate['pairwise_distance']`` has no producer
anywhere in the repo, so every pair fell to the default in
``new_pairwise_distance.get(existing_layer, 0.0)``. ``_count_diverse_parents``
then reads ``0.0 < _NEAR_DUPLICATE_JACCARD_THRESHOLD`` as a near-duplicate and
collapses the whole pool to a single parent — permanently blocking crossbreed
even though the admitted layers are geometrically diverse (live KG median
pairwise distance ~0.999). These tests pin the producer: distances come from
``voxel_features.scoring.pairwise_distance`` over the admitted store.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask
from voxel_features.store import GridSpec, VoxelStore

_GRID = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(0.02, 0.02, 20.0), shape=(4, 4, 2))


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "dataset_dir": str(tmp_path / "dataset"),
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "knowledge"),
            "artifact_dir": str(tmp_path / "artifacts"),
        }
    )


def _seed_experiment(kg_dir: Path, node_id: str, layer_name: str) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("a") as fh:
        fh.write(json.dumps({"node_id": node_id, "layer_name": layer_name, "bic_delta": -10.0}) + "\n")


def _distance_by_layer_pair(kg_dir: Path) -> dict[frozenset[str], float]:
    path = kg_dir / "pairwise_distance.jsonl"
    out: dict[frozenset[str], float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[frozenset((rec["layer_1"], rec["layer_2"]))] = rec["pairwise_distance"]
    return out


def test_producer_writes_real_distance_not_zero_sentinel(tmp_path: Path) -> None:
    admitted = tmp_path / "store" / "teniz_basin" / "admitted"
    store = VoxelStore(admitted, _GRID)
    host_a = np.zeros(_GRID.shape, dtype=np.float32)
    host_a[0:2, 0:2, :] = 0.7
    host_b = host_a.copy()
    host_b[0, 0, 0] = 0.71  # near-duplicate of host_a (one jittered voxel)
    fold_c = np.zeros(_GRID.shape, dtype=np.float32)
    fold_c[2:4, 2:4, :] = 0.7  # disjoint support from the host_* family
    store.add_layer(name="host_a", values=host_a, dtype="float")
    store.add_layer(name="host_b", values=host_b, dtype="float")
    store.add_layer(name="fold_c", values=fold_c, dtype="float")

    kg = tmp_path / "knowledge" / "teniz_basin"
    _seed_experiment(kg, "exp_a", "host_a")
    _seed_experiment(kg, "exp_c", "fold_c")

    task = _task(tmp_path)
    task._update_pairwise_distance_index(kg, "exp_b", "host_b", {}, admitted_dir=admitted)

    dists = _distance_by_layer_pair(kg)
    # disjoint layers → real distance ~1.0, NOT the old 0.0 sentinel.
    assert dists[frozenset(("host_b", "fold_c"))] > 0.15
    # near-duplicate layers → real distance stays below the near-dup threshold.
    assert dists[frozenset(("host_b", "host_a"))] < 0.15


def test_producer_skips_uncomputable_pair_instead_of_writing_zero(tmp_path: Path) -> None:
    # When a layer array is absent we must NOT write the 0.0 sentinel (which
    # _count_diverse_parents misreads as a near-duplicate). Leave it unknown.
    admitted = tmp_path / "store" / "teniz_basin" / "admitted"
    store = VoxelStore(admitted, _GRID)
    new = np.zeros(_GRID.shape, dtype=np.float32)
    new[0, 0, 0] = 1.0
    store.add_layer(name="new_layer", values=new, dtype="float")

    kg = tmp_path / "knowledge" / "teniz_basin"
    _seed_experiment(kg, "exp_ghost", "ghost_layer")  # no .npy on disk

    task = _task(tmp_path)
    task._update_pairwise_distance_index(kg, "exp_new", "new_layer", {}, admitted_dir=admitted)

    path = kg / "pairwise_distance.jsonl"
    rows = [l for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    assert rows == []
