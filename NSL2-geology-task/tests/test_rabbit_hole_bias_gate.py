"""Rabbit-hole-bias fix: bootstrap gate + greedy BIC initialisation.

Crossbreeding must not begin until every source has been visited at least once
AND the greedy BIC initialisation has run. Previously the -1.0 first-layer
sentinel flipped the pipeline to crossbreed after ~5/18 sources, collapsing the
pool to a monoculture. Ported from JenD86/file-rotation@72e3239.
"""

from __future__ import annotations

import json
from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import (
    _KAZAKHSTAN_SOURCE_FILES,
    FeatureHypothesisKazakhstanTask,
    FeatureHypothesisKazakhstanVariation,
)


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {"store_dir": str(tmp_path / "store_root"), "kg_dir": str(tmp_path / "kg_root")}
    )


def _variation(tmp_path: Path) -> FeatureHypothesisKazakhstanVariation:
    return FeatureHypothesisKazakhstanVariation(
        name="teniz_basin",
        description="test",
        dataset_dir=str(tmp_path / "data"),
        store_dir=str(tmp_path / "store" / "teniz_basin"),
        kg_dir=str(tmp_path / "kg" / "teniz_basin"),
    )


def _seed_admits(kg_dir: Path, n: int) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "experiments.jsonl").open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({"node_id": f"n{i}", "hypothesis": f"h{i}", "bic_delta": -(5.0 + i)}) + "\n")


def _seed_rotation(kg_dir: Path, n_visited: int) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    keys = [s["key"] for s in _KAZAKHSTAN_SOURCE_FILES]
    counts = {k: 1 for k in keys[:n_visited]}
    (kg_dir / "file_rotation_state.json").write_text(json.dumps({"counts": counts}))


def _mark_greedy_done(kg_dir: Path) -> None:
    (kg_dir / "greedy_init_complete.json").write_text(json.dumps({"status": "complete"}))


# ---------------------------------------------------------------------------
# _all_sources_visited
# ---------------------------------------------------------------------------


def test_all_sources_visited_requires_every_source(tmp_path: Path) -> None:
    task = _task(tmp_path)
    kg = tmp_path / "kg" / "teniz_basin"
    total = len(_KAZAKHSTAN_SOURCE_FILES)

    _seed_rotation(kg, n_visited=total - 1)
    assert task._all_sources_visited(str(kg)) is False

    _seed_rotation(kg, n_visited=total)
    assert task._all_sources_visited(str(kg)) is True


def test_all_sources_visited_false_when_no_state(tmp_path: Path) -> None:
    task = _task(tmp_path)
    assert task._all_sources_visited(str(tmp_path / "kg" / "teniz_basin")) is False


# ---------------------------------------------------------------------------
# populate() gating
# ---------------------------------------------------------------------------


def test_crossbreed_blocked_until_all_sources_visited(tmp_path: Path) -> None:
    """>=5 admits would historically trigger crossbreed; now it must stay survey
    until all sources are visited."""
    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)
    _seed_admits(kg, 6)
    _seed_rotation(kg, n_visited=5)  # only 5/18 — the rabbit-hole condition
    _mark_greedy_done(kg)  # isolate the source-coverage gate

    outcome = _task(tmp_path).populate([], variation)
    assert outcome.episode_context["workflow_kind"] == "survey", (
        "crossbreed must be blocked until every source is visited"
    )


def test_crossbreed_enabled_once_all_sources_visited_and_greedy_done(tmp_path: Path) -> None:
    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)
    _seed_admits(kg, 6)
    _seed_rotation(kg, n_visited=len(_KAZAKHSTAN_SOURCE_FILES))  # all 18
    _mark_greedy_done(kg)

    outcome = _task(tmp_path).populate([], variation)
    assert outcome.episode_context["workflow_kind"] == "crossbreed"


def test_greedy_init_completes_with_real_layers(tmp_path: Path) -> None:
    """End-to-end greedy selection over a real (small-grid) admitted store must
    COMPLETE (not silently 'skipped' from a store/scoring API mismatch)."""
    import numpy as np
    from voxel_features.spatial import SpatialVoxelStore
    from voxel_features.store import GridSpec

    grid = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(0.02, 0.02, 20.0), shape=(20, 20, 4))
    variation = _variation(tmp_path)
    variation.grid_spec = grid.to_dict()  # greedy uses GridSpec.from_dict(grid_spec)

    admitted = Path(variation.store_dir) / "admitted"
    admitted.mkdir(parents=True, exist_ok=True)
    store = SpatialVoxelStore(admitted, grid)
    rng = np.random.default_rng(0)
    for name in ("alpha", "beta", "gamma"):
        store.add_layer(name=name, values=rng.random((20, 20, 4)).astype(np.float32), dtype="float")

    _task(tmp_path)._run_greedy_bic_initialization(variation)

    flag = json.loads((Path(variation.kg_dir) / "greedy_init_complete.json").read_text())
    assert flag["status"] == "complete", f"greedy init did not complete: {flag}"
    assert set(flag["selected"]) | set(flag["not_selected"]) == {"alpha", "beta", "gamma"}
    assert len(flag["selected"]) >= 1


def test_populate_runs_greedy_init_when_sources_complete(tmp_path: Path) -> None:
    """With all sources visited and no flag yet, populate() must invoke greedy
    init, which writes greedy_init_complete.json (skipped is fine with no store)."""
    variation = _variation(tmp_path)
    kg = Path(variation.kg_dir)
    _seed_admits(kg, 6)
    _seed_rotation(kg, n_visited=len(_KAZAKHSTAN_SOURCE_FILES))

    _task(tmp_path).populate([], variation)
    assert (kg / "greedy_init_complete.json").exists(), (
        "greedy init must run (and write its flag) once all sources are visited"
    )
