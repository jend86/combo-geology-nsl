"""Integration: the first free (first_layer_auto) admit is hardened end-to-end
through ``FeatureHypothesisKazakhstanTask._admit_with_dedup``.

The unit tests in ``test_feature_hypothesis_kazakhstan_admit_gate.py`` pin the
pure ``_first_root_admission_ok`` gate; these drive it through the real
admission pipeline (scratch ``spatial.db`` + layer ``.npy`` → guards → KG) so
the wiring inside ``check_guards`` cannot silently regress.

Contract:
  - a degenerate first root (single uniform point op) is ``guard_rejected`` and
    never reaches ``experiments.jsonl`` / the admitted pool;
  - an all-creative_fallback first root is rejected by the SURVEY seed gate
    regardless of ``disallow_creative_fallback_admission`` (override-proof, and
    independent of the crossbreed-scoped knob);
  - a graded, multi-op, artifact-backed first root admits.

Crossbreed scope (separate contract, also pinned here):
  - by DEFAULT an all-creative_fallback crossbreed layer admits (the provenance
    guard is permissive); ``disallow_creative_fallback_admission=True`` restores
    the strict rejection in the crossbreed/normal path only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask
from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import GridSpec


def _task(tmp_path: Path, *, disallow_fallback: bool = False) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
            "dataset_dir": str(tmp_path / "data"),
            "disallow_creative_fallback_admission": disallow_fallback,
        }
    )


def _grid() -> GridSpec:
    return GridSpec(
        origin=(0.0, 0.0, 0.0),
        maximum=(4.0, 4.0, 2.0),
        shape=(4, 4, 2),
        crs="EPSG:4326",
    )


def _single_uniform_point(scratch_dir: Path, layer_name: str, *, coordinate_source: str) -> None:
    store = SpatialVoxelStore(scratch_dir, _grid())
    store.add_point_feature(
        name=layer_name,
        longitude=1.0,
        latitude=1.0,
        depth=0.5,
        value=1.0,
        radius_m=1.0,
        coordinate_source=coordinate_source,  # type: ignore[arg-type]
        source_file="analysis.csv" if coordinate_source == "artifact" else None,
        source_excerpt="row 3" if coordinate_source == "artifact" else None,
    )


def _multi_point(
    scratch_dir: Path,
    layer_name: str,
    *,
    uniform: bool = False,
    variant: int = 0,
) -> None:
    store = SpatialVoxelStore(scratch_dir, _grid())
    if variant == 0:
        coords = [(0.5, 0.5, 0.5), (1.5, 1.5, 0.5), (2.5, 2.5, 1.5), (3.5, 0.5, 1.5)]
    else:
        # A disjoint support (different lon-lat columns) so the candidate is a
        # genuinely distinct layer, not a near-duplicate of variant 0.
        coords = [(0.5, 3.5, 0.5), (1.5, 2.5, 1.5), (2.5, 0.5, 0.5), (3.5, 3.5, 1.5)]
    values = [1.0, 1.0, 1.0, 1.0] if uniform else [0.2, 0.5, 0.8, 1.0]
    for i, ((lon, lat, depth), value) in enumerate(zip(coords, values)):
        store.add_point_feature(
            name=layer_name,
            longitude=lon,
            latitude=lat,
            depth=depth,
            value=value,
            radius_m=1.0,
            coordinate_source="artifact",
            source_file="analysis.csv",
            source_excerpt=f"row {i}",
        )


def _multi_point_fallback(scratch_dir: Path, layer_name: str) -> None:
    # Multi-op, graded, but every op is an invented (creative_fallback) coordinate
    # with no source provenance — the shape the provenance guard keys on.
    store = SpatialVoxelStore(scratch_dir, _grid())
    coords = [(0.5, 0.5, 0.5), (1.5, 1.5, 0.5), (2.5, 2.5, 1.5), (3.5, 0.5, 1.5)]
    for i, (lon, lat, depth) in enumerate(coords):
        store.add_point_feature(
            name=layer_name,
            longitude=lon,
            latitude=lat,
            depth=depth,
            value=0.2 + 0.2 * i,
            radius_m=1.0,
            coordinate_source="creative_fallback",
        )


def _array_layer(scratch_dir: Path, layer_name: str, *, with_provenance: bool) -> None:
    store = SpatialVoxelStore(scratch_dir, _grid())
    values = np.zeros(_grid().shape, dtype=float)
    values[:, :, :] = 0.5
    if with_provenance:
        store.set_layer_array(
            layer_name,
            values,
            dtype="float",
            source_file="value_grid_array.npy",
            source_excerpt="code-phase value_grid artifact",
            coordinate_source="artifact",
        )
    else:
        store.add_layer(layer_name, values, dtype="float")


def _first_root_record(node_id: str, layer_name: str) -> dict:
    return {
        "node_id": node_id,
        "hypothesis": f"first root {layer_name}",
        "parent_node_1": None,
        "parent_node_2": None,
        "bic_delta": None,
        "admission_path": "first_layer_auto",
        "stage_completed": "first_layer_auto",
        "layer_name": layer_name,
    }


def _admit(task, kg_dir, store_dir, layer_name, record, *, seed_phase: bool = True) -> bool:
    scratch_dir = store_dir / "scratch" / layer_name
    admitted_dir = store_dir / "admitted"
    return task._admit_with_dedup(
        kg_dir,
        record,
        parents=[],
        hypothesis=record["hypothesis"],
        scratch_dir=scratch_dir,
        admitted_dir=admitted_dir,
        layer_name=layer_name,
        seed_phase=seed_phase,
    )


def test_degenerate_single_uniform_first_root_rejected(tmp_path: Path) -> None:
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "single_point_root"
    _single_uniform_point(store_dir / "scratch" / layer_name, layer_name, coordinate_source="artifact")
    record = _first_root_record("exp_single", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is False
    assert record["admission_tier"] == "guard_rejected"
    # Geometry/provenance floor: a single op is the reject reason (uniform value
    # and entropy are no longer gated under Approach 1).
    assert record["first_root_rejection_reason"] == "single_spatial_operation"
    assert not (kg_dir / "experiments.jsonl").exists()
    assert not (store_dir / "admitted" / "layers" / f"{layer_name}.npy").exists()


@pytest.mark.parametrize("disallow", [False, True])
def test_all_creative_fallback_first_root_rejected_regardless_of_disallow_flag(
    tmp_path: Path, disallow: bool
) -> None:
    # Crossbreed-only scope: the SURVEY seed gate rejects an all-fallback first
    # root whether or not disallow_creative_fallback_admission is set. The
    # permissive crossbreed default (disallow=False) must NOT leak into the seed
    # gate — the founder must still rest on real provenance.
    task = _task(tmp_path, disallow_fallback=disallow)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = f"fallback_root_{int(disallow)}"
    _single_uniform_point(store_dir / "scratch" / layer_name, layer_name, coordinate_source="creative_fallback")
    record = _first_root_record(f"exp_fallback_{int(disallow)}", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is False
    assert record["admission_tier"] == "guard_rejected"
    assert not (kg_dir / "experiments.jsonl").exists()


def test_crossbreed_all_creative_fallback_admits_by_default(tmp_path: Path) -> None:
    # Relaxed default: outside the survey seed window an all-creative_fallback
    # layer admits — the provenance guard no longer rejects on fallback.
    task = _task(tmp_path)  # disallow_creative_fallback_admission defaults False
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "xbreed_fallback"
    _multi_point_fallback(store_dir / "scratch" / layer_name, layer_name)
    record = _first_root_record("exp_xbreed_fallback", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record, seed_phase=False)

    assert admitted is True
    assert record["provenance_guard_passed"] is True
    assert record["provenance_rejection_reason"] == "none"
    assert record["translate_fallback_used"] is True
    assert (kg_dir / "experiments.jsonl").exists()


def test_crossbreed_all_creative_fallback_rejected_when_disallow(tmp_path: Path) -> None:
    # Opt-in stricten: disallow_creative_fallback_admission=True restores the
    # old rejection in the crossbreed/normal path.
    task = _task(tmp_path, disallow_fallback=True)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "xbreed_fallback_strict"
    _multi_point_fallback(store_dir / "scratch" / layer_name, layer_name)
    record = _first_root_record("exp_xbreed_fallback_strict", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record, seed_phase=False)

    assert admitted is False
    assert record["admission_tier"] == "guard_rejected"
    assert record["provenance_guard_passed"] is False
    assert record["provenance_rejection_reason"] == "all_creative_fallback"
    # The new wiring stamped the knob onto the record (absent on the old code path).
    assert record["disallow_creative_fallback_admission"] is True
    assert not (kg_dir / "experiments.jsonl").exists()


def test_graded_multi_op_first_root_admits(tmp_path: Path) -> None:
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "graded_root"
    _multi_point(store_dir / "scratch" / layer_name, layer_name)
    record = _first_root_record("exp_graded", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is True
    assert record["first_root_rejection_reason"] == "none"
    assert record["single_spatial_operation"] is False
    assert record["admission_tier"] in {"kg_evidence", "kg_parent_eligible"}
    assert (kg_dir / "experiments.jsonl").exists()


def test_artifact_backed_array_first_root_admits(tmp_path: Path) -> None:
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "array_root"
    _array_layer(store_dir / "scratch" / layer_name, layer_name, with_provenance=True)
    record = _first_root_record("exp_array", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is True
    assert record["first_root_rejection_reason"] == "none"
    assert record["spatial_operation_provenance_count"] == 1
    assert record["geometry_kind_counts"] == {"array": 1}
    assert record["single_spatial_operation"] is False
    assert (kg_dir / "experiments.jsonl").exists()


def test_array_without_operation_provenance_rejected(tmp_path: Path) -> None:
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "unprovenanced_array_root"
    _array_layer(store_dir / "scratch" / layer_name, layer_name, with_provenance=False)
    record = _first_root_record("exp_unprovenanced_array", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is False
    assert record["admission_tier"] == "guard_rejected"
    assert record["provenance_rejection_reason"] == "missing_spatial_operation_provenance"
    assert not (kg_dir / "experiments.jsonl").exists()


def test_second_root_with_positive_bic_admits_in_survey(tmp_path: Path) -> None:
    # The co-location stall: with one seed already in the pool, a distributed
    # candidate at a DIFFERENT support is rejected by the scorer (positive BIC).
    # In the survey phase the persist gate bypasses that, and the
    # geometry/provenance floor (multi-op, artifact-backed) lets it seed.
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"

    # Seed the pool with one admitted layer so this candidate is root #2.
    seed_name = "seed_root"
    _multi_point(store_dir / "scratch" / seed_name, seed_name)
    assert _admit(task, kg_dir, store_dir, seed_name, _first_root_record("exp_seed", seed_name))
    assert (store_dir / "admitted" / "layers" / f"{seed_name}.npy").exists()

    # Root #2: distributed, artifact-backed, but the scorer rejected it
    # (admitted=False, positive bic_delta, full two-stage scoring completed).
    layer_name = "second_root"
    _multi_point(store_dir / "scratch" / layer_name, layer_name, variant=1)
    record = {
        "node_id": "exp_second",
        "hypothesis": "second distinct distributed support",
        "parent_node_1": None,
        "parent_node_2": None,
        "bic_delta": 3.38,
        "admitted": False,
        "masking_test_passed": True,
        "admission_path": "two_stage_v2",
        "stage_completed": "mae_bic_completed",
        "layer_name": layer_name,
    }

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is True
    assert record["first_root_rejection_reason"] == "none"
    assert (store_dir / "admitted" / "layers" / f"{layer_name}.npy").exists()


def test_empty_layer_name_is_not_persisted_no_phantom(tmp_path: Path) -> None:
    # Phantom-record guard: a degenerate candidate whose layer never materialized
    # (no layer_name) must NOT write a KG experiment record. In the 2026-06-03 run
    # two such empty-layer_name phantoms reached experiments.jsonl and polluted the
    # diversity/pool reads. _admit_with_dedup is the chokepoint (it otherwise writes
    # unconditionally when no candidate .npy is present).
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    record = _first_root_record("exp_phantom", "")  # empty layer_name

    admitted = task._admit_with_dedup(
        kg_dir,
        record,
        parents=[],
        hypothesis="phantom degenerate layer",
        scratch_dir=None,
        admitted_dir=None,
        layer_name=None,  # call site passes `feature_layer_name or None`
        seed_phase=True,
    )

    assert admitted is False
    assert not (kg_dir / "experiments.jsonl").exists()


def test_distributed_uniform_multi_op_first_root_admits(tmp_path: Path) -> None:
    # Approach 1 relaxation: a multi-op, UNIFORM-valued distributed root is a
    # fine seed and must admit (this exact shape deadlocked the prior run).
    task = _task(tmp_path)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "uniform_distributed_root"
    _multi_point(store_dir / "scratch" / layer_name, layer_name, uniform=True)
    record = _first_root_record("exp_uniform_dist", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is True
    assert record["first_root_rejection_reason"] == "none"
    assert (kg_dir / "experiments.jsonl").exists()
