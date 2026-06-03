"""Integration: the first free (first_layer_auto) admit is hardened end-to-end
through ``FeatureHypothesisKazakhstanTask._admit_with_dedup``.

The unit tests in ``test_feature_hypothesis_kazakhstan_admit_gate.py`` pin the
pure ``_first_root_admission_ok`` gate; these drive it through the real
admission pipeline (scratch ``spatial.db`` + layer ``.npy`` → guards → KG) so
the wiring inside ``check_guards`` cannot silently regress.

Contract:
  - a degenerate first root (single uniform point op) is ``guard_rejected`` and
    never reaches ``experiments.jsonl`` / the admitted pool;
  - an all-creative_fallback first root is rejected even when the task is built
    with ``allow_creative_fallback_admission=True`` (override-proof);
  - a graded, multi-op, artifact-backed first root admits.
"""

from __future__ import annotations

from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask
from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import GridSpec


def _task(tmp_path: Path, *, allow_fallback: bool = False) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
            "dataset_dir": str(tmp_path / "data"),
            "allow_creative_fallback_admission": allow_fallback,
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


def _admit(task, kg_dir, store_dir, layer_name, record) -> bool:
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


def test_all_creative_fallback_first_root_rejected_even_with_override(tmp_path: Path) -> None:
    # Override globally enabled, yet the first root must still rest on real
    # provenance.
    task = _task(tmp_path, allow_fallback=True)
    kg_dir = tmp_path / "kg"
    store_dir = tmp_path / "store"
    layer_name = "fallback_root"
    _single_uniform_point(store_dir / "scratch" / layer_name, layer_name, coordinate_source="creative_fallback")
    record = _first_root_record("exp_fallback", layer_name)

    admitted = _admit(task, kg_dir, store_dir, layer_name, record)

    assert admitted is False
    assert record["admission_tier"] == "guard_rejected"
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


def test_second_root_with_positive_bic_admits_inside_window(tmp_path: Path) -> None:
    # The co-location stall: with one seed already in the pool, a distributed
    # candidate at a DIFFERENT support is rejected by the scorer (positive BIC).
    # Inside the first-K window the persist gate bypasses that, and the
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
