from __future__ import annotations

import numpy as np

from voxel_features.spatial import SpatialVoxelStore
from voxel_features.store import GridSpec


GRID = GridSpec(
    origin=(0.0, 0.0, 0.0),
    maximum=(10.0, 10.0, 50.0),
    shape=(10, 10, 5),
    crs="EPSG:4326",
)


def _store(tmp_path) -> SpatialVoxelStore:
    return SpatialVoxelStore(tmp_path / "store", GRID)


def test_box_fills_clipped_slice(tmp_path):
    store = _store(tmp_path)

    result = store.add_box_feature(
        name="clipped_box",
        min_longitude=-5.0,
        min_latitude=-1.0,
        min_depth_m=-10.0,
        max_longitude=2.1,
        max_latitude=3.1,
        max_depth_m=25.0,
        value=0.75,
    )

    assert result["success"] is True
    values = store.get_layer_values("clipped_box")
    expected = np.zeros(GRID.shape, dtype=float)
    expected[0:3, 0:4, 0:3] = 0.75
    np.testing.assert_array_equal(values, expected)
    assert result["affected_voxels"] == 36


def test_replace_layer_batch_is_idempotent_for_add_rule_and_provenance(tmp_path):
    store = _store(tmp_path)
    records = [
        {
            "record_id": "a",
            "geometry_kind": "box",
            "lon_min": 1.0,
            "lat_min": 1.0,
            "depth_min_m": 0.0,
            "lon_max": 2.1,
            "lat_max": 2.1,
            "depth_max_m": 10.0,
            "value": 1.0,
            "combination_rule": "add",
            "coordinate_source": "artifact",
        },
        {
            "record_id": "b",
            "geometry_kind": "box",
            "lon_min": 1.0,
            "lat_min": 1.0,
            "depth_min_m": 0.0,
            "lon_max": 2.1,
            "lat_max": 2.1,
            "depth_max_m": 10.0,
            "value": 2.0,
            "combination_rule": "add",
            "coordinate_source": "artifact",
        },
    ]

    first = store.add_geometry_batch("batch_layer", records, combination_rule="add")
    first_values = store.get_layer_values("batch_layer").copy()
    second = store.add_geometry_batch("batch_layer", records, combination_rule="add")
    second_values = store.get_layer_values("batch_layer")

    assert first["success"] is True
    assert second["success"] is True
    np.testing.assert_array_equal(first_values, second_values)
    assert float(second_values[1, 1, 0]) == 3.0
    ops = [op for op in store.get_spatial_operations() if op["feature_name"] == "batch_layer"]
    assert len(ops) == 2
    assert {op["record_id"] for op in ops} == {"a", "b"}
    assert all(op["operation_group_id"] == second["operation_group_id"] for op in ops)


def test_mixed_geometry_batch_reports_counts_and_logs_per_record(tmp_path):
    store = _store(tmp_path)
    records = [
        {
            "record_id": "pt-1",
            "geometry_kind": "point",
            "longitude": 5.0,
            "latitude": 5.0,
            "depth_m": 20.0,
            "radius_m": 80_000.0,
            "value": 0.4,
            "coordinate_source": "geonames",
        },
        {
            "record_id": "box-1",
            "geometry_kind": "box",
            "lon_min": 1.0,
            "lat_min": 1.0,
            "depth_min_m": 0.0,
            "lon_max": 2.1,
            "lat_max": 2.1,
            "depth_max_m": 10.0,
            "value": 0.8,
            "coordinate_source": "artifact",
        },
    ]

    result = store.add_geometry_batch("mixed_layer", records)

    assert result["success"] is True
    assert result["records_seen"] == 2
    assert result["records_applied"] == 2
    assert result["records_skipped"] == 0
    assert result["geometry_kind_counts"] == {"box": 1, "point": 1}
    assert result["coordinate_source_counts"] == {"artifact": 1, "geonames": 1}
    assert result["affected_voxels"] > 0
    ops = [op for op in store.get_spatial_operations() if op["feature_name"] == "mixed_layer"]
    assert len(ops) == 2
    assert {op["operation_type"] for op in ops} == {"point", "box"}
    assert all(op["affected_voxels"] > 0 for op in ops)


def test_batch_bounds_policy_skip_clip_and_fail(tmp_path):
    store = _store(tmp_path)
    record = {
        "record_id": "oob-box",
        "geometry_kind": "box",
        "lon_min": -1.0,
        "lat_min": 1.0,
        "depth_min_m": 0.0,
        "lon_max": 1.1,
        "lat_max": 2.1,
        "depth_max_m": 10.0,
        "value": 1.0,
        "coordinate_source": "artifact",
    }

    skipped = store.add_geometry_batch("skip_layer", [record], bounds_policy="skip")
    clipped = store.add_geometry_batch("clip_layer", [record], bounds_policy="clip")
    failed = store.add_geometry_batch("fail_layer", [record], bounds_policy="fail")

    assert skipped["success"] is True
    assert skipped["records_applied"] == 0
    assert skipped["records_skipped"] == 1
    assert clipped["success"] is True
    assert clipped["records_applied"] == 1
    assert clipped["affected_voxels"] > 0
    assert failed["success"] is False
    assert "outside grid bounds" in failed["error"]


def test_non_numeric_value_coerced_to_presence(tmp_path):
    # Agent mistake observed in the 2026-06-03 run: a categorical suite-name
    # string lands in the numeric `value` field. Records must NOT be silently
    # dropped (that produced 0-voxel degenerate layers ~85% of the time) — they
    # become presence (1.0) so the distributed layer still materializes.
    store = _store(tmp_path)
    records = [
        {
            "record_id": "s0",
            "geometry_kind": "point",
            "longitude": 3.0,
            "latitude": 3.0,
            "depth_m": 10.0,
            "radius_m": 80_000.0,
            "value": "Vladimirov Suite",
            "coordinate_source": "artifact",
        },
        {
            "record_id": "s1",
            "geometry_kind": "point",
            "longitude": 6.0,
            "latitude": 6.0,
            "depth_m": 20.0,
            "radius_m": 80_000.0,
            "value": "Kayraktin Suite",
            "coordinate_source": "artifact",
        },
    ]

    result = store.add_geometry_batch("suite_presence", records, dtype="float")

    assert result["success"] is True
    assert result["records_applied"] == 2  # coerced to presence, not skipped
    assert result["records_skipped"] == 0
    assert result["affected_voxels"] > 0
    assert result["value_min"] == 1.0
    assert result["value_max"] == 1.0
    assert any("presence" in w.lower() for w in result["warnings"])
    values = store.get_layer_values("suite_presence")
    assert float(values.max()) == 1.0
    assert int((values != 0).sum()) == result["affected_voxels"]


def test_none_value_defaults_to_presence(tmp_path):
    # An explicit None value (not just a missing key) is also coerced to presence
    # rather than skipping the record.
    store = _store(tmp_path)
    records = [
        {
            "record_id": "n0",
            "geometry_kind": "point",
            "longitude": 4.0,
            "latitude": 4.0,
            "depth_m": 10.0,
            "radius_m": 80_000.0,
            "value": None,
            "coordinate_source": "artifact",
        },
    ]

    result = store.add_geometry_batch("none_presence", records)

    assert result["success"] is True
    assert result["records_applied"] == 1
    assert result["value_min"] == 1.0
    assert result["value_max"] == 1.0
