"""A1 handoff: the Code phase produces feature_points.csv/feature_points_dataframe.csv;
get_experiment_summary must
surface its rows so the Translate agent (which has NO file-read capability) can map one
spatial_add_point per row instead of stamping a single blob.

These cover the pure CSV-loading helper.
"""
from __future__ import annotations

import numpy as np

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask as T
from src.task.types import CapabilityExecutionContext


def test_load_feature_points_from_full_path(tmp_path):
    p = tmp_path / "feature_points.csv"
    p.write_text(
        "longitude,latitude,depth_m,value,coordinate_source\n"
        "69.0,51.0,40,0.8,artifact\n"
        "70.1,50.2,20,0.5,geonames\n"
    )
    rows, truncated = T._load_feature_points([str(p)], str(tmp_path))
    assert truncated is False
    assert len(rows) == 2
    assert rows[0]["longitude"] == 69.0
    assert rows[0]["latitude"] == 51.0
    assert rows[0]["depth_m"] == 40.0
    assert rows[0]["coordinate_source"] == "artifact"
    assert rows[1]["value"] == 0.5


def test_load_feature_points_dataframe_capture_name(tmp_path):
    # the execution sandbox captures a `feature_points` DataFrame variable as
    # feature_points_dataframe.csv -- the helper MUST match this real artifact name
    p = tmp_path / "feature_points_dataframe.csv"
    p.write_text(
        "longitude,latitude,depth_m,value,coordinate_source\n"
        "68.04,51.99,40,1.0,artifact\n"
    )
    rows, truncated = T._load_feature_points([str(p)], str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["latitude"] == 51.99


def test_spatial_args_preserve_explicit_coordinate_source():
    args = {
        "longitude": 69.0,
        "latitude": 51.0,
        "depth_m": 40.0,
        "value": 0.8,
        "coordinate_source": "artifact",
    }
    resolved = T._prepare_spatial_provenance_args(args, {})
    assert resolved["coordinate_source"] == "artifact"


def test_load_feature_points_missing_returns_empty(tmp_path):
    rows, truncated = T._load_feature_points([str(tmp_path / "results.npy")], str(tmp_path))
    assert rows == []
    assert truncated is False


def test_load_feature_points_caps_rows(tmp_path):
    p = tmp_path / "feature_points.csv"
    header = "longitude,latitude,depth_m,value,coordinate_source"
    body = "\n".join("69.0,51.0,40,0.5,artifact" for _ in range(2100))
    p.write_text(header + "\n" + body + "\n")
    rows, truncated = T._load_feature_points([str(p)], str(tmp_path))
    assert len(rows) == 2000
    assert truncated is True


def test_load_feature_points_basename_via_directory(tmp_path):
    # artifact_files may carry a bare basename; fall back to artifact_directory join
    p = tmp_path / "feature_points.csv"
    p.write_text("longitude,latitude,depth_m,value,coordinate_source\n69.0,51.0,40,0.8,artifact\n")
    rows, truncated = T._load_feature_points(["feature_points.csv"], str(tmp_path))
    assert len(rows) == 1


def test_load_geometry_records_prefers_feature_geometry_over_feature_points(tmp_path):
    (tmp_path / "feature_points_dataframe.csv").write_text(
        "longitude,latitude,depth_m,value,coordinate_source\n"
        "69.0,51.0,40,0.8,artifact\n"
    )
    (tmp_path / "feature_geometry_dataframe.csv").write_text(
        "record_id,geometry_kind,lon_min,lat_min,depth_min_m,lon_max,lat_max,depth_max_m,value,coordinate_source\n"
        "box-1,box,68.0,50.0,0,69.0,51.0,20,0.6,artifact\n"
    )

    rows, truncated, total_count, artifact_path = T._load_geometry_records(
        ["feature_points_dataframe.csv", "feature_geometry_dataframe.csv"],
        str(tmp_path),
    )

    assert truncated is False
    assert total_count == 1
    assert artifact_path.endswith("feature_geometry_dataframe.csv")
    assert rows[0]["geometry_kind"] == "box"
    assert rows[0]["lon_min"] == 68.0


def test_load_geometry_records_accepts_legacy_feature_points(tmp_path):
    p = tmp_path / "feature_points_dataframe.csv"
    p.write_text(
        "longitude,latitude,depth_m,value,coordinate_source\n"
        "69.0,51.0,40,0.8,artifact\n"
    )

    rows, truncated, total_count, artifact_path = T._load_geometry_records([str(p)], str(tmp_path))

    assert truncated is False
    assert total_count == 1
    assert artifact_path == str(p)
    assert rows[0]["geometry_kind"] == "point"
    assert rows[0]["radius_m"] == 100.0


def test_spatial_batch_resolves_current_episode_artifact(tmp_path):
    task = T(
        {
            "store_dir": str(tmp_path / "store"),
            "kg_dir": str(tmp_path / "kg"),
            "dataset_dir": str(tmp_path / "data"),
        }
    )
    current = tmp_path / "current_ep"
    other = tmp_path / "other_ep"
    current.mkdir()
    other.mkdir()
    current_artifact = current / "feature_geometry_dataframe.csv"
    other_artifact = other / "feature_geometry_dataframe.csv"
    current_artifact.write_text(
        "record_id,geometry_kind,lon_min,lat_min,depth_min_m,lon_max,lat_max,depth_max_m,value,coordinate_source\n"
        "current-box,box,68.0,50.0,0,68.2,50.2,20,0.7,artifact\n"
    )
    other_artifact.write_text(
        "record_id,geometry_kind,lon_min,lat_min,depth_min_m,lon_max,lat_max,depth_max_m,value,coordinate_source\n"
        "wrong-box,box,70.0,52.0,0,70.2,52.2,20,0.2,artifact\n"
    )
    ctx = CapabilityExecutionContext(
        episode_id="ep_1",
        workflow_step="translate",
        episode_context={
            "episode_id": "ep_1",
            "store_dir": str(tmp_path / "store" / "teniz_basin"),
            "grid_spec": {
                "origin": [66.5, 49.5, 0.0],
                "maximum": [71.5, 52.5, 80.0],
                "shape": [20, 20, 4],
                "crs": "EPSG:4326",
            },
            "phase_records": {
                "code": {
                    "artifact_directory": str(current),
                    "artifact_files": ["feature_geometry_dataframe.csv"],
                }
            },
        },
    )

    result = task._exec_spatial_capability(
        [],
        {"name": "artifact_batch", "artifact_name": "auto", "bounds_policy": "clip"},
        ctx,
        "spatial_upsert_geometry_batch",
    )

    assert result.success is True
    assert result.output["records_applied"] == 1
    assert result.output["geometry_kind_counts"] == {"box": 1}
    translate = ctx.episode_context["phase_records"]["translate"]
    assert translate["feature_layer_name"] == "artifact_batch"
    assert translate["spatial_operation_provenance_count"] == 1
    assert translate["geometry_kind_counts"] == {"box": 1}

    from voxel_features.spatial import SpatialVoxelStore
    from voxel_features.store import GridSpec

    store = SpatialVoxelStore(
        tmp_path / "store" / "teniz_basin" / "scratch" / "ep_1",
        GridSpec.from_dict(ctx.episode_context["grid_spec"]),
    )
    values = store.get_layer_values("artifact_batch")
    assert np.nanmax(values) == 0.7
