"""A1 handoff: the Code phase produces feature_points.csv/feature_points_dataframe.csv;
get_experiment_summary must
surface its rows so the Translate agent (which has NO file-read capability) can map one
spatial_add_point per row instead of stamping a single blob.

These cover the pure CSV-loading helper.
"""
from __future__ import annotations

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask as T


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
