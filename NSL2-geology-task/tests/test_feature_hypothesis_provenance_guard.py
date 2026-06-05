"""FM4 regression: the provenance guard must find spatial ops logged under the BASE
layer name even though the admitted/scored layer carries a `_<timestamp>` suffix.

Before the fix, `_stamp_candidate_provenance` queried
`WHERE feature_name = '<base>_<timestamp>'` while `spatial_add_point` had logged ops under
`<base>`, so it always found 0 ops -> coordinate_source_counts={} -> the guard could never
fire (all_creative_fallback is False when there are no ops). That made the guard inert.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _make_spatial_db(
    scratch: Path,
    feature_name: str,
    coordinate_source: str,
    n: int,
    *,
    operation_type: str = "point",
) -> None:
    db = scratch / "spatial.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE spatial_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_type TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            coordinates TEXT NOT NULL,
            parameters TEXT NOT NULL,
            source_file TEXT,
            source_excerpt TEXT,
            coordinate_source TEXT NOT NULL DEFAULT 'creative_fallback',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for _ in range(n):
        conn.execute(
            "INSERT INTO spatial_operations (operation_type, feature_name, coordinates, parameters, coordinate_source) "
            "VALUES (?,?,?,?,?)",
            (operation_type, feature_name, "69.0,51.0,40", "radius_m=100,value=1.0", coordinate_source),
        )
    conn.commit()
    conn.close()


def test_provenance_lookup_matches_timestamped_layer_name(tmp_path):
    # ops logged under the BASE name; the scored layer is timestamped
    _make_spatial_db(tmp_path, "vladimirov_suite_prospects", coordinate_source="artifact", n=3)
    kg: dict = {}
    FeatureHypothesisKazakhstanTask._stamp_candidate_provenance(
        kg, scratch_dir=tmp_path, layer_name="vladimirov_suite_prospects_1780408167198"
    )
    assert kg["spatial_operation_provenance_count"] == 3
    assert kg["coordinate_source_counts"] == {"artifact": 3}
    assert kg["provenance_guard_passed"] is True


def test_provenance_all_creative_fallback_now_rejected(tmp_path):
    # with the lookup fixed, a layer built only from invented coords must be REJECTED
    _make_spatial_db(tmp_path, "blob", coordinate_source="creative_fallback", n=1)
    kg: dict = {}
    FeatureHypothesisKazakhstanTask._stamp_candidate_provenance(
        kg, scratch_dir=tmp_path, layer_name="blob_1780408167199"
    )
    assert kg["spatial_operation_provenance_count"] == 1
    assert kg["provenance_guard_passed"] is False


def test_missing_spatial_operation_provenance_rejected(tmp_path):
    kg: dict = {}
    FeatureHypothesisKazakhstanTask._stamp_candidate_provenance(
        kg, scratch_dir=tmp_path, layer_name="array_layer_1780408167200"
    )

    assert kg["spatial_operation_provenance_count"] == 0
    assert kg["coordinate_source_counts"] == {}
    assert kg["provenance_guard_passed"] is False
    assert kg["provenance_rejection_reason"] == "missing_spatial_operation_provenance"


def test_array_operation_artifact_provenance_passes(tmp_path):
    _make_spatial_db(
        tmp_path,
        "continuous_field",
        coordinate_source="artifact",
        n=1,
        operation_type="array",
    )
    kg: dict = {}
    FeatureHypothesisKazakhstanTask._stamp_candidate_provenance(
        kg, scratch_dir=tmp_path, layer_name="continuous_field_1780408167201"
    )

    assert kg["spatial_operation_provenance_count"] == 1
    assert kg["coordinate_source_counts"] == {"artifact": 1}
    assert kg["geometry_kind_counts"] == {"array": 1}
    assert kg["provenance_guard_passed"] is True
