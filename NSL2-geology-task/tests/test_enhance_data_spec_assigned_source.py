"""data_spec enhancement prepends the episode's assigned source.

When the code agent calls phase_get(phase='hypothesise'), the returned data_spec
has the episode's assigned source (from file rotation) prepended to file_specs,
so the code agent always works against the correct source even if the explore
agent's record_phase listed different files.
"""

from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask({
        "store_dir": str(tmp_path / "store_root"),
        "kg_dir": str(tmp_path / "kg_root"),
    })


class TestEnhanceDataSpecAssignedSource:
    def test_glob_assigned_source_prepended(self, tmp_path: Path) -> None:
        ctx = {"assigned_source": {
            "key": "smolianova_tectonics",
            "path": "36572_Smolianova_1984/chunks/",
            "glob_pattern": "*TECTONICS*.md",
        }}
        out = _task(tmp_path)._enhance_data_spec({"files": []}, ctx)
        first = out["file_specs"][0]["file"]
        assert first.startswith("/workspace/input/36572_Smolianova_1984/chunks")
        assert first.endswith("*TECTONICS*.md")

    def test_plain_assigned_source_prepended(self, tmp_path: Path) -> None:
        ctx = {"assigned_source": {
            "key": "copper_prospects_aoi",
            "path": "converted_spatial_data/copper_prospects_aoi.geojson",
        }}
        out = _task(tmp_path)._enhance_data_spec({"files": []}, ctx)
        assert out["file_specs"][0]["file"] == (
            "/workspace/input/converted_spatial_data/copper_prospects_aoi.geojson"
        )

    def test_no_episode_context_preserves_rich_catalogue(self, tmp_path: Path) -> None:
        # Backwards-compatible: no episode_context => no prepend, rich spec intact.
        out = _task(tmp_path)._enhance_data_spec({"files": []})
        assert "file_specs" in out and len(out["file_specs"]) > 0
        assert "kazakhstan_data_structure" in out

    def test_assigned_source_not_duplicated(self, tmp_path: Path) -> None:
        resolved = "/workspace/input/converted_spatial_data/copper_prospects_aoi.geojson"
        ctx = {"assigned_source": {
            "key": "copper_prospects_aoi",
            "path": "converted_spatial_data/copper_prospects_aoi.geojson",
        }}
        out = _task(tmp_path)._enhance_data_spec({"files": [resolved]}, ctx)
        files = [s.get("file") for s in out["file_specs"]]
        assert files.count(resolved) == 1
