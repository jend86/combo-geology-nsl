"""File rotation: source-sample pre-read (_read_source_sample).

At populate() time the assigned source is pre-read into a compact sample and
injected into the explore prompt, so the agent sees real data from its assigned
slice immediately rather than relying on parametric/memorised knowledge.
"""

import json
from pathlib import Path

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


def _task(tmp_path: Path) -> FeatureHypothesisKazakhstanTask:
    return FeatureHypothesisKazakhstanTask({
        "store_dir": str(tmp_path / "store_root"),
        "kg_dir": str(tmp_path / "kg_root"),
    })


class TestReadSourceSample:
    def test_glob_pattern_md_chunks_first_and_last(self, tmp_path: Path) -> None:
        ds = tmp_path / "data"
        chunks = ds / "36572_Smolianova_1984" / "chunks"
        chunks.mkdir(parents=True)
        for i in range(5):
            (chunks / f"Devonian_{i}.md").write_text(f"chunk {i} body text " * 4)
        assigned = {
            "key": "dev",
            "path": "36572_Smolianova_1984/chunks/",
            "glob_pattern": "*Devonian*.md",
        }
        sample = _task(tmp_path)._read_source_sample(assigned, str(ds))
        assert "Devonian_0.md" in sample  # first
        assert "Devonian_4.md" in sample  # last
        assert "chunk" in sample

    def test_geojson_schema_and_features(self, tmp_path: Path) -> None:
        ds = tmp_path / "data"
        sp = ds / "converted_spatial_data"
        sp.mkdir(parents=True)
        gj = {"features": [
            {"properties": {"Cu_pct": 1.2, "Name": f"P{i}"}, "geometry": {"type": "Point"}}
            for i in range(4)
        ]}
        (sp / "copper_prospects_aoi.geojson").write_text(json.dumps(gj))
        assigned = {"key": "cp", "path": "converted_spatial_data/copper_prospects_aoi.geojson"}
        sample = _task(tmp_path)._read_source_sample(assigned, str(ds))
        assert "Property columns" in sample
        assert "Cu_pct" in sample
        assert "[Point]" in sample

    def test_csv_header_plus_first_rows(self, tmp_path: Path) -> None:
        ds = tmp_path / "data"
        u = ds / "USGS"
        u.mkdir(parents=True)
        rows = ["col1,col2,col3"] + [f"{i},a,b" for i in range(20)]
        (u / "x.csv").write_text("\n".join(rows))
        assigned = {"key": "x", "path": "USGS/x.csv"}
        sample = _task(tmp_path)._read_source_sample(assigned, str(ds))
        assert "col1,col2,col3" in sample  # header always present
        assert sample.count("\n") <= 5  # header + up to 5 data rows

    def test_missing_path_returns_empty_string(self, tmp_path: Path) -> None:
        assigned = {"key": "nope", "path": "does/not/exist.geojson"}
        assert _task(tmp_path)._read_source_sample(assigned, str(tmp_path / "data")) == ""

    def test_sample_is_length_bounded(self, tmp_path: Path) -> None:
        ds = tmp_path / "data"
        chunks = ds / "36572_Smolianova_1984" / "chunks"
        chunks.mkdir(parents=True)
        for i in range(4):
            (chunks / f"Permian_{i}.md").write_text("x" * 5000)
        assigned = {
            "key": "perm",
            "path": "36572_Smolianova_1984/chunks/",
            "glob_pattern": "*Permian*.md",
        }
        sample = _task(tmp_path)._read_source_sample(assigned, str(ds))
        assert 0 < len(sample) <= 2600  # MAX_TOTAL 2500 + truncation marker
