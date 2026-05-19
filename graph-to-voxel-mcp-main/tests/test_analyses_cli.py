from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from graph_to_voxel.analyses import check_domain_closure, check_existence_presence, run_all_checks
from graph_to_voxel.cli import app
from graph_to_voxel.engine.loopstructural import GridSpec, build_voxel_field
from graph_to_voxel.graph import EntityGraph


def test_domain_and_existence_presence_checks(two_unit_graph_dict):
    two_unit_graph_dict["nodes"][0]["p_exists"] = 0.05
    graph = EntityGraph.from_dict(two_unit_graph_dict)
    field = build_voxel_field(
        graph,
        GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(8, 8, 8)),
    )

    assert check_domain_closure(field).severity == "pass"
    assert check_existence_presence(field, graph, tolerance=0.01).severity == "fail"

    results = run_all_checks(graph, field)
    assert {result.name for result in results} >= {"domain_closure", "existence_presence"}


def test_cli_build_and_check_smoke(tmp_path, two_unit_graph_dict):
    runner = CliRunner()
    graph_path = tmp_path / "graph.json"
    out_path = tmp_path / "field.zarr"
    graph_path.write_text(json.dumps(two_unit_graph_dict), encoding="utf-8")

    build = runner.invoke(
        app,
        [
            "build",
            str(graph_path),
            "--output",
            str(out_path),
            "--bounds",
            "0,10,0,10,0,10",
            "--shape",
            "5,5,5",
        ],
    )
    assert build.exit_code == 0, build.output

    check = runner.invoke(app, ["check", str(out_path), "--graph", str(graph_path)])
    assert check.exit_code == 0, check.output

    payload = json.loads(check.output)
    assert any(result["name"] == "domain_closure" and result["severity"] == "pass" for result in payload)

    render_dir = tmp_path / "renders"
    render = runner.invoke(app, ["render-slices", str(out_path), "--output-dir", str(render_dir)])
    assert render.exit_code == 0, render.output
    rendered = [Path(path) for path in json.loads(render.output)]
    assert len(rendered) == 6
    assert all(path.exists() for path in rendered)
    assert any("most_likely_unit_z" in path.name for path in rendered)
