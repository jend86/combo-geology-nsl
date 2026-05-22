from __future__ import annotations

import json
from pathlib import Path

import typer

from graph_to_voxel.analyses import run_all_checks
from graph_to_voxel.engine.loopstructural import GridSpec, build_voxel_field
from graph_to_voxel.graph import EntityGraph
from graph_to_voxel.voxel import load_zarr, run_ensemble, save_zarr


app = typer.Typer(help="Build and inspect graph-to-voxel v1 artifacts.")


@app.command()
def build(
    graph_path: Path = typer.Argument(..., help="Input graph JSON file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output zarr path."),
    bounds: str = typer.Option(..., "--bounds", help="Grid bounds: xmin,xmax,ymin,ymax,zmin,zmax."),
    shape: str = typer.Option(..., "--shape", help="Grid shape: nx,ny,nz."),
    epsg: int | None = typer.Option(None, "--epsg", help="Optional EPSG code stored on the artifact."),
) -> None:
    graph = EntityGraph.from_file(graph_path)
    field = build_voxel_field(graph, _parse_grid(bounds, shape), epsg=epsg)
    save_zarr(field, output)
    typer.echo(str(output))


@app.command()
def ensemble(
    graph_path: Path = typer.Argument(..., help="Input graph JSON file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output reduced zarr path."),
    bounds: str = typer.Option(..., "--bounds", help="Grid bounds: xmin,xmax,ymin,ymax,zmin,zmax."),
    shape: str = typer.Option(..., "--shape", help="Grid shape: nx,ny,nz."),
    n: int = typer.Option(32, "--n", min=1, help="Number of accepted realisations."),
    seed: int = typer.Option(0, "--seed", help="Root random seed."),
    epsg: int | None = typer.Option(None, "--epsg", help="Optional EPSG code stored on the artifact."),
) -> None:
    graph = EntityGraph.from_file(graph_path)
    grid = _parse_grid(bounds, shape)
    result = run_ensemble(graph, lambda realised: build_voxel_field(realised, grid, epsg=epsg), n=n, seed=seed)
    save_zarr(result.reduce(), output)
    typer.echo(json.dumps({"output": str(output), "accepted": len(result.realisations), "rejected": result.n_rejected}))


@app.command()
def check(
    voxel_path: Path = typer.Argument(..., help="Input zarr voxel artifact."),
    graph: Path = typer.Option(..., "--graph", help="Graph JSON used to build the artifact."),
) -> None:
    entity_graph = EntityGraph.from_file(graph)
    field = load_zarr(voxel_path)
    typer.echo(json.dumps([result.to_dict() for result in run_all_checks(entity_graph, field)]))


@app.command()
def viz(voxel_path: Path = typer.Argument(..., help="Input zarr voxel artifact.")) -> None:
    from graph_to_voxel.viz import show_units

    plotter = show_units(load_zarr(voxel_path))
    plotter.show()


@app.command()
def render_slices(
    voxel_path: Path = typer.Argument(..., help="Input zarr voxel artifact."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for SVG slices."),
) -> None:
    from graph_to_voxel.viz import write_standard_slices

    rendered = write_standard_slices(load_zarr(voxel_path), output_dir)
    typer.echo(json.dumps([str(item.path) for item in rendered]))


def main() -> None:
    app()


def _parse_grid(bounds: str, shape: str) -> GridSpec:
    bounds_values = _parse_floats(bounds, 6, "bounds")
    shape_values = _parse_ints(shape, 3, "shape")
    return GridSpec(
        bounds=(
            (bounds_values[0], bounds_values[1]),
            (bounds_values[2], bounds_values[3]),
            (bounds_values[4], bounds_values[5]),
        ),
        shape=(shape_values[0], shape_values[1], shape_values[2]),
    )


def _parse_floats(value: str, count: int, label: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != count:
        raise typer.BadParameter(f"{label} requires {count} comma-separated numbers")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must contain only numbers") from exc


def _parse_ints(value: str, count: int, label: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != count:
        raise typer.BadParameter(f"{label} requires {count} comma-separated integers")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must contain only integers") from exc
