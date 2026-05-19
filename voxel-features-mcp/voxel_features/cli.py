"""CLI for voxel feature store."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from voxel_features.store import VoxelStore, GridSpec, COE_FAIRBAIRN_GRID
from voxel_features.knowledge_graph import KnowledgeGraph
from voxel_features import scoring

app = typer.Typer(help="Voxel feature store CLI")


@app.command()
def init(
    store_path: Path = typer.Argument(..., help="Path to store directory"),
    grid: str = typer.Option("coe-fairbairn", help="Grid preset or JSON spec"),
):
    """Initialize a new voxel store."""
    if grid == "coe-fairbairn":
        grid_spec = COE_FAIRBAIRN_GRID
    else:
        grid_spec = GridSpec.from_dict(json.loads(grid))
    
    store = VoxelStore(store_path, grid_spec)
    typer.echo(f"Initialized store at {store_path}")
    typer.echo(f"Grid: {grid_spec.shape} voxels")


@app.command()
def info(store_path: Path = typer.Argument(..., help="Path to store directory")):
    """Show store info."""
    store = VoxelStore(store_path)
    
    typer.echo(f"Store: {store_path}")
    typer.echo(f"Grid: {store.grid.shape}")
    typer.echo(f"Layers: {len(store.layer_names)}")
    
    for layer in store.list_layers():
        typer.echo(f"  - {layer['name']} ({layer['dtype']})")


@app.command()
def mdl(store_path: Path = typer.Argument(..., help="Path to store directory")):
    """Compute MDL of store."""
    store = VoxelStore(store_path)
    mdl_bits = scoring.compute_mdl(store)
    
    typer.echo(f"MDL: {mdl_bits:.2f} bits")
    typer.echo(f"Layers: {len(store.layer_names)}")


@app.command()
def export_training(
    kg_path: Path = typer.Argument(..., help="Path to knowledge graph"),
    output: Path = typer.Argument(..., help="Output JSONL path"),
):
    """Export training data from knowledge graph."""
    kg = KnowledgeGraph(kg_path)
    count = kg.export_training_data(output)
    
    typer.echo(f"Exported {count} records to {output}")
    
    stats = kg.stats()
    typer.echo(f"Total experiments: {stats['total_experiments']}")
    typer.echo(f"Admission rate: {stats['admission_rate']:.1%}")


if __name__ == "__main__":
    app()
