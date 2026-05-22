from __future__ import annotations

from pathlib import Path

import xarray as xr

from graph_to_voxel.engine.voxel_field import VoxelField


def save_zarr(field: VoxelField, path: str | Path) -> None:
    field.to_xarray().to_zarr(Path(path), mode="w", consolidated=True)


def load_zarr(path: str | Path) -> VoxelField:
    dataset = xr.open_zarr(Path(path), consolidated=True)
    try:
        return VoxelField.from_xarray(dataset)
    finally:
        dataset.close()
