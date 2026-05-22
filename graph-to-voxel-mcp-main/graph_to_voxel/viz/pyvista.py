from __future__ import annotations

import numpy as np

from graph_to_voxel.engine.voxel_field import VoxelField


def to_pyvista_image_data(field: VoxelField):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("Install graph-to-voxel-mcp[viz] to use PyVista helpers") from exc

    grid = pv.ImageData()
    grid.dimensions = np.array(field.shape) + 1
    grid.origin = (_origin(field.x), _origin(field.y), _origin(field.z))
    grid.spacing = (_spacing(field.x), _spacing(field.y), _spacing(field.z))
    grid.cell_data["most_likely_unit"] = field.most_likely_unit.ravel(order="F")
    grid.cell_data["entropy"] = field.entropy.ravel(order="F")
    grid.cell_data["domain_mask"] = field.domain_mask.ravel(order="F").astype(np.uint8)
    for unit_idx, unit_id in enumerate(field.unit_ids):
        grid.cell_data[f"p_{unit_id}"] = field.unit_probs[unit_idx].ravel(order="F")
    return grid


def show_units(field: VoxelField, opacity_by_entropy: bool = True):
    grid = to_pyvista_image_data(field)
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("Install graph-to-voxel-mcp[viz] to use PyVista helpers") from exc
    plotter = pv.Plotter()
    opacity = "entropy" if opacity_by_entropy else 0.5
    plotter.add_volume(grid, scalars="most_likely_unit", opacity=opacity)
    return plotter


def show_entropy(field: VoxelField):
    grid = to_pyvista_image_data(field)
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("Install graph-to-voxel-mcp[viz] to use PyVista helpers") from exc
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="entropy")
    return plotter


def _spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    return float(np.mean(np.diff(values)))


def _origin(values: np.ndarray) -> float:
    return float(values[0] - _spacing(values) / 2.0)
