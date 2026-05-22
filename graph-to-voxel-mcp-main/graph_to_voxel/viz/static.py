from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from graph_to_voxel.engine.voxel_field import VoxelField


SliceAxis = Literal["x", "y", "z"]
SliceVariable = Literal["most_likely_unit", "entropy"]


@dataclass(frozen=True, slots=True)
class RenderedSlice:
    path: Path
    variable: SliceVariable
    axis: SliceAxis
    index: int


def write_standard_slices(field: VoxelField, output_dir: str | Path) -> list[RenderedSlice]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedSlice] = []
    for variable in ("most_likely_unit", "entropy"):
        for axis in ("x", "y", "z"):
            index = _axis_length(field, axis) // 2
            name = _slice_name(variable, axis, index, field)
            path = output / name
            write_slice_svg(field, path, variable=variable, axis=axis, index=index)
            rendered.append(RenderedSlice(path=path, variable=variable, axis=axis, index=index))
    return rendered


def write_slice_svg(
    field: VoxelField,
    path: str | Path,
    variable: SliceVariable = "most_likely_unit",
    axis: SliceAxis = "z",
    index: int | None = None,
    cell_size: int = 18,
) -> Path:
    selected_index = _axis_length(field, axis) // 2 if index is None else index
    data, x_label, y_label = _slice_data(field, variable, axis, selected_index)
    height_cells, width_cells = data.shape[1], data.shape[0]
    plot_width = width_cells * cell_size
    plot_height = height_cells * cell_size
    margin_left = 16
    margin_top = 44
    margin_bottom = 36
    legend_width = 220
    width = margin_left + plot_width + legend_width
    height = margin_top + plot_height + margin_bottom

    rects = []
    for ix in range(width_cells):
        for iy in range(height_cells):
            y = margin_top + (height_cells - iy - 1) * cell_size
            x = margin_left + ix * cell_size
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{_colour_for_value(field, variable, data[ix, iy])}" />'
            )

    legend = _legend_svg(field, variable, margin_left + plot_width + 18, margin_top)
    title = f"{variable} {axis}-slice index {selected_index}"
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{margin_left}" y="24" font-family="monospace" font-size="16" '
            f'fill="#111827">{_escape(title)}</text>',
            *rects,
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
            f'height="{plot_height}" fill="none" stroke="#111827" stroke-width="1" />',
            f'<text x="{margin_left}" y="{height - 12}" font-family="monospace" '
            f'font-size="12" fill="#374151">horizontal: {_escape(x_label)}; vertical: {_escape(y_label)}</text>',
            legend,
            "</svg>",
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")
    return destination


def _slice_name(variable: SliceVariable, axis: SliceAxis, index: int, field: VoxelField) -> str:
    if axis == "x":
        coord = field.x[index]
    elif axis == "y":
        coord = field.y[index]
    else:
        coord = field.z[index]
    return f"{variable}_{axis}{index:03d}_{coord:.3f}m.svg"


def _slice_data(
    field: VoxelField,
    variable: SliceVariable,
    axis: SliceAxis,
    index: int,
) -> tuple[np.ndarray, str, str]:
    data = getattr(field, variable)
    if index < 0 or index >= _axis_length(field, axis):
        raise IndexError(f"slice index {index} outside {axis} axis")
    if axis == "x":
        return data[index, :, :], "y", "z"
    if axis == "y":
        return data[:, index, :], "x", "z"
    return data[:, :, index], "x", "y"


def _axis_length(field: VoxelField, axis: SliceAxis) -> int:
    return {"x": field.shape[0], "y": field.shape[1], "z": field.shape[2]}[axis]


def _colour_for_value(field: VoxelField, variable: SliceVariable, value: float) -> str:
    if variable == "entropy":
        return _gradient(float(np.clip(value, 0.0, 1.0)))
    value_int = int(value)
    if value_int < 0:
        return "#d1d5db"
    palette = [
        "#d97706",
        "#2563eb",
        "#16a34a",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#4d7c0f",
        "#be185d",
    ]
    return palette[value_int % len(palette)]


def _gradient(value: float) -> str:
    start = np.array([8, 48, 107], dtype=float)
    mid = np.array([66, 146, 198], dtype=float)
    end = np.array([255, 255, 191], dtype=float)
    if value < 0.5:
        colour = start + (mid - start) * (value / 0.5)
    else:
        colour = mid + (end - mid) * ((value - 0.5) / 0.5)
    r, g, b = [int(round(component)) for component in colour]
    return f"#{r:02x}{g:02x}{b:02x}"


def _legend_svg(field: VoxelField, variable: SliceVariable, x: int, y: int) -> str:
    lines = [
        f'<text x="{x}" y="{y}" font-family="monospace" font-size="13" '
        f'fill="#111827">legend</text>'
    ]
    if variable == "entropy":
        for idx, value in enumerate(np.linspace(0.0, 1.0, 6)):
            yy = y + 22 + idx * 22
            lines.append(f'<rect x="{x}" y="{yy - 12}" width="18" height="18" fill="{_gradient(float(value))}" />')
            lines.append(
                f'<text x="{x + 26}" y="{yy + 2}" font-family="monospace" font-size="12" '
                f'fill="#374151">{value:.1f}</text>'
            )
        return "\n".join(lines)

    for idx, unit_id in enumerate(field.unit_ids):
        yy = y + 22 + idx * 22
        lines.append(
            f'<rect x="{x}" y="{yy - 12}" width="18" height="18" '
            f'fill="{_colour_for_value(field, variable, idx)}" />'
        )
        lines.append(
            f'<text x="{x + 26}" y="{yy + 2}" font-family="monospace" font-size="12" '
            f'fill="#374151">{idx}: {_escape(unit_id)}</text>'
        )
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
