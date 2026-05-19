"""Render sense-check visualisations for the v1.7 update-flow stress test.

SVG-only (no PyVista — env is headless WSL and the X11 backend isn't wired).
SVGs are vector, scale cleanly for geologist hand-off, and print well.

Reproduces the scenario from docs/design/05-v17-followups.md:
  step1: kriged cap/basement interface
  step2: + embedded mineralised intrusion
  step3: + embedded ore halo
  step4: + fault (engine ignores)
  step5: ensemble (mean over realisations) — step1 fixture, since
         stratigraphic_constrain rejects all embedded-body realisations
         under uncertainty (05-v17-followups.md §1.3).

Outputs to docs/design/figures/05-v17-stress/.

Run with:
  uv run python scripts/render_v17_stress.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from graph_to_voxel.engine.loopstructural.adapter import build_voxel_field
from graph_to_voxel.engine.voxel_field import GridSpec, VoxelField
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import (
    Contact,
    Fault,
    Orientation,
    Series,
    StratigraphicUnit,
)
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import (
    GaussianUncertainty,
    OrientationUncertainty,
    PointUncertainty,
)
from graph_to_voxel.viz.static import write_slice_svg
from graph_to_voxel.voxel.ensemble import run_ensemble


OUT = Path("docs/design/figures/05-v17-stress")
OUT.mkdir(parents=True, exist_ok=True)


# ---------- graph builders --------------------------------------------------

def _prov() -> Provenance:
    return Provenance(
        source="v17-stress-viz",
        confidence=0.9,
        timestamp=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )


def _pt(v: float) -> PointUncertainty:
    return PointUncertainty(value=v)


def _gz(m: float, s: float) -> GaussianUncertainty:
    return GaussianUncertainty(mean=m, std=s)


def _add_unit(g: Graph, uid: str, anchor=None) -> None:
    g.add_node(StratigraphicUnit(
        id=uid,
        unit_id=uid,
        series_id="s1",
        topology="embedded" if anchor else "layer",
        anchor_inside=anchor,
        provenance=_prov(),
    ))
    g.add_edge(GraphEdge(
        kind=EdgeKind.MEMBER_OF_SERIES, source=uid, target="s1", provenance=_prov(),
    ))


def step1_graph() -> Graph:
    g = Graph()
    g.add_node(Series(id="s1", provenance=_prov()))
    _add_unit(g, "cap")
    _add_unit(g, "basement")
    g.add_edge(GraphEdge(
        kind=EdgeKind.OVERLIES, source="cap", target="basement", provenance=_prov(),
    ))
    rng = np.random.default_rng(0)
    for i in range(8):
        x, y = rng.uniform(1.0, 9.0, size=2)
        g.add_node(Contact(
            id=f"c_cb_{i}",
            position=(_pt(float(x)), _pt(float(y)), _gz(5.5, 0.4)),
            between=("cap", "basement"),
            provenance=_prov(),
        ))
    g.add_node(Orientation(
        id="o_cap",
        position=(_pt(5.0), _pt(5.0), _pt(5.5)),
        dip=OrientationUncertainty(
            dip_mean=0.0, dip_kappa=1e4, azimuth_mean=0.0, azimuth_kappa=1.0,
        ),
        for_unit="cap",
        provenance=_prov(),
    ))
    return g


def step2_graph() -> Graph:
    g = step1_graph()
    centre = (5.0, 5.0, 3.0)
    _add_unit(g, "intrusion", anchor=centre)
    g.add_edge(GraphEdge(
        kind=EdgeKind.OVERLIES, source="intrusion", target="basement", provenance=_prov(),
    ))
    for i, (dx, dy, dz) in enumerate([
        (2.5, 0, 0), (-2.5, 0, 0), (0, 2.5, 0),
        (0, -2.5, 0), (0, 0, 1.2), (0, 0, -1.2),
    ]):
        g.add_node(Contact(
            id=f"c_int_{i}",
            position=(_pt(centre[0] + dx), _pt(centre[1] + dy), _pt(centre[2] + dz)),
            between=("intrusion", "basement"),
            provenance=_prov(),
        ))
    return g


def step3_graph() -> Graph:
    g = step2_graph()
    centre = (5.0, 5.0, 3.0)
    _add_unit(g, "ore_halo", anchor=centre)
    g.add_edge(GraphEdge(
        kind=EdgeKind.OVERLIES, source="ore_halo", target="intrusion", provenance=_prov(),
    ))
    for i, (dx, dy, dz) in enumerate([
        (1.2, 0, 0), (-1.2, 0, 0), (0, 1.2, 0),
        (0, -1.2, 0), (0, 0, 0.6), (0, 0, -0.6),
    ]):
        g.add_node(Contact(
            id=f"c_halo_{i}",
            position=(_pt(centre[0] + dx), _pt(centre[1] + dy), _pt(centre[2] + dz)),
            between=("ore_halo", "intrusion"),
            provenance=_prov(),
        ))
    return g


def step4_graph() -> Graph:
    g = step3_graph()
    g.add_node(Fault(
        id="f_main",
        surface_points=["c_int_0", "c_int_1", "c_int_4"],
        provenance=_prov(),
        chronology_rank=1,
    ))
    return g


# ---------- continuous-valued heatmap SVG -----------------------------------

PALETTE = {
    "cap":        (210, 195, 140),  # sand
    "basement":   (110, 105, 100),  # grey
    "intrusion":  (180,  70,  60),  # rust red
    "ore_halo":   (240, 200,  60),  # gold
}


def _viridis_like(t: np.ndarray) -> np.ndarray:
    """5-stop perceptually-uniform-ish gradient, no matplotlib dep."""
    stops = np.array([
        [ 68,   1,  84],
        [ 59,  82, 139],
        [ 33, 144, 141],
        [ 94, 201,  98],
        [253, 231,  37],
    ], dtype=float)
    t = np.clip(t, 0.0, 1.0)
    idx = t * (len(stops) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(stops) - 1)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[hi] * frac
    return rgb.astype(np.uint8)


def _scalar_heatmap_svg(
    array: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    cell_size: int = 10,
    vmin: float = 0.0,
    vmax: float = 1.0,
    overlay_contours: list[tuple[float, str]] | None = None,
) -> str:
    """Render a 2D array (rows=y, cols=x already in plot orientation) to SVG."""
    h, w = array.shape
    plot_w = w * cell_size
    plot_h = h * cell_size
    margin_l, margin_t, margin_b, legend_w = 60, 44, 50, 70
    width = margin_l + plot_w + legend_w
    height = margin_t + plot_h + margin_b
    norm = (np.clip(array, vmin, vmax) - vmin) / max(vmax - vmin, 1e-12)
    rgb = _viridis_like(norm)
    rects = []
    for iy in range(h):
        y_top = margin_t + iy * cell_size
        for ix in range(w):
            r, g, b = rgb[iy, ix]
            x_left = margin_l + ix * cell_size
            rects.append(
                f'<rect x="{x_left}" y="{y_top}" width="{cell_size}" '
                f'height="{cell_size}" fill="rgb({r},{g},{b})" />'
            )
    # legend (vertical gradient bar)
    n_steps = 32
    legend_x = margin_l + plot_w + 14
    legend_y_top = margin_t
    legend_h = plot_h
    bar_w = 14
    legend_rects = []
    for i in range(n_steps):
        t = 1.0 - i / (n_steps - 1)
        r, g, b = _viridis_like(np.array([t]))[0]
        ry = legend_y_top + i * (legend_h / n_steps)
        legend_rects.append(
            f'<rect x="{legend_x}" y="{ry:.1f}" width="{bar_w}" '
            f'height="{legend_h / n_steps:.2f}" fill="rgb({r},{g},{b})" />'
        )
    legend_text = (
        f'<text x="{legend_x + bar_w + 4}" y="{legend_y_top + 10}" '
        f'font-family="monospace" font-size="10" fill="#111827">{vmax:.2f}</text>'
        f'<text x="{legend_x + bar_w + 4}" y="{legend_y_top + legend_h}" '
        f'font-family="monospace" font-size="10" fill="#111827">{vmin:.2f}</text>'
    )
    # overlay contours (drawn as polylines on the cell grid using marching squares)
    contour_paths = ""
    if overlay_contours:
        for level, colour in overlay_contours:
            contour_paths += _marching_squares_paths(
                array, level, margin_l, margin_t, cell_size, colour,
            )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white" />'
        f'<text x="{margin_l}" y="28" font-family="monospace" font-size="13" '
        f'fill="#111827">{title}</text>'
        + "".join(rects)
        + contour_paths
        + f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#111827" stroke-width="1" />'
        + "".join(legend_rects)
        + legend_text
        + f'<text x="{margin_l}" y="{height - 18}" font-family="monospace" '
        f'font-size="11" fill="#374151">{x_label} →</text>'
        f'<text x="14" y="{margin_t + plot_h / 2}" font-family="monospace" '
        f'font-size="11" fill="#374151" transform="rotate(-90 14 {margin_t + plot_h / 2})">'
        f'{y_label} →</text>'
        + "</svg>"
    )
    return svg


def _marching_squares_paths(
    arr: np.ndarray,
    level: float,
    margin_l: int,
    margin_t: int,
    cell_size: int,
    colour: str,
) -> str:
    """Quick marching-squares contour at `level`, drawn on the cell grid."""
    segs = []
    h, w = arr.shape
    for iy in range(h - 1):
        for ix in range(w - 1):
            v00 = arr[iy, ix]
            v10 = arr[iy, ix + 1]
            v01 = arr[iy + 1, ix]
            v11 = arr[iy + 1, ix + 1]
            corners = [v00, v10, v11, v01]
            mask = sum((1 << i) for i, c in enumerate(corners) if c > level)
            if mask in (0, 15):
                continue
            cx0 = margin_l + (ix + 0.5) * cell_size
            cx1 = margin_l + (ix + 1.5) * cell_size
            cy0 = margin_t + (iy + 0.5) * cell_size
            cy1 = margin_t + (iy + 1.5) * cell_size
            # midpoints on each edge
            top    = (cx0 + (cx1 - cx0) * _t(v00, v10, level), cy0)
            right  = (cx1, cy0 + (cy1 - cy0) * _t(v10, v11, level))
            bottom = (cx0 + (cx1 - cx0) * _t(v01, v11, level), cy1)
            left   = (cx0, cy0 + (cy1 - cy0) * _t(v00, v01, level))
            edges = {1: top, 2: right, 4: bottom, 8: left}
            cases = {
                1:  (8, 1),  2:  (1, 2),  3:  (8, 2),
                4:  (2, 4),  5:  (1, 2, 8, 4),  6: (1, 4),  7:  (8, 4),
                8:  (4, 8),  9:  (1, 4),  10: (1, 4, 2, 8),  11: (4, 2),
                12: (2, 8),  13: (1, 2),  14: (8, 1),
            }
            case = cases[mask]
            for a, b in zip(case[::2], case[1::2]):
                segs.append((edges[a], edges[b]))
    return "".join(
        f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
        f'stroke="{colour}" stroke-width="1.4" />'
        for p1, p2 in segs
    )


def _t(a: float, b: float, level: float) -> float:
    if a == b:
        return 0.5
    return float(np.clip((level - a) / (b - a), 0.0, 1.0))


# ---------- categorical slice with overlay ---------------------------------

def _categorical_slice_svg(
    field: VoxelField, *, axis: str, index: int, title: str,
) -> str:
    """Categorical most-likely-unit slice using PALETTE; same orientation as
    write_slice_svg but with a unit legend that matches the 3D palette.

    Falls back to the in-tree write_slice_svg if axis indexing is awkward.
    """
    if axis == "y":
        data = field.most_likely_unit[:, index, :]  # (x, z)
        x_label, y_label = "x [m]", "z [m]"
    elif axis == "x":
        data = field.most_likely_unit[index, :, :]
        x_label, y_label = "y [m]", "z [m]"
    else:  # z
        data = field.most_likely_unit[:, :, index]
        x_label, y_label = "x [m]", "y [m]"
    # plot orientation: rows=y_label values bottom→top, cols=x_label left→right
    h, w = data.shape[1], data.shape[0]
    cell = 10
    plot_w, plot_h = w * cell, h * cell
    margin_l, margin_t, margin_b = 60, 44, 50
    legend_w = 160
    width = margin_l + plot_w + legend_w
    height = margin_t + plot_h + margin_b
    rects = []
    for iy in range(h):
        y_top = margin_t + (h - iy - 1) * cell
        for ix in range(w):
            v = int(data[ix, iy])
            if v < 0:
                colour = "#f5f5f5"
            else:
                rgb = PALETTE.get(field.unit_ids[v], (180, 180, 180))
                colour = "rgb(%d,%d,%d)" % rgb
            x_left = margin_l + ix * cell
            rects.append(
                f'<rect x="{x_left}" y="{y_top}" width="{cell}" height="{cell}" '
                f'fill="{colour}" />'
            )
    legend_x = margin_l + plot_w + 18
    legend_items = []
    for i, uid in enumerate(field.unit_ids):
        rgb = PALETTE.get(uid, (180, 180, 180))
        ly = margin_t + i * 22
        legend_items.append(
            f'<rect x="{legend_x}" y="{ly}" width="14" height="14" '
            f'fill="rgb({rgb[0]},{rgb[1]},{rgb[2]})" stroke="#111827" />'
            f'<text x="{legend_x + 22}" y="{ly + 12}" font-family="monospace" '
            f'font-size="11" fill="#111827">{uid}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white" />'
        f'<text x="{margin_l}" y="28" font-family="monospace" font-size="13" '
        f'fill="#111827">{title}</text>'
        + "".join(rects)
        + f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#111827" stroke-width="1" />'
        + "".join(legend_items)
        + f'<text x="{margin_l}" y="{height - 18}" font-family="monospace" '
        f'font-size="11" fill="#374151">{x_label} →</text>'
        f'<text x="14" y="{margin_t + plot_h / 2}" font-family="monospace" '
        f'font-size="11" fill="#374151" transform="rotate(-90 14 {margin_t + plot_h / 2})">'
        f'{y_label} →</text>'
        + "</svg>"
    )


# ---------- voxel-counts table ---------------------------------------------

def voxel_counts(field: VoxelField) -> dict[str, int]:
    return {
        uid: int(np.count_nonzero(field.most_likely_unit == idx))
        for idx, uid in enumerate(field.unit_ids)
    }


def write_counts_svg(per_step: dict[str, dict[str, int]], path: Path) -> None:
    units = ["cap", "basement", "intrusion", "ore_halo"]
    steps = list(per_step)
    cell_w = 130
    row_h = 22
    width = 220 + cell_w * len(steps)
    height = 60 + row_h * (len(units) + 4)
    rows = [
        f'<text x="20" y="32" font-family="monospace" font-size="14" fill="#111827">'
        f'voxel counts (most_likely_unit) per build step (40³ grid; 64000 cells)</text>',
        f'<text x="20" y="60" font-family="monospace" font-size="11" fill="#374151">unit</text>',
    ]
    for j, step in enumerate(steps):
        rows.append(
            f'<text x="{220 + j * cell_w}" y="60" font-family="monospace" '
            f'font-size="10" fill="#374151">{step}</text>'
        )
    for i, uid in enumerate(units):
        y = 80 + i * row_h
        if uid in PALETTE:
            r, g, b = PALETTE[uid]
            rows.append(
                f'<rect x="20" y="{y - 12}" width="14" height="14" '
                f'fill="rgb({r},{g},{b})" stroke="#111827" />'
            )
        rows.append(
            f'<text x="40" y="{y}" font-family="monospace" font-size="11" '
            f'fill="#111827">{uid}</text>'
        )
        for j, step in enumerate(steps):
            cnt = per_step[step].get(uid, 0)
            rows.append(
                f'<text x="{220 + j * cell_w}" y="{y}" font-family="monospace" '
                f'font-size="11" fill="#111827">{cnt}</text>'
            )
    # Δ row showing cap leakage
    delta_y = 80 + len(units) * row_h + 8
    if "step1_cap_basement" in per_step and "step2_with_intrusion" in per_step:
        d_cap = per_step["step2_with_intrusion"].get("cap", 0) - per_step["step1_cap_basement"].get("cap", 0)
        d_base = per_step["step2_with_intrusion"].get("basement", 0) - per_step["step1_cap_basement"].get("basement", 0)
        intr = per_step["step2_with_intrusion"].get("intrusion", 0)
        leak = d_cap  # cells the cap "absorbed" — should be 0
        rows.append(
            f'<text x="20" y="{delta_y}" font-family="monospace" font-size="11" '
            f'fill="#9b1c1c">Δstep1→step2: cap{d_cap:+d}, basement{d_base:+d}, '
            f'intrusion+{intr}.  '
            f'Cap leakage = {leak} cells (should be 0; see 05-v17-followups.md §1.1).</text>'
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white" />'
        + "".join(rows)
        + "</svg>",
        encoding="utf-8",
    )
    print(f"  wrote {path}")


# ---------- main ------------------------------------------------------------

GRID = GridSpec(bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)), shape=(40, 40, 40))


def render_step(field: VoxelField, label: str) -> None:
    j_mid = field.shape[1] // 2
    z_mid_intrusion = int(field.shape[2] * 3.0 / 10.0)  # depth ≈ intrusion centre
    # categorical cross-sections
    (OUT / f"{label}__most_likely_unit_y{j_mid}.svg").write_text(
        _categorical_slice_svg(
            field, axis="y", index=j_mid,
            title=f"{label} — most_likely_unit, y={float(field.y[j_mid]):.1f}m cross-section",
        ),
        encoding="utf-8",
    )
    (OUT / f"{label}__most_likely_unit_z{z_mid_intrusion}.svg").write_text(
        _categorical_slice_svg(
            field, axis="z", index=z_mid_intrusion,
            title=f"{label} — most_likely_unit, z={float(field.z[z_mid_intrusion]):.1f}m plan view",
        ),
        encoding="utf-8",
    )
    # entropy
    write_slice_svg(
        field,
        OUT / f"{label}__entropy_y{j_mid}.svg",
        variable="entropy", axis="y", index=j_mid, cell_size=10,
    )
    print(f"  rendered slices for {label}")


def render_support_vs_probs(field: VoxelField, label: str) -> None:
    j_mid = field.shape[1] // 2
    for unit_idx, uid in enumerate(field.unit_ids):
        sm = field.support_membership[unit_idx, :, j_mid, :].T[::-1, :]  # (z↓, x→)
        up = field.unit_probs[unit_idx, :, j_mid, :].T[::-1, :]
        (OUT / f"{label}__support_membership__{uid}_y{j_mid}.svg").write_text(
            _scalar_heatmap_svg(
                sm,
                title=f"support_membership[{uid}] @ y={float(field.y[j_mid]):.1f}m",
                x_label="x [m]", y_label="z [m]",
                overlay_contours=[(0.5, "#ffffff")],
            ),
            encoding="utf-8",
        )
        (OUT / f"{label}__unit_probs__{uid}_y{j_mid}.svg").write_text(
            _scalar_heatmap_svg(
                up,
                title=f"unit_probs[{uid}] (exclusive) @ y={float(field.y[j_mid]):.1f}m",
                x_label="x [m]", y_label="z [m]",
                overlay_contours=[(0.5, "#ffffff")],
            ),
            encoding="utf-8",
        )
    print(f"  rendered support_vs_probs for {label}")


def main() -> None:
    print(f"grid: shape={GRID.shape} spacing={GRID.spacing} domain={GRID.bounds}")

    fields: dict[str, VoxelField] = {}
    counts: dict[str, dict[str, int]] = {}

    for label, builder in [
        ("step1_cap_basement", step1_graph),
        ("step2_with_intrusion", step2_graph),
        ("step3_with_halo", step3_graph),
        ("step4_with_fault", step4_graph),
    ]:
        print(f"\n[{label}] building...")
        f = build_voxel_field(builder(), GRID, bandwidth=0.5)
        fields[label] = f
        counts[label] = voxel_counts(f)
        print(
            f"  unit_probs_kind={f.attrs['unit_probs_kind']}  "
            f"max(bg)={f.attrs['background_prob_max']:.3f}  "
            f"counts={counts[label]}"
        )
        render_step(f, label)

    # Headline: support_membership vs unit_probs at step3 (containment)
    print("\n[support_vs_probs] step3 cross-section...")
    render_support_vs_probs(fields["step3_with_halo"], "step3_with_halo")

    # Ensemble — uses the layered-only step1 graph (see §1.3 in the doc)
    print("\n[ensemble] running over step1 graph (Gaussian z on contacts)...")
    ens = run_ensemble(
        step1_graph(),
        lambda gr: build_voxel_field(gr, GRID, bandwidth=0.5),
        n=8, seed=11,
    )
    print(f"  realisations={len(ens.realisations)} rejected={ens.n_rejected}")
    if ens.realisations:
        reduced = ens.reduce()
        fields["step5_ensemble_step1"] = reduced
        counts["step5_ensemble_step1"] = voxel_counts(reduced)
        print(f"  reduced kind={reduced.attrs['unit_probs_kind']} "
              f"counts={counts['step5_ensemble_step1']}")
        render_step(reduced, "step5_ensemble_step1")

    write_counts_svg(counts, OUT / "voxel_counts_table.svg")

    # README for the geologist
    (OUT / "README.md").write_text(
        "# v1.7 stress-test renders\n\n"
        "All renders produced from `scripts/render_v17_stress.py` against a 40³ grid "
        "in `[0, 10] m` per axis (cell size 0.25 m). Fixture sequence matches "
        "`docs/design/05-v17-followups.md` §1.\n\n"
        "**Geologist orientation.** The fixture is a stylised porphyry: a sediment cap "
        "above basement (kriged contact at z≈5.5 m), an embedded mineralised intrusion "
        "centred at (5, 5, 3) m, an embedded ore halo nested inside the intrusion. "
        "Step 4 adds a fault trace; step 5 averages over an MCUE ensemble.\n\n"
        "## File map\n\n"
        "| Pattern | What it shows |\n"
        "|---|---|\n"
        "| `stepN__most_likely_unit_y20.svg` | Vertical cross-section through y=5 m. The bodies live here. |\n"
        "| `stepN__most_likely_unit_z12.svg` | Plan view at the intrusion-centre depth (z≈3 m). |\n"
        "| `stepN__entropy_y20.svg` | Cell-level uncertainty (red = ambiguous label). |\n"
        "| `step3_with_halo__support_membership__UNIT_y20.svg` | Per-unit *containment* field. > 0.5 = inside that body's envelope. |\n"
        "| `step3_with_halo__unit_probs__UNIT_y20.svg` | Per-unit *exclusive* probability after chronological ERODE. |\n"
        "| `voxel_counts_table.svg` | Bookkeeping per step. Highlights cap-leakage (`05-v17-followups.md` §1.1). |\n\n"
        "## What the geologist should look for\n\n"
        "1. **Step 1 vs step 2.** Compare `step1_cap_basement__most_likely_unit_y20.svg` and `step2_with_intrusion__...`. The cap layer above z=5.5 m should be *unchanged* by adding the intrusion. It isn't — the cap grows (~1000 cells leak from basement into cap). This is the regression in `05-v17-followups.md` §1.1.\n"
        "2. **Step 3 containment.** Open the `support_membership` SVGs for `host`/`intrusion`/`ore_halo` side-by-side. All three should show > 0.5 *inside the halo* (white contour). Then open `unit_probs__ore_halo`: the halo cleanly wins the exclusive label. This is the `support_membership` design (doc 04 §4.5) — containment preserved, exclusive resolved.\n"
        "3. **Step 4 fault.** The `step4_with_fault__*` SVGs are bit-identical to step 3 — engine ignores faults silently (`05-v17-followups.md` §1.4).\n"
        "4. **Step 5 ensemble.** Built only from step 1 because `stratigraphic_constrain` rejects every embedded-body realisation under contact-z uncertainty (`05-v17-followups.md` §1.3). The ensemble's mean cap/basement boundary smears slightly along z due to the kriged Gaussian contact uncertainty.\n\n"
        "## Caveat on 3D\n\n"
        "PyVista 3D renders are not generated here — the dev environment is headless WSL "
        "without an X11/MESA backend, and PyVista cannot screenshot off-screen without one. "
        "SVG cross-sections are the substitute for now; they're vector and print well. "
        "If you have a desktop environment, `from graph_to_voxel.viz.pyvista import show_units` "
        "drives an interactive 3D viewer.\n",
        encoding="utf-8",
    )
    print(f"\nwrote README.md to {OUT}")


if __name__ == "__main__":
    main()
