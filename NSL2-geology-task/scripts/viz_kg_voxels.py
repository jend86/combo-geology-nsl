#!/usr/bin/env python
"""2D visualisations of the admitted voxel layers in the current knowledge graph.

For every admitted layer (one .npy per KG node) this renders:
  - a per-layer figure: top-down map (max over depth), a lon-depth cross-section,
    and a histogram of nonzero voxel values;
and across the whole KG:
  - a montage of all top-down maps (chronological, annotated with BIC delta);
  - a co-location coverage map (# layers occupying each x,y cell);
  - a normalised aggregate-prospectivity map;
  - INDEX.md, a table of per-layer stats + hypothesis.

Axes: npy shape is (x=lon, y=lat, z=depth). Grid/CRS come from the store
index.json. Run inside the nix dev shell (`nix develop`) so numpy/matplotlib
find libstdc++.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path("data/kazakhstan/feature-hypothesis")
STORE = ROOT / "store/teniz_basin/admitted"
LAYERS_DIR = STORE / "layers"
INDEX_JSON = STORE / "index.json"
EXPERIMENTS = ROOT / "knowledge/teniz_basin/experiments.jsonl"
OUT = ROOT / "viz/kg_voxels_2026-06-06"

TS_RE = re.compile(r"_\d{13}$")


def short(name: str) -> str:
    return TS_RE.sub("", name)


def load_grid() -> dict:
    return json.loads(INDEX_JSON.read_text())["grid"]


def load_meta() -> dict[str, dict]:
    """layer_name -> latest experiment record (for annotations)."""
    meta: dict[str, dict] = {}
    for line in EXPERIMENTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ln = r.get("layer_name")
        if ln:
            meta[ln] = r  # last write wins
    return meta


def masked(a2d: np.ndarray) -> np.ndarray:
    return np.where(a2d != 0, a2d, np.nan)


def per_layer_figure(name: str, arr: np.ndarray, grid: dict, rec: dict, out_path: Path):
    o, m = grid["origin"], grid["maximum"]
    nz = arr[arr != 0]
    top = arr.max(axis=2)  # (lon, lat)
    section = arr.max(axis=1)  # (lon, depth)

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], wspace=0.32)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ededed")

    # top-down (max over depth)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        masked(top).T, origin="lower",
        extent=[o[0], m[0], o[1], m[1]], aspect="auto", cmap=cmap,
    )
    ax0.set_title("top-down  (max over depth)", fontsize=9)
    ax0.set_xlabel("lon"); ax0.set_ylabel("lat")
    fig.colorbar(im0, ax=ax0, shrink=0.85)

    # lon-depth section (max over lat); surface at top, depth increasing down
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        masked(section).T, origin="upper",
        extent=[o[0], m[0], m[2], o[2]], aspect="auto", cmap=cmap,
    )
    ax1.set_title("lon–depth section  (max over lat)", fontsize=9)
    ax1.set_xlabel("lon"); ax1.set_ylabel("depth (m)")
    fig.colorbar(im1, ax=ax1, shrink=0.85)

    # value histogram of nonzero voxels
    ax2 = fig.add_subplot(gs[0, 2])
    if nz.size:
        ax2.hist(nz, bins=40, color="#3b6ea5")
    ax2.set_title("nonzero voxel values", fontsize=9)
    ax2.set_xlabel("value"); ax2.set_ylabel("count")

    bic = rec.get("bic_delta")
    lift = rec.get("candidate_predictor_lift_mean")
    ent = rec.get("candidate_value_entropy")
    dlf = rec.get("depth_levels_filled")
    stat = (
        f"nonzero={int((arr != 0).sum())}/{arr.size}   "
        f"value=[{nz.min():.4g}, {nz.max():.4g}]" if nz.size else "EMPTY layer"
    )
    sub = (
        f"bicΔ={bic:+.4g}" if isinstance(bic, (int, float)) else "bicΔ=?"
    )
    if isinstance(lift, (int, float)):
        sub += f"   lift={lift:+.4g}"
    if isinstance(ent, (int, float)):
        sub += f"   entropy={ent:.3g}"
    if dlf is not None:
        sub += f"   depth_levels={dlf}"

    hyp = (rec.get("hypothesis") or "").strip().replace("\n", " ")
    if len(hyp) > 160:
        hyp = hyp[:157] + "..."
    fig.suptitle(f"{short(name)}\n{sub}    {stat}", fontsize=11, y=1.06)
    fig.text(0.5, -0.06, hyp, ha="center", va="top", fontsize=7.5, wrap=True, color="#444")

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def montage(names, arrays, grid, meta, out_path: Path):
    o, m = grid["origin"], grid["maximum"]
    n = len(names)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.7, nrows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ededed")
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off"); continue
        arr = arrays[i]
        ax.imshow(
            masked(arr.max(axis=2)).T, origin="lower",
            extent=[o[0], m[0], o[1], m[1]], aspect="auto", cmap=cmap,
        )
        bic = meta.get(names[i], {}).get("bic_delta")
        tag = f"\nbicΔ={bic:+.3g}" if isinstance(bic, (int, float)) else ""
        ax.set_title(f"{i+1}. {short(names[i])}{tag}", fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Teniz Basin KG — {n} admitted voxel layers (top-down, max over depth, chronological)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def coverage_and_aggregate(names, arrays, grid, out_cov: Path, out_agg: Path):
    o, m = grid["origin"], grid["maximum"]
    cov = np.zeros(arrays[0].shape[:2])
    agg = np.zeros(arrays[0].shape[:2])
    for arr in arrays:
        top = arr.max(axis=2)
        cov += (top != 0).astype(float)
        nz = top[top != 0]
        if nz.size:
            lo, hi = nz.min(), nz.max()
            norm = (top - lo) / (hi - lo) if hi > lo else (top != 0).astype(float)
            agg += np.where(top != 0, norm, 0.0)

    # coverage: how many layers occupy each x,y cell
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#ededed")
    im = ax.imshow(
        np.where(cov > 0, cov, np.nan).T, origin="lower",
        extent=[o[0], m[0], o[1], m[1]], aspect="auto", cmap=cmap,
        norm=Normalize(vmin=1, vmax=cov.max()),
    )
    ax.set_title(f"KG co-location: # of {len(names)} layers occupying each cell")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=ax, label="# layers")
    fig.savefig(out_cov, dpi=120, bbox_inches="tight"); plt.close(fig)

    # aggregate normalised prospectivity
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    cmap2 = plt.get_cmap("viridis").copy(); cmap2.set_bad("#ededed")
    im = ax.imshow(
        np.where(agg > 0, agg, np.nan).T, origin="lower",
        extent=[o[0], m[0], o[1], m[1]], aspect="auto", cmap=cmap2,
    )
    ax.set_title("KG aggregate (sum of per-layer min-max normalised top-down maps)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=ax, label="summed normalised value")
    fig.savefig(out_agg, dpi=120, bbox_inches="tight"); plt.close(fig)


def write_index(names, arrays, meta, grid, out_md: Path):
    lines = [
        "# Teniz Basin Knowledge Graph — admitted voxel layers",
        "",
        f"- Grid: shape {grid['shape']}, origin {grid['origin']}, maximum {grid['maximum']}, CRS {grid['crs']}",
        f"- Layers: {len(names)} (one .npy per admitted KG node)",
        "- Axes: (x=lon, y=lat, z=depth). Top-down = max over depth.",
        "",
        "| # | layer | bicΔ | lift_mean | entropy | nonzero | value range | depth lvls |",
        "|---|-------|------|-----------|---------|---------|-------------|-----------|",
    ]
    for i, (name, arr) in enumerate(zip(names, arrays), 1):
        r = meta.get(name, {})
        nz = arr[arr != 0]
        bic = r.get("bic_delta"); lift = r.get("candidate_predictor_lift_mean")
        ent = r.get("candidate_value_entropy"); dlf = r.get("depth_levels_filled")
        vr = f"[{nz.min():.3g}, {nz.max():.3g}]" if nz.size else "EMPTY"
        f = lambda v: f"{v:+.3g}" if isinstance(v, (int, float)) else "?"
        g = lambda v: f"{v:.3g}" if isinstance(v, (int, float)) else "?"
        lines.append(
            f"| {i} | {short(name)} | {f(bic)} | {f(lift)} | {g(ent)} | "
            f"{int((arr != 0).sum())} | {vr} | {dlf if dlf is not None else '?'} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main():
    grid = load_grid()
    meta = load_meta()
    index_layers = json.loads(INDEX_JSON.read_text())["layers"]
    # chronological order by added_timestamp
    order = sorted(index_layers, key=lambda n: index_layers[n].get("added_timestamp", ""))

    names, arrays = [], []
    for name in order:
        p = LAYERS_DIR / f"{name}.npy"
        if not p.exists():
            print("MISSING", p); continue
        names.append(name); arrays.append(np.load(p))

    (OUT / "layers").mkdir(parents=True, exist_ok=True)
    (OUT / "overview").mkdir(parents=True, exist_ok=True)

    for name, arr in zip(names, arrays):
        per_layer_figure(name, arr, grid, meta.get(name, {}), OUT / "layers" / f"{short(name)}.png")
        print("layer", short(name), arr.shape, "nz", int((arr != 0).sum()))

    montage(names, arrays, grid, meta, OUT / "overview" / "montage_topdown.png")
    coverage_and_aggregate(
        names, arrays, grid,
        OUT / "overview" / "kg_coverage.png",
        OUT / "overview" / "kg_aggregate_prospectivity.png",
    )
    write_index(names, arrays, meta, grid, OUT / "INDEX.md")
    print(f"\nDONE -> {OUT}  ({len(names)} layers)")


if __name__ == "__main__":
    main()
