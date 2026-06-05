#!/usr/bin/env python3
"""
Experiment 1 — Reproduce the co-location/monoculture failure mode by driving the
REAL current scorer on REAL voxel layers.

Validates §1 of spatial-coherence-scoring-unified-2026-06-05.md and the
"Corrections" arithmetic (where the candidate-as-target term, not the complexity
penalty, is the real sparse-layer killer).

Run:
  LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH" \
    NSL2-geology-task/.venv/bin/python \
    NSL2-geology-task/scratch/scoring_validation_2026-06-05/exp1_reproduce.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

REPO = Path("/home/elijah/combo-geology-nsl")
sys.path.insert(0, str(REPO / "voxel-features-mcp"))

from voxel_features.store import GridSpec
from voxel_features import scoring as S

SHAPE = (200, 200, 8)
GRID = GridSpec(origin=(66.5, 49.5, 0.0), maximum=(71.5, 52.5, 80.0), shape=SHAPE, crs="EPSG:4326")
SEED = 42  # matches the default effective_seed when VFM_EPISODE_ID is unset

ADMITTED = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin/admitted/layers"
REJECTED = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin/rejected"


def load_flat(path: Path) -> np.ndarray:
    return np.load(path).astype(float).flatten()


def n_cols(flat: np.ndarray) -> int:
    """distinct (x,y) columns with any nonzero across depth."""
    g = flat.reshape(SHAPE)
    return int(np.count_nonzero(g.any(axis=2)))


def score_candidate(pool_vals, pool_dtypes, cand_flat, cand_dtype, *, seed=SEED):
    """Faithful replication of evaluate_new_layer's before/after core + 2-stage gate.

    Returns a dict with bic_delta, stage gates, and a decomposition of where the
    delta comes from (existing-target MAE change vs candidate-as-target term).
    """
    L = len(pool_vals)
    # score_before (fixed pool). For L==1 the real code uses a null model; our pools are >=2.
    if L == 1:
        score_before = S._single_layer_null_bic(pool_vals[0], pool_dtypes[0], GRID, SHAPE, seed=seed)
    else:
        score_before = S.geological_coherence_score(pool_vals, pool_dtypes, GRID, SHAPE, seed=seed)
    all_vals = list(pool_vals) + [cand_flat]
    all_dtypes = list(pool_dtypes) + [cand_dtype]
    score_after = S.geological_coherence_score(all_vals, all_dtypes, GRID, SHAPE, seed=seed)

    n_eff_before = int(score_before.get("n_effective_samples", 0) or 0)
    n_eff_after = int(score_after.get("n_effective_samples", 0) or 0)
    cmp_neff = min(max(n_eff_before, n_eff_after, 1), S._MAX_EFFECTIVE_SAMPLES)
    bic_before = S._bic_with_common_effective_samples(score_before, L, cmp_neff)
    bic_after = S._bic_with_common_effective_samples(score_after, L + 1, cmp_neff)
    bic_delta_raw = bic_after - bic_before
    bic_delta = bic_delta_raw / cmp_neff

    mae_before = score_before.get("system_mae", None)
    mae_after = score_after.get("system_mae", 0.0)
    if L < 2 or mae_before is None:
        stage1 = True
        mae_impr = 0.0
    else:
        mae_impr = mae_before - mae_after
        tol_used = (mae_impr >= -S._STAGE1_MAE_TOLERANCE) and (bic_delta <= S._STAGE1_BIC_RESCUE_THRESHOLD)
        stage1 = (mae_impr > 0) or tol_used
    admitted = bool(stage1 and bic_delta < 0.0)

    # decomposition
    maes_before = np.asarray(score_before.get("relative_mae_by_target", []), float)
    maes_after = np.asarray(score_after.get("relative_mae_by_target", []), float)
    cand_as_target_mae = float(maes_after[-1]) if maes_after.size else float("nan")
    # existing targets' mean MAE shift (same L targets, before vs after the candidate column is added)
    if maes_before.size and maes_after.size >= maes_before.size:
        existing_mae_shift = float(np.mean(maes_after[:maes_before.size]) - np.mean(maes_before))
    else:
        existing_mae_shift = float("nan")
    return dict(
        L=L, bic_delta=bic_delta, bic_delta_raw=bic_delta_raw,
        cmp_neff=cmp_neff, n_eff_before=n_eff_before, n_eff_after=n_eff_after,
        mae_before=(None if mae_before is None else float(mae_before)),
        mae_after=float(mae_after), mae_impr=float(mae_impr),
        stage1=bool(stage1), admitted=admitted,
        cand_as_target_mae=cand_as_target_mae,
        existing_mae_shift=existing_mae_shift,
        moran_before=float(score_before.get("spatial_correction", 1.0)),
        moran_after=float(score_after.get("spatial_correction", 1.0)),
    )


def make_blob(col_xy, z_list=range(8), val=1.0):
    g = np.zeros(SHAPE, float)
    x, y = col_xy
    for z in z_list:
        g[x, y, z] = val
    return g.flatten()


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("EXP1 — Reproduce co-location/monoculture failure mode (REAL scorer)")
    print("=" * 78)

    # ---- DEMO A: monoculture re-entry vs distributed rejection (controlled pool) ----
    print("\n### DEMO A: synthetic monoculture pool (3 identical co-located blobs)")
    blob_pool = [make_blob((70, 51)), make_blob((70, 51)), make_blob((70, 51))]
    pool_dt = ["float", "float", "float"]
    r_clone = score_candidate(blob_pool, pool_dt, make_blob((70, 51)), "float")
    print(f"  candidate = 4th identical (70,51) blob  -> bic_delta={r_clone['bic_delta']:+.4f} "
          f"stage1={r_clone['stage1']} ADMITTED={r_clone['admitted']}  "
          f"(cand_as_target_mae={r_clone['cand_as_target_mae']:.3f})")
    # a near-but-offset blob (co-located-ish)
    r_near = score_candidate(blob_pool, pool_dt, make_blob((71, 52)), "float")
    print(f"  candidate = offset (71,52) blob         -> bic_delta={r_near['bic_delta']:+.4f} "
          f"stage1={r_near['stage1']} ADMITTED={r_near['admitted']}  "
          f"(cand_as_target_mae={r_near['cand_as_target_mae']:.3f})")
    # a real distributed layer as candidate against the blob pool
    dist_path = ADMITTED / "host_suite_prospects_1780510063231.npy"
    r_dist = score_candidate(blob_pool, pool_dt, load_flat(dist_path), "float")
    print(f"  candidate = real distributed (25 cols)  -> bic_delta={r_dist['bic_delta']:+.4f} "
          f"stage1={r_dist['stage1']} ADMITTED={r_dist['admitted']}  "
          f"(cand_as_target_mae={r_dist['cand_as_target_mae']:.3f})")

    # ---- DEMO B: real 19-layer admitted pool vs real rejected distributed candidates ----
    print("\n### DEMO B: REAL 19-layer admitted pool; score rejected distributed candidates")
    pool_paths = sorted(ADMITTED.glob("*.npy"))
    pool_vals = [load_flat(p) for p in pool_paths]
    pool_dtypes = ["float"] * len(pool_vals)
    pool_cols = [n_cols(v) for v in pool_vals]
    print(f"  pool = {len(pool_vals)} layers; cols per layer: "
          f"min={min(pool_cols)} med={int(np.median(pool_cols))} max={max(pool_cols)}")

    # collect rejected candidates with a range of support sizes (distributed-ish)
    cand_records = []
    for ep in sorted(REJECTED.glob("ep_*")):
        md = ep / "metadata.json"
        npys = list(ep.glob("*.npy"))
        if not md.exists() or not npys:
            continue
        try:
            meta = json.loads(md.read_text())
        except Exception:
            continue
        ev = meta.get("evaluate", {})
        flat = load_flat(npys[0])
        nc = n_cols(flat)
        cand_records.append((ev.get("layer_name", npys[0].stem), nc,
                             int(np.count_nonzero(flat)), flat,
                             ev.get("bic_delta"), ev.get("relative_mae_mean"),
                             ev.get("masking_test_passed"), ev.get("rejection_stage")))
    # pick distributed ones (>=10 cols), spread across support size, cap to 8
    distrib = sorted([c for c in cand_records if c[1] >= 10], key=lambda c: -c[1])
    pick = distrib[:8]
    print(f"  scoring {len(pick)} distributed rejected candidates against the real pool:\n")
    print(f"  {'layer':36s} {'cols':>4s} {'nz':>5s} | {'bic_d(new)':>10s} {'s1':>3s} {'adm':>4s} "
          f"{'candMAE':>7s} {'exShift':>7s} | {'orig_bic_d':>10s} {'orig_stg':>8s}")
    for name, nc, nz, flat, obd, omae, omtp, ostage in pick:
        r = score_candidate(pool_vals, pool_dtypes, flat, "float")
        obd_s = f"{obd:+.3f}" if isinstance(obd, (int, float)) else "n/a"
        print(f"  {name[:36]:36s} {nc:4d} {nz:5d} | {r['bic_delta']:+10.4f} "
              f"{str(r['stage1'])[0]:>3s} {str(r['admitted'])[0]:>4s} "
              f"{r['cand_as_target_mae']:7.3f} {r['existing_mae_shift']:+7.3f} | "
              f"{obd_s:>10s} {str(ostage):>8s}")

    # ---- DEMO C: decompose one candidate's bic_delta ----
    print("\n### DEMO C: decomposition of bic_delta (per-sample) for one distributed candidate")
    name, nc, nz, flat, *_ = pick[0]
    r = score_candidate(pool_vals, pool_dtypes, flat, "float")
    L = r["L"]
    dk = (L + 1) * max(L, 1) - L * max(L - 1, 1)  # = 2L
    neff = r["cmp_neff"]
    complexity_persample = dk * np.log(max(neff, L + 1)) / neff
    print(f"  candidate '{name}' ({nc} cols, {nz} nz)")
    print(f"  L={L}  n_eff(clamped)={neff}  Δk=2L={dk}")
    print(f"  -> per-sample COMPLEXITY term  = Δk·ln(n_eff)/n_eff = {complexity_persample:.5f}")
    print(f"  -> candidate-as-target rel_MAE = {r['cand_as_target_mae']:.4f}  "
          f"(=> ~+2·sc·(ln2r+1) tax on bic_delta)")
    print(f"  -> existing-targets MAE shift  = {r['existing_mae_shift']:+.5f}  (≈0 ⇒ no cross-lift)")
    print(f"  -> Moran sc before/after       = {r['moran_before']:.3f}/{r['moran_after']:.3f}")
    print(f"  -> total bic_delta             = {r['bic_delta']:+.4f}  (admit iff <0)")
    print(f"\n  INTERPRETATION: complexity≈{complexity_persample:.4f} is negligible at n_eff={neff};")
    print(f"  the positive bic_delta is driven by the candidate-as-target term + no cross-lift,")
    print(f"  exactly as the unified doc's Corrections section argues (D1 removes both).")


if __name__ == "__main__":
    main()
