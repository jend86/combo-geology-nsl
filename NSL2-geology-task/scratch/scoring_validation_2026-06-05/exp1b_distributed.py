#!/usr/bin/env python3
"""Exp1b — score MID-RANGE distributed rejected layers (the doc's 40-80 pt headline
regime) against the real 19-pool, plus the real admitted distributed layers as a
control. Reuses exp1.score_candidate (real scorer)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exp1_reproduce import (load_flat, n_cols, score_candidate, ADMITTED, REJECTED)

pool_paths = sorted(ADMITTED.glob("*.npy"))
pool_vals = [load_flat(p) for p in pool_paths]
pool_dtypes = ["float"] * len(pool_vals)

# gather rejected candidates with their recorded verdicts, mid-range support
recs = []
for ep in sorted(REJECTED.glob("ep_*")):
    md = ep / "metadata.json"; npys = list(ep.glob("*.npy"))
    if not md.exists() or not npys:
        continue
    try:
        meta = json.loads(md.read_text())
    except Exception:
        continue
    ev = meta.get("evaluate", {})
    flat = load_flat(npys[0]); nc = n_cols(flat)
    if 15 <= nc <= 400:  # sparse-distributed regime (exclude blobs<15 and blankets)
        recs.append((ev.get("layer_name", npys[0].stem), nc, int(np.count_nonzero(flat)),
                     flat, ev.get("bic_delta"), ev.get("relative_mae_mean"),
                     ev.get("masking_test_passed"), ev.get("rejection_stage")))

# spread across the support range, cap 8
recs.sort(key=lambda r: r[1])
pick = recs[:: max(1, len(recs) // 8)][:8] if recs else []

print(f"pool=19; {len(recs)} mid-range[15-400 col] rejected candidates available; scoring {len(pick)}")
print(f"{'layer':34s} {'cols':>4s} {'nz':>5s} | {'bic_d(new)':>10s} {'s1':>2s} {'adm':>3s} "
      f"{'candMAE':>7s} {'exShift':>7s} | {'orig_bic_d':>10s} {'orig_mae':>8s} {'orig_stg':>8s}")
adm = 0
for name, nc, nz, flat, obd, omae, omtp, ostage in pick:
    r = score_candidate(pool_vals, pool_dtypes, flat, "float")
    adm += int(r["admitted"])
    obd_s = f"{obd:+.3f}" if isinstance(obd, (int, float)) else "n/a"
    omae_s = f"{omae:.3f}" if isinstance(omae, (int, float)) else "n/a"
    print(f"{name[:34]:34s} {nc:4d} {nz:5d} | {r['bic_delta']:+10.4f} "
          f"{str(r['stage1'])[0]:>2s} {str(r['admitted'])[0]:>3s} "
          f"{r['cand_as_target_mae']:7.3f} {r['existing_mae_shift']:+7.3f} | "
          f"{obd_s:>10s} {omae_s:>8s} {str(ostage):>8s}")

# control: real ADMITTED distributed layers scored against the OTHER 18 (leave-one-out)
print(f"\nControl — leave-one-out on real ADMITTED layers (cols in [15,400]):")
print(f"{'layer':34s} {'cols':>4s} | {'bic_d':>10s} {'s1':>2s} {'adm':>3s} {'candMAE':>7s} {'exShift':>7s}")
for i, p in enumerate(pool_paths):
    flat = pool_vals[i]; nc = n_cols(flat)
    if not (15 <= nc <= 400):
        continue
    others = [v for j, v in enumerate(pool_vals) if j != i]
    odt = ["float"] * len(others)
    r = score_candidate(others, odt, flat, "float")
    print(f"{p.stem[:34]:34s} {nc:4d} | {r['bic_delta']:+10.4f} "
          f"{str(r['stage1'])[0]:>2s} {str(r['admitted'])[0]:>3s} "
          f"{r['cand_as_target_mae']:7.3f} {r['existing_mae_shift']:+7.3f}")
print(f"\nSUMMARY: {adm}/{len(pick)} mid-range rejected candidates would admit under CURRENT scorer.")
