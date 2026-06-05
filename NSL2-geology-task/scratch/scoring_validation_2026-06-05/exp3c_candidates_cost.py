#!/usr/bin/env python3
"""Exp3c — Part C (real REJECTED distributed candidates, calibrated) + Part D (cost),
using cached pool features so permutation calibration isn't 19x slower than needed."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from proto_objective import Cfg, score_predictor_lift, calibrated_admit, pool_features

REPO = Path("/home/elijah/combo-geology-nsl")
ADM = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin/admitted/layers"
REJ = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin/rejected"
SHAPE = (200, 200, 8)
CFG = Cfg(shape=SHAPE, scales_vox=(3.0, 8.0, 20.0), Rv_vox=0.8, truncate=2.0, n_blocks_xy=5,
          cv_buffer_vox=6, self_scales_vox=(3.0, 8.0), matched_zero_ratio=1.0, ridge_alpha=1e-2,
          tau_self=0.9, max_predictor_layers=6)
load3d = lambda p: np.load(p).astype(float).reshape(SHAPE)
ncol = lambda f: int(np.count_nonzero(f.any(axis=2)))

t0 = time.time()
paths = sorted(ADM.glob("*.npy"))
fields = [load3d(p) for p in paths]
names = [p.stem[:26] for p in paths]
pf = pool_features(fields, CFG)   # compute ONCE
print(f"pool=19 features cached in {time.time()-t0:.1f}s")

# Part C: real rejected distributed candidates
recs = []
for ep in sorted(REJ.glob("ep_*")):
    md = ep / "metadata.json"; npys = list(ep.glob("*.npy"))
    if not md.exists() or not npys:
        continue
    try:
        ev = json.loads(md.read_text()).get("evaluate", {})
    except Exception:
        continue
    f = load3d(npys[0]); c = ncol(f)
    if 15 <= c <= 400:
        recs.append((ev.get("layer_name", npys[0].stem)[:26], c, f, ev.get("bic_delta")))
recs.sort(key=lambda r: r[1])
pick = recs[:: max(1, len(recs) // 6)][:6]
print("\nC. REAL REJECTED distributed candidates (calibrated K=8, cached pool):")
print(f"   {'layer':26s} {'cols':>4s} {'self_mae':>8s} {'bic_delta':>10s} {'null_p5':>8s} {'cal.admit':>9s} {'orig':>6s}")
for name, c, f, obd in pick:
    r, adm, thr = calibrated_admit(fields, names, f, CFG, K=8, pf=pf)
    obd_s = f"{obd:+.2f}" if isinstance(obd, (int, float)) else "n/a"
    print(f"   {name:26s} {c:4d} {r.self_rel_mae:8.3f} {r.bic_delta:+10.4f} {thr:+8.3f} {adm!s:>9s} {obd_s:>6s}")

# Part D: cost dense vs sparse
print("\nD. COST (cached pool, single candidate score):")
blanket = next((f for f in fields if ncol(f) > 10000), None)
sparse = next((f for f in fields if 15 <= ncol(f) <= 100), None)
for tag, cand in [("sparse(~tens cols)", sparse), ("dense blanket(>10k)", blanket)]:
    if cand is None:
        continue
    t = time.time()
    score_predictor_lift(fields, names, cand, CFG, precomputed_pool_feats=pf)
    print(f"   {tag:22s} cols={ncol(cand):6d}  score={time.time()-t:.2f}s")
print(f"\n[total {time.time()-t0:.1f}s]")
