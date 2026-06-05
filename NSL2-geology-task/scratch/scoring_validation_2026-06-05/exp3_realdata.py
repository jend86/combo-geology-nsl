#!/usr/bin/env python3
"""
Exp3 — Prototype objective on REAL Teniz Basin layers (make-or-break).

A. Pairwise cross-coherence matrix of the real 19-layer admitted pool:
   does the basin actually contain exploitable cross-layer spatial structure?
   (If not, cross-only predictor-lift admits nothing -> redesign changes the
   rejection REASON but does not unblock the run. This is the key risk.)
B. Leave-one-out calibrated admission of the real distributed admitted layers.
C. Real REJECTED distributed candidates scored (calibrated) against the pool.
D. Dense-blanket vs sparse cost (O(N) full-field conv claim).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proto_objective as P
from proto_objective import (Cfg, score_predictor_lift, calibrated_admit, self_validity_mae,
                             conv_features, buffered_block_relmae, eval_indices,
                             union_bbox, make_block_labels, _stack)

REPO = Path("/home/elijah/combo-geology-nsl")
ADMITTED = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin/admitted/layers"
REJECTED = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin/rejected"
SHAPE = (200, 200, 8)

# real cell ~2.2 km horiz; scales ~ {7,18,44} km ; regional ~20vox is the distant lever
CFG = Cfg(shape=SHAPE, scales_vox=(3.0, 8.0, 20.0), Rv_vox=0.8, truncate=2.0,
          n_blocks_xy=5, cv_buffer_vox=6, self_scales_vox=(3.0, 8.0),
          matched_zero_ratio=1.0, ridge_alpha=1e-2, tau_self=0.9, max_predictor_layers=6)


def load3d(path):
    return np.load(path).astype(float).reshape(SHAPE)


def cols(f):
    return int(np.count_nonzero(f.any(axis=2)))


def pairwise_coherence(target, predictor, cfg, bbox, labels):
    """coherence = 1 - rel_mae(target | single predictor), held out + matched zeros."""
    idx = eval_indices(target, bbox, cfg.matched_zero_ratio)
    if len(idx) < 12:
        return float("nan")
    y = target[idx[:, 0], idx[:, 1], idx[:, 2]]
    feats = conv_features(predictor, cfg.scales_vox, cfg.Rv_vox, cfg.truncate)
    X = _stack(feats, idx)
    rb, _ = buffered_block_relmae(idx, y, X, labels, cfg.cv_buffer_vox, cfg.ridge_alpha)
    return 1.0 - rb


def main():
    np.set_printoptions(precision=3, suppress=True)
    t0 = time.time()
    paths = sorted(ADMITTED.glob("*.npy"))
    fields = [load3d(p) for p in paths]
    names = [p.stem[:26] for p in paths]
    ncols = [cols(f) for f in fields]
    print("REAL POOL (19 admitted layers):")
    for n, c in sorted(zip(names, ncols), key=lambda t: -t[1]):
        print(f"   {c:6d} cols  {n}")

    # ---------- A. pairwise cross-coherence ----------
    print("\n" + "=" * 76 + "\nA. PAIRWISE CROSS-COHERENCE  (1 - rel_mae(i | j), held-out)\n" + "=" * 76)
    bbox = union_bbox(fields)
    labels = make_block_labels(SHAPE, bbox, CFG.n_blocks_xy)
    # restrict to non-empty, non-blanket targets (degenerate otherwise)
    keep = [i for i, c in enumerate(ncols) if 5 <= c <= 5000]
    print(f"   bbox={bbox}  scorable targets (5<=cols<=5000): {len(keep)}/{len(fields)}")
    M = np.full((len(keep), len(keep)), np.nan)
    pairs = []
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            if i == j:
                continue
            coh = pairwise_coherence(fields[i], fields[j], CFG, bbox, labels)
            M[a, b] = coh
            if np.isfinite(coh):
                pairs.append((coh, names[i], names[j]))
    finite = M[np.isfinite(M)]
    print(f"   coherence stats over {finite.size} ordered pairs: "
          f"max={finite.max():.3f} p95={np.percentile(finite,95):.3f} "
          f"median={np.median(finite):.3f} frac>0.05={np.mean(finite>0.05):.2f} "
          f"frac>0.15={np.mean(finite>0.15):.2f}")
    pairs.sort(reverse=True)
    print("   strongest cross-predictive pairs (target <- predictor):")
    for coh, ti, pj in pairs[:8]:
        print(f"      coh={coh:+.3f}  {ti:26s} <- {pj}")

    # ---------- B. leave-one-out calibrated admission of real distributed layers ----------
    print("\n" + "=" * 76 + "\nB. LEAVE-ONE-OUT calibrated admission of real DISTRIBUTED admitted layers\n" + "=" * 76)
    distrib = [i for i in keep if 15 <= ncols[i] <= 400][:8]
    print(f"   {'layer':28s} {'cols':>4s} {'self_mae':>8s} {'bic_delta':>10s} {'null_p5':>8s} {'cal.admit':>9s}")
    nadm = 0
    for i in distrib:
        others = [fields[j] for j in keep if j != i]
        onames = [names[j] for j in keep if j != i]
        r, adm, thr = calibrated_admit(others, onames, fields[i], CFG, K=8)
        nadm += int(adm)
        print(f"   {names[i]:28s} {ncols[i]:4d} {r.self_rel_mae:8.3f} {r.bic_delta:+10.4f} "
              f"{thr:+8.3f} {adm!s:>9s}")
    print(f"   -> {nadm}/{len(distrib)} real admitted distributed layers re-admit under prototype")

    # ---------- C. real REJECTED distributed candidates ----------
    print("\n" + "=" * 76 + "\nC. REAL REJECTED distributed candidates (calibrated) vs current-scorer reject\n" + "=" * 76)
    recs = []
    for ep in sorted(REJECTED.glob("ep_*")):
        md = ep / "metadata.json"; npys = list(ep.glob("*.npy"))
        if not md.exists() or not npys:
            continue
        try:
            ev = json.loads(md.read_text()).get("evaluate", {})
        except Exception:
            continue
        f = load3d(npys[0]); c = cols(f)
        if 15 <= c <= 400:
            recs.append((ev.get("layer_name", npys[0].stem)[:28], c, f, ev.get("bic_delta")))
    recs.sort(key=lambda r: r[1])
    pick = recs[:: max(1, len(recs) // 6)][:6]
    print(f"   {len(recs)} mid-range rejected candidates; scoring {len(pick)} (calibrated K=8)")
    print(f"   {'layer':28s} {'cols':>4s} {'self_mae':>8s} {'bic_delta':>10s} {'null_p5':>8s} {'cal.admit':>9s} {'orig_bicd':>9s}")
    for name, c, f, obd in pick:
        r, adm, thr = calibrated_admit(fields, names, f, CFG, K=8)
        obd_s = f"{obd:+.2f}" if isinstance(obd, (int, float)) else "n/a"
        print(f"   {name:28s} {c:4d} {r.self_rel_mae:8.3f} {r.bic_delta:+10.4f} "
              f"{thr:+8.3f} {adm!s:>9s} {obd_s:>9s}")

    # ---------- D. cost ----------
    print("\n" + "=" * 76 + "\nD. COST: dense blanket vs sparse candidate (O(N) full-field conv)\n" + "=" * 76)
    blanket = next((f for f in fields if cols(f) > 10000), None)
    sparse = next((f for f in fields if 15 <= cols(f) <= 100), None)
    for tag, cand in [("sparse(~tens cols)", sparse), ("dense blanket(>10k cols)", blanket)]:
        if cand is None:
            continue
        t = time.time()
        score_predictor_lift(fields[:8], names[:8], cand, CFG)
        print(f"   {tag:26s} cols={cols(cand):6d}  score time={time.time()-t:.2f}s")

    print(f"\n[total elapsed {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
