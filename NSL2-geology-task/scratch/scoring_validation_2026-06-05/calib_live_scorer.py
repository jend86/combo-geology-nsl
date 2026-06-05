#!/usr/bin/env python3
"""calib_live_scorer.py — calibrate the LIVE spatial_predictor_lift_v1 scorer
(voxel_features.scoring) against real Teniz voxel artifacts.

Unlike proto_objective.py (a standalone reference), this imports the ACTUAL
shipped scorer so the numbers are what the live run will produce.

modes:  validity | cost | founder | loo | all
  validity : self_validity_score over ALL real candidates, bucketed by size
             -> is the trivial-junk filter (tau_self) doing its job?
  cost     : wall-clock of a single full predictor-lift score (sparse vs dense)
  founder  : admission rate of distributed REJECTED candidates vs the 19-pool
  loo      : leave-one-out re-admission of the 19 admitted layers
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/elijah/combo-geology-nsl")
sys.path.insert(0, str(REPO / "voxel-features-mcp"))
from voxel_features import scoring as S  # noqa: E402

SHAPE = (200, 200, 8)
BASE = REPO / "NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin"
ADM = BASE / "admitted/layers"
REJ = BASE / "rejected"
TAU = S._SPATIAL_TAU_SELF


def load3d(p):
    return np.load(p).astype(float).reshape(SHAPE)


def ncol(f):
    return int(np.count_nonzero(f.any(axis=2)))


def nvox(f):
    return int(np.count_nonzero(f))


def load_admitted():
    return [(p.stem[:34], load3d(p)) for p in sorted(ADM.glob("*.npy"))]


def load_rejected():
    out = []
    for ep in sorted(REJ.glob("ep_*")):
        npys = list(ep.glob("*.npy"))
        if not npys:
            continue
        try:
            f = load3d(npys[0])
        except Exception:
            continue
        out.append((ep.name[5:18], f))
    return out


def bucket(c):
    if c == 0:
        return "0 empty"
    if c <= 2:
        return "1 point(1-2)"
    if c <= 30:
        return "2 small(3-30)"
    if c <= 400:
        return "3 mid(31-400)"
    return "4 large(>400)"


def validity_sweep():
    print(f"=== VALIDITY SWEEP (tau_self={TAU}) ===")
    adm = load_admitted()
    rej = load_rejected()
    print(f"admitted={len(adm)}  rejected_eps={len(rej)}")
    agg = defaultdict(lambda: [0, 0])
    smae_by = defaultdict(list)
    for tag, lst in [("ADM", adm), ("REJ", rej)]:
        for name, f in lst:
            c = ncol(f)
            smae = S.self_validity_score(f, SHAPE)
            b = bucket(c)
            agg[(tag, b)][1] += 1
            if smae < TAU:
                agg[(tag, b)][0] += 1
            smae_by[(tag, b)].append(smae)
    print(f"{'tag/bucket':24s} {'pass':>5s}/{'tot':<5s} {'pass%':>6s} {'med_smae':>9s}")
    for key in sorted(agg):
        tag, b = key
        p, t = agg[key]
        med = float(np.median(smae_by[key]))
        print(f"{tag + ' ' + b:24s} {p:>5d}/{t:<5d} {100 * p / max(t, 1):6.1f} {med:9.3f}")


def cost_test():
    print("\n=== COST (single full score) ===")
    adm = load_admitted()
    pool = [f for _, f in adm[:8]]
    names = [n for n, _ in adm[:8]]
    sparse = min(adm, key=lambda x: ncol(x[1]))[1]
    dense = max(adm, key=lambda x: ncol(x[1]))[1]
    for tag, cand in [("sparse", sparse), ("dense", dense)]:
        t = time.time()
        r = S.spatial_predictor_lift_score(
            [x.flatten() for x in pool], names, cand.flatten(), SHAPE
        )
        print(
            f"  {tag:8s} cols={ncol(cand):6d} L={len(pool)} -> {time.time() - t:5.2f}s "
            f"admitted={r['admitted']} note={r['score_note']}"
        )


def founder(sample=40):
    print("\n=== ADMISSION vs founder pool (19 admitted) ===")
    adm = load_admitted()
    rej = load_rejected()
    pool = [f.flatten() for _, f in adm]
    names = [n for n, _ in adm]
    rej = [(n, f) for n, f in rej if ncol(f) >= 3]
    rej.sort(key=lambda x: ncol(x[1]))
    step = max(1, len(rej) // sample)
    rej = rej[::step][:sample]
    n_adm = 0
    admits = []
    for n, f in rej:
        r = S.spatial_predictor_lift_score(pool, names, f.flatten(), SHAPE)
        if r["admitted"]:
            n_adm += 1
            admits.append(
                (n, ncol(f), r["self_relative_mae"],
                 r["candidate_predictor_lift_mean"], r["bic_delta"])
            )
    print(f"sampled={len(rej)} (distributed, cols>=3)  admitted={n_adm} "
          f"({100 * n_adm / max(len(rej), 1):.1f}%)")
    for a in admits[:30]:
        print(f"  ADMIT {a[0]:14s} cols={a[1]:5d} smae={a[2]:.3f} "
              f"lift={a[3]:+.4f} bicd={a[4]:+.4f}")


def loo():
    print("\n=== LOO re-admission on 19 admitted ===")
    adm = load_admitted()
    n = 0
    res = []
    for i, (name, f) in enumerate(adm):
        pool = [x.flatten() for j, (_, x) in enumerate(adm) if j != i]
        names = [nm for j, (nm, _) in enumerate(adm) if j != i]
        r = S.spatial_predictor_lift_score(pool, names, f.flatten(), SHAPE)
        if r["admitted"]:
            n += 1
        res.append((name, ncol(f), r["validity_passed"], r["admitted"],
                    r["candidate_predictor_lift_mean"], r["bic_delta"]))
    print(f"re-admitted {n}/{len(adm)}")
    for name, c, v, a, lift, bd in res:
        print(f"  {name:34s} cols={c:5d} valid={str(v):5s} admit={str(a):5s} "
              f"lift={lift:+.4f} bicd={bd:+.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("validity", "all"):
        validity_sweep()
    if mode in ("cost", "all"):
        cost_test()
    if mode in ("founder", "all"):
        founder()
    if mode in ("loo", "all"):
        loo()
    print("\n[done]")
