"""READ-ONLY: would re-applying the calibrated criteria to the already-rejected
candidates grow the KG? Re-scores each set-aside reject against the CURRENT KG
with the new scorer, applies the lift-primary gate, and checks novelty. No writes.

Answers "is a retro beneficial": admitted-AND-novel = real KG gain; admitted-but-
near-dup = no gain (novelty would dedup it in a write-back too).
"""
import glob
import json
import os
import re

import numpy as np
from voxel_features import scoring
from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask

STORE = "/home/elijah/combo-geology-nsl/NSL2-geology-task/data/kazakhstan/feature-hypothesis/store/teniz_basin"
REJECT = "/home/elijah/combo-geology-nsl/NSL2-geology-task/scratch/scoring_validation_2026-06-05/review_layers/rejected"
NOVELTY = 0.15
ADMIT_MIN_LIFT = 0.005

grid = json.load(open(f"{STORE}/admitted/index.json"))["grid"]
shape = tuple(grid["shape"])

kg_files = sorted(glob.glob(f"{STORE}/admitted/layers/*.npy"))
kg_names = [os.path.basename(f)[:-4] for f in kg_files]
kg_vals = [np.load(f).ravel() for f in kg_files]
kg_arrs = [np.load(f) for f in kg_files]
print(f"current KG = {len(kg_vals)} layers")

strip_ep = re.compile(r"^ep_\d+_[0-9a-f]+__")
results = []
for f in sorted(glob.glob(f"{REJECT}/*.npy")):
    a = np.load(f)
    name = strip_ep.sub("", os.path.basename(f)[:-4])
    nz = int((a != 0).sum())
    if nz == 0:
        continue
    try:
        sc = scoring.spatial_predictor_lift_score(
            kg_vals, kg_names, a.ravel(), shape, ridge_alpha=1e-2, null_permutations=0
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  score-fail {name}: {exc}")
        continue
    lift = float(sc.get("candidate_predictor_lift_mean", 0.0) or 0.0)
    bic = float(sc.get("bic_delta", 0.0) or 0.0)
    valid = bool(sc.get("validity_passed", False))
    admit = scoring.predictor_lift_admission_decision(
        validity_passed=valid, lift_mean=lift, bic_delta=bic, admit_min_lift=ADMIT_MIN_LIFT
    )
    min_d, novel = None, False
    if admit:
        dists = []
        for ka in kg_arrs:
            if ka.shape != a.shape:
                continue
            try:
                dists.append(float(FeatureHypothesisKazakhstanTask._normalised_pairwise_distance(a, ka)))
            except Exception:  # noqa: BLE001
                pass
        min_d = min(dists) if dists else 1.0
        novel = min_d >= NOVELTY
    results.append((name, nz, lift, bic, valid, admit, min_d, novel))

would_grow = [r for r in results if r[7]]
admit_dup = [r for r in results if r[5] and not r[7]]
print(f"\nre-scored {len(results)} non-empty rejects vs current KG")
print(f"new gate ADMITS: {sum(1 for r in results if r[5])}")
print(f"  -> NOVEL (would actually GROW the KG): {len(would_grow)}")
print(f"  -> near-dup (admitted but novelty would dedup): {len(admit_dup)}")
print("\n=== RETRO CANDIDATES (admit + novel -> real KG gain) ===")
for r in sorted(would_grow, key=lambda r: -r[1]):
    print(f"  {r[0][:46]:46} nz={r[1]:5} lift={r[2]:+.4f} bic={r[3]:+.4f} min_novelty_dist={r[6]:.3f}")
print("\n=== admitted-but-near-dup (NO KG gain; novelty dedups) ===")
for r in sorted(admit_dup, key=lambda r: -r[1])[:12]:
    md = f"{r[6]:.3f}" if r[6] is not None else "n/a"
    print(f"  {r[0][:46]:46} nz={r[1]:5} lift={r[2]:+.4f} min_novelty_dist={md}")
