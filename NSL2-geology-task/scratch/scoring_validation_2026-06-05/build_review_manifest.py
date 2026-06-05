"""Build a review-ready manifest for the set-aside voxel layers.

For every admitted + rejected .npy in review_layers/, compute geometry stats,
join the held-out lift / bic_delta from the run logs, derive each gate's verdict,
and emit:
  - MANIFEST.csv            one row per layer (the review index)
  - csv/<set>__<layer>.csv  compact nonzero voxels as (lon,lat,depth_m,value)
  - png/<set>__<layer>.png  top-down (max-over-depth) map, if matplotlib present
"""
import csv
import glob
import json
import os
import re

import numpy as np

REVIEW = "/home/elijah/combo-geology-nsl/NSL2-geology-task/scratch/scoring_validation_2026-06-05/review_layers"
LOGS = [
    "/tmp/claude-1000/-home-elijah-combo-geology-nsl/ce77ece6-9676-4f9c-ba56-5e6721ab76bc/tasks/bhmi8240q.output",
    "/tmp/claude-1000/-home-elijah-combo-geology-nsl/ce77ece6-9676-4f9c-ba56-5e6721ab76bc/tasks/bv195i8vh.output",
]

grid = json.load(open(f"{REVIEW}/admitted/admitted_index.json"))["grid"]
o, m, s = grid["origin"], grid["maximum"], grid["shape"]


def to_lld(i, j, k):
    return (
        o[0] + (i + 0.5) * (m[0] - o[0]) / s[0],
        o[1] + (j + 0.5) * (m[1] - o[1]) / s[1],
        o[2] + (k + 0.5) * (m[2] - o[2]) / s[2],
    )


# lift / bic_delta per layer_name, scraped from the run logs
p_name = re.compile(r"'layer_name': '([^']+)'")
p_lift = re.compile(r"'candidate_predictor_lift_mean': (-?[\d.eE+-]+)")
p_bic = re.compile(r"'bic_delta': (-?[\d.eE+-]+)")
meta = {}
for log in LOGS:
    if not os.path.exists(log):
        continue
    for line in open(log):
        if "'candidate_predictor_lift_mean'" not in line:
            continue
        nm, lf, bc = p_name.search(line), p_lift.search(line), p_bic.search(line)
        if nm and lf and bc:
            try:
                meta[nm.group(1)] = (float(lf.group(1)), float(bc.group(1)))
            except ValueError:
                pass

os.makedirs(f"{REVIEW}/csv", exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    do_png = True
    os.makedirs(f"{REVIEW}/png", exist_ok=True)
except Exception:
    do_png = False

strip_ep = re.compile(r"^ep_\d+_[0-9a-f]+__")
rows = []
for sset in ("admitted", "rejected"):
    for f in sorted(glob.glob(f"{REVIEW}/{sset}/*.npy")):
        a = np.load(f)
        lname = strip_ep.sub("", os.path.basename(f)[:-4])
        nz = int((a != 0).sum())
        idx = np.argwhere(a != 0)
        rec = dict(set=sset, layer=lname, nz=nz, fill=round(nz / a.size, 6))
        if len(idx):
            llds = [to_lld(*p) for p in idx]
            lons, lats, deps = zip(*llds)
            vals = [float(a[i, j, k]) for i, j, k in idx]
            rec.update(
                vmin=round(float(min(vals)), 4), vmax=round(float(max(vals)), 4),
                lon_min=round(min(lons), 4), lon_max=round(max(lons), 4),
                lat_min=round(min(lats), 4), lat_max=round(max(lats), 4),
                dep_min=round(min(deps), 1), dep_max=round(max(deps), 1),
            )
            with open(f"{REVIEW}/csv/{sset}__{lname}.csv", "w", newline="") as ch:
                w = csv.writer(ch)
                w.writerow(["longitude", "latitude", "depth_m", "value"])
                for lo, la, de, va in zip(lons, lats, deps, vals):
                    w.writerow([f"{lo:.5f}", f"{la:.5f}", f"{de:.2f}", va])
            if do_png:
                plt.figure(figsize=(4, 3))
                plt.imshow(a.max(axis=2).T, origin="lower",
                           extent=[o[0], m[0], o[1], m[1]], aspect="auto", cmap="viridis")
                plt.title(f"{lname[:34]}\n{sset}  nz={nz}", fontsize=7)
                plt.colorbar(shrink=0.8)
                plt.savefig(f"{REVIEW}/png/{sset}__{lname}.png", dpi=80, bbox_inches="tight")
                plt.close()
        lift, bic = meta.get(lname, (None, None))
        rec["lift"] = lift
        rec["bic_delta"] = bic
        rec["OLD_gate_bic<0"] = (bic is not None and bic < 0)
        rec["NEW_gate_lift>0.005"] = (lift is not None and lift > 0.005)
        rows.append(rec)

cols = ["set", "layer", "nz", "fill", "vmin", "vmax", "lon_min", "lon_max",
        "lat_min", "lat_max", "dep_min", "dep_max", "lift", "bic_delta",
        "OLD_gate_bic<0", "NEW_gate_lift>0.005"]
with open(f"{REVIEW}/MANIFEST.csv", "w", newline="") as mh:
    w = csv.DictWriter(mh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["set"], -r["nz"])):
        w.writerow(r)

n_adm = sum(1 for r in rows if r["set"] == "admitted")
print(f"manifest rows={len(rows)} (admitted={n_adm}, rejected={len(rows)-n_adm})  png={do_png}")
print(f"with lift/bic joined: {sum(1 for r in rows if r['lift'] is not None)}/{len(rows)}")
