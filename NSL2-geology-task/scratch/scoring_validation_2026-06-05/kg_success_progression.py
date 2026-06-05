#!/usr/bin/env python3
"""Success-rate progression (binned) + KG admission timeline for the live run."""
import json
import os
from pathlib import Path

import numpy as np

FH = Path("/home/elijah/combo-geology-nsl/NSL2-geology-task/data/kazakhstan/feature-hypothesis")
GEN = FH / "generations/generation_0"
ST = FH / "store/teniz_basin"
SHAPE = (200, 200, 8)


def cat(d):
    return (d.get("trajectory") or {}).get("termination_category", "?")


def nofeat(d):
    return (d.get("task_breakdown") or {}).get("no_feature")


recs = [json.loads(l) for l in open(GEN / "all_episodes.jsonl") if l.strip()]
recs.sort(key=lambda d: (d.get("completed_at") or ""))
n = len(recs)
print(f"=== SUCCESS-RATE PROGRESSION ({n} episodes, in completion order, windows of 25) ===")
print(f"{'window':14s} {'n':>3s} {'full-admit':>10s} {'SFT-yield':>9s} {'timeout':>8s} {'no_feat':>8s} {'med_dur':>8s}")
W = 25
for i in range(0, n, W):
    w = recs[i:i + W]
    m = len(w)
    fa = sum(1 for d in w if d.get("success") is True)
    sft = sum(1 for d in w if isinstance(d.get("score"), (int, float)) and d["score"] > 0)
    to = sum(1 for d in w if cat(d) == "inference_timeout")
    nf = sum(1 for d in w if nofeat(d) is True)
    durs = sorted(d.get("duration_seconds") for d in w if isinstance(d.get("duration_seconds"), (int, float)))
    md = int(durs[len(durs) // 2]) if durs else 0
    print(f"ep[{i:3d}-{i+m-1:3d}]    {m:>3d} {100*fa//m:>9d}% {100*sft//m:>8d}% {100*to//m:>7d}% {100*nf//m:>7d}% {md:>7d}s")

# overall
fa = sum(1 for d in recs if d.get("success") is True)
sft = sum(1 for d in recs if isinstance(d.get("score"), (int, float)) and d["score"] > 0)
to = sum(1 for d in recs if cat(d) == "inference_timeout")
print(f"{'OVERALL':14s} {n:>3d} {100*fa//n:>9d}% {100*sft//n:>8d}% {100*to//n:>7d}%")

print("\n=== KG PROGRESSION (admission timeline; index.json added_timestamp) ===")
idxjson = json.load(open(ST / "admitted/index.json"))
layers = idxjson.get("layers", {})
tl = sorted((v.get("added_timestamp", ""), k) for k, v in layers.items())
for ts, name in tl:
    p = ST / "admitted/layers" / f"{name}.npy"
    cols = vox = "?"
    if p.exists():
        f = np.load(p).astype(float).reshape(SHAPE)
        cols = int(f.any(axis=2).sum())
        vox = int((f != 0).sum())
    print(f"  +{ts}  {name[:42]:42s} cols={cols} vox={vox}")
print(f"KG size: {len(layers)} (crossbreed flips at min_features=6)")
print(f"train_data on disk: {os.popen(f'du -sh {FH}/train_data 2>/dev/null').read().split()[0] if os.path.exists(FH/'train_data') else '?'}")
