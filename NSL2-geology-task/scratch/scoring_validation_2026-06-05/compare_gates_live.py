"""Compare what the OLD admission gate (bic_delta < 0) vs the NEW gate
(lift > 0.005) admit, on the SAME scored candidates pulled from live run logs.

Goal: defend (or honestly qualify) "the new admits are better". The operational
criterion for "better": admit spatially-DISTRIBUTED layers with genuine held-out
cross-layer predictive lift; reject trivial low-voxel self-blobs. We can measure
voxel count (distribution) and lift (coherence proxy) per candidate; we do NOT
have ground-truth geology, so this is a proxy argument, stated as such.
"""
import re
import statistics as st

LOGS = [
    # prior run window (contains the rich crossbreed *_intersection children)
    "/tmp/claude-1000/-home-elijah-combo-geology-nsl/ce77ece6-9676-4f9c-ba56-5e6721ab76bc/tasks/bhmi8240q.output",
    # current calibrated 28-slot run
    "/tmp/claude-1000/-home-elijah-combo-geology-nsl/ce77ece6-9676-4f9c-ba56-5e6721ab76bc/tasks/bv195i8vh.output",
]

p_name = re.compile(r"'layer_name': '([^']+)'")
p_nz = re.compile(r"'candidate_nonzero_voxels': (\d+)")
p_lift = re.compile(r"'candidate_predictor_lift_mean': (-?[\d.eE+-]+)")
p_bic = re.compile(r"'bic_delta': (-?[\d.eE+-]+)")
p_fill = re.compile(r"'candidate_fill_fraction': (-?[\d.eE+-]+)")

rows = []
for log in LOGS:
    try:
        fh = open(log)
    except FileNotFoundError:
        continue
    for line in fh:
        if "'candidate_predictor_lift_mean'" not in line:
            continue
        nz, lf, bc = p_nz.search(line), p_lift.search(line), p_bic.search(line)
        if not (nz and lf and bc):
            continue
        nm = p_name.search(line)
        try:
            rows.append((
                nm.group(1) if nm else "?",
                int(nz.group(1)), float(lf.group(1)), float(bc.group(1)),
            ))
        except ValueError:
            continue

# de-dup identical (name,nz,lift,bic) tuples (same layer re-logged)
seen, uniq = set(), []
for r in rows:
    k = (r[0], r[1], round(r[2], 6), round(r[3], 5))
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)

def summ(s):
    if not s:
        return "n=0"
    nzs = [r[1] for r in s]
    return (f"n={len(s):2}  mean_nz={st.mean(nzs):6.0f}  median_nz={st.median(nzs):5.0f}  "
            f"min_nz={min(nzs)}  max_nz={max(nzs)}")

old = [r for r in uniq if r[3] < 0]          # OLD gate: bic_delta < 0
new = [r for r in uniq if r[2] > 0.005]      # NEW gate: lift > 0.005

print(f"Distinct scored candidates: {len(uniq)}\n")
print("ADMIT SET sizes & voxel distribution (voxels = how spatially distributed):")
print("  OLD gate  bic<0       :", summ(old))
print("  NEW gate  lift>0.005   :", summ(new))

unblocked = sorted([r for r in uniq if r[2] > 0.005 and r[3] >= 0], key=lambda r: -r[1])
nowrej = sorted([r for r in uniq if r[3] < 0 and r[2] <= 0.005], key=lambda r: r[1])

print(f"\n[A] NEW admits / OLD rejects  (lift>0.005 AND bic>=0) -- 'unblocked rich layers': {len(unblocked)}")
for r in unblocked:
    print(f"     {r[0][:46]:46} nz={r[1]:5} lift={r[2]:+.4f} bic={r[3]:+.4f}")
print(f"\n[B] OLD admits / NEW rejects  (bic<0 AND lift<=0.005) -- 'trivial now-blocked': {len(nowrej)}")
for r in nowrej:
    print(f"     {r[0][:46]:46} nz={r[1]:5} lift={r[2]:+.4f} bic={r[3]:+.4f}")

if unblocked and nowrej:
    print(f"\nSUMMARY: layers the NEW gate ADDS are median nz="
          f"{st.median([r[1] for r in unblocked]):.0f}; layers it DROPS are median nz="
          f"{st.median([r[1] for r in nowrej]):.0f}. "
          f"If ADDS >> DROPS in voxels, the swap trades trivial blobs for distributed layers.")
