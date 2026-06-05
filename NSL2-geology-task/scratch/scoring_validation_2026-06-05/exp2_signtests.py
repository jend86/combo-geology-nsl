#!/usr/bin/env python3
"""
Exp2 (v2) — Prototype sign tests (the §11 TDD cases) + mechanism checks, using the
CORRECTED feature mechanism (full-field cross-features + buffered block CV;
leave-self-out only for the validity gate).

A FAIL is a design/implementation issue caught before writing production code.
Also documents the block-jackknife FEATURE-COLLAPSE finding that motivated the
correction.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proto_objective as P
from proto_objective import (Cfg, score_predictor_lift, self_validity_mae, calibrated_admit,
                             conv_features, blank, blob, scatter, shifted, union_bbox,
                             make_block_labels, eval_indices)
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, iterate_structure

SHAPE = (200, 200, 8)


def multiblob(centers, r=8, val=1.0, depths=range(2, 6), shape=SHAPE):
    f = blank(shape)
    for (cx, cy) in centers:
        f = np.maximum(f, blob((cx, cy), r=r, val=val, depths=depths, shape=shape))
    return f


def banner(s):
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)


CENTERS = [(60, 70), (90, 120), (130, 80), (150, 150), (75, 150), (120, 50)]
CFG = Cfg()  # defaults: scales (2,5,12), cv_buffer 4, self scales (2,5)/buffer 1


def show(tag, r, expect):
    ok = expect(r)
    lift = np.mean(list(r.lift_by_target.values())) if r.lift_by_target else float("nan")
    print(f"  [{'PASS' if ok else '**FAIL**'}] {tag:42s} bic_delta={r.bic_delta:+8.4f} "
          f"admit={r.admit!s:5s} self_mae={r.self_rel_mae:.3f} valid={r.valid!s:5s} meanlift={lift:+.4f}")
    return ok


def main():
    np.set_printoptions(precision=4, suppress=True)
    results = []
    A = multiblob(CENTERS, r=8)

    banner("SIGN TESTS (prototype of §11 TDD cases) — corrected mechanism")
    onept = blank(); onept[100, 100, 4] = 1.0
    for tag, cand, exp, label in [
        ("T1 offset-related (shift 6)", shifted(A, 0, 6), lambda r: r.admit, "T1 offset-related admits via lift"),
        ("T4 constant blanket", np.ones(SHAPE), lambda r: not r.valid, "T4 constant blanket rejected by validity"),
        ("T6a single point", onept, lambda r: not r.valid, "T6a single point rejected by validity"),
        ("T6b uniform scatter", scatter(120, 1, bbox=(30, 170, 30, 170)), lambda r: not r.valid, "T6b uniform scatter rejected by validity"),
    ]:
        r = score_predictor_lift([A], ["A"], cand, CFG)
        results.append((label, show(tag, r, exp)))

    # T5 distance sweep — RAW bic_delta<0 is too lax: unrelated structure earns a
    # noise-floor lift by marking where the target ISN'T. Shows why §6.6 calibration.
    banner("T5 DISTANCE SWEEP (raw threshold) — note the spurious noise floor")
    print(f"  {'offset(vox)':>11s} {'~km':>5s} {'bic_delta':>10s} {'meanlift':>9s} {'raw_admit':>9s}")
    for d in [4, 10, 18, 30, 50]:
        r = score_predictor_lift([A], ["A"], blob((60, 70 - d), r=6), CFG)
        lift = np.mean(list(r.lift_by_target.values())) if r.lift_by_target else float("nan")
        print(f"  {d:11d} {d*2.2:5.0f} {r.bic_delta:+10.4f} {lift:+9.4f} {r.admit!s:>9s}")

    # CALIBRATED admission: candidate must beat its own permuted-placement null (§6.6)
    banner("PERMUTATION-NULL CALIBRATION (§6.6 / Alt-4) — separates real lift from noise floor")
    print(f"  {'case':28s} {'bic_delta':>10s} {'null_p5':>9s} {'valid':>6s} {'cal.admit':>10s}")
    cal_cases = [
        ("T1 related (shift 6)", shifted(A, 0, 6), True),
        ("T2 isolated blob (far)", blob((178, 22), r=6), False),
        ("T5-far blob (offset 50)", blob((60, 20), r=6), False),
        ("wide false-positive blob", blob((100, 100), r=45), False),
    ]
    for label, cand, want in cal_cases:
        r, adm, thr = calibrated_admit([A], ["A"], cand, CFG, K=10)
        ok = (adm == want)
        print(f"  [{'PASS' if ok else '**FAIL**'}] {label:24s} {r.bic_delta:+10.4f} {thr:+9.4f} "
              f"{r.valid!s:>6s} {adm!s:>10s}")
        results.append((f"CAL {label}", ok))

    # T3 clone into N>=2 identical pool -> redundant, no admit
    r = score_predictor_lift([A.copy(), A.copy(), A.copy()], ["A1", "A2", "A3"], A.copy(), CFG)
    results.append(("T3 clone into 3 identical: no admit", show("T3 clone into 3 identical", r, lambda r: not r.admit)))
    # T3b clone of sole layer DOES admit (only the near-dup guard catches it; per design)
    r = score_predictor_lift([A], ["A"], A.copy(), CFG)
    results.append(("T3b clone of sole layer admits (guard's job)", show("T3b clone of sole layer", r, lambda r: r.admit)))

    banner("MATCHED-ZEROS EFFECT (F7)")
    wide = blob((100, 100), r=45)
    r_mz = score_predictor_lift([A], ["A"], wide, Cfg(matched_zero_ratio=1.0))
    r_so = score_predictor_lift([A], ["A"], wide, Cfg(matched_zero_ratio=0.0))
    print(f"  wide positive blob (covers A's zeros):")
    print(f"    signal-only (ratio=0):   bic_delta={r_so.bic_delta:+.4f} admit={r_so.admit}")
    print(f"    matched-zeros (ratio=1): bic_delta={r_mz.bic_delta:+.4f} admit={r_mz.admit}")
    mz_ok = r_mz.bic_delta > r_so.bic_delta
    print(f"  [{'PASS' if mz_ok else '**FAIL**'}] matched-zeros penalizes false-positive blanket "
          f"(Δ={r_mz.bic_delta - r_so.bic_delta:+.4f})")
    results.append(("F7 matched-zeros penalizes blanket", mz_ok))

    # ---- M1: documents the block-jackknife FEATURE COLLAPSE that motivated the fix ----
    banner("MECHANISM: block-jackknife (buffer>=support) collapses held-out features")
    field = A
    bbox = union_bbox([field]); labels = make_block_labels(SHAPE, bbox, CFG.n_blocks_xy)
    sigma = (12.0, 12.0, 0.8); support = int(np.ceil(CFG.truncate * 12.0))  # regional
    # full-field feature (v2): nonzero at held-out voxels
    full = conv_features(field, (12.0,), CFG.Rv_vox, CFG.truncate)[0]
    # block-jackknife with buffer>=support (doc §3.3): mask block+buffer, read held-out block
    obs = (field != 0).astype(float)
    num_full = gaussian_filter(field, sigma, truncate=CFG.truncate, mode="constant")
    den_full = gaussian_filter(obs, sigma, truncate=CFG.truncate, mode="constant")
    st = iterate_structure(generate_binary_structure(2, 1), support)
    jk = np.zeros_like(field)
    for b in np.unique(labels[labels >= 0]):
        col = (labels == b); buf = binary_dilation(col, structure=st)
        m3 = np.repeat(buf[:, :, None], SHAPE[2], axis=2).astype(float)
        nb = num_full - gaussian_filter(field * m3, sigma, truncate=CFG.truncate, mode="constant")
        db = den_full - gaussian_filter(obs * m3, sigma, truncate=CFG.truncate, mode="constant")
        gb = nb / (db + P.EPS); sel = np.repeat(col[:, :, None], SHAPE[2], axis=2)
        jk[sel] = gb[sel]
    sig = field != 0
    full_frac = float(np.mean(np.abs(full[sig]) > 1e-6))
    jk_frac = float(np.mean(np.abs(jk[sig]) > 1e-6))
    print(f"  regional sigma=12 (support={support}), block width ~{(bbox[1]-bbox[0])//CFG.n_blocks_xy}")
    print(f"  fraction of SIGNAL voxels with nonzero regional feature:")
    print(f"    full-field (v2 mechanism):           {full_frac:6.1%}")
    print(f"    block-jackknife buffer>=support (doc): {jk_frac:6.1%}   <-- collapse")
    m1_ok = (full_frac > 0.9) and (jk_frac < 0.5)
    print(f"  [{'PASS' if m1_ok else '**FAIL**'}] full-field keeps features; block-jackknife collapses them")
    results.append(("M1 block-jackknife feature collapse documented", m1_ok))

    # ---- M2: leave-self-out behaves (single point ->0, blob ->coherent) ----
    banner("MECHANISM: leave-self-out validity gate")
    one = blank(); one[100, 100, 4] = 1.0
    smae_pt = self_validity_mae(one, CFG)
    smae_blob = self_validity_mae(blob((100, 100), r=8), CFG)
    smae_A = self_validity_mae(A, CFG)
    print(f"  self_mae: single_point={smae_pt:.3f} (expect ~1), solid_blob={smae_blob:.3f} (expect <0.9), "
          f"multiblob_A={smae_A:.3f}")
    m2_ok = (smae_pt > 0.9) and (smae_blob < 0.9)
    print(f"  [{'PASS' if m2_ok else '**FAIL**'}] leave-self-out: point unpredictable, blob coherent")
    results.append(("M2 leave-self validity gate behaves", m2_ok))

    banner("SUMMARY")
    npass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else '**FAIL**'}  {name}")
    print(f"\n  {npass}/{len(results)} checks passed")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\n[elapsed {time.time()-t0:.1f}s]")
