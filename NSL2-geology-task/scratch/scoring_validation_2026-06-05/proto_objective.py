#!/usr/bin/env python3
"""
Prototype of the UNIFIED objective (Approach A) — v2, after the first validation
pass revealed that the doc's block-jackknife FEATURE machinery (§3.1/§3.3/§6.3)
is (a) unnecessary under D1 and (b) self-contradictory as written.

KEY CORRECTION (validated empirically in exp2):
  Under D1 (cross-only predictor-lift) the predictor features are smooths of
  OTHER layers, so they contain NO target signal. Therefore:
    * CROSS features can be computed on the FULL field (no per-block masking).
      The only leakage that matters is in the FIT, handled by a *buffered* block
      train/test split (standard spatial CV; buffer ~ autocorrelation range, NOT
      >= kernel support).
    * "buffer >= largest kernel support" (doc §3.3) is WRONG for held-out
      *features*: it makes the compact kernel unable to reach any unmasked voxel,
      so held-out features collapse to 0 and nothing is predictable.
    * LEAVE-SELF-OUT is needed only for the SELF-VALIDITY gate (where the feature
      IS a smooth of the target); there we use small local scales + a tiny buffer.

This v2 keeps the design's *objective* (cross-only predictor-lift, self=validity
gate, multi-scale kernel, matched zeros, honest per-target BIC) but fixes the
feature/leakage mechanism. Differences from the doc are reported as findings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import (gaussian_filter, binary_dilation,
                           generate_binary_structure, iterate_structure)

EPS = 1e-9
RELATIVE_MAE_FLOOR = 1e-3


@dataclass
class Cfg:
    shape: tuple = (200, 200, 8)
    scales_vox: tuple = (2.0, 5.0, 12.0)   # cross-feature horizontal sigma (local/neigh/regional)
    Rv_vox: float = 0.8                     # vertical sigma (a few depth voxels max)
    truncate: float = 2.0
    n_blocks_xy: int = 4
    cv_buffer_vox: int = 4                   # buffered spatial-CV gap (~autocorr range), NOT >=support
    self_scales_vox: tuple = (2.0, 5.0)      # validity gate scales (local/neighborhood)
    self_buffer_vox: int = 1                 # tiny: leave-self handled analytically, this guards CV
    matched_zero_ratio: float = 1.0
    ridge_alpha: float = 1e-2
    tau_self: float = 0.9
    max_predictor_layers: int | None = None   # §6.5 parsimony cap; None = use all pool layers


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def union_bbox(fields, pad=0):
    acc = np.zeros(fields[0].shape[:2], bool)
    for f in fields:
        acc |= f.any(axis=2)
    xs, ys = np.where(acc)
    if xs.size == 0:
        nx, ny = fields[0].shape[:2]
        return 0, nx, 0, ny
    nx, ny = fields[0].shape[:2]
    return (max(0, int(xs.min()) - pad), min(nx, int(xs.max()) + 1 + pad),
            max(0, int(ys.min()) - pad), min(ny, int(ys.max()) + 1 + pad))


def make_block_labels(shape, bbox, k):
    nx, ny, nz = shape
    x0, x1, y0, y1 = bbox
    lab = -np.ones((nx, ny), int)
    xe = np.linspace(x0, x1, k + 1).astype(int)
    ye = np.linspace(y0, y1, k + 1).astype(int)
    bid = 0
    for i in range(k):
        for j in range(k):
            lab[xe[i]:xe[i + 1], ye[j]:ye[j + 1]] = bid
            bid += 1
    return lab


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _center_weight(sigma, truncate, nz):
    n = 21
    z = max(nz, 1)
    imp = np.zeros((n, n, z))
    imp[n // 2, n // 2, z // 2] = 1.0
    out = gaussian_filter(imp, sigma, truncate=truncate, mode="constant")
    return float(out[n // 2, n // 2, z // 2])


def conv_features(field, scales, Rv, truncate, leave_self=False):
    """Multi-scale normalized convolution on the FULL field.
    leave_self=True subtracts the voxel's own contribution (for the self gate)."""
    obs = (field != 0).astype(float)
    feats = []
    for s in scales:
        sigma = (s, s, Rv)
        num = gaussian_filter(field, sigma, truncate=truncate, mode="constant")
        den = gaussian_filter(obs, sigma, truncate=truncate, mode="constant")
        if leave_self:
            k0 = _center_weight(sigma, truncate, field.shape[2])
            num = num - k0 * field
            den = den - k0 * obs
        feats.append(num / (den + EPS))
    return feats


def _stack(feat_list, idx):
    cols = [G[idx[:, 0], idx[:, 1], idx[:, 2]] for G in feat_list]
    return np.column_stack(cols) if cols else np.zeros((len(idx), 0))


# --------------------------------------------------------------------------- #
# ridge + buffered block CV
# --------------------------------------------------------------------------- #
def _ridge(X, y, alpha):
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    A = Xc.T @ Xc
    reg = alpha * np.eye(p + 1); reg[0, 0] = 0.0
    return np.linalg.solve(A + reg, Xc.T @ y)


def _predict(X, beta):
    return np.column_stack([np.ones(X.shape[0]), X]) @ beta


def buffered_block_relmae(idx, y, X, labels, buffer_vox, alpha):
    """K-fold-by-block held-out relative MAE with a buffered train/test gap.
    Train rows within `buffer_vox` (column distance) of the test block are dropped.
    """
    block_of = labels[idx[:, 0], idx[:, 1]]
    blocks = np.unique(block_of[block_of >= 0])
    st = iterate_structure(generate_binary_structure(2, 1), buffer_vox) if buffer_vox > 0 else None
    err_p = err_n = 0.0
    n_used = 0
    for b in blocks:
        test = block_of == b
        if test.sum() < 1:
            continue
        col_b = (labels == b)
        buf2d = binary_dilation(col_b, structure=st) if st is not None else col_b
        in_buf = buf2d[idx[:, 0], idx[:, 1]]
        train = (~test) & (~in_buf) & (block_of >= 0)
        if train.sum() < 5:
            continue
        if X.shape[1] == 0:
            pred = np.full(int(test.sum()), y[train].mean())
        else:
            beta = _ridge(X[train], y[train], alpha)
            pred = _predict(X[test], beta)
        null = y[train].mean()
        err_p += np.abs(y[test] - pred).sum()
        err_n += np.abs(y[test] - null).sum()
        n_used += int(test.sum())
    if err_n <= 1e-10 or n_used == 0:
        return 1.0, n_used
    return max(err_p / err_n, RELATIVE_MAE_FLOOR), n_used


# --------------------------------------------------------------------------- #
# eval voxel set: signal + matched zeros within union bbox
# --------------------------------------------------------------------------- #
def eval_indices(target_field, bbox, ratio):
    x0, x1, y0, y1 = bbox
    region = np.zeros(target_field.shape, bool)
    region[x0:x1, y0:y1, :] = True
    sig = (target_field != 0) & region
    pos = np.array(np.where(sig)).T
    zero = (target_field == 0) & region
    zidx = np.array(np.where(zero)).T
    n_neg = int(ratio * len(pos))
    if len(zidx) > n_neg and n_neg > 0:
        stride = max(1, len(zidx) // n_neg)
        zidx = zidx[::stride][:n_neg]
    return np.vstack([pos, zidx]) if len(pos) else zidx


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    bic_delta: float
    n_total: int
    rel_mae_before: dict = field(default_factory=dict)
    rel_mae_after: dict = field(default_factory=dict)
    self_rel_mae: float = float("nan")
    valid: bool = True
    lift_by_target: dict = field(default_factory=dict)
    admit: bool = False
    note: str = ""


def _laplace_bic(rel_maes, n_list, k_list):
    rel = np.clip(np.asarray(rel_maes, float), RELATIVE_MAE_FLOOR, None)
    n = np.asarray(n_list, float)
    k = np.asarray(k_list, float)
    logL = float(np.sum(-n * (np.log(2.0 * rel) + 1.0)))
    return -2.0 * logL + float(np.sum(k * np.log(np.maximum(n, 2.0))))


def self_validity_mae(field, cfg: Cfg):
    """Validity = does the leave-self-out neighborhood predict presence?

    IN-SAMPLE relative MAE with LEAVE-SELF features (the feature already excludes
    the voxel, so there is no self-leakage). We are testing *coherence*, not
    generalization, so no block CV is needed — which also makes the gate robust
    for COMPACT structures (block-CV folds go degenerate on a tiny bbox). Only the
    cross predictor-LIFT needs folds (to avoid overfitting a useless column).
    """
    bbox = union_bbox([field], pad=int(2 * max(cfg.self_scales_vox)))
    feats = conv_features(field, cfg.self_scales_vox, cfg.Rv_vox, cfg.truncate, leave_self=True)
    idx = eval_indices(field, bbox, cfg.matched_zero_ratio)
    if len(idx) < 8:
        return 1.0
    y = field[idx[:, 0], idx[:, 1], idx[:, 2]]
    X = _stack(feats, idx)
    if X.shape[1] == 0 or np.allclose(X, 0.0):
        return 1.0
    beta = _ridge(X, y, cfg.ridge_alpha)
    pred = _predict(X, beta)
    err_p = float(np.abs(y - pred).sum())
    err_n = float(np.abs(y - y.mean()).sum())
    if err_n <= 1e-10:
        return 1.0
    return max(err_p / err_n, RELATIVE_MAE_FLOOR)


def pool_features(pool_fields, cfg: Cfg):
    """Precompute full-field cross-features for a fixed pool (reuse across candidates)."""
    return [conv_features(f, cfg.scales_vox, cfg.Rv_vox, cfg.truncate) for f in pool_fields]


def score_predictor_lift(pool_fields, pool_names, candidate_field, cfg: Cfg, self_gate=True,
                         precomputed_pool_feats=None):
    """Cross-only predictor-lift admission (Approach A), v2 mechanism.

    precomputed_pool_feats: pass pool_features(pool_fields, cfg) once to avoid
    recomputing pool convolutions on every candidate (e.g. permutation calibration).
    """
    L = len(pool_fields)
    all_fields = list(pool_fields) + [candidate_field]
    bbox = union_bbox(all_fields)
    labels = make_block_labels(cfg.shape, bbox, cfg.n_blocks_xy)

    # validity gate (self-prediction, leave-self, small scales/buffer)
    self_mae = self_validity_mae(candidate_field, cfg)
    valid = self_mae < cfg.tau_self
    if self_gate and not valid:
        return Result(float("inf"), 0, self_rel_mae=self_mae, valid=False, admit=False,
                      note="rejected_by_validity_gate")

    # cross features (FULL field) for pool + candidate
    feats_pool = precomputed_pool_feats if precomputed_pool_feats is not None \
        else [conv_features(f, cfg.scales_vox, cfg.Rv_vox, cfg.truncate) for f in pool_fields]
    feats_cand = conv_features(candidate_field, cfg.scales_vox, cfg.Rv_vox, cfg.truncate)

    rb_, ra_, n_, kb_, ka_, lift = {}, {}, {}, {}, {}, {}
    for t in range(L):
        name = pool_names[t]
        idx = eval_indices(pool_fields[t], bbox, cfg.matched_zero_ratio)
        if len(idx) < 8:
            continue
        y = pool_fields[t][idx[:, 0], idx[:, 1], idx[:, 2]]
        others = [j for j in range(L) if j != t]
        if cfg.max_predictor_layers is not None and len(others) > cfg.max_predictor_layers:
            # §6.5 parsimony: keep the K predictor layers most relevant to this
            # target (|corr| of their neighborhood-scale feature with y). Selection
            # uses POOL layers only, so before/after stay nested (after=before+cand).
            mid = min(1, len(cfg.scales_vox) - 1)
            rel = []
            for j in others:
                fj = feats_pool[j][mid][idx[:, 0], idx[:, 1], idx[:, 2]]
                s = np.std(fj)
                c = 0.0 if s < 1e-12 else abs(np.corrcoef(fj, y)[0, 1])
                rel.append((0.0 if np.isnan(c) else c, j))
            rel.sort(reverse=True)
            others = [j for _, j in rel[:cfg.max_predictor_layers]]
        other = [feats_pool[j] for j in others]
        Xb = np.column_stack([_stack(fo, idx) for fo in other]) if other else np.zeros((len(idx), 0))
        Xc = _stack(feats_cand, idx)
        Xa = np.column_stack([Xb, Xc]) if Xb.shape[1] else Xc
        rb, nb = buffered_block_relmae(idx, y, Xb, labels, cfg.cv_buffer_vox, cfg.ridge_alpha)
        ra, na = buffered_block_relmae(idx, y, Xa, labels, cfg.cv_buffer_vox, cfg.ridge_alpha)
        rb_[name], ra_[name], n_[name] = rb, ra, nb
        kb_[name], ka_[name] = Xb.shape[1] + 1, Xa.shape[1] + 1
        lift[name] = rb - ra
    if not n_:
        return Result(float("inf"), 0, self_rel_mae=self_mae, valid=valid, admit=False,
                      note="no_scorable_targets")
    names = list(n_.keys())
    bic_b = _laplace_bic([rb_[t] for t in names], [n_[t] for t in names], [kb_[t] for t in names])
    bic_a = _laplace_bic([ra_[t] for t in names], [n_[t] for t in names], [ka_[t] for t in names])
    n_total = int(sum(n_.values()))
    bic_delta = (bic_a - bic_b) / max(n_total, 1)
    return Result(bic_delta, n_total, rb_, ra_, self_mae, valid, lift,
                  admit=bool(bic_delta < 0), note="scored")


# --------------------------------------------------------------------------- #
# permutation-null calibration (§6.6 / Alt-4): is the candidate's lift better
# than that of the SAME structure placed at random locations?  Deterministic
# shifts (no RNG) so replay parity holds.
# --------------------------------------------------------------------------- #
def permutation_null_deltas(pool_fields, pool_names, candidate, cfg: Cfg, K=10, pf=None):
    nx, ny, _ = candidate.shape
    deltas = []
    for k in range(K):
        dx = (37 * (k + 1)) % (nx - 1)
        dy = (53 * (k + 2)) % (ny - 1)
        cand_k = np.roll(np.roll(candidate, dx, axis=0), dy, axis=1)
        r = score_predictor_lift(pool_fields, pool_names, cand_k, cfg, self_gate=False,
                                 precomputed_pool_feats=pf)
        if np.isfinite(r.bic_delta):
            deltas.append(r.bic_delta)
    return deltas


def calibrated_admit(pool_fields, pool_names, candidate, cfg: Cfg, K=10, q=5.0, pf=None):
    """Admit iff (a) valid and (b) bic_delta below the q-th percentile of the
    permuted-placement null (significantly better than random placement).
    pf: optional precomputed pool_features(pool_fields, cfg)."""
    r = score_predictor_lift(pool_fields, pool_names, candidate, cfg, precomputed_pool_feats=pf)
    if not r.valid:
        return r, False, float("nan")
    null = permutation_null_deltas(pool_fields, pool_names, candidate, cfg, K, pf=pf)
    thr = float(np.percentile(null, q)) if null else 0.0
    thr = min(thr, 0.0)  # never admit a positive delta
    return r, bool(r.bic_delta < thr), thr


# --------------------------------------------------------------------------- #
# synthetic builders
# --------------------------------------------------------------------------- #
def blank(shape=(200, 200, 8)):
    return np.zeros(shape, float)


def blob(center_xy, r=2, shape=(200, 200, 8), val=1.0, depths=range(8)):
    f = blank(shape)
    cx, cy = center_xy
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                x, y = cx + dx, cy + dy
                if 0 <= x < shape[0] and 0 <= y < shape[1]:
                    for z in depths:
                        f[x, y, z] = val
    return f


def scatter(n, seed, shape=(200, 200, 8), bbox=(60, 140, 60, 140), val=1.0):
    rng = np.random.default_rng(seed)
    f = blank(shape)
    x0, x1, y0, y1 = bbox
    f[rng.integers(x0, x1, n), rng.integers(y0, y1, n), rng.integers(0, shape[2], n)] = val
    return f


def shifted(field, dx, dy, noise=0.0, seed=0):
    out = np.roll(np.roll(field, dx, axis=0), dy, axis=1)
    if noise > 0:
        rng = np.random.default_rng(seed)
        nz = np.array(np.where(out != 0)).T
        drop = rng.random(len(nz)) < noise
        for i in np.where(drop)[0]:
            out[nz[i, 0], nz[i, 1], nz[i, 2]] = 0.0
    return out
