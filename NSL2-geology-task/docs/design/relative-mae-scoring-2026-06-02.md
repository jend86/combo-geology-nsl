# Relative MAE Scoring Redesign — 2026-06-02

## Problem

`spatial_add_box` creates large constant-value voxel regions. This breaks the
scoring pipeline in two places:

1. **`_single_layer_null_bic`**: computes intra-layer MAD over non-zero voxels.
   A constant box has MAD ≈ 0 → astronomical Laplace log-likelihood → very
   negative null BIC. First-layer `bic_delta = -null_bic` explodes to +1.5M.

2. **`geological_coherence_score`** (cross-layer): predicting a *constant*
   target from any predictor gives MAE = 0 (mean predictor is perfect). Same
   astronomical likelihood, same explosion.

Root cause: absolute MAE with uncapped `n_eff` rewards constant/near-constant
layers regardless of whether they have genuine geological signal.

## Solution: Relative MAE

Normalize each cross-layer MAE by the target layer's naive prediction error
(predict-by-mean MAE = MAD of the target):

```
relative_MAE = MAE_predicted / MAE_null
where MAE_null = mean(|target_test - mean(target_train)|)
```

Key properties:
- `relative_MAE ∈ [0, 1]` — naturally bounded
- Constant target: `MAE_null ≈ 0` → return 1.0 (no improvement possible)
- Perfect predictor: `relative_MAE ≈ 0` → log(0+ε) = large negative → good BIC
- n_eff only affects BIC penalty term `k*log(n)`, so the likelihood blowup
  disappears regardless of n_eff value

This makes `_single_layer_null_bic` unnecessary for the `score_before`
baseline. The `bic=0` sentinel (what `geological_coherence_score` returns for
n_layers==1) becomes the correct baseline again, because:
- Good second layer → relative_MAE < 1 → negative BIC → bic_delta < 0 → admitted ✓
- Constant/noisy second layer → relative_MAE ≈ 1 → BIC ≈ k*log(n) > 0 → bic_delta > 0 → rejected ✓

## Changes Required

### 1. `voxel-features-mcp/voxel_features/scoring.py`

**`compute_out_of_sample_mae`** (line ~913):
- After computing `mae_predicted`, compute `mae_null` (predict-by-mean MAE on
  test set)
- If `mae_null < 1e-10`: return `1.0` (constant target, no improvement possible)
- Otherwise: return `mae_predicted / mae_null`
- Rename to clarify return type (or add a comment — return is now dimensionless [0,1])

**`mae_to_laplace_likelihood`** (line ~963) — **REPLACE ENTIRELY, do not adapt**:
- The Laplace formula `−n*log(2*mae)−n` was calibrated for geological units
  (MAE ≈ 0.01–0.5). With dimensionless relative MAE ∈ (0, 1], a 50%
  improvement (relative_mae=0.5) still produces positive BIC — the formula
  breaks entirely in this range.
- New formula:
  ```
  log_likelihood = -n_eff * mean(log(relative_mae_values))
  ```
  Where `log(relative_mae) ≤ 0` for values ≤ 1, so log_likelihood ≥ 0 for
  good layers, and BIC = -2 * log_likelihood + k * log(n_eff) is negative.
  Concrete check: relative_mae=0.5 → log_likelihood=6930 → BIC≈-13851 ✓
                  relative_mae=1.0 → log_likelihood=0   → BIC≈+9      ✓
- Rename or repurpose: keep signature compatible but replace body.

**`geological_coherence_score`** (line ~1560):
- Cap `effective_samples = min(total_non_zero, max(n_layers * 10, 10_000))`
  — n_eff still needed for BIC penalty term; cap prevents blowup even for
  any edge case that slips through

**`_single_layer_null_bic`** (line ~1055):
- Keep function (may be useful for diagnostics/future)
- But remove from the hot path in `evaluate_new_layer`

**`evaluate_new_layer`** (line ~1969):

*First layer (no existing layers):*
```python
# Was: compute _single_layer_null_bic, return bic_delta = -null_bic
# Now: admit unconditionally, return fixed bic_delta = -1.0
# Rationale: cannot evaluate a single layer's cross-layer relevance;
#            -1.0 gives a small reward signal without spurious magnitude
```

*One existing layer (`score_before` baseline):*
```python
# Was: score_before = _single_layer_null_bic(existing_values[0], ...)
# Now: score_before = geological_coherence_score([existing_values[0]], ...)
#      which returns bic=0.0 for n_layers==1
# This is now correct because relative MAE handles the degenerate cases
```

Remove the `if len(existing_values) == 1: _single_layer_null_bic(...)` branch.

### 2. Task files: reward recalibration

`NSL2-geology-task/tasks/feature_hypothesis.py`  
`NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py`

BIC scale change — worked out numerically from the new formula:
- relative_mae = 0.5 (50% improvement): bic_delta_raw ≈ -13,851,
  normalized by n_eff=10,000 → bic_delta ≈ -1.386
- relative_mae = 1.0 (neutral): bic_delta ≈ +0.001 (tiny positive penalty)

**Correct divisor = 1.0** (not 0.1 or 0.5):
- `stage2_reward = -bic_delta / 1.0`
- 50% improvement → reward ≈ 1.386 → clamped to 1.0. Good.
- First layer `bic_delta = -1.0` → reward = 1.0. Acceptable.

**Double-normalization check (important):**
`compute_geological_bic` returns raw BIC (not per-sample).
`evaluate_new_layer` divides by n_eff at line 2102 to get per-sample delta.
When rewriting `compute_geological_bic`, ensure output remains raw BIC —
do NOT move the division into `compute_geological_bic` or the division
in `evaluate_new_layer` will double-normalize.

**Recalibrate after first real run.**

### 3. Tests

`NSL2-geology-task/tests/test_scoring_two_stage.py`

Add/update:
- Test: constant box layer → `relative_MAE = 1.0`, `bic_delta > 0`, rejected
- Test: variable layer predicts another variable layer → `relative_MAE < 1`,
  `bic_delta < 0`, admitted
- Test: first layer → admitted unconditionally, `bic_delta = -1.0`
- Test: no n_eff blowup for large constant layer (assert `|bic_delta| < 10`)

Existing test `test_second_layer_bic_uses_null_baseline` will need updating
(the null baseline mechanism changes).

## Execution Order

1. `scoring.py` — `compute_out_of_sample_mae` (relative MAE)
2. `scoring.py` — `geological_coherence_score` (n_eff cap)
3. `scoring.py` — `evaluate_new_layer` (drop `_single_layer_null_bic` path,
   first-layer fixed delta)
4. Run existing tests, fix breakages
5. Task files — reward divisor update
6. Add new regression tests

## What Does NOT Change

- Stage 1 MAE gate (`mae_improvement > 0`). Note: `system_mae` in
  `geological_coherence_score` is currently computed from the *absolute*
  off-diagonal MAE matrix mean. After this change it will be computed from
  *relative* MAE mean (range ≈ [0,1]). The gate `mae_improvement > 0` still
  works correctly (sign is what matters). Threshold semantics change slightly:
  an improvement of 0.05 now means "5% reduction in relative prediction error"
  which is more interpretable than the previous unit-dependent value.
- Interpolation, Moran's I spatial correction, crossbreeding selection —
  all unchanged.
- BIC delta sign convention: negative = good, positive = bad. Unchanged.
