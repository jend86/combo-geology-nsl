# Feature-Hypothesis Scoring Fix, KG Cleanup, and Retroactive Replay

**Date:** 2026-05-25 (revised 2026-05-25 after pulled fix + post-fix data review; implementation completed 2026-05-25)
**Status:** **DONE.** Scoring fix landed (commit `1754bf3`) + calibration, RNG determinism, dual-copy sync, version bump, KG-A cleanup, 2nd-layer null baseline, regression tests (8), replay tests (3), and `scripts/replay_scoring.py` all applied in this branch. A real-data replay of run `20260524-rg26xw` (44 admitted rows; 18 with surviving .npy) shows 16/18 (88.9%) still survive the fixed gate.
**Scope:** `NSL2-geology-task/src/voxel_features/scoring.py`, `voxel-features-mcp/voxel_features/scoring.py`, `NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py`, `NSL2-geology-task/tasks/feature_hypothesis.py`, `voxel-features-mcp/voxel_features/knowledge_graph.py`, `voxel-features-mcp/voxel_features/mcp/tools/experiment_tools.py`, new `scripts/replay_scoring.py`, new `tests/test_scoring_two_stage.py` + `tests/test_replay_scoring.py`.
**Related:**
- `NSL2-geology-task/docs/design/kazakhstan-variance-and-throughput-2026-05-24.md` — orthogonal diagnosis of low submission variance.
- `docs/design/feature_hypothesis_duplicate_handling_and_bootstrap_ramp.md` — the dedup gate that has been carrying quality filtering by default.
- Pulled commit: `1754bf3 "fix: Scoring rubber stamp - four-bug fix"`.

---

## 0. TL;DR

- **Original problem:** Pre-fix scoring rubber-stamped every submission (90/90 admits, mean `bic_delta = −46K`). The pulled commit `1754bf3` directionally fixed Bugs 1-5 but left calibration off by ~1000× on Stage 1 and ~10× on Stage 2; gradient was bimodal (~0 for rejects, ~0.60 for admits).
- **Implementation outcome (this branch):**
  - **Calibration:** `stage1_reward` divisor `0.02 → 1e-4`; `stage2_reward` divisor `0.1 → 1.0` — applied to both Kazakhstan and Australia tasks.
  - **RNG determinism:** `evaluate_new_layer` and the downstream call graph (`geological_coherence_score`, `compute_geological_interpolation`, `compute_moran_correction`, `create_geological_cv_split`, `create_spatial_mask`, `compute_pairwise_mae`) accept an explicit `seed` / `rng`. Default is `seed=42` when no override, with auto-derivation from `VFM_EPISODE_ID` for per-episode reproducibility.
  - **`scoring_version`** bumped `two_stage_v1 → two_stage_v2` (4 sites).
  - **Dual `scoring.py`** copies (`NSL2-geology-task/src/voxel_features/scoring.py` + mcp copy) re-synced byte-identical.
  - **KG-A cleanup:** deleted `knowledge_graph.py`, `experiment_tools.py`, the 6 `experiment.*` MCP tool registrations + `_kg`/`_get_kg()` in `server.py`, the `export_training` CLI command, the orphan `test_naming_and_deduplication.py`, and the broken `KnowledgeGraph(COE_FAIRBAIRN_GRID)` block in `feature_hypothesis.py`. Australia now reads the JSONL ledger directly (parity with Kazakhstan).
  - **2nd-layer null baseline:** new `_single_layer_null_bic` (mean-predictor BIC) replaces the `bic=0` artefact baseline when `len(existing_layers) == 1`.
  - **`evaluate_bidirectional_prediction`** marked deprecated (still in tree for the future Approach B successor); `test_two_stage_scoring.py` moved to `tests/legacy/`.
  - **Tests:** 11 added — 8 in `tests/test_scoring_two_stage.py` (uninformative/redundant/informative gate behaviour, first-layer fields, null baseline, grid invariance, reward gradient, seeded determinism) + 3 in `tests/test_replay_scoring.py` (reproducibility, missing layer .npy, input immutability). All pass under `nix develop`.
  - **`scripts/replay_scoring.py`** walks `experiments.jsonl`, reconstructs the VoxelStore at each admit step, and re-scores under the fixed code with a stable per-`node_id` seed. Does not mutate inputs. Smoke run on run `20260524-rg26xw`: 88.9% of replayable admits survive the new gate; 25/44 rows skipped because their `.npy` predates the current `admitted/layers/` path.
- **Remaining work:** see §8 — only "decide SFT-data filtering policy" is open (a policy call, not a code change).

---

## 1. Why this exists

In run `20260524-rg26xw/generation_0` pre-fix (`scoring.py` < commit `1754bf3`), 90 of 90 successful episodes passed both Stage 1 and Stage 2; 85 of 90 scored ≥ 0.99; mean `bic_delta = −46,352`; min `−75,927`. The scoring gate was a rubber stamp — any episode producing a submission earned a near-saturated reward.

Five concrete defects, all in `NSL2-geology-task/src/voxel_features/scoring.py` (byte-identical copy in `voxel-features-mcp/voxel_features/scoring.py` pre-fix; *now divergent* — see §4 and §8):

| # | Bug | Location | Effect |
|---|-----|----------|--------|
| B1 | `masking_test_passed` hardcoded `True` | `scoring.py:1556` (pre-fix) | Stage 1 could not fail except for the "no data at all" branch. The 250-line `evaluate_bidirectional_prediction` was defined but never called. |
| B2 | `masking_test_improvement` was a state metric, not a delta | `scoring.py:1546-1547` (pre-fix) | Reported as "Stage 1 improvement" but it was `1 − mean(off-diagonal MAE)` of the *final* layer set. Saturated near 1.0 whenever any submission landed. |
| B3 | 2nd-layer BIC compared against `bic=0` baseline | `scoring.py:1882-1887` (pre-fix) | The second admit always compared "real BIC vs 0" — guaranteeing a large negative delta from baseline artifact, not from added information. |
| B4 | Stage 2 reward saturated at `bic_delta < −1000` | `feature_hypothesis_kazakhstan.py:2384` (pre-fix) | 88 of 90 episodes were below −1000; gradient was dead beyond that threshold. |
| B5 | First-layer auto-admit returned `bic_delta = −1.0` with no `masking_test_improvement` key | `scoring.py:1858-1875` (pre-fix) | Caller defaulted the missing key to `0.0`, so first-layer episodes anomalously scored ≈ 0.0006 instead of 1.0 — the only mechanism producing low rewards. |

Provenance from `git log` on `scoring.py`:

- `3c8ffc1` (2026-05-23) — "Implement two-stage geological scoring system": introduces `evaluate_bidirectional_prediction` as the Stage 1 gate, called from `geological_coherence_score`.
- `e9a2ede8` (2026-05-23) — "MAJOR: Replace R² with MAE + Laplace BIC": rewrites `geological_coherence_score` to use MAE+Laplace BIC and inlines a simplified `masking_test_passed: True` + `masking_test_improvement = 1 - system_MAE`. The bidirectional function survived, unreferenced.
- `1754bf3` (2026-05-25) — "fix: Scoring rubber stamp - four-bug fix": addresses B1-B5 via a different path than this doc originally proposed. See §4.

The MAE refactor in `e9a2ede8` was substituting a *coherence summary* for a *predictive test* without flagging the semantic change. Only direct test was `/test_two_stage_scoring.py`, which exercised the dead code, not the live path. `evaluate_new_layer` had no correctness tests at all.

---

## 2. What "correct" means here

The scorer's job: answer **"does this new layer add geological information beyond what the existing pool already represents?"** with a bounded scalar usable as a reward signal. Two requirements:

- **Discrimination:** uninformative layers (random noise, copies of existing layers, all-zeros) must be rejected and earn low reward.
- **Gradient:** informative layers must produce reward that varies monotonically with how informative — no saturation across the realistic range of layer quality.

Pre-fix failed both. Post-fix passes (a) at the gate boundary, fails (b) among admits.

---

## 3. Three approaches considered (kept for record)

```
APPROACH A: Restore original intent — bidirectional masked holdout + normalized BIC
  Summary: Wire `evaluate_bidirectional_prediction` back into `geological_coherence_score`;
           change `stage_1_improvement` to be the actual held-out delta; replace the bic=0
           baseline with a constant-mean null model; normalize bic_delta by n_effective.
  Complexity: Low
  Risk:    Low
  Pros: Honors original design; minimal code churn; backward-compatible schema; replay is straightforward.
  Cons: "Layers predict layers" is a coherence proxy, not a geological-truth proxy. n_effective is capped
        at ~10K by `compute_geological_interpolation`'s `max_targets`.
  Reuses: `evaluate_bidirectional_prediction`, `compute_pairwise_mae`, `compute_geological_bic`.

APPROACH B: Score against held-out ground truth (copper-prospect prediction)
  Summary: Pre-split GeoJSON prospects into train/holdout. For each new layer, measure whether
           adding it to the existing pool improves prediction of holdout prospect locations
           (binary AUC or grade regression MAE).
  Complexity: Med
  Risk:    Med
  Pros: Real signal — measures actual exploration-relevant prediction. Bounded, interpretable.
        Stable across grid size. Less gameable.
  Cons: Requires curated train/holdout split kept out of agent-visible data — non-trivial
        plumbing. Changes the gate semantics, not just the math. Per-region holdout splits.

APPROACH C: Pool-relative percentile (rank-based marginal contribution)
  Summary: Marginal BIC contribution of the new layer → percentile against historical
           marginal contributions in this run's experiments.jsonl. Reward = percentile rank.
  Complexity: Med
  Risk:    High
  Pros: Self-calibrating; no thresholds; grid-size invariant; bounded; rewards diversity.
  Cons: Bootstrap problem; gameable (flood pool with weak layers); non-replayable (depends
        on insertion order). Disqualifying for SFT training-data provenance.
```

**Decision:** Approach A. The pulled fix (commit `1754bf3`) is a leaner implementation of the same intent — different mechanism, same direction. Approach B remains the recommended v3 successor once the holdout-split plumbing is designed; this doc does not specify that work.

---

## 4. What landed (commit `1754bf3`) and how it differs from the proposal

The commit touched three files (+107 / −68):

- `voxel-features-mcp/voxel_features/scoring.py` (+110)
- `NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py` (+43)
- `NSL2-geology-task/tasks/feature_hypothesis.py` (+22, Australia counterpart kept in sync)

It did **not** touch `NSL2-geology-task/src/voxel_features/scoring.py`, any test file, the `scoring_version` constant, or any RNG-seeding surface.

| Item | Original proposal | What landed | Status |
|------|------------------|-------------|--------|
| Bug 1 — Stage 1 gate | Re-wire `evaluate_bidirectional_prediction` into `geological_coherence_score` | Compute Stage 1 inside `evaluate_new_layer` as `mae_before − mae_after`, using a newly-returned `system_mae` field from `geological_coherence_score`. Bidirectional function remains dead code. | **Fixed differently.** Bug closed; dead code remains. |
| Bug 2 — Improvement saturation | Return the real held-out delta from the bidirectional test | Return the system-MAE delta (`mae_before − mae_after`) | **Fixed.** Observed post-fix range: `[8e-06, 1e-04]`. |
| Bug 3 — BIC baseline artifact | Add `_single_layer_null_bic` constant-mean baseline | Always call `geological_coherence_score` for `score_before` *and* normalize `bic_delta /= n_effective_samples`. The `n_layers==1` early-return still returns `bic=0`, so the apples-to-oranges baseline persists — but the magnitude is no longer pathological. | **Partially fixed.** Magnitude clipped, comparison still imperfect. |
| Bug 4 — Reward saturation | `stage2_reward = tanh(max(0, -bic_delta / n_eff))` | Linear clamp: `min(1.0, max(0.0, -bic_delta / 0.1))`. Per-sample bic_delta. | **Partially fixed.** Threshold too loose — saturates on 83% of post-fix admits (5 of 6 observed). |
| Bug 5 — First-layer credit | Return `bic_delta=0.0`, `masking_test_improvement=1.0`, direction `"first_layer"` | Keep `bic_delta=-1.0` sentinel; add direction `"first_layer"`; reward function special-cases `direction in ("first_layer", "auto_pass")` to give `stage1_reward = 1.0` | **Fixed differently.** Same observable effect — first-layer admits now earn ~1.0. |
| stage1_reward calibration | Match divisor to observed scale | Divisor = `0.02` (2% MAE delta = max reward) | **Mis-calibrated.** Observed improvements are ~`1e-5` — divisor is 100-1000× too large. `stage1_reward ≤ 0.005` for every observed admit. |
| RNG determinism | Seed `compute_geological_interpolation` and `create_spatial_mask`; seed = hash(episode_id) | Not done | **Not applied.** `np.random.choice` still unseeded; `mae_before` and `mae_after` use different random CV splits. Stage 1 gate is noisy and non-reproducible. |
| `scoring_version` bump | `"two_stage_v1"` → `"two_stage_v2"` | Not done | **Not applied.** Post-fix `experiments.jsonl` rows still emit `"two_stage_v1"`. Only discriminator between pre/post-fix data is the presence of `"region"` in `task_breakdown` — fragile. |
| Dual-copy sync | Sync `NSL2-geology-task/src/voxel_features/scoring.py` to the mcp copy | Not done | **Diverged.** mcp copy: 2025 lines, 13 occurrences of `n_effective_samples`. NSL2 copy: 2001 lines, 0 occurrences. Container builds copy the mcp version (correct), but local/test imports of the NSL2 copy hit unfixed code. |
| Regression tests | 11 failing tests (8 for scoring, 3 for replay) | None added | **Not applied.** Commit message references a smoke test but it wasn't committed. Future regression of any of B1-B5 lands silently. |
| Retroactive replay | `scripts/replay_scoring.py` walks `experiments.jsonl`, rebuilds store state at each admit, re-scores under fixed code | Not done | **Not applied.** The 393 pre-fix episodes (and 9 admitted layers from before the fix) cannot be re-scored without it. |

### 4.1 Post-fix data evidence (N=10 with new `region` task-breakdown tag, 6 successful)

Sample of the three most recent post-fix successful episodes (completed 2026-05-25 14:59-15:09):

```
ep_gen0_0409  bic_delta=-0.806  stage_1_improvement=7.85e-06  stage1_reward=0.000393  stage2_reward=1.0  final=0.600
ep_gen0_0410  bic_delta=-0.442  stage_1_improvement=6.18e-05  stage1_reward=0.003088  stage2_reward=1.0  final=0.601
ep_gen0_0406  bic_delta=-0.620  stage_1_improvement=9.06e-06  stage1_reward=0.000453  stage2_reward=1.0  final=0.600
```

Aggregate over 6 successful + 4 Stage-1-rejected:

- **Bug 1 closed:** 4/10 (40%) now fail Stage 1 with `no_predictive_value: True`. Pre-fix: 0/90.
- **Bug 2 closed:** stage_1_improvement is now real delta. Range `[8e-06, 1e-04]`.
- **Bug 3 mitigated:** bic_delta range `[-0.91, -0.03]`. Pre-fix: `[-77K, -1]`.
- **Bug 4 partial:** stage2_reward saturates 5/6 (83%) of admits.
- **Calibration bug introduced:** stage1_reward range `[0.0004, 0.005]` — three orders of magnitude below the `0.02` divisor expects.
- **Net reward shape:** 5 of 6 successful admits within `0.001` of `0.600`. One outlier at `0.191`. Bimodal: reject ≈ 0, admit ≈ 0.6.

The fix replaced "everything ≈ 0.997" with "everything ≈ 0.600 (or 0)." Gradient gained at the admit/reject boundary, lost among admits.

---

## 5. Adjacent finding: orphaned `KnowledgeGraph` class

Surfaced during reviewer follow-up ("KG data is never persisted"). The complaint is half-right: two parallel KG systems exist, and one of them is dead.

### System A — hand-rolled JSONL (alive)

- Writer: `tasks/feature_hypothesis_kazakhstan.py:_admit_with_dedup`
- Files (host paths):
  - `data/kazakhstan/feature-hypothesis/knowledge/teniz_basin/experiments.jsonl` — line-delimited, 32 rows, 91 KB, growing
  - sibling `crossbreed_index.jsonl`, `crossbreed_queue.jsonl`, `admitted_index.json`, `bootstrap_state.json`, `kg.lock`
- Locking via the `kg.lock` file
- 13 admitted layer arrays at `store/teniz_basin/admitted/layers/*.npy`

### System B — `KnowledgeGraph` class (dead for Kazakhstan, broken for Australia)

- Definition: `voxel-features-mcp/voxel_features/knowledge_graph.py` (321 lines) — class with `record()`, `_save()`, `list_admitted()`, `list_rejected()`, `get_crossbreed_pairs()` (with MI-orthogonality scoring), `export_training_data()`, `stats()`
- MCP server instantiates one: `voxel-features-mcp/voxel_features/mcp/server.py:74` — `_kg = KnowledgeGraph(kg_path)`
- Container env: `VFM_KG_PATH=/workspace/knowledge`. Docker-compose bind-mounts that to `data/kazakhstan/feature-hypothesis/knowledge/` on host — so the path *would* persist.
- Would write **`experiments.json`** (singular — a single JSON object mapping `id → record`), **not** `experiments.jsonl`.
- Three MCP tools exposed: `experiment.record`, `experiment.list_admitted`, `experiment.get_crossbreed_pairs` (`server.py:196, 229, 234`).

Evidence the writer never fires:

```
$ find /home/elijah/combo-geology-nsl -name "experiments.json"
(no results — singular file doesn't exist anywhere)

$ grep -c "experiment.record\|experiment_record" all_episodes.jsonl  # 408 episodes
0

$ grep -o "experiment[._][a-z_]*" all_episodes.jsonl | sort -u
(no results — agent's tool set never contained the experiment.* namespace)

$ grep "experiment.record\|experiment_record" tasks/feature_hypothesis_kazakhstan.py
(no results — task code never invokes it)
```

The agent's system prompt lists Phases 1-5 capabilities (`analysis_shell`, `hypothesis_create`, `execution_*`, `spatial_*`, `record_phase`). No `experiment.*` tools are exposed to the agent.

### Australia footgun

`tasks/feature_hypothesis.py:2526`:

```python
from voxel_features.knowledge_graph import KnowledgeGraph
from voxel_features.store import COE_FAIRBAIRN_GRID
kg = KnowledgeGraph(COE_FAIRBAIRN_GRID)   # Pass a GridSpec where Path is expected
all_experiments = kg.list_all()
```

`KnowledgeGraph.__init__` expects `store_path: Path | str` (`knowledge_graph.py:135`). `COE_FAIRBAIRN_GRID` is a `GridSpec` dataclass (`store.py:395`). The call constructs a nonsense path or crashes at `Path(...)`, then `self.store_path.mkdir(...)` does something silly. The whole block is wrapped in `try: ... except Exception:` (`feature_hypothesis.py:2548`). Silent failure: Australia's "AVOID REPEATING RECENT EXPERIMENTS" prompt context **is never injected**.

### Why this matters

1. **Two implementations of the same concept.** MI-orthogonality crossbreed selection, training-data export, admit/reject filters — all reimplemented in `_admit_with_dedup`/`_update_crossbreed_index`/`_recent_admitted_hypotheses`. Two surfaces to maintain; already divergent (different file extensions, different schemas, different locking).
2. **`experiments.json` ↔ `experiments.jsonl` is a footgun.** If anyone wires `experiment.record` into the agent's tool list (intentionally or by accident — the MCP dispatcher is already wired), records start landing in a different file at the same parent directory, with no warning.
3. **Same "silently swallowed" pattern as the Stage 1 scoring bug.** Code exists, looks invoked, returns valid-looking shapes, has no effect. The codebase tolerates dead branches by convention.
4. **The Australia context-injection regression** has probably been silently active since the Australia/Kazakhstan split — no telemetry exists to detect it.

### Two coherent paths

```
APPROACH KG-A: Delete System B
  Summary: Remove voxel_features/knowledge_graph.py, mcp/tools/experiment_tools.py,
           server.py KG instantiation + three tool registrations, and the orphaned attempt
           at feature_hypothesis.py:2515-2548 (replace with a one-call to the jsonl reader).
  Complexity: Low
  Risk:    Low
  Pros: ~600 LOC deleted; no behavior change for Kazakhstan; eliminates the dual-system
        ambiguity; fixes Australia's silent context regression by replacing the broken
        KnowledgeGraph call with a `_recent_admitted_hypotheses`-style jsonl read.
  Cons: Loses the MCP-tool surface — if a future task wants to write KG records from the
        agent side, it'll need to be rebuilt. Acceptable: current and planned tasks all
        use the framework-side `_admit_with_dedup` path.

APPROACH KG-B: Unify on System B
  Summary: Replace `_admit_with_dedup`'s raw file I/O with KnowledgeGraph.record() calls;
           fix the experiments.json↔jsonl mismatch (migrate data, or change the class to
           emit jsonl); add locking to KnowledgeGraph._save(); fix the Australia GridSpec
           bug; wire experiment.record into the agent's MCP tool surface.
  Complexity: Med
  Risk:    Med
  Pros: One tested codepath; better OO surface for future tasks.
  Cons: Schema migration; concurrent-writer race on _save() needs lock; agent-facing
        change to tool surface; touches Australia + Kazakhstan + MCP simultaneously.
```

**Recommendation: KG-A.** The hand-rolled jsonl path has dedup, locking, lineage tracking, and crossbreed scoring — all wired up. The class is a graveyard of unused abstractions whose only practical effect today is confusing reviewers.

---

## 6. Detailed spec for remaining scoring work (Approach A continuation)

What still needs to happen to finish the scoring fix:

### 6.1 Recalibrate reward thresholds (highest priority)

In `tasks/feature_hypothesis_kazakhstan.py` (and mirror in `feature_hypothesis.py` for Australia):

```python
# Current (mis-calibrated):
stage1_reward = min(1.0, max(0.0, masking_test_improvement / 0.02))
stage2_reward = min(1.0, max(0.0, -bic_delta / 0.1))

# Proposed (recalibrated to observed post-fix scales):
stage1_reward = min(1.0, max(0.0, masking_test_improvement / 1e-4))
stage2_reward = min(1.0, max(0.0, -bic_delta / 1.0))
```

Rationale:
- Observed post-fix `masking_test_improvement` range: `[8e-06, 1e-04]`. Saturating at `1e-4` puts max reward at the top of the observed admit distribution.
- Observed post-fix `bic_delta` range: `[-0.91, -0.03]`. Saturating at `1.0` keeps the top of the distribution within the linear region; values approaching `-1.0` are rare and represent unusually strong improvements.
- The choice of denominator should not be hardcoded — set it from the 90th percentile of the *passing* distribution after ~100 post-fix episodes have run. The numbers above are a first-pass estimate from N=6 and will need a second pass.

### 6.2 Seed the CV-split RNG

In `voxel-features-mcp/voxel_features/scoring.py`:

```python
def compute_geological_interpolation(layer_values, grid, shape, influence_radius_m=None, *, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # ... use rng.choice instead of np.random.choice
```

Same shape change for `create_spatial_mask`, `compute_pairwise_mae`, and `create_geological_cv_split`. Pipe a fixed seed (derived from `episode_id`) through `evaluate_new_layer`. Without this, `mae_before` and `mae_after` are computed on *different random CV splits*, so a layer can pass or fail Stage 1 depending on RNG state — and replay is impossible.

### 6.3 Bump `scoring_version`

In `tasks/feature_hypothesis_kazakhstan.py:1653, 1722`: `"two_stage_v1"` → `"two_stage_v2"`. Add `"two_stage_v2"` recognition wherever `scoring_version` is read. Leaves `"two_stage_v1"` rows in `experiments.jsonl` untouched as historical data, but makes pre/post-fix discriminable by a stable field rather than the fragile `"region"` key.

### 6.4 Sync the dual scoring.py copies

Either:
- Copy `voxel-features-mcp/voxel_features/scoring.py` over `NSL2-geology-task/src/voxel_features/scoring.py` byte-for-byte, OR
- Delete `NSL2-geology-task/src/voxel_features/scoring.py` and replace with an import shim from the mcp package.

The second is preferable long-term (no future drift), but requires touching `PYTHONPATH` in tests and CI. The first is one `cp`.

### 6.5 Fix the 2nd-layer baseline (lower priority)

The `n_layers==1` branch in `geological_coherence_score` still returns `bic=0`, so the 2nd-layer admit still compares "real BIC vs 0." Normalization by `n_effective_samples` clips the magnitude but doesn't fix the comparison. Add a single-layer null-model baseline:

```python
def _single_layer_null_bic(layer_values, layer_dtype, grid, shape):
    # Predict the layer with its own mean; compute Laplace likelihood + complexity penalty.
    # Returns the same dict shape as geological_coherence_score so callers don't branch.
```

Plumb into `evaluate_new_layer`'s `score_before` selection when `len(existing_values) == 1`. ~30 LOC.

### 6.6 Optionally retire the dead `evaluate_bidirectional_prediction` function

Since the landed fix took a different route, the 250-line function is now permanently orphaned. Two options:
- Delete it (and `test_two_stage_scoring.py` which is the only caller).
- Keep it as the basis for Approach B's future ground-truth path.

Recommendation: keep but mark `# DEPRECATED: see scoring-fix-and-replay-2026-05-25.md` and move the test to `tests/legacy/`. Approach B will want to reuse the masked-holdout machinery.

---

## 7. Retroactive replay

### 7.1 What's available

Audit findings (paths under `NSL2-geology-task/data/kazakhstan/feature-hypothesis/`):

- `store/teniz_basin/admitted/layers/*.npy` — 13 admitted layer arrays (was 9 at original audit), dtype `float32`, shape `(200, 200, 8)`, ~2.6 MB each.
- `store/teniz_basin/admitted/index.json` — grid spec + per-layer metadata: `name`, `dtype`, `added_timestamp`, `content_hash`. `added_timestamp` gives admit order.
- `knowledge/teniz_basin/experiments.jsonl` — 32 admitted rows (3 with legacy `stage_completed="stage_2_completed"`, 29 with `"mae_bic_completed"`). 7 are post-fix.
- `generations/<run_id>/generation_N/successful/ep_*.json` — per-episode records.
- `store/teniz_basin/admitted/spatial.db` — sqlite log; **not needed** for replay since `.npy` files are post-translation.

### 7.2 What's not available

- **Rejected layers** — only admitted `.npy` files survive. Cannot replay scoring on rejections.
- **Per-episode random seeds** — pre-fix code unseeded, so original BIC values cannot be exactly reproduced. Replay produces values from a *fixed* RNG; the comparison is "old (non-deterministic) score" vs "new (deterministic, post-recalibration) score." Acceptable for the purpose of retroactive auditing.
- **Pre-fix `experiments.jsonl` rows** lack the `n_effective_samples` field. Replay must recompute.

### 7.3 `scripts/replay_scoring.py` design (unchanged from original doc)

Standalone script (~250 LOC). Walks `experiments.jsonl` in `added_timestamp` order, reconstructs `VoxelStore` state with prior layers from `.npy` files, re-runs `evaluate_new_layer` under the fixed (and recalibrated) code with seeded RNG (seed = stable hash of `episode_id`). Writes a parallel `replay_report.jsonl` and `replay_summary.json`. **Does not mutate originals.**

```
python scripts/replay_scoring.py \
    --run-id 20260524-rg26xw \
    --store-dir data/kazakhstan/feature-hypothesis/store/teniz_basin \
    --kg-dir data/kazakhstan/feature-hypothesis/knowledge/teniz_basin \
    --generations-dir data/kazakhstan/feature-hypothesis/generations/20260524-rg26xw \
    --out replay_report.jsonl
```

### 7.4 What replay can and cannot tell us

| Question | Replay answers? |
|----------|-----------------|
| Of the 13 currently-admitted layers, how many would survive the fixed gate? | Yes |
| Is the rank order of admitted layers stable under the new scoring? | Yes |
| Would the agent have produced different submissions under the new reward? | **No** — would require a re-run |
| What fraction of the original ~119 "successful" episodes had truly low-quality submissions? | Partial — only admitted ones; rejected layers are gone |
| Should we discard the existing SFT training data and regenerate? | Replay produces the evidence; the decision is policy, not data |

### 7.5 Writing-back policy

Do **not** mutate `experiments.jsonl` or episode JSONs. The originals describe what the system actually rewarded the agent for — that's what the SFT data was trained on. Replay outputs are *evaluative*, not corrective. Any SFT data filtering based on replay results is a downstream pipeline decision.

---

## 8. Remaining work, prioritized

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | **Recalibrate `stage1_reward` and `stage2_reward` thresholds** (§6.1) | DONE | `0.02 → 1e-4` and `0.1 → 1.0` applied to Kazakhstan + Australia. |
| 2 | **Seed CV-split RNG** (§6.2) | DONE | `evaluate_new_layer` accepts `seed`; piped through all RNG sites. Default `42`, auto-derived from `VFM_EPISODE_ID`. |
| 3 | **Delete dead `KnowledgeGraph` class (KG-A)** (§5) | DONE | ~600 LOC removed across both mcp + NSL2 mirrors; Australia survey-prompt now reads JSONL via `_read_jsonl_records`. |
| 4 | **Bump `scoring_version` to `"two_stage_v2"`** (§6.3) | DONE | 4 sites updated. |
| 5 | **Sync dual `scoring.py` copies** (§6.4) | DONE | Re-synced byte-identical after each scoring edit. |
| 6 | **Add the 11 regression tests** (§9) | DONE | 8 in `tests/test_scoring_two_stage.py`, 3 in `tests/test_replay_scoring.py`; all pass under `nix develop`. |
| 7 | **Implement `scripts/replay_scoring.py`** (§7) | DONE | Smoke-tested on real run; 88.9% of replayable admits survive. |
| 8 | **Fix 2nd-layer null baseline** (§6.5) | DONE | New `_single_layer_null_bic` (mean-predictor BIC). |
| 9 | **Decide on `evaluate_bidirectional_prediction`** (§6.6) | DONE | Marked DEPRECATED with cross-ref; legacy test moved to `tests/legacy/`. |
| 10 | **Run replay, decide SFT-data filtering policy** | OPEN | Replay evidence in `replay_report.jsonl`; the policy call (whether to retrain/filter) is outside this doc's scope. |

---

## 9. TDD plan (unchanged from original doc, still not implemented)

Failing tests first, in this order. Each test must fail against current `scoring.py` and pass after the corresponding fix.

In `NSL2-geology-task/tests/test_scoring_two_stage.py` (new file):

1. **`test_uninformative_layer_fails_stage1`** — 2-layer setup, second layer independent random noise. Assert `evaluate_new_layer(...)["masking_test_passed"] is False`.
2. **`test_redundant_layer_fails_stage1`** — Second layer noisy copy of first. Assert `masking_test_improvement < threshold`.
3. **`test_informative_layer_passes_stage1`** — Second layer `f(layer1) + ε`. Assert `masking_test_passed is True` and `masking_test_improvement > threshold`.
4. **`test_first_layer_returns_stage1_fields`** — First-layer auto-admit must include `masking_test_improvement`, `masking_test_passed`, `masking_test_direction`, `stage_completed`. (Landed fix did include `masking_test_direction="first_layer"`; test would verify the full dict shape.)
5. **`test_second_layer_bic_uses_null_baseline`** — Add a second layer identical to the first. Assert `bic_delta >= 0`. **Today (post-fix) still fails** — because the null baseline isn't fixed (§6.5).
6. **`test_bic_delta_grid_invariance`** — Same predictive relationship on 50×50×8 and 200×200×8 grids should yield comparable per-sample BIC delta (within a factor of 2). Today: should pass post-fix because of n_eff normalization.
7. **`test_reward_gradient_above_threshold`** — Two layers, one strong improvement, one very strong. Assert `reward(strong) < reward(very_strong)`. **Today (post-fix) still fails** — stage2_reward saturates (§6.1).
8. **`test_scoring_deterministic_with_seed`** — Same inputs + same seed yield identical `bic_delta`. **Today still fails** — RNG not seeded (§6.2).

In `NSL2-geology-task/tests/test_replay_scoring.py` (new file):

9. **`test_replay_reproduces_seeded_scoring`** — Build a fake run dir with 3 admits, assert per-layer `bic_delta` matches direct `evaluate_new_layer` on the same store-state-at-admit.
10. **`test_replay_handles_missing_layer_file`** — Missing `.npy` → skip with warning, don't crash.
11. **`test_replay_does_not_mutate_inputs`** — `experiments.jsonl` and episode JSONs byte-identical after replay.

Per TDD policy: write all 11 first, confirm they fail (1, 2, 3 pass post-fix; 4 likely passes post-fix; 5, 6, 7, 8 still fail; 9-11 fail for absence of script), then implement.

---

## 10. Risks and open questions

### Post-fix risks observed

- **Calibration is wrong.** Without item 1 from §8, every admit earns ~0.60 reward. SFT trained on this signal will not learn to differentiate good admits from mediocre ones.
- **Stage 1 is noisy.** Without item 2, 40% rejection rate is partly coin flip. Same layer submitted twice in succession could land on either side of the gate. The agent's tool-use trajectory learning will see incoherent reward signal.
- **`scoring_version` doesn't discriminate eras.** Without item 4, the only way to filter pre/post-fix `experiments.jsonl` rows is the presence of the `"region"` key in episode `task_breakdown` — *fragile, and not present in `experiments.jsonl` at all*. Downstream aggregation across runs is currently impossible.

### Original risks (still valid)

- **A clever agent might still satisfy the MAE delta gate** by submitting a smoothed/scaled copy of an existing layer with just enough novel structure to clear the threshold. Mitigation is the dedup gate (`_admit_with_dedup`), which catches near-duplicates — but its sensitivity has never been tested under adversarial submissions. If it doesn't, this points toward Approach B.
- **`compute_geological_interpolation`'s `max_targets=10000` cap** ceilings `n_effective`. The per-sample normalization fixes reward sensitivity but the test's *statistical power* is also affected. Worth a separate look — possibly raise the cap, possibly use a different sample-size estimate.
- **Bootstrap-mode interaction.** `bootstrap_active=True` episodes follow the same reward path. After recalibration, absolute reward will drop for low-quality submissions during bootstrap. If pacing depends on minimum reward, recheck `_acquire_bootstrap_permit`.
- **Cross-run scoring drift.** `v1` (broken) and `v2` (after item 4) rows will coexist in `experiments.jsonl` indefinitely. Any aggregation across runs must filter by `scoring_version` — which is item 4's whole point.

### Approach B prerequisites (for future)

If we move to Approach B later, we need to design the train/holdout split *before* the agent reads the GeoJSON. That means changes to `tasks/feature_hypothesis_kazakhstan.py`'s data-loading path and the system prompt — not in scope for this doc; flagged for a successor design doc.

---

## 11. What this doc deliberately does not address

- The duplicate-handling gate (covered by `docs/design/feature_hypothesis_duplicate_handling_and_bootstrap_ramp.md`).
- The system prompt's effect on submission variance (covered by `kazakhstan-variance-and-throughput-2026-05-24.md`).
- Whether the entire two-stage framework is the right philosophical structure (Approach B is the partial answer; full discussion deferred).
- SFT-data triage policy after replay (separate decision once the replay evidence exists).
- Migration of any data already written to System B (none exists — confirmed in §5).
