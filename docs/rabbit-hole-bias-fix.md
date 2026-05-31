# Rabbit-Hole Bias Fix: Bootstrap Gate + Greedy BIC Initialisation

## Problem

The pipeline was exhibiting "rabbit-hole" behaviour: the model fixated on whichever sources it happened to investigate first, then spent the majority of episodes crossbreeding variations of those same ideas rather than exploring the full dataset.

Root cause analysis on a 20-episode test run showed:

- Only **5 of 18 sources** were visited before crossbreeding began
- All 20 episodes recorded `bic_delta = -1.0` — a hardcoded sentinel, not a real score
- Episodes 6–20 all had `bootstrap_active = False`, meaning the model was already in crossbreed mode with essentially no information about most of the dataset

The sentinel was the trigger. When the very first layer was added to an empty store, `evaluate_new_layer` returned a hardcoded `bic_delta = -1.0` rather than computing a real BIC score. This immediately set `has_admission = True` in the knowledge graph, which satisfied the crossbreed threshold (`admitted_count >= 5`) after just five episodes and switched the pipeline into crossbreed mode permanently.

---

## Changes

### 1. Real null-model BIC for the first layer (`voxel-features-mcp/voxel_features/scoring.py`)

**What changed:** The `evaluate_new_layer` function previously returned a hardcoded `bic_delta = -1.0` sentinel when the store was empty (first layer). This has been replaced with a real call to `_single_layer_null_bic`.

**Why:** The `-1.0` sentinel had two downstream effects:
1. It triggered premature crossbreeding (as above)
2. It gave the greedy BIC initialisation step (see below) no real signal to rank layers with — all first layers looked identical

**What the null-model BIC measures:** `_single_layer_null_bic` computes the BIC of a "predict by mean" model for the layer. Higher null-model BIC = the layer has more spatial variance and is harder to compress = more informative. This gives each bootstrap layer a genuine, comparable score.

**BIC delta semantics for first layers:**
- `bic_before` = null-model BIC of the new layer (cost of encoding it by its mean)
- `bic_after` = 0.0 (empty store has no pairwise prediction cost)
- `bic_delta` = `−null_bic` (negative, so the layer is correctly marked as admitted)

The layer is still admitted unconditionally — the first layer always enters the store. The change is that the score is now real and negative rather than a fixed sentinel.

The seed resolution block was also moved before the early return so the null-model BIC call is reproducibly seeded.

---

### 2. Bootstrap gate: visit all sources before crossbreeding (`NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py`)

**What changed:** Added `_all_sources_visited(kg_dir)`, a method that returns `True` only when every key in `_KAZAKHSTAN_SOURCE_FILES` (all 18 sources) has a visit count ≥ 1 in `file_rotation_state.json`.

The crossbreed condition in `populate()` now requires this:

```
crossbreed = (
    crossbreed_enabled
    AND n_features >= min_features
    AND all_sources_visited          ← new
    AND greedy_init_complete         ← new (see below)
    AND has_crossbreed_pairs
)
```

**Why:** The existing file-rotation logic assigned sources round-robin, but crossbreeding could begin before the rotation had completed a full cycle. The new gate ensures the model has at least one exposure to every source before it starts combining ideas.

---

### 3. Greedy BIC initialisation (`NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py`)

**What changed:** Added `_run_greedy_bic_initialization(variation)`, which runs a **forward greedy BIC selection** over all admitted bootstrap layers once all sources have been visited.

**Why:** After 18 bootstrap episodes, the admitted pool contains one layer per source — but not all of these layers are complementary. Some may be redundant or collinear. Starting crossbreeding from the full unfiltered pool wastes episodes recombining layers that add no new information. The greedy selection identifies the subset of layers that together minimise the geological coherence BIC, then writes a completion flag so crossbreeding starts from that curated foundation.

**Algorithm (O(N²) = 153 evaluations for N = 18):**

- **Round 1:** Score each candidate layer with `_single_layer_null_bic`. Pick the layer with the **highest** null-model BIC as the foundation — this is the most spatially variable (most informative) single layer.
- **Rounds 2+:** For each remaining layer, compute `geological_coherence_score(selected + [candidate])`. Add whichever candidate most reduces the system BIC (most negative delta). Stop when no remaining layer improves BIC.

This is the standard forward-greedy approach for subset selection. It correctly handles complementarity (unlike pairwise comparison) without the exponential cost of all-subsets search (2¹⁸ ≈ 262,000 evaluations).

**Why not pairwise?** Pairwise comparison ranks layers by how well they predict each other in isolation, but misses the case where layer A and layer B are individually uncorrelated with the rest yet together explain a third variable. Greedy sequential addition captures these interaction effects.

**Why the first layer has no artificial advantage:** In the original scoring code, the first layer was always admitted with a synthetic score, making it look like the "best" foundation regardless of its actual information content. With real null-model BIC, the greedy Round 1 selects the objectively most variable layer from the whole bootstrap pool.

**Concurrency safety:** The method is guarded by `_kg_lock` and a `greedy_init_complete.json` flag file. Under `parallel_episodes = 10`, only one episode ever executes the selection; the others see the flag and skip immediately.

**Output:** `greedy_init_complete.json` in the KG directory, containing:
- `selected`: list of layer names that form the initial model
- `not_selected`: layers excluded as redundant
- `final_bic`: the geological coherence BIC of the selected set

---

### 4. Layer name fix in knowledge graph (`NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py`)

**What changed:** `_exec_scoring_capability` previously stored `args.get("name")` as the layer name in the episode's phase record. This is the bare name the agent submitted (e.g. `copper_concentration`). The actual `.npy` file written by `evaluate_new_layer` has a timestamp suffix (e.g. `copper_concentration_1780199863749`).

The fix uses `result.get("layer_name") or args.get("name", "")` — the scoring tool's return value is authoritative, since it reflects the name actually used when writing the file.

**Why it matters:** The layer name stored in phase records flows into every KG write in `_exec_submit_rewrite`:
- `experiments.jsonl` — `layer_name` field
- Training pair records — `layer_name` field
- Knowledge graph artifact links — `store/teniz_basin/admitted/layers/{name}.npy`
- `_admit_with_dedup` — looks up `{name}.npy` in the scratch directory to promote it

With a mismatched name, promotion silently fails (file not found), and the artifact link in the KG points to a nonexistent path.

---

## Files Modified

| File | Change |
|------|--------|
| `voxel-features-mcp/voxel_features/scoring.py` | Replace first-layer sentinel with real null-model BIC; move seed resolution before early return |
| `NSL2-geology-task/tasks/feature_hypothesis_kazakhstan.py` | Add `_all_sources_visited()`; add `_run_greedy_bic_initialization()`; update `populate()` crossbreed condition; fix layer name capture in `_exec_scoring_capability` |

---

## Expected Behaviour After Fix

1. The pipeline runs in survey mode until all 18 sources have each been visited at least once (~18 episodes at `parallel_episodes = 10`, possibly more due to concurrency)
2. Once all sources are visited, `_run_greedy_bic_initialization` runs once, selecting the complementary subset of bootstrap layers (typically fewer than 18)
3. `greedy_init_complete.json` is written; subsequent episodes switch to crossbreed mode using the curated layer pool
4. All KG records contain the correct timestamped layer name and valid artifact links
