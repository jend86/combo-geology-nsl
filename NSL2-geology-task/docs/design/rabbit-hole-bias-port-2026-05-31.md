# Rabbit-Hole-Bias Fix — Port into combo (merge-file-rotation)

**Date:** 2026-05-31
**Status:** Ported (TDD), all tests green. Not yet run end-to-end (waiting to stop the live run).
**Source:** `JenD86/combo-geology-nsl@file-rotation` commit `72e3239` ("Fix rabbit-hole bias: bootstrap gate + greedy BIC initialisation"). Upstream rationale: `docs/rabbit-hole-bias-fix.md` on that branch.

## Why we ported this

It is the root-cause fix for the saturation/monoculture collapse we measured (see
`sft-explore-boundary-resplit-2026-05-31.md §7`): crossbreed hypotheses collapsed to a single
copper-redox-Kireyskaya template and crossbreed admission fell to 0%. The chain:

> `evaluate_new_layer` returned a hardcoded **`bic_delta = -1.0` sentinel** for the first
> (empty-store) layer → it auto-satisfied admission → the crossbreed floor (`admitted_count >= 5`)
> was hit after only **~5 of 18 sources** → bootstrap ended early → the seed pool was tiny and
> monocultured → crossbreed compounded it forever.

Confirmed our branch had the bug (`scoring.py:2028` sentinel; no source-coverage gate in
`populate()`). This is a bootstrap-coverage bug, not a prompting problem — so it supersedes the
de-priming/abductive prompt work (re-measure diversity after this lands before doing more there).

## Surgical, not a cherry-pick

`72e3239`'s 4233-line diff also rewrote the SFT transform (upstream's `ExperimentReasoningRows`),
which we've independently and heavily changed. A merge would clobber our SFT/boundary/C work, so we
ported only the 5 rabbit-hole changes. Pleasantly, **no adaptation was needed** — our
`file_rotation_state.json` is already `{"counts": {...}}` and our admitted store is already
`SpatialVoxelStore(store_dir/"admitted", grid)`, both matching upstream.

## What was ported

1. **`scoring.py` `evaluate_new_layer`** — first layer is now scored with a real predict-by-mean
   `_single_layer_null_bic` (`bic_before=null_bic`, `bic_delta=-null_bic`, direction
   `null_model_baseline`), not the `-1.0` sentinel. Seed resolution moved before the early return
   so the null-BIC is reproducibly seeded.
2. **`_all_sources_visited(kg_dir)`** — True only when every `_KAZAKHSTAN_SOURCE_FILES` key has
   count ≥ 1 in `file_rotation_state.json`.
3. **`_run_greedy_bic_initialization(variation)`** — forward-greedy BIC selection over admitted
   bootstrap layers (round 1 = highest null-BIC foundation; rounds 2+ add the layer that most
   reduces `geological_coherence_score`), lock-guarded, writes `greedy_init_complete.json` once.
   Ported verbatim minus upstream's `sys.path` hack (our package imports cleanly).
4. **`populate()` gate** — crossbreed now also requires `all_sources_done AND greedy_done`;
   greedy init is invoked once all sources are visited.
5. **`_exec_scoring_capability`** — records the scoring tool's timestamped `result["layer_name"]`
   (the real `.npy` name) instead of the bare `args["name"]`. Plus `run_episode.py --max-episodes`
   for sequential bootstrap testing.

## Tests (all green under `LD_LIBRARY_PATH=…nix-ld…/lib`)

- `voxel-features-mcp/tests/test_store.py::test_first_layer_gets_real_null_bic_not_sentinel` — first
  layer gets a real negative bic_delta, not `-1.0`.
- `NSL2-geology-task/tests/test_rabbit_hole_bias_gate.py` — `_all_sources_visited` semantics;
  crossbreed blocked until all sources visited; greedy init invoked + flag written; **functional**
  greedy selection completes over a real 3-layer small-grid store (guards against silent skip from
  an API mismatch).
- Updated `test_scoring_two_stage.py::test_first_layer_returns_stage1_fields` and
  `test_crossbreed_diversity_steering.py::test_populate_assigns_rotation_source_for_crossbreed` to the
  new behaviour (direction rename; gate now requires all-sources + greedy flag).
- Full affected sweep: 102 passed. (Run NSL2 tests from `NSL2-geology-task/` — the task's
  docker-compose path is relative to that dir.)

## End-to-end readiness

The live run is at 0% admission and still loaded the old code. When ready: stop it, then start a
fresh run — bootstrap should now visit **all 18 sources** (vs 5) before crossbreeding, with real
first-layer BIC scores and a greedy-selected foundation. Re-measure crossbreed diversity with
`scripts/inspect/compare_diversity.py` and admission rate from `all_episodes.jsonl`.
