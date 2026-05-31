# SFT Explore-Phase Boundary Re-split — Implementation

**Date:** 2026-05-31
**Status:** Implemented (TDD), verified against live run `20260531-f2jcpm`. Not yet published into a run dir.
**Implements:** the boundary principle from `sft-reasoning-decomposition-deep-dive-2026-05-30.md` §4–§5 for the merged `explore` phase.
**Touches:** `tasks/feature_hypothesis_kazakhstan.py` (`ExperimentReasoningRows`, `_exec_record_phase`, `_phase_records_from_tool_outputs`), `tests/test_explore_boundary_resplit.py`, `scripts/inspect/preview_sft_rows.py`.

---

## 1. Problem (observed on `20260531-f2jcpm`)

Two defects in the SFT rows synthesized from the new merged `explore` episodes:

1. **`analysis_plan` (T3) targets degenerated to `Required files: <path>`.** The explore
   prompt asks the agent for `data_spec = {analysis, files, output}`, but
   `_format_data_spec` only understood the legacy `{target_feature, required_files,
   analysis_steps}` schema, so `analysis` and `output` were silently dropped. ~38 % of
   T3 targets were a bare files line; the rest fell back to a raw `json.dumps`. (Looked
   bootstrap-correlated only because the run was still in its survey/bootstrap window —
   it is a universal schema mismatch.)

2. **The source the agent read was on the wrong side of the cut.** Every `explore` row's
   `raw_response` is empty; the agent's file reads accumulate as tool-output blobs in the
   `explore` *prompts*. The synthesized `dataset_hypothesis` (T2) query dumped the entire
   raw explore prompt — system preamble, `record_phase(...)` schema, task-constraint
   scaffolding — while the hypothesis it trained on was really built from the source
   excerpt. The model was asked to *generate* a hypothesis that depended on evidence it
   couldn't see.

## 2. Decision

Chosen: **B + C + source-side capture + dataset backfill** (user steer, 2026-05-31).

- **C (re-cut T3):** move `data_spec.analysis` (the observation of what was read) to the
  **query**; the T3 **target** is the planned feature (`output` → `Target feature:`) plus
  required files (+ legacy `analysis_steps` when present).
- **B (enrich queries):** put a clean **`Source examined — <section>: <excerpt>`** block on
  the query side of both T2 (de-novo hypothesis) and T3 (analysis plan).
- **Backfill from the dataset:** evidence is sourced, in priority order:
  1. the actual `data_spec.files` chunk read from disk (`dataset_dir`, best-effort), then
  2. the source-side `source_excerpt` persisted at `record_phase` (P1, robust/portable), then
  3. the `SAMPLE CONTENT` block regex-parsed from the explore prompt.
  All evidence is BIC/outcome-sanitized (leakage guard) and capped at
  `max_evidence_chars` (2500).
- **Source-side capture (P1, future runs only):** `_exec_record_phase` now persists
  `source_excerpt` + `assigned_section` on the `hypothesise` phase record so future
  exports don't depend on transcript archaeology or disk availability. **Does not affect
  the running generation** (old code is loaded in-process); takes effect next run.

Rejected alternatives (per `sft-reasoning-decomposition-deep-dive` §3): A (T2-only,
formatter fix) was too narrow given the steer; a pure source-side rewrite (B-ideal alone)
can't re-export already-collected episodes.

## 3. Resulting pair shapes

```
dataset_hypothesis (T2)
  QUERY : Source examined — <section>:\n<file excerpt, ≤2500ch, sanitized>
          Task: From the source above, propose a falsifiable hypothesis...
  TARGET: Hypothesis: <child hypothesis>

analysis_plan (T3)
  QUERY : Hypothesis: <h>
          Observations from source: <data_spec.analysis>     ← moved from target (C)
          Source examined — <section>:\n<file excerpt>        ← evidence (B)
          Available files: <files>
          Task: Design the target feature and analysis plan.
  TARGET: Target feature: <data_spec.output>                  ← no longer dropped (#1)
          Required files: <files>   [+ Analysis steps: ... for legacy schema]
```

## 4. Verification

- `tests/test_explore_boundary_resplit.py` (12 tests): formatter new+legacy schema, T2
  excerpt-on-query / no scaffolding leak / BIC-sanitized, T3 observation+excerpt on query /
  observation absent from target, dataset_dir disk enrichment, and source-side capture
  (read + write). All red on `bc09407`, green after.
- Existing `tests/test_experiment_reasoning_synth.py` and the broader transform suite stay
  green (one pre-existing failure, `test_kazakhstan_variant_also_synthesizes`, is an
  environment `libstdc++`/numpy import break, unrelated).
- Live rebuild of `20260531-f2jcpm` (15 successful episodes) via the in-memory preview:
  0/15 degenerate analysis_plan targets (was ~38 %), every target carries `Target
  feature:`, no numeric BIC in any query, max prompt 202k→3.5k. Disk-backfill resolved
  (evidence is the real `data_spec.files` chunk).

## 5. How to re-export (when the run is finalized)

In-process re-export uses the *current* code, so it must be done out-of-band (the live run
loaded the old code):

- **Preview, no side effects:** `scripts/inspect/preview_sft_rows.py build <gen_dir>` —
  builds rows in memory, never writes into the run dir.
- **Publish:** `src.training_data.transforms.regenerate_sft_export(<gen_dir>, task)` —
  writes `exports/sft/<id>/` and updates `latest.json`. Do **not** run against a generation
  whose process is still live (it would race the run's own finalize export).

## 6. Notes / follow-ups

- `config()` adds `max_evidence_chars` (changes the export recipe hash — expected for a new
  transform version); `dataset_dir` is deliberately excluded so the recipe hash stays
  machine-independent.
- Minor: the new explore system prompt has a template gap — `"A feature layer is  if
  bic_delta < 0."` (missing word). Cosmetic, source-side; fix in the prompt builder.
- Crossbreed episodes don't set `assigned_source`/`source_sample`, so their T2 keeps the
  generic dataset-context fallback and their evidence comes from disk/regex when available.
  (Addressed by §7 — crossbreed now gets a rotated source too.)

## 7. Follow-up: crossbreed diversity regression (Approach C)

**Found while verifying the resume.** The file-rotation pulldown left **crossbreed** episodes
— the dominant episode type in steady state — with **no diversity steering**:

- the **novelty/family nudge** (`_novelty_block_for`) was deprecated for all episodes
  (dormant, never called); and
- the new **file rotation** diversity mechanism was wired **survey-only** (`populate()`'s
  `if workflow_kind == "survey"`).

Pre-pulldown (`d29c14e`), crossbreed ran the survey step as its entry *with* the novelty
nudge, so it was steered toward diversity. The merge dropped both for crossbreed.

**Measured** (`scripts/inspect/compare_diversity.py`, raw pre-curation hypotheses,
crossbreed-only, matched n=14):

| | current `f2jcpm` (no steering) | previous `r2ligp` (nudge on) |
|---|---|---|
| distinct families | **4** | **~13** |
| mean pairwise Jaccard (N-robust) | **0.391** | 0.265 |
| top-family share | **0.43** | 0.06 |

All 14 `f2jcpm` crossbreed hypotheses were near-verbatim variants of one template
("Copper mineralization … preferentially concentrated at redox/reduced-facies contacts").
`f2jcpm`'s *survey* episodes were healthy (Jaccard 0.122) — rotation works; the collapse
was specific to the unsteered crossbreed path.

**Fix (Approach C), generation-side:**
1. **Extend file rotation to crossbreed** — `populate()` assigns a least-explored source +
   pre-read sample for `workflow_kind in ("survey", "crossbreed")` (`_assign_rotation_source`).
   The crossbreed prompt grounds in it *in addition to* the parents
   (`_assigned_source_blocks`, shared with the survey prompt so SFT evidence extraction is
   identical).
2. **Re-wire the family-balance signal** — inject `_novelty_block_for(variation)` (which lists
   saturated families + a mechanism-family summary) into the crossbreed prompt, with a
   "take a genuinely different angle from the saturated families" instruction.

The two levers reinforce: the nudge says *which* family dominates; rotation supplies concrete
off-family material to diverge with. The novelty knobs were already enabled
(`novelty_nudge_enabled=True`) — only the call sites were missing.

Tests: `tests/test_crossbreed_diversity_steering.py`. **Generation-side only** — it changes
what future crossbreed episodes are *asked*, so the diversity gain shows up only in a new run;
re-measure with `compare_diversity.py`.

### 7.1 Result — novelty nudge REVERTED, file rotation kept (same day)

The restarted run (process @16:09) picked up C. Dissecting the post-C crossbreed episodes:

- **The nudge backfired.** Proposed-hypothesis mechanism mix went *more* monocultured, not
  less: geochemical 62% → **71%**, drillhole 1% → **0%**; all 48 post-C proposals open with the
  identical *"Copper mineralization in the Teniz Basin is preferentially concentrated…"*.
  Cause: the block lists the saturated families verbatim → **negation-priming** (showing redox
  examples primes redox). The rotated source couldn't override the parent anchor either.
- **The real failure mode is saturation, not C.** Episodes don't crash — they reach
  `stage_2_completed` and return `no_feature` (feature built but **not admitted**). Crossbreed
  success declines monotonically across the run (17.5% → 0% by ep ~200, *before* the C restart):
  the pool is saturated with copper-redox features, so new near-variants add no information and
  fail the gate. 0% admission ⇒ no new training data.

Action: the explicit novelty/"be a different family" nudge is **removed** from the crossbreed
prompt (it was net-negative). `_novelty_block_for` + helpers/knobs are retained but unwired
(analysis only). **File rotation on crossbreed is kept** — it may work better without the nudge
priming against it.

Diversity-aware **parent pairing is ruled out** (not viable). The goal instead: crossbreed
hypotheses that are a *different family* yet *explain* the parents, emerging **organically**
(no explicit "differ" instruction) — pursued via source-led/abductive prompt structure +
reduced parent-prose priming, not an instruction. See the next design note.
