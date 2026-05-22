# test_fixed_workflow.sh — Debug Report & Fixes

**Date:** 2026-05-21
**Repo:** `/root/combo-geology-nsl/` (NSL2-geology-task)
**Goal:** Get a full set of passing indicators from `test_fixed_workflow.sh`.
**Result:** ✅ **Full pass achieved — confirmed on the real Coe Fairbairn dataset.**

---

## Final test result

Final run (`test_fixed_workflow.sh`, real dataset present at `../Coe Fairbairn/`):

```
📊 Step 5: Analyze results
✅ Agent called scoring_create_feature_layer
✅ MCP function was invoked
✅ BIC evaluation completed
✅ Reached rewrite phase
✅ Episode completed successfully

💾 Step 6: Check data persistence
✅ Training data saved               📝 Training pairs in database: 1
                                     📊 Latest BIC delta: -1.0   🎯 admitted: True
✅ Knowledge graph data saved        📚 Knowledge entries: 1

📋 Step 7: Summary
🎉 SUCCESS: Fixed workflow is working!
✅ Agent called terminator capability
✅ BIC evaluation triggered
✅ Data persistence working
✅ Episode completed successfully
```

`Episode completed - Success: True`. Episode wall time ~2 min; the agent loaded the real
drill-hole CSV and formed a data-grounded hypothesis (see "Observations" below).

`Episode completed - Success: True`

---

## Root causes (none of it was "cached workflows")

### 1. Missing `docker/aiq/requirements.lock`
- `docker/aiq/Dockerfile` does `COPY requirements.lock /tmp/requirements.lock`.
- `.gitignore` contains `*.lock`, so the file was **never committed** — it only ever
  existed on whatever machine first built the image.
- The test script runs `run_episode.py --rebuild-harness`, which forces a fresh build →
  `docker build failed: COPY failed: file not found in build context: requirements.lock`.
- This is why a fresh checkout fails while a machine that already had the file passed.

### 2. NAT MCP-client 5-second httpx timeout (crashed the harness)
- `nat/plugins/mcp/client/client_base.py` builds `httpx.AsyncClient(...)` with **no
  `timeout`**, so httpx's **5s default** applies to every capability tool-call POST.
- Host-side capabilities exceed 5s (e.g. `spatial_add_line` took ~7s; the first spatial
  call also paid a ~3s cold import). Result: `httpx.ReadTimeout` →
  `[nsl] aiq failed: ExceptionGroup` → harness container exits 1 (`agent_failure`) →
  episode dies in the translate phase.

### 3. Cold-import blocking the MCP-bridge event loop
- `_exec_spatial_capability` / `_exec_scoring_capability` import `voxel_features.spatial`
  (pulls in `geopandas`/`pyproj`/`shapely`, ~3s) **synchronously on the host MCP-bridge
  event loop** on first use, starving other MCP traffic.

### 4. `submit_rewrite` nested-object parameter (terminal capability never executed)
- `submit_rewrite` was the **only** capability with a nested object parameter
  (`training_pair: {prompt, response}`). All others use flat scalars.
- NAT's MCP client generates the nested model type **twice** (two distinct
  `TrainingPairInputSchema` classes); its own `model_validate` then rejects its own
  parsed value:
  `ValidationError: training_pair — Input should be ... instance of
  TrainingPairInputSchema [input_type=TrainingPairInputSchema]`.
- The tool call errored, LangChain's ToolNode swallowed it into an error message, the
  graph ended, and the terminal capability was recorded as "never invoked" →
  `Episode ... terminal capability 'submit_rewrite' was never invoked - forcing
  success=False`.

### 5. `_exec_submit_rewrite` wrote to non-existent directories
- It writes `data/feature-hypothesis/training/training_pairs.pkl` and
  `data/feature-hypothesis/knowledge/coe_fairbairn/experiments.jsonl` without
  `mkdir`-ing the parent dirs → `FileNotFoundError`, data never persisted.

### 6. `test_fixed_workflow.sh` timeout too short + buffered output lost
- `timeout 300s` wraps **both** the harness rebuild and the episode. A real
  rebuild + 5-phase episode does not fit in 300s.
- On the `timeout` SIGTERM, Python had block-buffered its stdout through the `tee`
  pipe, so the entire episode log was discarded — making **every** Step 5/6 indicator
  read as a false failure even when the episode was fine.

### Non-code factors
- **OpenRouter key** was over its `$3` spend cap (`limit_remaining: 0`) — the episode
  got `403 Key limit exceeded`. Resolved by the user supplying a fresh key.
- **A second checkout of this repo** at
  `/root/michael-folder/geology/model-training/combo-geology-nsl/` was running episodes
  **concurrently**. Both checkouts share the global Docker image tag `nsl/aiq:0.1.0`
  and the named containers `feature-hypothesis-compose-{agent,vfm,analysis}-1`, so each
  copy rebuilt the image with its own build-context hash (making the other see it as
  "stale" and force a no-cache rebuild) and the episodes interfered. This is the real
  source of the "staleness churn" and flakiness — *not* caching.

---

## File changes

All changes are in `/root/combo-geology-nsl/`. **Uncommitted.**

| File | Status | Change |
|------|--------|--------|
| `NSL2-geology-task/docker/aiq/requirements.lock` | new (gitignored by `*.lock`) | Regenerated with `uv pip compile requirements.in -o requirements.lock` (588 pinned deps). Required by the Dockerfile's `COPY`. |
| `NSL2-geology-task/docker/aiq/run.py` | modified | Monkeypatch `httpx.AsyncClient.__init__` to inject a 120s default timeout when a caller passes none — fixes NAT's 5s-default crash. |
| `NSL2-geology-task/tasks/feature_hypothesis.py` | modified | (a) `_prewarm_voxel_features()` called from `__init__` to warm `voxel_features` imports. (b) `submit_rewrite` capability schema flattened from nested `training_pair` object to top-level `prompt`/`response` strings. (c) `_exec_submit_rewrite` reads the flat args and `mkdir`s the `training/` and `knowledge/coe_fairbairn/` directories before writing. |
| `NSL2-geology-task/test_fixed_workflow.sh` | modified | `timeout 300s` → `timeout 900s`; added `PYTHONUNBUFFERED=1` so episode output survives. |
| `NSL2-geology-task/.env` | new (gitignored) | Created from environment vars so the script's `.env` precondition passes. Contains the OpenRouter key. |

`src/genner/OAI.py` had temporary diagnostic logging during debugging; it was reverted —
the file is unchanged from origin.

`git status` (tracked): `docker/aiq/run.py`, `tasks/feature_hypothesis.py`,
`test_fixed_workflow.sh` modified.

---

## The debugging story

The fix wasn't one change — it was a chain of "run it → hit a wall → fix that wall →
run it again", where each fix uncovered the next problem hiding behind it.

1. **Ran the script. The build wouldn't even start.** `--rebuild-harness` tried to
   build the Docker image and died instantly: `COPY requirements.lock` — no such file.
   The Dockerfile needs a lock file that `.gitignore` (`*.lock`) had quietly excluded
   from the repo. **Fix:** regenerated `requirements.lock` with `uv pip compile`.

2. **Ran it again. The build worked — but ate the whole clock.** The 588-package
   install took ~5 minutes, and the script's `timeout 300s` wraps *both* the build and
   the episode, so the episode never even ran. (Worse: on the timeout's SIGTERM, Python
   threw away its buffered output, so the log looked empty and every check failed.)
   This told me the script's timeout was mis-budgeted — noted for later.

3. **Ran the episode directly to see past the build.** It got further — built the
   image, started up — then hit `403 Key limit exceeded`: the OpenRouter key was over
   its $3 cap. Not something I could fix in code. **You supplied a fresh key.**

4. **Ran again with the new key. New wall: the harness crashed mid-episode.** It got
   through Phases 1–3, then in Phase 4 (translate) the container died with
   `httpx.ReadTimeout`. Tracing it: NAT's MCP client creates its HTTP client with no
   timeout, so it inherits httpx's 5-second default — and the voxel-store capabilities
   genuinely take longer than 5s (cold imports, a ~7s spatial op). **Fix:** a prewarm
   for the slow imports, plus a 120s default timeout injected in `run.py`.

5. **Ran again. Now all 5 phases completed** — scoring ran, BIC evaluation ran, the
   container exited cleanly. So close. But the very last step, `submit_rewrite`, failed
   two runs in a row in two *different* ways (once the model wrote prose instead of
   calling the tool; once it called the tool but it errored). That smelled flaky, so I
   suspected something environmental…

6. **…and found it.** A *second checkout of this repo* was running its own episodes at
   the same time, and both copies were fighting over the same Docker image and
   containers. I flagged it; you stopped the other run. Then I ran one more clean
   episode — and `submit_rewrite` **failed the exact same way again**. So it wasn't the
   concurrency. It was a real bug: `submit_rewrite` was the only capability with a
   *nested object* parameter, and NAT's MCP client mangles those (generates the type
   twice, then rejects its own value). **Fix:** flattened it to two plain string args.

7. **Ran the episode — `Success: True`** at last. One loose end: it couldn't *save* the
   results because the `training/` directory didn't exist. **Fix:** `mkdir` it.

8. **Ran the full script one final time** (with the timeout bumped to 900s and
   unbuffered output, from step 2's note) — **every indicator green.** At this point
   the dataset was still absent, so the episode ran on agent-fabricated coordinates.

9. **Dataset arrived — ran once more on real data.** The `Coe Fairbairn` dataset was
   placed at `../Coe Fairbairn/`. Inspected it (1297 drill-hole rows + 3711 surface
   rows, all within the voxel grid, columns matching the task's expectations), then
   re-ran the full script. The agent loaded the real `geochemDrillhole.csv`, formed a
   data-grounded hypothesis, and the script passed **every indicator again**.

Each run answered exactly one question and revealed the next. The "cached workflows"
hunch turned out to be a missing generated file plus a concurrent second checkout —
real, mechanical causes, not caching.

## Run history

Coarse progression:

| Stage | Outcome |
|-------|---------|
| Baseline | Build failed: `COPY failed: requirements.lock not found`. |
| After lock fix | `timeout 300s` consumed entirely by the 588-pkg image build; episode never ran. |
| Direct episode | OpenRouter `403 Key limit exceeded` — key over cap. |
| After new key | Episode crashed in translate phase: `httpx.ReadTimeout` (NAT 5s timeout). |
| After httpx fix + prewarm | All 5 phases ran, `scoring_create_feature_layer` + BIC eval worked, container exited cleanly — but `submit_rewrite` failed (nested-object NAT bug). |
| After `submit_rewrite` flatten + `mkdir` | `Episode completed - Success: True`; training + knowledge data persisted. |
| `test_fixed_workflow.sh` (no dataset) | 🎉 Full pass — all Step 5/6/7 indicators ✅ (agent-fabricated coordinates). |
| **Final `test_fixed_workflow.sh` (real dataset)** | **🎉 Full pass — all indicators ✅, agent loaded real `geochemDrillhole.csv`.** |

### Per-run / per-episode status

`script` = full `test_fixed_workflow.sh` run; `episode` = `run_episode.py` run directly
(unbuffered, no `timeout`) for clean diagnosis.

| # | Type | Run with | Episode result | Failure point |
|---|------|----------|----------------|---------------|
| 1 | script | baseline | ✗ never started | Harness build failed — `COPY requirements.lock`: file not in build context. |
| 2 | script | + `requirements.lock` | ✗ never ran | `timeout 300s` fully consumed by the 588-package nocache image build. |
| 3 | script | (retry) | ✗ never ran | Same — build ate the 300s; SIGTERM dropped buffered output. |
| 4 | episode | direct, 900s budget | ✗ aborted pre-Phase 1 | OpenRouter `403 Key limit exceeded` ($3 cap hit). **→ user supplied a new key.** |
| 5 | script | + new key | ✗ never ran | `timeout 300s` killed it during the harness build. |
| 6 | episode | direct, new key | ✗ aborted pre-Phase 1 | Tool-contract probe: `wrong tool name` — transient OpenRouter routing glitch (probe later verified 15/15 reliable). |
| 7 | episode | direct | ✗ `Success: False` | Probe ✓, Phases 1–4 ran; **crashed in Phase 4 (translate)** after `spatial_add_point` — `httpx.ReadTimeout`, harness container exited 1 (`agent_failure`). |
| 8 | episode | + voxel-import prewarm | ✗ `Success: False` | Got one step further — crashed after `spatial_add_line` (2nd spatial op), still `httpx.ReadTimeout` (NAT 5s default). |
| 9 | episode | + httpx 120s timeout fix | ✗ `Success: False` | **All 5 phases ran**, `scoring_create_feature_layer` + BIC eval ✓, container exited cleanly. Phase 5: model wrote the rewrite as plain text instead of calling `submit_rewrite` → terminal capability never invoked. |
| 10 | episode | direct (retry) | ✗ `Success: False` | Model *did* call `submit_rewrite`, but it errored — NAT pydantic `ValidationError` on the nested `training_pair` object. |
| 11 | script | (retry) | ✗ false-fail | `timeout 300s` killed it; buffered output lost → all indicators read ✗ even though the run was progressing. |
| 12 | episode | direct, clean (other checkout stopped, fresh image) | ✗ `Success: False` | **Reproduced** the `submit_rewrite` nested-object `ValidationError` — confirmed systematic, not concurrency/flakiness. |
| 13 | script | (retry) | ✗ false-fail | `timeout 300s` killed it again; output lost. |
| 14 | episode | + `submit_rewrite` flattened to `prompt`/`response` | ⚠️ `Success: True` | Episode succeeded end-to-end, but `_exec_submit_rewrite` failed to *save* — `training/` directory not created. |
| 15 | episode | + `mkdir(parents=True)` for training/knowledge dirs | ✅ `Success: True` | Verified — episode completes, training pair + knowledge-graph node persisted. |
| 16 | script | + `timeout 900s` + `PYTHONUNBUFFERED=1` | ✅ **Full pass** | All Step 5/6/7 indicators ✅ — but dataset still absent (agent fabricated coordinates). |
| 17 | **script** | + `Coe Fairbairn` dataset placed at `../Coe Fairbairn/` | ✅ **Full pass** | **All indicators ✅ on real data** — agent loaded `geochemDrillhole.csv`, real schema, data-grounded hypothesis. |

Notes:
- Episodes that reached BIC evaluation reported `bic_delta: -1.0`, `admitted: True`.
  Note `-1.0` is a floor/sentinel value (`bic_before = bic_after = 0.0`), not a
  data-driven score — see "Observations" below.
- Several `script` runs (#2, #3, #5, #11, #13) "failed" only because `timeout 300s`
  SIGTERM-killed the process and Python's buffered stdout was discarded — the indicators
  were false negatives, not real workflow failures.

---

## Observations (final real-data run, #17)

These are not failures — `test_fixed_workflow.sh` passed every indicator — but they are
worth knowing about the *quality* of an episode, as distinct from its plumbing.

- **The real dataset is read.** The survey-phase agent loaded
  `/workspace/input/amalgamated_csvs/geochemDrillhole.csv` and printed its true schema
  (`longitude`, `latitude`, `maxdepth_drill`, `cu_ppm`, `au_ppm`, ~90 element columns).
  The hypothesis it formed referenced real files:
  *"Copper mineralization concentrations show spatial correlation with fault lineaments
  in the near-surface region (0-40m depth)"*, `data_spec.files =
  [geochemDrillhole.csv, geochemSurface.csv]`.

- **The agent does file-path trial-and-error.** The run log shows 4
  `FileNotFoundError: No such file or directory` hits — the agent first guessed paths
  like `/workspace/input/geochemDrillhole.csv`, failed, then listed the directory,
  found `amalgamated_csvs/`, and read the file correctly. It self-corrects; not a
  workflow failure. Could be reduced by stating the exact path in the survey prompt.

- **`bic_delta` is a floor value, not a real score.** Every scored episode reports
  `bic_before: 0.0, bic_after: 0.0, bic_delta: -1.0`. The translate phase only emits a
  couple of `spatial_add_point` / `spatial_add_line` features, so the voxel model is too
  sparse for BIC ridge-regression to produce a meaningful delta — it returns the `-1.0`
  floor. The scoring *machinery* works; it just isn't being fed a rich enough feature
  layer to score. Making episodes scientifically meaningful (denser feature layers, BIC
  deltas that actually reflect the data) is follow-up workflow-depth work, separate from
  the plumbing fixes in this report.

- **Dataset quirks the agent's analysis code should handle:** `au_ppm` contains sentinel
  values (drill-hole max `5960`, surface min `-5556` — below-detection placeholders);
  `cu_ppm` is sparse (~14% populated in drill-hole, ~57% in surface). Structurally the
  data is fine and all 1297 + 3711 rows fall within the voxel grid bounds.

---

## Notes / caveats worth knowing

- **Two checkouts cannot run episodes at the same time.** They share the global Docker
  image `nsl/aiq:0.1.0` and the `feature-hypothesis-compose-*` containers. Run only one
  checkout's episodes at a time, or give each isolated Docker resource names.
- **First rebuild is slow (~5 min).** `run_episode.py --rebuild-harness` uses Docker's
  *legacy* builder, while `scripts/build_harness_images.py` uses *BuildKit* — different
  caches, so a build can't reuse the other path's cached layers and re-runs the full
  `pip install`. Subsequent runs through the same path reuse the cache and are fast.
  The new `timeout 900s` comfortably covers a cold rebuild + episode.
- **The Coe Fairbairn dataset is now present** at `/root/combo-geology-nsl/Coe Fairbairn/`
  (`amalgamated_csvs/geochemDrillhole.csv`, `geochemSurface.csv`, tenement bundles, WAMEX
  JSON chunks). It is bind-mounted read-only into the analysis container at
  `/workspace/input`. The final run (#17) used it. If this folder is removed, episodes
  still complete and the test still passes, but the agent fabricates coordinates from
  geological priors instead of real assays.
- **`requirements.lock` is gitignored** (`*.lock`). It will be missing again on the next
  clean checkout. Consider committing it (e.g. add `!docker/aiq/requirements.lock` to
  `.gitignore`) so `--rebuild-harness` works out of the box.
- The OpenRouter key in `.env` is a live secret; `.env` is gitignored — keep it that way.
