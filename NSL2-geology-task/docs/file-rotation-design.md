# File Rotation & Episode Diversity — Design Notes

## The Problem: Agent Hyperfocus

When a hypothesis-generation agent runs many episodes without explicit guidance, it tends to
repeatedly fixate on the same data sources. This happens for two reasons:

1. **Context priming.** If the prompt mentions past experiments ("avoid repeating these"), the
   agent is still primed with those concepts and tends to riff on them rather than explore fresh
   territory.

2. **Free browsing.** Without a specific assignment, the agent opens whichever file it finds most
   salient from prior context — usually the same one it opened last time.

The result is a cluster of semantically similar hypotheses generated from a narrow slice of the
available data, regardless of how much other material exists.

---

## Solution: Source-File Rotation

### Core mechanism

Each episode is **assigned a specific source file or data group** at `populate()` time, before
the agent sees any prompt. The assignment is written into `episode_context["assigned_source"]`
and displayed prominently at the top of the survey prompt:

```
YOUR ASSIGNED SOURCE FILE FOR THIS EPISODE
  Path   : converted_spatial_data/anticlines_synclines.geojson
  Details: 33 geological fold structures (anticlines and synclines) ...

You MUST derive your hypothesis candidate from this file.
Do not freely browse other files during the survey phase.
```

The agent is also shown a neutral coverage map (counts only, no hypothesis text):

```
SOURCE COVERAGE (episodes completed per file — for situational awareness only):
  converted_spatial_data/copper_prospects.geojson: 3 episode(s)
  converted_spatial_data/anticlines_synclines.geojson: 1 episode(s)  ← assigned this episode
  ...
```

The coverage map shows visit counts, not what hypotheses were generated. This gives the agent
situational awareness about which sources have been covered without priming it with past concepts
(the "pink elephant" problem — telling an agent "don't think about X" makes it think about X).

### Rotation state

Assignment is tracked in `{kg_dir}/file_rotation_state.json`:

```json
{
  "counts": {
    "copper_prospects": 3,
    "anticlines_synclines": 1,
    "assessment_tract": 2
  }
}
```

Each episode picks the **least-explored entry** (ties broken by list order). This is a simple
round-robin across the source list that naturally spreads coverage without any randomness or
complex scheduling.

### Why not random assignment?

Random assignment would eventually cover all sources but with no guarantee of balance. The
least-explored-first approach is deterministic, auditable, and ensures every source gets visited
before any is revisited.

---

## Source File Lists

### Coe Fairbairn (Australian dataset) — `_COE_FAIRBAIRN_SOURCE_FILES`

5 entries covering the main data types:

| key | description |
|---|---|
| `drillhole` | Geochemistry drillhole CSV (80+ element columns) |
| `surface` | Surface geochemistry CSV |
| `tenements` | Tenement boundary polygons |
| `description_maps` | Geological map descriptions |
| `wamex_reports` | WAMEX OCR'd exploration report chunks |

These are naturally at the right granularity — each is a distinct data type. The WAMEX reports
directory contains chunks from many different exploration reports (different companies, areas,
eras), so rotating at the directory level is appropriate; rotating by section within a single
report would scatter context across reports rather than focusing it.

### Kazakhstan dataset — `_KAZAKHSTAN_SOURCE_FILES`

Currently 6 entries (the original directory-level list). A **section-level expansion** is
planned but not yet applied:

#### Current entries (6)

| key | path |
|---|---|
| `copper_prospects` | `converted_spatial_data/copper_prospects.geojson` |
| `anticlines_synclines` | `converted_spatial_data/anticlines_synclines.geojson` |
| `assessment_tract` | `converted_spatial_data/assessment_tract.geojson` |
| `copper_prospects_aoi` | `converted_spatial_data/copper_prospects_aoi.geojson` |
| `smolianova_survey` | `36572_Smolianova_1984/` (entire directory) |
| `usgs_assessment` | `USGS/` (entire directory) |

#### Planned section-level expansion (19 entries)

The Smolianova 1984 Russian survey is a single comprehensive report with 329 text chunks
organised into clear geological chapters. Assigning the whole directory means the agent
free-browses all 329 chunks and tends to fixate on whatever it opens first.

The planned expansion replaces the two directory entries with **15 section-level entries** using
a `glob_pattern` field to scope each assignment to one chapter's chunks:

- `smolianova_geological_study` — Ch. II, geological-geophysical survey
- `smolianova_physical_properties` — Ch. IV, rock physical properties
- `smolianova_stratigraphy_early` — Ch. V, Proterozoic/Cambrian/Ordovician
- `smolianova_stratigraphy_devonian` — Ch. V, Devonian
- `smolianova_stratigraphy_carboniferous` — Ch. V, Carboniferous (copper-hosting sequences)
- `smolianova_stratigraphy_permian` — Ch. V, Permian
- `smolianova_magmatic` — Ch. VI, magmatic formations
- `smolianova_tectonics` — Ch. VII, structural geology
- `smolianova_useful_minerals` — Ch. VIII, economic minerals
- `smolianova_prognosis` — Ch. IX, ore-forming regularities and prognosis
- `smolianova_hydrogeology` — Ch. X, groundwater
- `smolianova_mineral_catalogue` — Textual appendix, mineral deposits catalogue
- `smolianova_drill_holes` — `drill_holes_data/` directory, 60+ borehole logs
- `usgs_report` — USGS text report chunks
- `usgs_figures` — USGS figure descriptions

This is appropriate for Kazakhstan (but not for Coe Fairbairn) because the Smolianova survey
is one deep structured document, not many shallow ones. Section-level rotation gives the agent
a bounded, topically coherent slice to work from each episode.

---

## Crossbreed Threshold

### What crossbreeding is

Once enough survey episodes have produced admitted features, the workflow switches to
**crossbreed mode**: the agent is shown two previously successful experiments and asked to
propose a hypothesis that combines or builds on them. This is intended to synthesise insights
across the dataset rather than generate them independently.

### Previous threshold: 2 admitted features

With `admitted_count >= 2` as the gate, crossbreeding could start after as few as 2 successful
episodes. With 19 rotation slots (planned Kazakhstan expansion) and a typical ~30–50% admission
rate, this means the system would start recombining after exploring only 4–7 sources — barely
scratching the available material before the workflow shifts mode.

### New threshold: 5 admitted features

Changed to `admitted_count >= 5` in `_has_crossbreed_pairs()` in both task files.

With a 30–50% admission rate this implies 10–17 survey episodes before crossbreeding begins,
which is enough to cycle through the full source list at least once. The crossbreed stage then
has meaningfully diverse material to work from rather than recombining two features from
adjacent data sources.

---

## Survey Step Budget

The survey is a single `WorkflowStep` with no per-step turn limit. It runs until the agent
calls `record_phase(phase='survey', candidates=[...])`, drawing from the shared episode pool
of 120 LLM turns across all 6 phases (survey → hypothesise → code → translate → evaluate →
rewrite).

The framework supports per-step `StepConstraints` overrides (as used in `geology_graph.py`)
but none are set for the survey step in these tasks. The current approach is to let the agent
self-regulate and observe whether it explores enough of the assigned source before forming
candidates. If future runs show consistently shallow survey phases (agent reads one chunk and
terminates), adding either:

- A minimum exploration instruction in the prompt ("read at least 3 files before forming
  candidates"), or
- A `step_overrides={"survey": StepConstraints(...)}` budget allocation

would be the appropriate fix.

---

## Files Modified

| file | changes |
|---|---|
| `tasks/feature_hypothesis.py` | Added `_COE_FAIRBAIRN_SOURCE_FILES`; modified `populate()` to inject `assigned_source`; updated `_survey_workflow()` to pass `episode_context`; rewrote `_generate_survey_prompt_with_context()`; added `_pick_assigned_source()`; raised crossbreed threshold 2→5 |
| `tasks/feature_hypothesis_kazakhstan.py` | Same set of changes for Kazakhstan; added `_KAZAKHSTAN_SOURCE_FILES` (6-entry directory-level list, section expansion pending) |

---

## Combo integration notes (pulled down 2026-05-31)

This design was ported into the combo repo from `JenD86/combo-geology-nsl@file-rotation`.
A few things differ from the description above, reflecting later commits on that branch and
the merge policy used:

- **Kazakhstan `_KAZAKHSTAN_SOURCE_FILES` is the 18-entry section-level list** (the section
  expansion that was "pending" above shipped in a later commit). Glob-pattern entries point at
  Smolianova chapters; the agent enumerates only its assigned section.
- **The survey + hypothesise merge into a single `explore` step is Kazakhstan-ONLY.** The Coe
  Fairbairn task (`feature_hypothesis.py`) keeps `survey` and `hypothesise` as separate steps;
  it gets file rotation + the counts-only coverage map only (no merged step, no pre-read sample).
- **Crossbreed still grounds in data.** Combo keeps an `explore`-named entry step in crossbreed
  mode (it runs `analysis_shell` before hypothesising) rather than jumping straight to a bare
  hypothesise prompt — preserving the grounding guarantee.
- **Source-sample pre-read** (injecting a compact sample of the assigned source into the prompt)
  is Kazakhstan-only, matching upstream.
- **Scoring is combo's, not file-rotation's.** `scoring.py` is kept from combo (a strict superset
  of file-rotation's four-bug fix, plus `pairwise_distance`, null-layer BIC, and RNG/replay
  support), so file-rotation's scoring change was intentionally dropped.
- **Spatial search tools** (`search_web_geological`, `search_geonames_lookup`) were adopted and
  wired into the Kazakhstan translate step; they are Kazakhstan-hardcoded and not added to Coe.
