"""Feature hypothesis task for Kazakhstan geological dataset.

Agents explore the Kazakhstan Teniz Basin geological dataset, hypothesise about
informative feature layers, write code to test hypotheses, and have features
evaluated via BIC on ridge CV.

Sibling of tasks.feature_hypothesis — same workflow + dedup gate + bootstrap
permit machinery, only the grid spec, system prompt, dataset overview, and
default paths differ.

Architecture:
- Hypothesis Agent: Survey → Hypothesise → (wait) → Translate
- Coding Agent: Code (stateless, isolated from raw data)
- Framework: Evaluate (automated BIC/MI scoring)
- Rewriting Agent: Rewrite (creates training pairs and knowledge graph nodes)
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from docker.models.containers import Container
from loguru import logger

from src.container import container_to_service
from src.task.base import TaskEnvironmentError, TaskSpec
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityExecutionContext,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    FinalizationContext,
    PopulationOutcome,
    PopulationResult,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    TaskReward,
    Variation,
    Workflow,
    WorkflowStep,
)
from tasks.common.dedup_ledger import JsonDedupLedger
from tasks.common.foundry_exec import coerce_exec_result, exec_run_with_timeout
from tasks.common.ordered_pair_queue import OrderedPairQueue


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain JSON-serializable types.

    The voxel BIC scoring computes with numpy, so its result dict carries
    ``np.float64`` / ``np.bool_`` values. ``np.float64`` survives ``json.dumps``
    (it subclasses ``float``) but ``np.bool_`` does not — it raises
    "Object of type bool is not JSON serializable" when a capability result
    (scoring / get_experiment_summary / submit_rewrite) is serialized over the
    MCP bridge. Coercing at the scoring source keeps every downstream consumer
    clean.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if type(obj).__module__.startswith("numpy"):
        import numpy as np

        if isinstance(obj, np.generic):  # any numpy scalar (bool_, float64, int64, …)
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return obj


_ROLE_SERVICE = {
    "agent": "agent",
    "vfm": "vfm",  # voxel-features-mcp
    "analysis": "analysis",
}

_SUMMARY_CODE_MAX_CHARS = 2_000
_SUMMARY_RESULT_MAX_CHARS = 2_500
_FEATURE_POINT_NUMERIC_COLUMNS = {"longitude", "latitude", "depth_m", "value"}
_VALID_COORDINATE_SOURCES = {"artifact", "geonames", "web", "creative_fallback"}


def _safe_artifact_component(value: Any) -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe or "unknown"


def _compact_agent_text(value: Any, max_chars: int) -> tuple[str, bool]:
    if not isinstance(value, str):
        return "", False
    if max_chars <= 0 or len(value) <= max_chars:
        return value, False
    head_chars = max(1, int(max_chars * 0.65))
    tail_chars = max(1, max_chars - head_chars)
    omitted = len(value) - head_chars - tail_chars
    return (
        value[:head_chars].rstrip()
        + f"\n...[truncated {omitted} chars; see artifact files/full trace]...\n"
        + value[-tail_chars:].lstrip(),
        True,
    )

_ANALYSIS_INPUT = "/workspace/input"
_ANALYSIS_OUT = "/workspace/out"

# Filenames inside `variation.kg_dir`. Centralised so the task code, tests,
# and any external tooling (e.g. analytics scripts) reference one source.
_KG_EXPERIMENTS = "experiments.jsonl"
_KG_CROSSBREED_INDEX = "crossbreed_index.jsonl"          # legacy MI index — read for back-compat only
_KG_PAIRWISE_DISTANCE = "pairwise_distance.jsonl"        # new orthogonality index (Jaccard/MAE)
_KG_ADMITTED_INDEX = "admitted_index.json"
_KG_BOOTSTRAP_STATE = "bootstrap_state.json"
_KG_INTERWEAVE_STATE = "interweave_state.json"
_KG_QUEUE = "crossbreed_queue.jsonl"
_KG_LOCK = "kg.lock"

# Queue selection knobs. The effective score of an entry is
#   score / (1 + α · attempt_count) / Π (1 + γ · uses(parent_i)),
# with an extra · β multiplier when the pair is "consummated" (already has ≥1
# admitted crossbreed child). α decays repeatedly-tried *pairs*; γ decays
# repeatedly-tried *parents* (the lever that breaks monoculture where
# 32 fold-pairs share the top slope); β puts consummated pairs in a slow lane
# without banning them — LLM hypothesis generation is high-variance and the
# surrounding feature pool keeps growing, so a second attempt at the same
# pair is a genuinely different experiment.
_PAIR_ATTEMPT_DECAY = 0.5
_PARENT_USE_DECAY = 0.05     # γ — safety rail while tensor novelty guards bed
                             # in. With 360 historical uses this gives an ~19×
                             # divisor: strong enough to break monoculture
                             # without hard-banning high-value parents.
_PAIR_DISTANCE_WEIGHT = 2.0  # λ for the orthogonality term in the score prior
_CONSUMMATED_DISCOUNT = 0.25


# Kazakhstan Teniz Basin grid specification - Regional scale for basin analysis
_KAZAKHSTAN_TENIZ_GRID = {
    "origin": [66.5, 49.5, 0.0],      # 66°30'E, 49°30'N, 0m depth
    "maximum": [71.5, 52.5, 80.0],    # 71°30'E, 52°30'N, 80m depth
    "shape": [200, 200, 8],            # ~1.75km x 1.75km x 10m resolution, 320k total voxels
    "crs": "EPSG:4326",
}


_SYSTEM_PROMPT = """You are analyzing Kazakhstan mineral prospects.

Grid: lon 66.5–71.5°E, lat 49.5–52.5°N, depth 0–80m (200×200×8 voxels, ~1.75km/voxel).

Feature admission uses the real two-stage gate:
- Stage 1: once at least two admitted layers exist, adding the candidate must improve system-wide MAE.
- Stage 2: the candidate must improve normalized BIC (`bic_delta < 0`).

Optimize for both gates, but match spatial support to the geological claim. Dense basin-scale surfaces are appropriate only when evidence and mechanism justify broad support; sparse localized layers are acceptable when the theory is localized.
"""


_DATASET_OVERVIEW = """## Kazakhstan Teniz Basin Dataset Overview

This dataset has three corpus classes. A useful survey samples at least one
source from each — vector geometry alone has been observed to bias hypotheses
toward fold-axis distance and miss redox, lithology, host-suite, and drill-log
mechanisms documented in the text and tabular sources.

**Vector data (GeoJSON) — /workspace/input/converted_spatial_data/:**
- copper_prospects.geojson: 113 Point features — sediment-hosted Cu prospects.
  Properties include Latitude, Longitude, Type, Subtype, Age_Ma, Tonnage_Mt,
  Cu_pct, Ag_g_t, Co_pct, Agehost, Unit (host suite — e.g. Vladimirov,
  Kayraktin, Kirey), HostRocks, Mineralogy (e.g. chalcopyrite), Comments.
- copper_prospects_aoi.geojson: 112 Point features — near-duplicate of the
  prospects, same property schema (NOT an AOI polygon).
- anticlines_synclines.geojson: 58 LineString fold-axis traces. Properties:
  id, Name, Type, Number.
- assessment_tract.geojson: 1 MultiPolygon — Teniz Basin tract (49,714 km²).
  Properties include descr, Area_km2, Geology, Age, Dep_type, GT_model,
  Asmt_depth, N_known, N_expected, DepDensity.

**Tabular data (CSV) — /workspace/input/USGS/:**
- TZ_ssCu_Prospects.csv: 113 rows × 32 columns. Same fields as the prospects
  geojson, easier to scan with polars/pandas for value distributions over
  Mineralogy, Agehost, Unit, Age_Ma.
- TZ_ssCu_Tract.csv: 1-row tract summary.

**Text corpora (English + Russian — REQUIRED for non-structural hypotheses):**
- USGS/chunks/*.md (7 files): USGS Sandstone Copper assessment.
  Includes the textbook redox-zoning model
  (pyrite → chalcopyrite → bornite → chalcocite → hematite; oxidized red beds
  overlying chemically reduced gray/green/black strata) and deposit subtype
  taxonomy (reduced facies / sandstone Cu / red bed). The single highest-density
  source of non-structural mechanisms.
- USGS/descriptions/*.md (13 files): figure descriptions from the USGS report.
- 36572_Smolianova_1984/chunks/*.md (328 files): Soviet survey covering
  stratigraphy, tectonics, magmatism, physical properties, mineral evaluation.
  Chunk titles are descriptive (e.g. "STRATIGRAPHY — Proterozoic — Karaashevka
  is predominantly dark green and greenish-gray …").
- 36572_Smolianova_1984/drill_holes_data/*.description.md (63 files): per-well
  descriptions of Soviet wireline log sheets — SP (spontaneous polarisation),
  apparent resistivity (КС), gamma-ray, neutron curves, lithology columns,
  and per-depth spectral-analysis (Pb/Cu/Zn/Mo/Sn) assays. Direct proxies
  for redox boundaries and lithology contacts at metre-scale depth.

**Scale note:** Each voxel covers ~1.75 km × 1.75 km × 10 m, suitable for
regional features. Drill-log signals are sub-voxel and must be aggregated
(e.g. per-borehole mean assay, depth-of-first-anomaly) before becoming a
voxel-grid feature.
"""


# Ordered list of distinct source files/groups for round-robin episode assignment.
# Each episode is assigned the least-explored entry so agents are forced to
# derive hypotheses from different data sources rather than free-roaming and
# fixating on whatever the context history primes them toward.
#
# Entries with a "glob_pattern" field (str or list[str]) point to a directory;
# the agent is shown a code snippet to enumerate only that section's files.
# Entries without "glob_pattern" are read as a single file or plain directory.
_KAZAKHSTAN_SOURCE_FILES = [
    # --- Spatial GeoJSON ---
    {
        "key": "copper_prospects_aoi",
        "path": "converted_spatial_data/copper_prospects_aoi.geojson",
        "description": (
            "112 sediment-hosted copper prospect points (area-of-interest subset) — "
            "coordinates, tonnage (Tonnage_Mt), Cu% grade (Cu_pct), Ag content, "
            "deposit classification."
        ),
    },
    {
        "key": "anticlines_synclines",
        "path": "converted_spatial_data/anticlines_synclines.geojson",
        "description": (
            "58 geological fold structures (anticlines and synclines) — fold axes, "
            "structure names, geological ages."
        ),
    },
    # --- Smolianova 1984 Russian survey — section-level ---
    {
        "key": "smolianova_geological_study",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*II_GEOLOGICAL-GEOPHYSICAL*.md",
        "description": (
            "Smolianova 1984 Ch. II — geological-geophysical survey (~10 chunks): "
            "regional setting, survey methods, geophysical interpretation."
        ),
    },
    {
        "key": "smolianova_stratigraphy_early",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": ["*Proterozoic*.md", "*Cambrian*.md", "*Ordovician*.md"],
        "description": (
            "Smolianova 1984 Ch. V — early stratigraphy (~20 chunks): "
            "Proterozoic, Cambrian, Ordovician sequences and basement lithology."
        ),
    },
    {
        "key": "smolianova_stratigraphy_devonian",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*Devonian*.md",
        "description": (
            "Smolianova 1984 Ch. V — Devonian stratigraphy (~16 chunks): "
            "sedimentary sequences, redbeds, lithology descriptions."
        ),
    },
    {
        "key": "smolianova_stratigraphy_carboniferous_lower",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*Carboniferous_Lower*.md",
        "description": (
            "Smolianova 1984 Ch. V — Lower Carboniferous (~31 chunks): "
            "key copper-hosting sedimentary sequences and diagenetic features."
        ),
    },
    {
        "key": "smolianova_stratigraphy_carboniferous_middle",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*Carboniferous_Middle*.md",
        "description": (
            "Smolianova 1984 Ch. V — Middle Carboniferous (~10 chunks): "
            "stratigraphy, lithology, and depositional environment."
        ),
    },
    {
        "key": "smolianova_stratigraphy_carboniferous_upper",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*Carboniferous_Upper*.md",
        "description": (
            "Smolianova 1984 Ch. V — Upper Carboniferous (~15 chunks): "
            "stratigraphy, lithology, and lateral facies variation."
        ),
    },
    {
        "key": "smolianova_stratigraphy_permian",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*STRATIGRAPHY_Permian*.md",
        "description": (
            "Smolianova 1984 Ch. V — Permian stratigraphy (~27 chunks): "
            "red-bed sequences, evaporites, post-orogenic basin evolution."
        ),
    },
    {
        "key": "smolianova_magmatic",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*MAGMATIC_FORMATIONS*.md",
        "description": (
            "Smolianova 1984 Ch. VI — Magmatic formations (~49 chunks): "
            "igneous intrusions, volcanic sequences, geochemistry."
        ),
    },
    {
        "key": "smolianova_tectonics",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*TECTONICS*.md",
        "description": (
            "Smolianova 1984 Ch. VII — Tectonics (~8 chunks): "
            "structural geology, fault systems, basin architecture, fold belts."
        ),
    },
    {
        "key": "smolianova_useful_minerals",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*USEFUL_MINERALS*.md",
        "description": (
            "Smolianova 1984 Ch. VIII — Useful minerals (~21 chunks): "
            "economic mineralogy, Cu/Ag occurrences, mineral paragenesis."
        ),
    },
    {
        "key": "smolianova_prognosis",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*REGULARITIES_AND_PROGNOSIS*.md",
        "description": (
            "Smolianova 1984 Ch. IX — Ore-forming regularities and prognosis (~8 chunks): "
            "spatial controls on mineralisation, exploration targeting criteria."
        ),
    },
    {
        "key": "smolianova_geological_work",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*GEOLOGICAL_WORK_IN_APPENDICES*.md",
        "description": (
            "Smolianova 1984 Ch. XI — Geological work in appendices (~14 chunks): "
            "detailed analytical results, laboratory data, supporting studies."
        ),
    },
    {
        "key": "smolianova_textual_appendix",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*TEXTUAL_APPENDIX*.md",
        "description": (
            "Smolianova 1984 Textual appendix (~74 chunks): mineral deposits catalogue "
            "with deposit name, location, type, tonnage, and grade data."
        ),
    },
    {
        "key": "smolianova_nrs_review",
        "path": "36572_Smolianova_1984/chunks/",
        "glob_pattern": "*NRS_REVIEW*.md",
        "description": (
            "Smolianova 1984 — NRS review & commission appendix (~9 chunks): "
            "expert commentary on survey conclusions and recommended follow-up."
        ),
    },
    {
        "key": "smolianova_drill_holes",
        "path": "36572_Smolianova_1984/drill_holes_data/",
        "description": (
            "Smolianova 1984 — 60+ borehole logs: lithology descriptions, depth "
            "intervals, core recovery, mineralisation intercepts."
        ),
    },
    # --- USGS sandstone copper assessment ---
    {
        "key": "usgs_report",
        "path": "USGS/chunks/",
        "description": (
            "USGS sandstone copper assessment (~7 chunks): English-language report, "
            "quantitative resource estimates, deposit model methodology, figure "
            "descriptions (13 figures)."
        ),
    },
]


@dataclass
class FeatureHypothesisKazakhstanVariation(Variation):
    """Variation configuration for feature hypothesis task."""

    dataset_dir: str = ""
    store_dir: str = ""
    kg_dir: str = ""
    grid_spec: dict[str, Any] = field(default_factory=lambda: dict(_KAZAKHSTAN_TENIZ_GRID))
    min_features: int = 0  # minimum features before crossbreeding
    crossbreed_enabled: bool = True
    # Crossbreed-pool dedup keeps near-identical experiments from flooding
    # `experiments.jsonl`. When enabled, an admitted record's fingerprint
    # (ordered parents + hypothesis) must be unseen; duplicates remain
    # successes for reward purposes but are silently skipped from the pool.
    dedup_enabled: bool = True
    # Optional upper bound on concurrent bootstrap (= survey) episodes. The
    # default 0 disables task-layer gating so `GenerationConfig.parallel_episodes`
    # is used from the first bootstrap episode. Set a positive value only when
    # a deliberate survey-phase cap is required.
    bootstrap_concurrency_cap: int = 0
    # Legacy ramp knobs retained so older variation configs still deserialize;
    # when permit gating is enabled, bootstrap starts at the full cap.
    bootstrap_window_size: int = 0
    bootstrap_min_concurrency_fraction: float = 1.0
    bootstrap_permit_timeout_s: float = 600.0
    bootstrap_permit_stale_after_s: float = 1800.0
    # Novelty nudge: surface the last K admitted hypotheses in the proposer
    # prompt as a "do not propose variants of these" block. Counters the
    # diversity collapse pattern (e.g. 3 unique fingerprints across 245
    # episodes) at the prompt layer, complementing the lexical dedup gate.
    novelty_nudge_enabled: bool = True
    novelty_recent_k: int = 8
    # Per-entry cap so the block stays bounded under long hypotheses; the
    # rendering uses an ellipsis when exceeded.
    novelty_max_chars_per_hypothesis: int = 280
    # Once steady-state crossbreed has stalled for this many completed attempts
    # without a fresh KG admit, enter a short survey burst before returning to
    # crossbreed. This interweaves fresh data-grounded hypotheses without
    # changing scoring or adding an explicit novelty prompt.
    interweave_bootstrap_enabled: bool = True
    interweave_failed_episode_threshold: int = 30
    interweave_survey_burst_episodes: int = 15


@dataclass
class FeatureHypothesisKazakhstanState:
    """Episode state for feature hypothesis task."""
    
    episode_id: str = ""
    workflow_kind: str = "survey"  # survey, crossbreed
    n_features: int = 0
    
    # Phase artifacts collected during episode
    survey_candidates: list[str] = field(default_factory=list)
    hypothesis: str = ""
    hypothesis_uri: str = ""
    data_spec: dict[str, Any] = field(default_factory=dict)
    code_executed: str = ""
    result_summary: str = ""
    feature_layer_name: str | None = None
    feature_values: list | None = None
    
    # Scoring (framework-computed after Translate)
    bic_before: float | None = None
    bic_after: float | None = None
    bic_delta: float | None = None
    cv_mse_delta: float | None = None
    mutual_info: dict[str, float] = field(default_factory=dict)
    admitted: bool = False
    
    # Two-stage scoring results
    masking_test_passed: bool = True
    masking_test_improvement: float = 0.0
    masking_test_direction: str = "not_applicable"
    stage_1_tolerance_used: bool = False
    stage_1_mae_tolerance: float | None = None
    stage_1_bic_rescue_threshold: float | None = None
    stage_completed: str = "stage_2_completed"
    admission_path: str = "normal"
    proposal_evidence_tier: str = "mixed"
    confidence: float | None = None
    evidence_strength: float | None = None
    admission_tier: str | None = None
    crossbreed_parent_eligible: bool | None = None
    
    # Crossbreeding context
    parent_experiments: list[str] = field(default_factory=list)

    # Training data
    prompt_response_pair: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BIC-result appendix pattern injected by _exec_submit_rewrite; outcome
# narrative completions should not train on this harness-added suffix.
# ---------------------------------------------------------------------------
_BIC_RESULT_RE = re.compile(
    r"\s*Result:\s*-?\d+(?:\.\d+)?\s+BIC delta\.\s*(?:Admitted|Not admitted)\.",
    re.IGNORECASE,
)

# Stop-words for the novelty heuristic that decides whether to emit a row
# asking for a new hypothesis from dataset context alone.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "to", "for", "with",
        "is", "are", "was", "were", "that", "this", "it", "be", "by",
        "at", "as", "on", "from", "can", "will", "may", "has", "have",
    }
)

PAIR_KIND_PARENT_HYPOTHESIS = "parent_hypothesis"
PAIR_KIND_DATASET_HYPOTHESIS = "dataset_hypothesis"
PAIR_KIND_ANALYSIS_PLAN = "analysis_plan"
PAIR_KIND_OUTCOME_NARRATIVE = "outcome_narrative"


# Parse the assigned-source section name and the pre-read sample block out of a
# survey/explore prompt. The merged ``explore`` step injects an "ASSIGNED SOURCE"
# header plus a "SAMPLE CONTENT" excerpt of the file the episode was anchored to;
# both describe what the agent actually read. We lift them onto the QUERY side of
# synthesized rows so the model sees the source evidence as input while the
# completion contains only what the agent had to produce: reasoning, hypothesis,
# and plan. This prevents harness-injected grounding from being trained as if it
# were model output.
_ASSIGNED_SECTION_RE = re.compile(
    r"ASSIGNED SOURCE FOR THIS EPISODE\s*\n\s*(?:Section|Path)\s*:\s*(.+)"
)
_SAMPLE_BLOCK_RE = re.compile(
    r"SAMPLE CONTENT FROM YOUR ASSIGNED SOURCE\s*\n[-─—_]+\s*\n(.*?)"
    r"(?:\n\s*\n\s*Use analysis_shell|\n\s*Use analysis_shell|\Z)",
    re.DOTALL,
)


def _content_words(text: str) -> set[str]:
    """Return lower-cased non-stop words from *text*."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _is_novel_vs_parents(hypothesis: str, parent_hypotheses: list[str]) -> bool:
    """Return True when >50% of the child hypothesis content-words are absent
    from all parent hypotheses combined.  Falls back to True when there are no
    parents (genuinely de-novo survey episode)."""
    if not parent_hypotheses:
        return True
    child_words = _content_words(hypothesis)
    if not child_words:
        return False
    parent_words: set[str] = set()
    for ph in parent_hypotheses:
        parent_words |= _content_words(ph)
    novel_count = sum(1 for w in child_words if w not in parent_words)
    return novel_count / len(child_words) > 0.50


class ExperimentReasoningRows:
    """Synthesize prompt-completion rows from successful geology episodes.

    Each successful episode may produce these row kinds:

    - ``parent_hypothesis``: parent findings -> child hypothesis and rationale.
      Skipped when parent context is not recoverable.
    - ``dataset_hypothesis``: dataset context -> new hypothesis and rationale.
      Only emitted when the hypothesis has enough vocabulary not seen in parent
      hypotheses.
    - ``analysis_plan``: hypothesis and available files -> data_spec plan.
    - ``outcome_narrative``: hypothesis and built feature -> explanatory
      narrative. These rows are tagged ``faithfulness = "post_hoc"`` because
      they are reconstructed after evaluation, and the harness-added BIC result
      appendix is stripped from the completion.

    Prompts never include the BIC delta value or admitted/not-admitted outcome.
    Curation collapses exact prompt/completion duplicates and caps dominant
    hypothesis families while preserving dataset-context hypothesis rows.
    """

    def __init__(
        self,
        *,
        max_per_family: int = 5,
        novelty_threshold: float = 0.50,
        max_pair_chars: int = 12_000,
        dataset_dir: str = "",
        max_evidence_chars: int = 2_500,
    ) -> None:
        self._max_per_family = max_per_family
        self._novelty_threshold = novelty_threshold
        self._max_pair_chars = max_pair_chars
        # Optional host dataset root (maps /workspace/input/X -> dataset_dir/X)
        # used for best-effort evidence enrichment; "" disables disk reads.
        self._dataset_dir = dataset_dir
        self._max_evidence_chars = max_evidence_chars
        self._source_payload_cache_path: Path | None = None
        self._source_payload_cache_identity: tuple[int, int] | None = None
        self._source_payload_cache_offset = 0
        self._source_payload_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # TrainingDataTransform protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ExperimentReasoningRows[v1]"

    def config(self) -> dict[str, Any]:
        # dataset_dir is deliberately excluded: it is an environment detail, not
        # a recipe parameter, and including it would make the export recipe hash
        # machine-specific (breaking the resume/export-recipe-hash guard).
        return {
            "max_per_family": self._max_per_family,
            "novelty_threshold": self._novelty_threshold,
            "max_pair_chars": self._max_pair_chars,
            "max_evidence_chars": self._max_evidence_chars,
        }

    def transform_export_rows(
        self,
        context: Any,
        episodes: list[Any],
    ) -> list[Any]:
        from src.training_data.transforms import EpisodeTrainingRows

        source_payloads = self._load_source_episode_payloads(context)
        raw: list[tuple[EpisodeTrainingRows, list[dict[str, Any]], float | None]] = []

        for episode in episodes:
            source_payload = source_payloads.get(getattr(episode, "episode_id", ""), {})
            record = self._backfill_record(episode, source_payload)
            if not record.get("training_success", True):
                raw.append((episode, [], None))
                continue

            rows = self._synthesize_rows(episode, record)
            raw.append((episode, rows, record.get("bic_delta")))

        # Curate rows by exact pair de-duplication and family balance.
        curated = self._curate(raw)

        # Preserve empty groups for failed episodes so the caller sees the same
        # episode count.
        out: list[EpisodeTrainingRows] = []
        for episode, rows, _bic in curated:
            out.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=episode.episode_score,
                    rows=rows,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_source_payload_cache(
        self,
        source_path: Path,
        identity: tuple[int, int],
    ) -> None:
        self._source_payload_cache_path = source_path
        self._source_payload_cache_identity = identity
        self._source_payload_cache_offset = 0
        self._source_payload_cache = {}

    def _load_source_episode_payloads(self, context: Any) -> dict[str, dict[str, Any]]:
        path = getattr(context, "source_all_episodes_path", None)
        if path is None:
            return {}
        source_path = Path(path)
        try:
            stat = source_path.stat()
        except FileNotFoundError:
            return {}
        identity = (int(stat.st_dev), int(stat.st_ino))
        if (
            self._source_payload_cache_path != source_path
            or self._source_payload_cache_identity != identity
            or stat.st_size < self._source_payload_cache_offset
        ):
            self._reset_source_payload_cache(source_path, identity)

        try:
            with source_path.open("rb") as handle:
                handle.seek(self._source_payload_cache_offset)
                while True:
                    line_start = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        self._source_payload_cache_offset = handle.tell()
                        break
                    if not raw_line.endswith(b"\n"):
                        self._source_payload_cache_offset = line_start
                        break
                    self._source_payload_cache_offset = handle.tell()
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line.decode("utf-8"))
                    if isinstance(payload, dict) and isinstance(payload.get("episode_id"), str):
                        self._source_payload_cache[payload["episode_id"]] = payload
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ExperimentReasoningRows: failed to inspect all_episodes.jsonl: {exc}")
        return dict(self._source_payload_cache)

    def _backfill_record(
        self,
        episode: Any,
        source_payload: dict[str, Any],
    ) -> dict[str, Any]:
        episode_context = self._episode_context(episode, source_payload)
        rows = list(getattr(episode, "rows", []))
        meta_record = self._first_meta_record(rows)
        transcript_phase_records = self._phase_records_from_tool_outputs(rows)

        phase_records = self._first_dict(
            meta_record.get("phase_records"),
            episode_context.get("phase_records"),
            meta_record.get("experiment_record", {}).get("phase_records")
            if isinstance(meta_record.get("experiment_record"), dict)
            else None,
            transcript_phase_records,
        )
        terminal_record = self._first_dict(
            meta_record.get("terminal_record"),
            episode_context.get("terminal_record"),
            meta_record.get("experiment_record", {}).get("terminal_record")
            if isinstance(meta_record.get("experiment_record"), dict)
            else None,
        )
        graph_node = self._first_dict(
            terminal_record.get("graph_node") if isinstance(terminal_record, dict) else None,
            meta_record.get("graph_node"),
        )
        crossbreed_context = self._first_dict(
            episode_context.get("crossbreed_context"),
            meta_record.get("crossbreed_context"),
        )

        hypothesise = self._first_dict(phase_records.get("hypothesise"))
        code = self._first_dict(phase_records.get("code"))
        translate = self._first_dict(phase_records.get("translate"))
        evaluate = self._first_dict(phase_records.get("evaluate"))
        outcome = self._first_dict(graph_node.get("outcome"))
        task_breakdown = self._first_dict(source_payload.get("task_breakdown"))

        explore_row = self._row_for_step(rows, "explore")
        hypothesise_row = self._row_for_step(rows, "hypothesise") or explore_row
        survey_row = self._row_for_step(rows, "survey") or explore_row
        rewrite_row = self._rewrite_output_row(rows)

        hypothesis, hypothesis_source = self._first_text_with_source(
            (hypothesise.get("hypothesis"), "phase_records"),
            (graph_node.get("hypothesis"), "graph_node"),
            (meta_record.get("hypothesis"), "record_meta"),
            (self._parse_hypothesis(hypothesise_row), "transcript"),
        )
        data_spec, data_spec_source = self._first_value_with_source(
            (hypothesise.get("data_spec"), "phase_records"),
            (graph_node.get("data_spec"), "graph_node"),
            (meta_record.get("data_spec"), "record_meta"),
            (self._parse_data_spec(hypothesise_row), "transcript"),
        )
        if not isinstance(data_spec, dict):
            data_spec = {}
        parent_ids = self._string_list(hypothesise.get("parent_experiments"))
        if not parent_ids:
            parent_ids = self._string_list(meta_record.get("parent_experiments"))
        if not parent_ids:
            parent_ids = self._string_list(crossbreed_context.get("parent_ids"))
        parent_context, parents_source = self._parent_context(
            episode_context,
            hypothesise,
            hypothesise_row,
            hypothesis,
            crossbreed_context,
        )
        parent_hypotheses = self._parent_hypotheses(hypothesise, parent_context)
        bic_delta, outcome_source = self._first_float_with_source(
            (evaluate.get("bic_delta"), "phase_records"),
            (outcome.get("bic_delta"), "graph_node"),
            (task_breakdown.get("bic_delta"), "task_breakdown"),
            (meta_record.get("bic_delta"), "record_meta"),
        )

        narrative, narrative_source = self._first_text_with_source(
            (rewrite_row.get("raw_response") if rewrite_row else None, "rewrite_output"),
            (
                terminal_record.get("training_pair", {}).get("response")
                if isinstance(terminal_record.get("training_pair"), dict)
                else None,
                "terminal_record",
            ),
        )
        narrative_clean, outcome_appended = self._strip_outcome_appendix(narrative)
        survey_context = self._survey_context(survey_row)
        if survey_row is explore_row and parent_ids:
            survey_context = (
                "Kazakhstan Teniz Basin dataset context: vector prospect points, "
                "fold-axis traces, tract geometry, tabular prospect and tract "
                "data, USGS sandstone-copper report chunks, Smolianova survey "
                "chunks, and drill-hole descriptions."
            )

        # Lift the source the agent read (assignment + pre-read sample, optionally
        # enriched from the on-disk dataset) onto the query side; keep the
        # data_spec 'analysis' as the observation that grounds the analysis plan.
        explore_prompt_text = self._row_text(explore_row, "prompt")
        data_spec_files = self._data_spec_files(data_spec)
        captured_excerpt = ""
        captured_section = ""
        if isinstance(hypothesise, dict):
            captured_excerpt = str(hypothesise.get("source_excerpt") or "").strip()
            captured_section = str(hypothesise.get("assigned_section") or "").strip()
        assigned_section, source_evidence = self._source_evidence(
            explore_prompt_text,
            data_spec_files,
            captured_excerpt=captured_excerpt,
            captured_section=captured_section,
        )
        observation = ""
        if isinstance(data_spec, dict):
            analysis_value = data_spec.get("analysis")
            if isinstance(analysis_value, str):
                observation = analysis_value.strip()

        return {
            "training_success": bool(episode_context.get("success", True)),
            "duplicate_rejected": bool(episode_context.get("duplicate_rejected", False)),
            "hypothesis": hypothesis,
            "data_spec": data_spec,
            "parent_ids": parent_ids,
            "parent_context": parent_context,
            "parent_hypotheses": parent_hypotheses,
            "survey_context": survey_context,
            "assigned_section": assigned_section,
            "source_evidence": source_evidence,
            "observation": observation,
            "hypothesise_response": self._row_text(hypothesise_row, "raw_response"),
            "feature_layer_name": str(
                translate.get("feature_layer_name")
                or graph_node.get("feature_layer_name")
                or ""
            ).strip(),
            "result_summary": str(
                code.get("result_summary")
                or graph_node.get("experiment_summary")
                or ""
            ).strip(),
            "bic_delta": bic_delta,
            "narrative": narrative_clean,
            "outcome_appended": outcome_appended,
            "source_rows": {
                "survey": survey_row,
                "hypothesise": hypothesise_row,
                "explore": explore_row,
                "rewrite": rewrite_row,
            },
            "provenance": {
                "hypothesis": hypothesis_source,
                "data_spec": data_spec_source,
                "parents": parents_source,
                "outcome": outcome_source,
                "narrative": narrative_source,
            },
        }

    @staticmethod
    def _episode_context(episode: Any, source_payload: dict[str, Any]) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        raw_ctx = getattr(episode, "episode_context", None)
        if isinstance(raw_ctx, dict):
            ctx.update(raw_ctx)
        if source_payload:
            ctx.setdefault("success", bool(source_payload.get("success", True)))
            ctx.setdefault("task_breakdown", source_payload.get("task_breakdown", {}))
            ctx.setdefault("trajectory", source_payload.get("trajectory", {}))
        return ctx

    @staticmethod
    def _first_meta_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in rows:
            meta = row.get("record_meta")
            if not isinstance(meta, dict):
                continue
            experiment_record = meta.get("experiment_record")
            if isinstance(experiment_record, dict):
                out.update(experiment_record)
            for key in (
                "phase_records",
                "terminal_record",
                "graph_node",
                "hypothesis",
                "data_spec",
                "parent_experiments",
                "bic_delta",
            ):
                if key in meta and key not in out:
                    out[key] = meta[key]
        return out

    @staticmethod
    def _first_dict(*values: Any) -> dict[str, Any]:
        for value in values:
            if isinstance(value, dict) and value:
                return value
        return {}

    @staticmethod
    def _row_for_step(rows: list[dict[str, Any]], step: str) -> dict[str, Any]:
        for row in rows:
            if row.get("workflow_step") == step:
                return row
        return {}

    @staticmethod
    def _rewrite_output_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            if row.get("workflow_step") == "rewrite" and ExperimentReasoningRows._row_text(
                row,
                "raw_response",
            ):
                return row
        return ExperimentReasoningRows._row_for_step(rows, "rewrite")

    @staticmethod
    def _row_text(row: dict[str, Any], field: str) -> str:
        value = row.get(field) if isinstance(row, dict) else None
        return value.strip() if isinstance(value, str) else ""

    @classmethod
    def _phase_records_from_tool_outputs(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        phase_records: dict[str, Any] = {}
        for row in rows:
            prompt = cls._row_text(row, "prompt")
            if "[tool]" not in prompt:
                continue
            for output in cls._iter_tool_outputs(prompt):
                if any(
                    key in output
                    for key in (
                        "hypothesis",
                        "data_spec",
                        "parent_experiments",
                        "source_excerpt",
                    )
                ):
                    phase_records.setdefault("hypothesise", {}).update(
                        {
                            key: output[key]
                            for key in (
                                "hypothesis",
                                "data_spec",
                                "parent_experiments",
                                "source_excerpt",
                                "assigned_section",
                            )
                            if key in output
                        }
                    )
                if any(key in output for key in ("code_executed", "result_summary", "artifact_files")):
                    phase_records.setdefault("code", {}).update(
                        {
                            key: output[key]
                            for key in (
                                "code_executed",
                                "result_summary",
                                "artifact_directory",
                                "artifact_files",
                            )
                            if key in output
                        }
                    )
                feature_layer_name = output.get("feature_layer_name")
                if isinstance(feature_layer_name, str) and feature_layer_name.strip():
                    phase_records.setdefault("translate", {})["feature_layer_name"] = (
                        feature_layer_name.strip()
                    )
                if any(key in output for key in ("bic_delta", "admitted", "mutual_info")):
                    phase_records.setdefault("evaluate", {}).update(
                        {
                            key: output[key]
                            for key in ("bic_delta", "admitted", "mutual_info")
                            if key in output
                        }
                    )
        return phase_records

    @staticmethod
    def _iter_tool_outputs(text: str) -> Iterator[dict[str, Any]]:
        decoder = json.JSONDecoder()
        marker = "[tool]\n"
        cursor = 0
        while True:
            marker_index = text.find(marker, cursor)
            if marker_index < 0:
                return
            start = marker_index + len(marker)
            chunk = text[start:].lstrip()
            try:
                payload, end = decoder.raw_decode(chunk)
            except ValueError:
                cursor = start
                continue
            cursor = start + end
            if not isinstance(payload, dict):
                continue
            output = payload.get("output")
            if isinstance(output, dict):
                yield output

    @staticmethod
    def _parse_hypothesis(row: dict[str, Any]) -> str:
        text = ExperimentReasoningRows._row_text(row, "raw_response")
        if not text:
            return ""
        match = re.search(r"Hypothesis:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_data_spec(row: dict[str, Any]) -> dict[str, Any]:
        text = ExperimentReasoningRows._row_text(row, "raw_response")
        if not text:
            return {}
        match = re.search(r"DataSpec:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return {}
        return {"text": match.group(1).strip()}

    @staticmethod
    def _first_text_with_source(*items: tuple[Any, str]) -> tuple[str, str]:
        for value, source in items:
            if isinstance(value, str) and value.strip():
                return value.strip(), source
        return "", "missing"

    @staticmethod
    def _first_value_with_source(*items: tuple[Any, str]) -> tuple[Any, str]:
        for value, source in items:
            if isinstance(value, dict) and value:
                return value, source
            if isinstance(value, str) and value.strip():
                return {"text": value.strip()}, source
        return {}, "missing"

    @staticmethod
    def _first_float_with_source(*items: tuple[Any, str]) -> tuple[float | None, str]:
        for value, source in items:
            if value is None:
                continue
            try:
                return float(value), source
            except (TypeError, ValueError):
                continue
        return None, "missing"

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item]

    def _parent_context(
        self,
        episode_context: dict[str, Any],
        hypothesise: dict[str, Any],
        hypothesise_row: dict[str, Any],
        child_hypothesis: str,
        crossbreed_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        parent_context = hypothesise.get("parent_context")
        if isinstance(parent_context, list) and parent_context:
            rendered = self._format_parent_summary(parent_context)
            sanitized = self._sanitize_prompt_context(rendered, child_hypothesis)
            if sanitized:
                return sanitized, "phase_records"

        crossbreed_context = crossbreed_context or episode_context.get("crossbreed_context")
        if isinstance(crossbreed_context, dict):
            prompt = crossbreed_context.get("prompt")
            if isinstance(prompt, str):
                sanitized = self._sanitize_prompt_context(prompt, child_hypothesis)
                if sanitized:
                    return sanitized, "crossbreed_context"

        prompt = self._row_text(hypothesise_row, "prompt")
        if prompt and re.search(r"parent|prior experiment|experiment\s+\d", prompt, re.IGNORECASE):
            sanitized = self._sanitize_prompt_context(prompt, child_hypothesis)
            if sanitized:
                return sanitized, "transcript"
        return "", "missing"

    @staticmethod
    def _parent_hypotheses(hypothesise: dict[str, Any], parent_context: str) -> list[str]:
        parent_records = hypothesise.get("parent_context")
        if isinstance(parent_records, list):
            out = [
                item.get("hypothesis", "").strip()
                for item in parent_records
                if isinstance(item, dict) and isinstance(item.get("hypothesis"), str)
            ]
            if out:
                return out
        quoted = re.findall(r"Experiment\s+\d+:\s*\"(.+?)\"", parent_context)
        return [item.strip() for item in quoted if item.strip()]

    @staticmethod
    def _strip_outcome_appendix(text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        cleaned, count = _BIC_RESULT_RE.subn("", text)
        return cleaned.rstrip(), count > 0

    @staticmethod
    def _sanitize_prompt_context(text: str, child_hypothesis: str = "") -> str:
        cleaned = _BIC_RESULT_RE.sub("", text)
        cleaned = re.sub(
            r"^.*\bBIC (?:delta|improvement)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        cleaned = re.sub(
            r"\s*-?\d+(?:\.\d+)?\s+BIC (?:delta|improvement)\b\.?:?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\bnot admitted\b|\badmitted\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if child_hypothesis and child_hypothesis.strip() in cleaned:
            return ""
        return cleaned

    @staticmethod
    def _survey_context(survey_row: dict[str, Any]) -> str:
        prompt = ExperimentReasoningRows._row_text(survey_row, "prompt")
        response = ExperimentReasoningRows._row_text(survey_row, "raw_response")
        if survey_row.get("workflow_step") == "explore":
            return prompt or _DATASET_OVERVIEW
        if prompt and response:
            return f"{prompt}\n\nSurvey notes:\n{response}"
        return prompt or response or _DATASET_OVERVIEW

    def _synthesize_rows(self, episode: Any, record: dict[str, Any]) -> list[dict[str, Any]]:
        hypothesis = str(record.get("hypothesis") or "").strip()
        if not hypothesis:
            return []
        rows: list[dict[str, Any]] = []
        provenance = dict(record.get("provenance") or {})
        source_rows = dict(record.get("source_rows") or {})
        hypothesise_response = str(record.get("hypothesise_response") or "").strip()
        hypothesis_target = hypothesise_response if hypothesis in hypothesise_response else f"Hypothesis: {hypothesis}"

        parent_context = str(record.get("parent_context") or "").strip()
        if parent_context and record.get("parent_ids"):
            rows.append(
                self._make_row(
                    episode=episode,
                    source_row=source_rows.get("hypothesise", {}),
                    row_suffix=PAIR_KIND_PARENT_HYPOTHESIS,
                    prompt=(
                        "Prior experiment findings:\n"
                        f"{parent_context}\n\n"
                        "Task: Compose a new geological hypothesis that extends or "
                        "contrasts with these findings."
                    ),
                    raw_response=hypothesis_target,
                    pair_kind=PAIR_KIND_PARENT_HYPOTHESIS,
                    hypothesis=hypothesis,
                    provenance=provenance,
                    extra_meta={"parent_ids": record.get("parent_ids", [])},
                )
            )

        if self._is_novel(hypothesis, list(record.get("parent_hypotheses") or [])):
            evidence = str(record.get("source_evidence") or "").strip()
            section = str(record.get("assigned_section") or "").strip()
            if evidence:
                header = f"Source examined — {section}:" if section else "Source examined:"
                dataset_prompt = (
                    f"{header}\n{evidence}\n\n"
                    "Task: From the source above, propose a falsifiable geological "
                    "hypothesis grounded in this evidence."
                )
            else:
                survey_context = self._sanitize_prompt_context(
                    str(record.get("survey_context") or "")
                )
                dataset_prompt = (
                    "Dataset context:\n"
                    f"{survey_context}\n\n"
                    "Task: Propose a de-novo geological hypothesis from the available data."
                )
            rows.append(
                self._make_row(
                    episode=episode,
                    source_row=source_rows.get("survey", {}),
                    row_suffix=PAIR_KIND_DATASET_HYPOTHESIS,
                    prompt=dataset_prompt,
                    raw_response=hypothesis_target,
                    pair_kind=PAIR_KIND_DATASET_HYPOTHESIS,
                    hypothesis=hypothesis,
                    provenance=provenance,
                    extra_meta={
                        "novelty_routed": True,
                        "evidence_on_query": bool(evidence),
                    },
                )
            )

        data_spec = record.get("data_spec")
        if isinstance(data_spec, dict) and data_spec:
            files = self._data_spec_files(data_spec)
            plan_target = self._format_data_spec_target(data_spec)
            if plan_target.strip():
                observation = str(record.get("observation") or "").strip()
                evidence = str(record.get("source_evidence") or "").strip()
                section = str(record.get("assigned_section") or "").strip()
                query_parts = [f"Hypothesis: {hypothesis}"]
                if observation:
                    # The observation of what was read grounds the plan; it lives
                    # on the query side, not in the target (boundary re-split).
                    query_parts.append(f"Observations from source: {observation}")
                if evidence:
                    header = (
                        f"Source examined — {section}:" if section else "Source examined:"
                    )
                    query_parts.append(f"{header}\n{evidence}")
                query_parts.append(
                    "Available files: "
                    f"{', '.join(files) if files else 'see dataset context'}"
                )
                query_parts.append(
                    "Task: Design the target feature and analysis plan for testing "
                    "the hypothesis."
                )
                rows.append(
                    self._make_row(
                        episode=episode,
                        source_row=source_rows.get("hypothesise", {}),
                        row_suffix=PAIR_KIND_ANALYSIS_PLAN,
                        prompt="\n\n".join(query_parts),
                        raw_response=plan_target,
                        pair_kind=PAIR_KIND_ANALYSIS_PLAN,
                        hypothesis=hypothesis,
                        provenance=provenance,
                        extra_meta={"observation_on_query": bool(observation)},
                    )
                )

        narrative = str(record.get("narrative") or "").strip()
        if narrative:
            feature = str(record.get("feature_layer_name") or "").strip()
            rows.append(
                self._make_row(
                    episode=episode,
                    source_row=source_rows.get("rewrite", {}),
                    row_suffix=PAIR_KIND_OUTCOME_NARRATIVE,
                    prompt=(
                        f"Hypothesis: {hypothesis}\n\n"
                        + (f"Feature built: {feature}\n\n" if feature else "")
                        + "Task: Interpret whether the experiment was informative for "
                        "geological model compression. Do not cite any BIC number."
                    ),
                    raw_response=narrative,
                    pair_kind=PAIR_KIND_OUTCOME_NARRATIVE,
                    hypothesis=hypothesis,
                    provenance=provenance,
                    extra_meta={
                        "faithfulness": "post_hoc",
                        "outcome_appended": False,
                        "stripped_outcome_appendix": bool(record.get("outcome_appended")),
                    },
                )
            )
        return [row for row in rows if len(row["prompt"]) + len(row["raw_response"]) <= self._max_pair_chars]

    def _is_novel(self, hypothesis: str, parent_hypotheses: list[str]) -> bool:
        if not parent_hypotheses:
            return True
        child_words = _content_words(hypothesis)
        if not child_words:
            return False
        parent_words: set[str] = set()
        for ph in parent_hypotheses:
            parent_words |= _content_words(ph)
        novel_count = sum(1 for word in child_words if word not in parent_words)
        return novel_count / len(child_words) > self._novelty_threshold

    def _make_row(
        self,
        *,
        episode: Any,
        source_row: dict[str, Any],
        row_suffix: str,
        prompt: str,
        raw_response: str,
        pair_kind: str,
        hypothesis: str,
        provenance: dict[str, Any],
        extra_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one fully-validated training row dict."""
        prompt = self._sanitize_prompt_context(prompt)
        record_meta: dict[str, Any] = {
            "task_kind": pair_kind,
            "pair_kind": pair_kind,
            "hypothesis": hypothesis,
            "field_provenance": provenance,
            **extra_meta,
        }
        source_row_id = source_row.get("row_id") if isinstance(source_row, dict) else None
        source_row_index = source_row.get("source_row_index", 0) if isinstance(source_row, dict) else 0
        source_interaction_type = (
            source_row.get("interaction_type") if isinstance(source_row, dict) else None
        ) or "synthesized"
        return {
            "row_id": f"{episode.episode_id}:synth:{row_suffix}",
            "parent_row_id": source_row_id if isinstance(source_row_id, str) and source_row_id else None,
            "prompt": prompt,
            "raw_response": raw_response,
            "interaction_type": "synthesized",
            "source_interaction_type": source_interaction_type,
            "timestamp": self._source_timestamp(source_row, episode),
            "success": True,
            "error_message": None,
            "episode_id": episode.episode_id,
            "episode_index": episode.episode_index,
            "generation_id": episode.generation_id,
            "episode_score": episode.episode_score,
            "episode_score_scope": "whole_episode",
            "source_episode_id": episode.episode_id,
            "source_row_index": int(source_row_index) if isinstance(source_row_index, int) else 0,
            "workflow_step": f"synth_{pair_kind.lower()}",
            "actor_role": "synthesizer",
            "record_meta": record_meta,
        }

    @staticmethod
    def _source_timestamp(source_row: dict[str, Any], episode: Any) -> str:
        timestamp = source_row.get("timestamp") if isinstance(source_row, dict) else None
        if isinstance(timestamp, str) and timestamp:
            return timestamp
        return str(getattr(episode, "generation_id", ""))

    def _format_parent_summary(self, parent_context: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, pc in enumerate(parent_context, 1):
            hyp = pc.get("hypothesis", "")
            finding = pc.get("finding") or pc.get("response") or pc.get("result") or ""
            line = f"Parent {i}: {hyp}"
            if finding:
                line += f"\nFinding: {finding}"
            parts.append(line)
        return "\n".join(parts)

    def _format_data_spec_target(self, data_spec: Any) -> str:
        """Render the analysis-plan TARGET (the planned feature + files + steps).

        Understands both the merged-explore schema ``{analysis, files, output}``
        (``output`` -> target feature) and the legacy
        ``{target_feature, required_files, analysis_steps}`` schema. The
        ``analysis`` observation is intentionally NOT rendered here: it is the
        source-read content and is placed on the query side instead. Free-form
        agent dicts fall back to a readable feature summary rather than a raw
        JSON dump.
        """
        if not isinstance(data_spec, dict):
            return str(data_spec)
        target = self._data_spec_target_feature(data_spec)
        files = self._data_spec_files(data_spec)
        steps = data_spec.get("analysis_steps", [])
        lines: list[str] = []
        if target:
            lines.append(f"Target feature: {target}")
        if files:
            lines.append(f"Required files: {', '.join(files)}")
        if isinstance(steps, list) and steps:
            lines.append("Analysis steps:")
            for j, step in enumerate(steps, 1):
                lines.append(f"  {j}. {step}")
        if lines:
            return "\n".join(lines)
        summary = self._freeform_target_summary(data_spec)
        if summary:
            return summary
        return json.dumps(data_spec, sort_keys=True)

    @staticmethod
    def _data_spec_target_feature(data_spec: dict[str, Any]) -> str:
        for key in ("target_feature", "output", "feature_layer_name", "name"):
            value = data_spec.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _freeform_target_summary(data_spec: dict[str, Any]) -> str:
        """Best-effort readable target for free-form ``{features|feature_layers}``
        dicts the agent sometimes emits instead of the prompted schema."""
        for key in ("features", "feature_layers"):
            value = data_spec.get(key)
            if not isinstance(value, list) or not value:
                continue
            lines: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    name = str(
                        item.get("name")
                        or item.get("feature_layer_name")
                        or ""
                    ).strip()
                    desc = str(item.get("description") or "").strip()
                    rendered = " — ".join(part for part in (name, desc) if part)
                    if rendered:
                        prefix = "Target feature: " if not lines else "  - "
                        lines.append(f"{prefix}{rendered}")
                elif isinstance(item, str) and item.strip():
                    prefix = "Target feature: " if not lines else "  - "
                    lines.append(f"{prefix}{item.strip()}")
            if lines:
                return "\n".join(lines)
        return ""

    @staticmethod
    def _data_spec_files(data_spec: dict[str, Any]) -> list[str]:
        for key in ("files", "required_files"):
            value = data_spec.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, str) and item]
        return []

    # ------------------------------------------------------------------
    # Source-read evidence (query-side grounding for explore rows)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_assigned_source(prompt: str) -> tuple[str, str]:
        """Extract (section/path, pre-read sample) from an explore prompt."""
        section = ""
        match = _ASSIGNED_SECTION_RE.search(prompt)
        if match:
            section = match.group(1).strip()
        sample = ""
        sample_match = _SAMPLE_BLOCK_RE.search(prompt)
        if sample_match:
            sample = sample_match.group(1).strip()
        return section, sample

    def _read_dataset_excerpt(self, files: list[str]) -> str:
        """Best-effort read of the assigned file from the on-disk dataset.

        Maps a container path ``/workspace/input/X`` to ``dataset_dir/X``.
        Returns "" if dataset_dir is unset or no file resolves; never raises.
        """
        dataset_dir = getattr(self, "_dataset_dir", "") or ""
        if not dataset_dir:
            return ""
        base = Path(dataset_dir)
        prefix = _ANALYSIS_INPUT.rstrip("/") + "/"
        for file_path in files:
            if not isinstance(file_path, str) or not file_path:
                continue
            rel = file_path
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
            elif rel.startswith(_ANALYSIS_INPUT):
                rel = rel[len(_ANALYSIS_INPUT):]
            rel = rel.lstrip("/")
            if not rel:
                continue
            try:
                candidate = base / rel
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return ""

    def _source_evidence(
        self,
        explore_prompt: str,
        files: list[str],
        captured_excerpt: str = "",
        captured_section: str = "",
    ) -> tuple[str, str]:
        """Return (assigned_section, bounded sanitized evidence excerpt).

        Source priority: the fuller on-disk file when ``dataset_dir`` is set,
        then the source-side ``captured_excerpt`` persisted at record_phase
        (P1, robust/portable), then the transcript SAMPLE CONTENT block parsed
        from the explore prompt. The excerpt is BIC/outcome sanitized (leakage
        guard) and length-capped.
        """
        parsed_section, parsed_sample = self._parse_assigned_source(explore_prompt)
        section = captured_section or parsed_section
        disk = self._read_dataset_excerpt(files)
        excerpt = disk or captured_excerpt or parsed_sample
        if not excerpt:
            return section, ""
        excerpt = self._sanitize_prompt_context(excerpt)
        if len(excerpt) > self._max_evidence_chars:
            excerpt = excerpt[: self._max_evidence_chars].rstrip() + " …"
        return section, excerpt

    def _hypothesis_head(self, hypothesis: str) -> str:
        """Return a coarse lexical family key for balance capping."""
        words = [word for word in re.findall(r"[a-z]+", hypothesis.lower()) if word not in _STOP_WORDS]
        return " ".join(words[:7]) or "other"

    def _curate(
        self,
        raw: list[tuple[Any, list[dict[str, Any]], float | None]],
    ) -> list[tuple[Any, list[dict[str, Any]], float | None]]:
        """Deduplicate exact pairs, then cap dominant hypothesis families.

        Dataset-context hypothesis rows are always preserved at the family cap.
        """
        best_by_pair: dict[str, tuple[int, int, float, dict[str, Any]]] = {}
        for episode_index, (_episode, rows, bic) in enumerate(raw):
            strength = abs(bic) if bic is not None else 0.0
            for row_index, row in enumerate(rows):
                key = self._pair_key(row)
                current = best_by_pair.get(key)
                if current is None or strength > current[2]:
                    best_by_pair[key] = (episode_index, row_index, strength, row)

        rows_by_episode: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(raw))}
        for episode_index, row_index, _strength, row in sorted(best_by_pair.values()):
            rows_by_episode[episode_index].append(row)

        family_counts: dict[str, int] = {}
        result: list[tuple[Any, list[dict[str, Any]], float | None]] = []
        for index, (episode, _rows, bic) in enumerate(raw):
            rows = rows_by_episode.get(index, [])
            if not rows:
                result.append((episode, [], bic))
                continue
            hypothesis = self._extract_hypothesis_from_rows(rows) or episode.episode_id
            family = self._hypothesis_head(hypothesis)
            if family_counts.get(family, 0) >= self._max_per_family:
                rows = [
                    row
                    for row in rows
                    if row.get("record_meta", {}).get("pair_kind")
                    == PAIR_KIND_DATASET_HYPOTHESIS
                ]
            else:
                family_counts[family] = family_counts.get(family, 0) + 1
            result.append((episode, rows, bic))
        return result

    @staticmethod
    def _pair_key(row: dict[str, Any]) -> str:
        prompt = re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip().lower()
        response = re.sub(r"\s+", " ", str(row.get("raw_response", ""))).strip().lower()
        return hashlib.sha256(f"{prompt}\n---\n{response}".encode("utf-8")).hexdigest()

    def _extract_hypothesis_from_rows(self, rows: list[dict[str, Any]]) -> str:
        """Extract the hypothesis string from the analysis-plan row metadata."""
        for row in rows:
            if row.get("record_meta", {}).get("pair_kind") == PAIR_KIND_ANALYSIS_PLAN:
                meta = row.get("record_meta", {})
                if isinstance(meta, dict) and isinstance(meta.get("hypothesis"), str):
                    return meta["hypothesis"].strip()
                prompt: str = row.get("prompt", "")
                # Analysis-plan prompts start with "Hypothesis: <hypothesis>".
                m = re.match(r"Hypothesis:\s*(.+?)(?:\n|$)", prompt)
                if m:
                    return m.group(1).strip()
        return ""


class FeatureHypothesisKazakhstanProposerRows:
    """SFT transform: keep proposer-persona turns, drop pure executor turns.

    Twin of :class:`tasks.feature_hypothesis.FeatureHypothesisProposerRows`
    for the Kazakhstan variation. Kept as a sibling class (rather than
    re-using the Australian one) so the recipe hash recorded in
    ``export_recipe.json`` makes the source task unambiguous when SFT data
    from both variants ends up in the same downstream sweep.
    """

    DEFAULT_INCLUDED_WORKFLOW_STEPS: tuple[str, ...] = (
        "explore",
        "translate",
        "rewrite",
    )

    def __init__(self, included_workflow_steps: tuple[str, ...] | None = None) -> None:
        self._included: tuple[str, ...] = tuple(
            included_workflow_steps
            if included_workflow_steps is not None
            else self.DEFAULT_INCLUDED_WORKFLOW_STEPS
        )

    @property
    def name(self) -> str:
        return "FeatureHypothesisKazakhstanProposerRows[v1]"

    def config(self) -> dict[str, Any]:
        return {"included_workflow_steps": list(self._included)}

    def transform_export_rows(
        self,
        context: Any,
        episodes: list[Any],
    ) -> list[Any]:
        del context
        from src.training_data.transforms import EpisodeTrainingRows

        allowed = set(self._included)
        out: list[EpisodeTrainingRows] = []
        for episode in episodes:
            kept: list[dict[str, Any]] = []
            for row in episode.rows:
                step = row.get("workflow_step")
                if step is None:
                    raise ValueError(
                        "feature_hypothesis_kazakhstan export row is missing "
                        f"workflow_step (row_id={row.get('row_id')!r})"
                    )
                if step in allowed:
                    kept.append(row)
            out.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=episode.episode_score,
                    rows=kept,
                )
            )
        return out


class FeatureHypothesisKazakhstanTask(TaskSpec[FeatureHypothesisKazakhstanState]):
    """Feature hypothesis discovery task."""
    
    name = "feature-hypothesis-kazakhstan"
    description = "Discover informative feature layers from Kazakhstan Teniz Basin geological data through hypothesis-driven exploration."
    metric_name = "bic_improvement"
    metric_unit = "nats"
    higher_is_better = False  # Lower BIC is better
    agent_service_name = "agent"
    
    def __init__(self, task_config: dict[str, Any]) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        
        # Dataset paths - Kazakhstan data
        default_dataset = repo_root.parent / "Kazkhstan_data"
        self._dataset_dir = Path(task_config.get("dataset_dir", default_dataset)).resolve()

        # Store paths - Kazakhstan regional structure
        default_store = repo_root / "data" / "kazakhstan" / "feature-hypothesis" / "store"
        self._store_dir = Path(task_config.get("store_dir", default_store)).resolve()

        default_kg = repo_root / "data" / "kazakhstan" / "feature-hypothesis" / "knowledge"
        self._kg_dir = Path(task_config.get("kg_dir", default_kg)).resolve()

        default_artifacts = self._kg_dir.parent / "train_data" / "artifacts"
        self._artifact_dir = Path(
            task_config.get("artifact_dir", default_artifacts)
        ).resolve()
        self._allow_creative_fallback_admission = bool(
            task_config.get("allow_creative_fallback_admission", False)
        )

        self._docker_compose_dir = task_config.get(
            "docker_compose_dir", "docker/feature-hypothesis-kazakhstan-compose"
        )

        # Pre-create the per-variation store + kg dirs as the calling user.
        # Otherwise docker compose up's bind-mount auto-creates the missing
        # path as root (daemon UID), then the host-side Python in
        # _kg_lock().mkdir() / _save_index() etc. fails with PermissionError
        # on subsequent runs. Idempotent (exist_ok=True).
        for sub in ("teniz_basin",):
            (self._store_dir / sub / "admitted" / "layers").mkdir(
                parents=True, exist_ok=True
            )
            (self._store_dir / sub / "scratch").mkdir(parents=True, exist_ok=True)
            (self._kg_dir / sub).mkdir(parents=True, exist_ok=True)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        # Pre-warm voxel-features imports. _exec_spatial_capability /
        # _exec_scoring_capability import voxel_features.spatial (which pulls
        # in geopandas/pyproj/shapely — a ~3s cold import) synchronously on the
        # host MCP-bridge event loop. Left cold, the first in-episode capability
        # call blocks that loop long enough for NAT's MCP client (httpx, 5s
        # default timeout) to drop the agent connection mid tool-call and crash
        # the harness container with a ReadTimeout. Warming here keeps every
        # capability call sub-second.
        self._prewarm_voxel_features()

    @staticmethod
    def _prewarm_voxel_features() -> None:
        """Import voxel-features-mcp modules once, at task construction."""
        import sys

        vfm_path = str(
            Path(__file__).resolve().parent.parent.parent / "voxel-features-mcp"
        )
        if vfm_path not in sys.path:
            sys.path.append(vfm_path)
        try:
            import voxel_features.spatial  # noqa: F401
            import voxel_features.store  # noqa: F401
            import voxel_features.mcp.tools.spatial_tools  # noqa: F401
            import voxel_features.mcp.tools.scoring_tools  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"voxel-features prewarm failed; first capability call may be "
                f"slow enough to trip the MCP client timeout: {exc}"
            )

    @property
    def docker_compose_dir(self) -> str:
        return self._docker_compose_dir
    
    def list_variations(self) -> list[Variation]:
        return [
            FeatureHypothesisKazakhstanVariation(
                name="teniz_basin",
                description="Kazakhstan Teniz Basin - discover regional geological features.",
                dataset_dir=str(self._dataset_dir),
                store_dir=str(self._store_dir / "teniz_basin"),
                kg_dir=str(self._kg_dir / "teniz_basin"),
                grid_spec=dict(_KAZAKHSTAN_TENIZ_GRID),
            ),
        ]
    
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    
    def populate(
        self,
        containers: list[Container],
        variation: Variation,
    ) -> PopulationOutcome:
        if not isinstance(variation, FeatureHypothesisKazakhstanVariation):
            raise TypeError("FeatureHypothesisKazakhstanTask requires FeatureHypothesisKazakhstanVariation")

        episode_id = f"ep_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        # Check existing features to decide workflow. Crossbreeding is gated on
        # FULL source coverage + greedy BIC initialisation (rabbit-hole-bias fix,
        # ported from JenD86/file-rotation@72e3239): the -1.0 first-layer BIC
        # sentinel used to flip the pipeline to crossbreed after only ~5/18
        # sources, collapsing the pool to a single hypothesis family.
        n_features = self._count_features(variation)
        all_sources_done = self._all_sources_visited(variation.kg_dir)

        # Once all sources are visited, attempt greedy BIC init (no-op if already
        # complete or another parallel episode beat us to it).
        if all_sources_done:
            self._run_greedy_bic_initialization(variation)

        greedy_done = (
            Path(variation.kg_dir) / "greedy_init_complete.json"
        ).exists()

        crossbreed_ready = (
            variation.crossbreed_enabled and
            n_features >= variation.min_features and
            all_sources_done and
            greedy_done and
            self._has_crossbreed_pairs(variation)
        )
        interweave_bootstrap = False
        interweave_state_after_claim: dict[str, Any] | None = None
        if crossbreed_ready:
            kg_dir_path = Path(variation.kg_dir)
            interweave_bootstrap = self._claim_interweave_bootstrap(
                kg_dir_path,
                threshold=int(getattr(variation, "interweave_failed_episode_threshold", 0) or 0),
                episode_id=episode_id,
                enabled=bool(getattr(variation, "interweave_bootstrap_enabled", True)),
                burst_episodes=int(getattr(variation, "interweave_survey_burst_episodes", 1) or 1),
            )
            if interweave_bootstrap:
                interweave_state_after_claim = self._read_interweave_state(kg_dir_path)
        workflow_kind = "survey" if interweave_bootstrap else (
            "crossbreed" if crossbreed_ready else "survey"
        )

        episode_context: dict[str, Any] = {
            "episode_id": episode_id,
            "variation_name": variation.name,
            "workflow_kind": workflow_kind,
            "dataset_dir": variation.dataset_dir,
            "store_dir": variation.store_dir,
            "kg_dir": variation.kg_dir,
            "grid_spec": variation.grid_spec,
            "n_features": n_features,
        }
        if interweave_bootstrap:
            episode_context.update({
                "interweave_bootstrap": True,
                "interweave_reason": str(
                    (interweave_state_after_claim or {}).get(
                        "last_interweave_reason", "crossbreed_plateau"
                    )
                ),
                "interweave_failed_episode_threshold": int(
                    getattr(variation, "interweave_failed_episode_threshold", 0) or 0
                ),
                "interweave_survey_burst_episodes": int(
                    getattr(variation, "interweave_survey_burst_episodes", 1) or 1
                ),
            })
            if interweave_state_after_claim is not None:
                episode_context["interweave_survey_remaining"] = int(
                    interweave_state_after_claim.get("interweave_survey_remaining", 0)
                )

        # Crossbreed: serve a distinct ordered (parent_a, parent_b) pair from
        # the queue so concurrent slots do not collide on the same parents.
        if workflow_kind == "crossbreed":
            crossbreed_ctx: dict[str, Any] = {}
            kg_dir_path = Path(variation.kg_dir)
            if variation.dedup_enabled:
                pair = self._queue_pop_pair(kg_dir_path)
                if pair is not None:
                    crossbreed_ctx = self._build_crossbreed_context_for_pair(
                        kg_dir_path, pair[0], pair[1]
                    )
            if not crossbreed_ctx:
                # Fall back to the legacy single-best selection (or a no-op
                # empty context if no experiments). Keeps the task usable when
                # dedup_enabled is False or the queue couldn't yield a pair.
                crossbreed_ctx = self._get_crossbreed_context(variation)
            episode_context["crossbreed_context"] = crossbreed_ctx

        # File rotation is survey-only. Crossbreed follows numpy-slices parity:
        # parent experiments plus a generic instruction to inspect relevant
        # sources, without injecting an assigned source/sample block.
        if workflow_kind == "survey":
            self._assign_rotation_source(episode_context, variation)

        # Survey (= bootstrap): cap at the configured full slot count from the
        # first episode. The older N/2 -> N bootstrap ramp is obsolete.
        if (
            workflow_kind == "survey"
            and variation.dedup_enabled
            and variation.bootstrap_concurrency_cap > 0
        ):
            permit_slot_id = f"slot_{episode_id}_{uuid.uuid4().hex[:6]}"
            acquired = self._acquire_bootstrap_permit(
                Path(variation.kg_dir),
                permit_slot_id,
                configured_slots=variation.bootstrap_concurrency_cap,
                window_size=variation.bootstrap_window_size,
                min_fraction=variation.bootstrap_min_concurrency_fraction,
                timeout_s=variation.bootstrap_permit_timeout_s,
                stale_after_s=variation.bootstrap_permit_stale_after_s,
            )
            if acquired:
                episode_context["bootstrap_permit_slot_id"] = permit_slot_id

        results = [
            PopulationResult(
                container_id=getattr(container, "id", ""),
                variation_name=variation.name,
                description=variation.description,
                success=True,
                details={"service": container_to_service(container)},
            )
            for container in containers
        ]
        
        return PopulationOutcome(results=results, episode_context=episode_context)
    
    def verify_population(
        self,
        containers: list[Container],
        variation: Variation,
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> bool:
        return bool(episode_context.get("episode_id"))
    
    # ------------------------------------------------------------------
    # Prompt and workflow
    # ------------------------------------------------------------------
    
    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> TaskPromptSpec:
        assert isinstance(variation, FeatureHypothesisKazakhstanVariation)
        
        workflow_kind = episode_context.get("workflow_kind", "survey")
        n_features = episode_context.get("n_features", 0)
        
        if workflow_kind == "crossbreed":
            crossbreed_ctx = episode_context.get("crossbreed_context", {})
            mission = (
                f"Crossbreed mode: {n_features} features exist. "
                f"Combine insights from successful experiments to propose a new hypothesis.\n\n"
                f"Crossbreed prompt:\n{crossbreed_ctx.get('prompt', '')}"
            )
        else:
            mission = (
                f"Survey mode: {n_features} features exist. "
                "Explore the dataset and propose a hypothesis about an informative feature layer."
            )
        
        env_context = (
            f"Episode: {episode_context.get('episode_id')}\n"
            f"Dataset: {_ANALYSIS_INPUT}\n"
            f"Grid: {json.dumps(variation.grid_spec)}\n"
            f"Features in store: {n_features}\n"
        )
        
        system_instruction = _SYSTEM_PROMPT + "\n\n" + _DATASET_OVERVIEW
        
        return TaskPromptSpec(
            system_instruction=system_instruction,
            environment_context=env_context,
            capabilities=self.list_capabilities(variation, episode_context),
        )
    
    def workflow(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> Workflow | None:
        assert isinstance(variation, FeatureHypothesisKazakhstanVariation)
        
        workflow_kind = episode_context.get("workflow_kind", "survey")
        
        if workflow_kind == "crossbreed":
            return self._crossbreed_workflow(variation, episode_context)
        return self._survey_workflow(variation, episode_context)
    
    def episode_constraints(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> EpisodeConstraints:
        return EpisodeConstraints(
            # Episode-wide budget shared across ALL phases (explore+hypothesise+code+translate+
            # rewrite). The old 60/45 starved the Translate phase: after survey reading + code
            # execution, only a few turns remained, forcing single-point "blob" layers. Raised for
            # an abundance of tool calls so Translate can emit many per-record spatial ops.
            # Per-turn context growth (not turn count) is the overflow lever -- in-loop compaction
            # (context_compaction_*) + tool_output_max_chars handle that; watch overflow rate.
            budgets=BudgetConstraints(max_task_tool_calls=250, max_llm_turns=150),
            success=SuccessConstraints(terminal_capability_for_success="submit_rewrite"),
        )

    def training_data_transforms(self) -> tuple[ExperimentReasoningRows, ...]:
        return (
            ExperimentReasoningRows(
                dataset_dir=str(getattr(self, "_dataset_dir", "") or "")
            ),
        )

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------
    
    def _survey_workflow(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Standard workflow: Survey → Hypothesise → Code → Translate → Evaluate → Rewrite"""

        # Survey and Hypothesise are merged into a single `explore` step that
        # terminates on record_phase(phase='hypothesise'). The explore body
        # (file-rotation assignment + pre-read sample) is built by
        # _generate_explore_prompt. Both the survey and crossbreed workflows
        # run an explore step (see _crossbreed_workflow, which swaps only the
        # prompt — grounding is preserved either way).
        #
        # NOTE: the novelty nudge (self._novelty_block_for) is NOT injected into
        # any prompt. A 2026-05-31 crossbreed experiment injected it and was
        # reverted the same day: it backfired via negation-priming and did not
        # diversify proposals. Survey diversity relies on file rotation; the
        # crossbreed prompt intentionally follows numpy-slices generic grounding.
        explore_prompt = self._generate_explore_prompt(episode_context)

        return Workflow(
            steps=(
                # EXPLORE + HYPOTHESISE AGENT: Phase 1 (survey + hypothesise merged)
                WorkflowStep(
                    name="explore",
                    is_entry=True,
                    prompt=explore_prompt,
                    inherit_all_capabilities=False,
                    capabilities=(
                        "analysis_shell",
                        "record_phase",
                    ),
                    terminator_capabilities=("record_phase",),
                    next_steps=("code",),
                ),

                # CODING AGENT: Phase 3 (isolated, stateless)
                WorkflowStep(
                    name="code",
                    prompt=(
                        "Phase 3: Code (Async Execution with Budget Control)\n\n"
                        "Use async execution tools only.\n\n"
                        "Analyze data to test the hypothesis. Focus on producing data artifacts.\n\n"
                        "**EXECUTION BUDGET**: You have 3 execution attempts. Use them strategically!\n\n"
                        "**WORKFLOW**:\n"
                        "1. Call phase_get(phase='hypothesise') to get enhanced hypothesis and data_spec\n"
                        "2. Write analysis code that:\n"
                        "   - Loads and examines data from data_spec\n"
                        "   - Performs statistical analysis, correlation, classification\n"
                        "   - Creates filtered DataFrames, computed arrays, summary statistics\n"
                        "   - Tests geological relationships and patterns\n"
                        "   - Creates artifact outputs: top-level DataFrames/arrays are auto-saved, and files\n"
                        "     written under /workspace/out are collected as artifacts.\n"
                        "   - KEY DELIVERABLE (the spatial spec): when your analysis localizes the feature, build a\n"
                        "     pandas DataFrame named EXACTLY `feature_points` with EXACTLY these columns:\n"
                        "     longitude, latitude, depth_m, value, coordinate_source -- ONE ROW PER LOCATION the\n"
                        "     feature occupies (every located prospect/contact/sample your analysis supports), NOT a\n"
                        "     single representative point. It is auto-saved as an artifact for the next phase. 'value'\n"
                        "     carries the per-location quantity (concentration, probability, score); 'coordinate_source'\n"
                        "     is 'artifact' when the coordinate came from the dataset, else 'geonames'/'web'/\n"
                        "     'creative_fallback'. The next phase maps this table directly to voxels. (This is a data\n"
                        "     table, not voxel creation.)\n"
                        "3. Submit code for async execution:\n"
                        "   execution_submit(code='your_code_here', timeout_s=300)\n\n"
                        "4. Monitor execution progress:\n"
                        "   execution_status(execution_id='...')  # Check status/progress\n"
                        "   execution_status(execution_id='...')  # Keep checking until 'completed'\n\n"
                        "5. Get results and confirm artifacts exist:\n"
                        "   execution_results(execution_id='...')  # MUST show artifact_files non-empty\n"
                        "   If artifact_files is empty: fix your code to save files, then resubmit.\n\n"
                        "6. Finalize successful execution:\n"
                        "   execution_finalize(execution_id='...', success=True, summary='Brief summary of results')\n\n"
                        "**RETRY STRATEGY**:\n"
                        "- If execution fails/times out: analyze the error, modify code, try again\n"
                        "- If no artifacts produced: focus next attempt on creating data outputs\n"
                        "- If budget exhausted (3 attempts): step will fail and restart with new hypothesis\n\n"
                        "**REQUIREMENTS**:\n"
                        "- Available libraries: pandas, numpy, scipy\n"
                        "- Use try/except blocks for robust file handling\n"
                        "- DO NOT attempt 3D interpolation or voxel creation\n"
                        "- MUST produce at least one artifact file to proceed\n\n"
                        "**SUCCESS CRITERIA**: At least one artifact file created + execution_results shows success"
                    ),
                    context_mode="isolated",
                    inherit_all_capabilities=False,
                    capabilities=(
                        "phase_get",
                        "execution_submit",
                        "execution_status", 
                        "execution_results",
                        "execution_cancel",
                        "execution_finalize",
                    ),
                    terminator_capabilities=("execution_finalize",),
                    next_steps=("translate",),
                ),
                
                # HYPOTHESIS AGENT: Phase 4 (isolated)
                WorkflowStep(
                    name="translate",
                    prompt=(
                        "Phase 4: Translate (Spatial Data Processor)\n\n"
                        "🚨 CRITICAL: You are NOT in analysis mode. Your role is now to convert existing analysis artifacts into spatial feature commands as best you can.\n"
                        "The system automatically maps these to a voxel grid.\n\n"
                        "1. Call get_experiment_summary() to get hypothesis, data_spec, and analysis results\n"
                        "1b. PRIMARY PATH -- if it returns feature_points (the Code phase's table), that IS your\n"
                        "    spatial spec: call spatial_add_point ONCE PER ROW, passing that row's longitude,\n"
                        "    latitude, depth_m, value, and coordinate_source, all under ONE shared layer name. Emit\n"
                        "    every row -- do NOT collapse them to a single point. Then go to scoring (step 5). Use the\n"
                        "    steps below only for locations feature_points does not cover, or if it is empty.\n"
                        "2. Generate spatial commands based on analysis findings:\n"
                        "   Grid bounds: lon 66.5°-71.5°E, lat 49.5°-52.5°N, depth 0-80m\n"
                        "   Resolution: ~1.75km × 1.75km × 10m per voxel (200×200×8 total)\n\n"
                        "   **For prospect/drill data with coordinates:**\n"
                        "   spatial_add_point(name='string', longitude=float, latitude=float, depth_m=float, value=float, radius_m=float)\n\n"
                        "   **For geological structures (faults, anticlines, basins):**\n"
                        "   spatial_add_line(name='string', start_longitude=float, start_latitude=float, start_depth_m=float, end_longitude=float, end_latitude=float, end_depth_m=float, value=float, width_m=float)\n\n"
                        "   **For text-based locations without coordinates:**\n"
                        "   1. Extract spatial references from analysis: formation names, map sheets, localities\n"
                        "   2. Use search tools as needed (no fixed call budget):\n"
                        "      • search_web_geological('Vladimirovskoye geological formation')\n"
                        "      • search_geonames_lookup('M42-I', 'Kazakhstan')\n"
                        "   3. If search yields coordinates → use them\n"
                        "   4. If search fails or is ambiguous → BE CREATIVE and make geological sense:\n"
                        "      • 'southeastern' → bottom-right 25% of grid (lat<50.75°, lon>69.0°)\n"
                        "      • 'northern edge' → top 12.5% (lat>51.75°)\n"
                        "      • When in doubt, distribute spatially and document your reasoning\n"
                        "3. Create exactly ONE coherent feature layer:\n"
                        "   - ALL spatial operations must use the SAME layer name\n"
                        "   - value is the cell's content: set it per operation to carry the quantity your analysis produced (e.g. concentration, probability, a distance-derived score). A constant value on every operation discards that analysis.\n"
                        "   - Emit one operation per located record from your analysis, all under this one layer name, so the layer reflects the data's actual spatial distribution rather than a single placeholder point.\n"
                        "   - Set the remaining fields deliberately: radius_m / width_m to the feature's real extent (the defaults are placeholders); combination_rule to resolve overlaps (max when signals compete, add when they accumulate, mean when they average); coordinate_source to record provenance (artifact for data-derived coordinates, geonames/web when looked up) -- reserve the creative fallback for coordinates you genuinely had to infer.\n"
                        "4. Validate coordinates using spatial_coord_to_voxel() to check grid bounds\n\n"
                        "5. **MANDATORY TO COMPLETE THIS PHASE**:\n"
                        "   🚨 When you are done YOU MUST CALL scoring_create_feature_layer(name='your_layer_name') 🚨\n"
                        "   Example workflow:\n"
                        "   1. spatial_add_point(name='name', ...)\n"
                        "   2. spatial_add_line(name='name', ...)\n"
                        "   3. scoring_create_feature_layer(name='name')  ← REQUIRED!\n"
                        "   \n"
                        "Focus on regional geological intelligence for Kazakhstan basin analysis!"
                    ),
                    context_mode="isolated",
                    inherit_all_capabilities=False,
                    capabilities=(
                        "get_experiment_summary",
                        "spatial_add_point",
                        "spatial_add_line",
                        "spatial_query_region",
                        "spatial_coord_to_voxel",
                        "spatial_get_operations_log",
                        "scoring_create_feature_layer",
                        "search_web_geological",
                        "search_geonames_lookup",
                    ),
                    terminator_capabilities=("scoring_create_feature_layer",),
                    next_steps=("rewrite",),
                ),
                
                # REWRITING AGENT: Phase 5
                WorkflowStep(
                    name="rewrite",
                    prompt=(
                        "Phase 5: Rewrite\n\n"
                        "An experiment was conducted. Write it up as a training pair.\n\n"
                        "1. Call get_experiment_summary() to retrieve the experiment data.\n"
                        "2. Call submit_rewrite(prompt=..., response=...) to close the phase.\n\n"
                        "Pass two string arguments:\n"
                        "  prompt:   A description of the dataset context and the hypothesis "
                        "being tested. What patterns in the data suggested this hypothesis?\n"
                        "  response: What analysis was performed, what was found, and why "
                        "the result is or isn't informative for mineral exploration.\n\n"
                        "Do NOT include the BIC score in your response — "
                        "it will be appended automatically."
                    ),
                    context_mode="isolated",
                    inherit_all_capabilities=False,
                    capabilities=(
                        "get_experiment_summary",
                        "submit_rewrite",
                    ),
                    terminator_capabilities=("submit_rewrite",),
                    next_steps=(),
                ),
            ),
        )
    
    def _crossbreed_workflow(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Crossbreed workflow: same Explore → Code → Translate → Rewrite chain
        as the standard workflow, but the entry `explore` step combines parent
        experiments instead of picking a fresh survey candidate.

        Per the merge decision, crossbreed episodes STILL ground themselves in
        the data: we keep an `explore`-named entry step (so the agent runs
        analysis_shell before hypothesising) rather than jumping straight to a
        bare hypothesise prompt. Only the explore step's prompt differs; the
        rest of the chain is reused verbatim from _survey_workflow.
        """

        crossbreed_ctx = episode_context.get("crossbreed_context", {})
        parent_ids = crossbreed_ctx.get("parent_ids", [])

        base_workflow = self._survey_workflow(variation, episode_context)

        # Crossbreed stays parent-driven and uses generic dataset grounding, not
        # survey assigned-source/sample anchoring. This matches numpy-slices and
        # avoids giving the crossbreed phase a second, unrelated hard anchor.
        crossbreed_grounding = (
            "First, use analysis_shell to ground yourself in the dataset — open a\n"
            "few relevant sources and confirm what the data actually shows.\n"
            "Then, building on the parent findings above together with what you\n"
            "observed, propose ONE hypothesis that combines or extends them.\n\n"
        )
        crossbreed_prompt = (
            "Phase 1: Explore + Hypothesise (Crossbreed Mode)\n\n"
            f"Parent experiments: {parent_ids}\n\n"
            f"{crossbreed_ctx.get('prompt', '')}\n\n"
            f"{crossbreed_grounding}"
            "Include a data_spec as before.\n\n"
            "Declare evidence_tier='weak' when support is indirect, text-only, single-record, or contradicted by prospect metadata. Declaring weak costs no reward; it only raises the system's parentage caution.\n\n"
            "Close with:\n"
            "  record_phase(phase='hypothesise', hypothesis=..., data_spec=..., evidence_tier=..., self_assessment=..., "
            f"parent_experiments={parent_ids})"
        )

        # Keep the full Explore → … → Rewrite chain; only swap the entry
        # `explore` step's prompt for the crossbreed variant. It stays is_entry
        # and keeps analysis_shell so grounding still happens in crossbreed mode.
        crossbreed_explore = WorkflowStep(
            name="explore",
            is_entry=True,
            prompt=crossbreed_prompt,
            inherit_all_capabilities=False,
            capabilities=(
                "analysis_shell",
                "record_phase",
            ),
            terminator_capabilities=("record_phase",),
            next_steps=("code",),
        )
        new_steps = tuple(
            crossbreed_explore if s.name == "explore" else s
            for s in base_workflow.steps
        )
        return Workflow(steps=new_steps)
    
    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    
    def list_capabilities(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> list[Capability]:
        return [
            Capability(
                name="analysis_shell",
                description=(
                    "Execute Python code in an analysis sandbox. "
                    "Has polars, duckdb, scipy, numpy. "
                    "Data is mounted at /workspace/input/."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            ),
            Capability(
                name="record_phase",
                description="Record completion of a workflow phase.",
                schema={
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string"},
                        "candidates": {"type": "array", "items": {"type": "string"}},
                        "corpora_sampled": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Survey-phase only: subset of "
                                "['vector','tabular','text'] indicating which "
                                "corpus classes were actually inspected."
                            ),
                        },
                        "hypothesis": {"type": "string"},
                        "data_spec": {"type": "object"},
                        "evidence_tier": {
                            "type": "string",
                            "enum": ["weak", "mixed", "strong"],
                            "description": "Agent-declared evidence strength; telemetry only.",
                        },
                        "self_assessment": {
                            "type": "object",
                            "description": "Evidence confidence/basis/limitations/sources; telemetry only.",
                        },
                        "feature_layer_name": {"type": "string"},
                        "parent_experiments": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["phase"],
                },
            ),
            Capability(
                name="phase_get",
                description="Retrieve artifacts from a previous phase.",
                schema={
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string"},
                    },
                    "required": ["phase"],
                },
            ),
            Capability(
                name="get_experiment_summary",
                description=(
                    "Retrieve all phase data for the current experiment in one call. "
                    "Returns hypothesis, data_spec, result_summary, feature_layer_name, "
                    "dtype, bic_delta, admitted, and mutual_info."
                ),
                schema={"type": "object", "properties": {}},
            ),
            Capability(
                name="execution_finalize",
                description="Store execution results and complete the code phase.",
                schema={
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string", "description": "Execution ID to finalize"},
                        "success": {"type": "boolean", "description": "Whether execution succeeded"},
                        "summary": {"type": "string", "description": "Brief summary of what was accomplished"},
                    },
                    "required": ["execution_id", "success", "summary"],
                },
            ),
            Capability(
                name="execution_submit",
                description="Submit code for async execution with budget control.",
                schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "timeout_s": {"type": "integer", "description": "Execution timeout in seconds"},
                        "session_id": {"type": "string", "description": "Optional session ID for budget tracking"},
                        "max_attempts": {"type": "integer", "description": "Maximum execution attempts for this session"},
                    },
                    "required": ["code"],
                },
            ),
            Capability(
                name="execution_status",
                description="Check status and progress of async execution.",
                schema={
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string", "description": "Execution ID to check"},
                    },
                    "required": ["execution_id"],
                },
            ),
            Capability(
                name="execution_results",
                description="Get results and artifacts from completed execution.",
                schema={
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string", "description": "Execution ID to get results for"},
                    },
                    "required": ["execution_id"],
                },
            ),
            Capability(
                name="execution_cancel",
                description="Cancel a running execution.",
                schema={
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string", "description": "Execution ID to cancel"},
                    },
                    "required": ["execution_id"],
                },
            ),
            Capability(
                name="submit_rewrite",
                description=(
                    "Submit the training pair for this experiment. "
                    "The knowledge graph node is generated automatically."
                ),
                # `prompt` and `response` are passed as flat top-level string
                # arguments. A nested object parameter (training_pair={...})
                # trips a NAT MCP-client bug: it generates the nested model
                # type twice and its own validation rejects its own parsed
                # value (SubmitRewriteInputSchema.training_pair). Flat scalar
                # params — like every other capability here — avoid it.
                schema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Description of the dataset context and the "
                                "hypothesis being tested."
                            ),
                        },
                        "response": {
                            "type": "string",
                            "description": (
                                "What analysis was performed, what was found, "
                                "and why the result is or isn't informative."
                            ),
                        },
                    },
                    "required": ["prompt", "response"],
                },
            ),
            # Spatial tool capabilities
            Capability(
                name="spatial_add_point",
                description="Add a point feature at geographic coordinates with radius of effect.",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Feature layer name"},
                        "longitude": {"type": "number", "description": "Longitude in degrees"},
                        "latitude": {"type": "number", "description": "Latitude in degrees"},
                        "depth_m": {"type": "number", "description": "Depth in meters"},
                        "value": {"type": "number", "description": "Feature value"},
                        "radius_m": {"type": "number", "description": "Radius of effect in meters"},
                        "dtype": {"type": "string", "enum": ["float", "categorical", "boolean"]},
                        "combination_rule": {"type": "string", "enum": ["replace", "max", "add", "mean"]},
                        "coordinate_source": {"type": "string", "enum": sorted(_VALID_COORDINATE_SOURCES)},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "longitude", "latitude", "depth_m", "value"],
                },
            ),
            Capability(
                name="spatial_add_line",
                description="Add a line feature between two geographic points (e.g., fault, vein).",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Feature layer name"},
                        "start_longitude": {"type": "number", "description": "Start longitude in degrees"},
                        "start_latitude": {"type": "number", "description": "Start latitude in degrees"},
                        "start_depth_m": {"type": "number", "description": "Start depth in meters"},
                        "end_longitude": {"type": "number", "description": "End longitude in degrees"},
                        "end_latitude": {"type": "number", "description": "End latitude in degrees"},
                        "end_depth_m": {"type": "number", "description": "End depth in meters"},
                        "value": {"type": "number", "description": "Feature value"},
                        "width_m": {"type": "number", "description": "Width of line in meters"},
                        "dtype": {"type": "string", "enum": ["float", "categorical", "boolean"]},
                        "combination_rule": {"type": "string", "enum": ["replace", "max", "add", "mean"]},
                        "coordinate_source": {"type": "string", "enum": sorted(_VALID_COORDINATE_SOURCES)},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "start_longitude", "start_latitude", "start_depth_m", 
                                "end_longitude", "end_latitude", "end_depth_m", "value"],
                },
            ),
            Capability(
                name="spatial_query_region",
                description="Query existing features within a geographic region.",
                schema={
                    "type": "object",
                    "properties": {
                        "center_longitude": {"type": "number", "description": "Center longitude in degrees"},
                        "center_latitude": {"type": "number", "description": "Center latitude in degrees"},
                        "center_depth_m": {"type": "number", "description": "Center depth in meters"},
                        "radius_m": {"type": "number", "description": "Query radius in meters"},
                    },
                    "required": ["center_longitude", "center_latitude", "center_depth_m", "radius_m"],
                },
            ),
            Capability(
                name="spatial_coord_to_voxel",
                description="Convert geographic coordinates to voxel indices for validation.",
                schema={
                    "type": "object",
                    "properties": {
                        "longitude": {"type": "number", "description": "Longitude in degrees"},
                        "latitude": {"type": "number", "description": "Latitude in degrees"},
                        "depth_m": {"type": "number", "description": "Depth in meters"},
                    },
                    "required": ["longitude", "latitude", "depth_m"],
                },
            ),
            Capability(
                name="spatial_get_operations_log",
                description="Get history of spatial operations for debugging and review.",
                schema={"type": "object", "properties": {}},
            ),
            Capability(
                name="scoring_create_feature_layer",
                description="Extract spatial layer and evaluate with BIC scoring.",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string", 
                            "description": "Name of existing spatial layer to evaluate"
                        },
                        "dtype": {
                            "type": "string",
                            "enum": ["float", "categorical", "boolean"],
                            "default": "float",
                            "description": "Data type for evaluation"
                        },
                    },
                    "required": ["name"],
                },
            ),
            Capability(
                name="search_web_geological",
                description="Search for geological location information using web search.",
                schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'Vladimirovskoye geological formation')"
                        },
                    },
                    "required": ["query"],
                },
            ),
            Capability(
                name="search_geonames_lookup",
                description="Look up geographical coordinates using OpenStreetMap.",
                schema={
                    "type": "object",
                    "properties": {
                        "place_name": {
                            "type": "string",
                            "description": "Name to search for (e.g., 'Vladimirovskoye', 'M42-I')"
                        },
                        "region": {
                            "type": "string",
                            "description": "Geographic region to constrain search",
                            "default": "Kazakhstan"
                        },
                    },
                    "required": ["place_name"],
                },
            ),
        ]

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: Variation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Execute a capability invocation."""
        
        name = invocation.name
        args = invocation.input or {}
        
        if name == "analysis_shell":
            # Block analysis_shell in the code step - force use of async execution tools
            current_step = ctx.episode_context.get("current_step", "")
            if current_step == "code":
                return CapabilityResult(
                    "analysis_shell", 
                    success=False, 
                    error="analysis_shell is disabled in code step. Use execution_submit/status/results/finalize instead."
                )
            return self._exec_analysis_shell(containers, args, ctx)
        elif name == "record_phase":
            return self._exec_record_phase(args, ctx)
        elif name == "phase_get":
            return self._exec_phase_get(args, ctx)
        elif name == "get_experiment_summary":
            return self._exec_get_experiment_summary(containers, ctx)
        elif name == "execution_finalize":
            return self._exec_execution_finalize(args, ctx)
        elif name.startswith("execution_"):
            return self._exec_execution_capability(containers, args, ctx, name)
        elif name == "submit_rewrite":
            return self._exec_submit_rewrite(containers, args, ctx)
        elif name.startswith("spatial_"):
            return self._exec_spatial_capability(containers, args, ctx, name)
        elif name == "scoring_create_feature_layer":
            return self._exec_scoring_capability(containers, args, ctx, name)
        elif name.startswith("search_"):
            return self._exec_search_capability(containers, args, ctx, name)
        else:
            return CapabilityResult(name, success=False, error=f"Unknown capability: {name}")
    
    def _exec_analysis_shell(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Execute Python code in analysis container."""
        code = args.get("code", "")
        if not code:
            return CapabilityResult("analysis_shell", success=False, error="No code provided")
        
        analysis = self._pick_container(containers, "analysis")
        if analysis is None:
            return CapabilityResult("analysis_shell", success=False, error="No analysis container")
        
        # Check if this is translate phase - if so, auto-detect voxel arrays
        current_step = ctx.episode_context.get("current_step", "")
        if "translate" in current_step:
            return self._exec_analysis_shell_with_voxel_detection(analysis, code, ctx)
        
        # Normal execution for other phases
        cmd = ["python", "-c", code]
        try:
            result = exec_run_with_timeout(analysis, cmd, timeout_s=60)
            exit_code, raw = coerce_exec_result(result)
            stdout = raw.decode(errors="replace")
            return CapabilityResult(
                "analysis_shell",
                output={"stdout": stdout, "stderr": ""},
                success=exit_code == 0,
                error=stdout if exit_code != 0 else None,
            )
        except Exception as e:
            return CapabilityResult("analysis_shell", success=False, error=str(e))
    
    def _exec_analysis_shell_with_voxel_detection(
        self,
        analysis: Container,
        code: str,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Execute analysis_shell code with automatic voxel array detection and feature layer creation."""
        
        # Wrap user code to automatically detect and save (200, 200, 8) voxel arrays
        wrapped_code = '''
import numpy as np
import pickle
import os

# Store original locals to compare later
_original_locals = set(locals().keys())

try:
    # Execute user's analysis code
''' + code + '''
    
    print("\\n" + "="*50)
    print("VOXEL DETECTION - CHECKING FOR (200,200,8) ARRAYS")
    print("="*50)
    
    # Check for voxel arrays in final namespace
    _final_locals = locals().copy()
    _voxel_arrays_found = []
    
    for var_name, obj in _final_locals.items():
        if (not var_name.startswith('_') and 
            var_name not in _original_locals and 
            var_name not in ['np', 'pickle', 'os']):
            
            try:
                if isinstance(obj, np.ndarray) and obj.shape == (200, 200, 8):
                    print(f"Found voxel array '{var_name}': shape {obj.shape}, dtype {obj.dtype}")
                    _voxel_arrays_found.append({
                        'name': var_name,
                        'array': obj,
                        'dtype': str(obj.dtype)
                    })
                elif isinstance(obj, np.ndarray) and len(obj.shape) > 3:
                    print(f"WARNING: Multidimensional array '{var_name}' with shape {obj.shape} detected")
                    print(f"  Only (200,200,8) arrays are accepted. Please reduce to single feature.")
                    
            except Exception as check_err:
                print(f"Error checking '{var_name}': {check_err}")
    
    print(f"\\nTotal voxel arrays detected: {len(_voxel_arrays_found)}")
    
    # Auto-save voxel arrays as feature layers
    _feature_layers_created = []
    for voxel_info in _voxel_arrays_found:
        try:
            # Convert 3D voxel array to feature layer
            voxel_array = voxel_info['array']
            layer_name = voxel_info['name']
            dtype = 'float' if 'float' in voxel_info['dtype'] else 'int'
                
            # Convert to Python list and save
            voxel_list = voxel_array.tolist()
            
            # Here we would call create_feature_layer, but we'll mark it for the wrapper
            print(f"AUTO-SAVE: {layer_name} -> feature layer (dtype: {dtype})")
            _feature_layers_created.append({
                'name': layer_name,
                'values': voxel_list,
                'dtype': dtype
            })
            
        except Exception as save_err:
            print(f"Failed to auto-save '{voxel_info['name']}': {save_err}")
    
    print(f"\\nFEATURE_LAYERS_TO_CREATE: {len(_feature_layers_created)}")
    for layer in _feature_layers_created:
        print(f"  - {layer['name']} ({layer['dtype']})")
    print("="*50)
    
except Exception as user_code_error:
    print(f"ERROR in user analysis code: {user_code_error}")
    import traceback
    traceback.print_exc()
'''
        
        # Execute wrapped code
        cmd = ["python", "-c", wrapped_code]
        try:
            result = exec_run_with_timeout(analysis, cmd, timeout_s=60)
            exit_code, raw = coerce_exec_result(result)
            stdout = raw.decode(errors="replace")
            
            
            return CapabilityResult(
                "analysis_shell",
                output={"stdout": stdout, "stderr": ""},
                success=exit_code == 0,
                error=stdout if exit_code != 0 else None,
            )
            
        except Exception as e:
            return CapabilityResult("analysis_shell", success=False, error=str(e))
    
    def _exec_record_phase(
        self,
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Record phase completion."""
        phase = args.get("phase", "")
        
        # Store in episode context
        phase_records = ctx.episode_context.setdefault("phase_records", {})
        record = {
            "candidates": args.get("candidates"),
            "corpora_sampled": args.get("corpora_sampled"),
            "hypothesis": args.get("hypothesis"),
            "data_spec": args.get("data_spec"),
            "evidence_tier": self._normalise_evidence_tier(args.get("evidence_tier")),
            "self_assessment": args.get("self_assessment") if isinstance(args.get("self_assessment"), dict) else {},
            "feature_layer_name": args.get("feature_layer_name"),
            "parent_experiments": args.get("parent_experiments"),
            "timestamp": time.time(),
        }
        # Source-side evidence capture (P1): persist the assigned source and the
        # pre-read excerpt alongside the hypothesise phase so the SFT export can
        # place the read evidence on the query side WITHOUT regex-parsing the
        # transcript or reading the dataset from disk. Survey episodes set these
        # in episode_context at populate(); they are absent for crossbreed.
        if phase == "hypothesise":
            assigned = ctx.episode_context.get("assigned_source") or {}
            source_sample = ctx.episode_context.get("source_sample")
            if isinstance(source_sample, str) and source_sample.strip():
                record["source_excerpt"] = source_sample
            section = assigned.get("key") or assigned.get("path") or ""
            if section:
                record["assigned_section"] = section
        phase_records[phase] = record

        return CapabilityResult(
            "record_phase",
            output={"phase": phase, "recorded": True},
            success=True,
        )
    
    def _exec_phase_get(
        self,
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Retrieve artifacts from a previous phase."""
        phase = args.get("phase", "")
        phase_records = ctx.episode_context.get("phase_records", {})
        
        if phase not in phase_records:
            return CapabilityResult(
                "phase_get",
                success=False,
                error=f"Phase '{phase}' not found",
            )
        
        output = phase_records[phase].copy()
        
        # Auto-enhance data_spec for coding phase, prepending the episode's
        # assigned source (file rotation) so the code agent always sees it.
        if phase == "hypothesise" and "data_spec" in output:
            output["data_spec"] = self._enhance_data_spec(
                output["data_spec"], ctx.episode_context
            )
        
        return CapabilityResult(
            "phase_get",
            output=output,
            success=True,
        )
    
    def _enhance_data_spec(
        self, data_spec: dict[str, Any], episode_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Enhance data_spec with Kazakhstan-specific file guidance and correct paths.

        Property lists and counts here are surfaced directly to the coding
        agent — they must match the on-disk schemas. Prior versions advertised
        a lossy subset of ``copper_prospects.geojson`` properties (no
        Mineralogy / Agehost / Unit / HostRocks), which hid the redox-relevant
        columns from prompt context and biased hypotheses toward
        coordinate-only structural framings.
        """
        enhanced = data_spec.copy()
        files = enhanced.get("files", [])

        kazakhstan_geojson_files = [
            {
                "file": "converted_spatial_data/copper_prospects.geojson",
                "full_path": "/workspace/input/converted_spatial_data/copper_prospects.geojson",
                "type": "geojson",
                "geometry": "Point",
                "count": 113,
                "properties": [
                    "Coded_ID", "Tract_name", "Name", "Latitude", "Longitude",
                    "Type", "Subtype", "Age_Ma", "Comm_major",
                    "Tonnage_Mt", "Cu_pct", "Ag_g_t", "Co_pct",
                    "Agehost", "Unit", "HostRocks", "Mineralogy",
                    "SiteStatus", "SiteStat2", "Comments", "Ref_short",
                ],
                "note": (
                    "113 sediment-hosted Cu prospects. Property fields incl. "
                    "Unit (host suite — Vladimirov/Kayraktin/Kirey/etc.), "
                    "Agehost, Mineralogy (e.g. chalcopyrite), and tonnage/grade. "
                    "Use geopandas.read_file(full_path). Many numeric fields use "
                    "-9999 as a sentinel for missing values."
                ),
                "description": "Sediment-hosted copper prospects (vector + tabular surrogate)",
            },
            {
                "file": "converted_spatial_data/copper_prospects_aoi.geojson",
                "full_path": "/workspace/input/converted_spatial_data/copper_prospects_aoi.geojson",
                "type": "geojson",
                "geometry": "Point",
                "count": 112,
                "properties": [
                    "Coded_ID", "Tract_name", "Name", "Latitude", "Longitude",
                    "Type", "Subtype", "Age_Ma", "Comm_major",
                    "Tonnage_Mt", "Cu_pct", "Ag_g_t", "Co_pct",
                    "Agehost", "Unit", "HostRocks", "Mineralogy",
                ],
                "note": (
                    "112 Point features — near-duplicate of copper_prospects, "
                    "filtered to the AOI. Same property schema (NOT an AOI polygon)."
                ),
                "description": "Copper prospects filtered to area of interest",
            },
            {
                "file": "converted_spatial_data/anticlines_synclines.geojson",
                "full_path": "/workspace/input/converted_spatial_data/anticlines_synclines.geojson",
                "type": "geojson",
                "geometry": "LineString",
                "count": 58,
                "properties": ["id", "Name", "Type", "Number"],
                "note": (
                    "58 LineString fold-axis traces. The Type field distinguishes "
                    "anticline vs syncline. Use geopandas.read_file(full_path)."
                ),
                "description": "Regional anticline and syncline axis traces",
            },
            {
                "file": "converted_spatial_data/assessment_tract.geojson",
                "full_path": "/workspace/input/converted_spatial_data/assessment_tract.geojson",
                "type": "geojson",
                "geometry": "MultiPolygon",
                "count": 1,
                "properties": [
                    "descr", "Area_km2", "Tract_name", "Country", "Commodity",
                    "Dep_type", "GT_model", "Geology", "Age",
                    "Asmt_date", "Asmt_depth", "N_known", "N_expected",
                    "N90", "N50", "N10", "DepDensity",
                ],
                "area_km2": 49714,
                "note": (
                    "Single MultiPolygon — Teniz Basin assessment tract (49,714 km²). "
                    "Carries Geology, Age, Dep_type (deposit type), and USGS expected-"
                    "deposit-count fields (N90/N50/N10/N_expected)."
                ),
                "description": "USGS assessment tract boundary + tract-level metadata",
            },
        ]

        usgs_tabular_files = [
            {
                "file": "USGS/TZ_ssCu_Prospects.csv",
                "full_path": "/workspace/input/USGS/TZ_ssCu_Prospects.csv",
                "type": "csv",
                "count": 113,
                "columns": [
                    "X", "Y", "Coded_ID", "Tract_name", "Name", "SiteStatus",
                    "Latitude", "Longitude", "Type", "Subtype", "Age_Ma",
                    "Comm_major", "Tonnage_Mt", "Cu_pct", "Ag_g_t",
                    "Co_pct", "Agehost", "Unit", "HostRocks", "Mineralogy",
                    "Comments", "Ref_short",
                ],
                "note": (
                    "Tabular twin of copper_prospects.geojson — easier to scan "
                    "value distributions over Mineralogy / Unit (host suite) / "
                    "Agehost / Age_Ma with polars or pandas. -9999 = missing."
                ),
                "description": "Tabular sediment-hosted Cu prospects",
            },
            {
                "file": "USGS/TZ_ssCu_Tract.csv",
                "full_path": "/workspace/input/USGS/TZ_ssCu_Tract.csv",
                "type": "csv",
                "count": 1,
                "note": "Single-row tract metadata (twin of assessment_tract properties).",
                "description": "Tabular tract metadata",
            },
        ]

        usgs_text_files = [
            {
                "file": "USGS/chunks/",
                "full_path": "/workspace/input/USGS/chunks/",
                "type": "text_corpus",
                "count": 7,
                "language": "English",
                "note": (
                    "7 USGS Sandstone Copper assessment chunks. Highest-density "
                    "source of NON-STRUCTURAL mechanisms: chunk sir2010-5090_001 "
                    "contains the textbook redox-zoning model (pyrite → chalcopyrite "
                    "→ bornite → chalcocite → hematite; oxidized red beds overlying "
                    "chemically reduced gray/green/black strata) and the deposit "
                    "subtype taxonomy (reduced facies / sandstone Cu / red bed). "
                    "Use os.listdir() then open(); each file < 10 KB."
                ),
                "description": "USGS technical narrative — redox model, subtypes, controls",
            },
            {
                "file": "USGS/descriptions/",
                "full_path": "/workspace/input/USGS/descriptions/",
                "type": "text_corpus",
                "count": 13,
                "language": "English",
                "note": "13 figure-description files from the USGS report.",
                "description": "USGS figure descriptions",
            },
        ]

        russian_survey_files = [
            {
                "file": "36572_Smolianova_1984/chunks/",
                "full_path": "/workspace/input/36572_Smolianova_1984/chunks/",
                "type": "text_corpus",
                "count": 328,
                "language": "English (translated)",
                "note": (
                    "328 chunked sections of the Soviet survey (Smolianova 1984): "
                    "stratigraphy (Proterozoic→Mesozoic), tectonics, magmatism, "
                    "physical properties (density, magnetic susceptibility), "
                    "geochemistry, mineral evaluation. Chunk filenames are "
                    "self-describing (e.g. STRATIGRAPHY_*, METHODS_*, "
                    "PHYSICAL_PROPERTIES_*). Sample 2-3 representative chunks "
                    "per topic; full read of all chunks is unnecessary."
                ),
                "description": "Soviet geological survey — lithology, stratigraphy, geophysics",
            },
            {
                "file": "36572_Smolianova_1984/drill_holes_data/",
                "full_path": "/workspace/input/36572_Smolianova_1984/drill_holes_data/",
                "type": "text_corpus",
                "count": 63,
                "language": "English (translated from Russian wireline log sheets)",
                "note": (
                    "63 *.description.md files — one per Soviet borehole (скв_NNN). "
                    "Each describes a wireline log sheet: SP (spontaneous polarisation), "
                    "apparent resistivity (КС), gamma-ray, neutron curves; a lithology "
                    "column (sandstone/clay/marl interbeds); and per-depth spectral-"
                    "analysis (Pb/Cu/Zn/Mo/Sn) assays. Direct metre-scale proxies for "
                    "redox boundaries and lithology contacts. Borehole IDs do not "
                    "carry coordinates here — would need joining."
                ),
                "description": "Per-borehole wireline-log descriptions with assays",
            },
        ]

        # Combine known specs, then append any agent-supplied extras the
        # enhancer doesn't already cover.
        file_specs = list(
            kazakhstan_geojson_files
            + usgs_tabular_files
            + usgs_text_files
            + russian_survey_files
        )
        known_file_keys = {spec["file"] for spec in file_specs}
        for file_path in files:
            if not any(known in file_path for known in known_file_keys):
                file_specs.append({"file": file_path, "type": "unknown", "note": "Additional file - check format"})

        # Prepend the episode's assigned source (file rotation) so the code
        # agent always sees it first, even if the explore agent's record_phase
        # listed different files.
        assigned = (episode_context or {}).get("assigned_source", {})
        if assigned:
            apath = assigned.get("path", "")
            aglob = assigned.get("glob_pattern")
            if aglob:
                patterns = [aglob] if isinstance(aglob, str) else aglob
                for p in patterns:
                    resolved = f"/workspace/input/{apath}/{p}"
                    if not any(resolved in str(s.get("file", "")) for s in file_specs):
                        file_specs.insert(0, {
                            "file": resolved,
                            "type": "md_chunks",
                            "note": f"Assigned section — {assigned.get('key', '')}: use glob to enumerate",
                        })
            elif apath:
                resolved = f"/workspace/input/{apath}"
                if not any(resolved in str(s.get("file", "")) for s in file_specs):
                    file_specs.insert(0, {
                        "file": resolved,
                        "full_path": resolved,
                        "note": f"Assigned source — {assigned.get('key', '')}",
                    })

        enhanced["file_specs"] = file_specs
        enhanced["kazakhstan_data_structure"] = {
            "geojson_files": 4,
            "csv_files": 2,
            "usgs_text_chunks": 7,
            "usgs_figure_descriptions": 13,
            "smolianova_text_chunks": 328,
            "drill_hole_descriptions": 63,
            "copper_prospects": 113,
            "fold_axis_traces": 58,
            "assessment_area_km2": 49714,
            "data_languages": ["English", "Russian (translated)"],
        }
        return enhanced
    
    @staticmethod
    def _load_feature_points(
        artifact_files,
        artifact_directory,
        *,
        max_rows: int = 2000,
    ) -> tuple[list[dict], bool]:
        """A1 handoff: read the Code phase's feature-points artifact (columns
        longitude,latitude,depth_m,value,coordinate_source) and return its rows so the Translate
        agent -- which has no file-read capability -- can map ONE spatial op per row instead of
        stamping a single blob. Returns (rows, truncated). Robust to full-path or basename entries
        in artifact_files; falls back to known names in artifact_directory."""
        import csv as _csv
        import os as _os
        import re as _re

        # The execution sandbox captures a top-level `feature_points` DataFrame as
        # 'feature_points_dataframe.csv' (varname + type suffix), so match both that and a
        # plainly-written 'feature_points.csv'.
        def _is_fp(name) -> bool:
            return bool(_re.fullmatch(r"feature_points(_dataframe)?\.csv", _os.path.basename(str(name))))

        path = None
        for f in artifact_files or []:
            if _is_fp(f):
                if _os.path.exists(str(f)):
                    path = str(f)
                    break
                if artifact_directory:
                    cand = _os.path.join(artifact_directory, _os.path.basename(str(f)))
                    if _os.path.exists(cand):
                        path = cand
                        break
        if path is None and artifact_directory:
            for _nm in ("feature_points.csv", "feature_points_dataframe.csv"):
                cand = _os.path.join(artifact_directory, _nm)
                if _os.path.exists(cand):
                    path = cand
                    break
        if path is None:
            return [], False

        rows: list[dict] = []
        truncated = False
        try:
            with open(path, newline="") as fh:
                for i, row in enumerate(_csv.DictReader(fh)):
                    if i >= max_rows:
                        truncated = True
                        break
                    rows.append(FeatureHypothesisKazakhstanTask._coerce_feature_point_row(row))
        except Exception:
            return [], False
        return rows, truncated

    @staticmethod
    def _coerce_feature_point_row(row: dict) -> dict:
        coerced = {}
        for key, value in dict(row).items():
            if key is None:
                continue
            clean_key = str(key).strip()
            clean_value = value.strip() if isinstance(value, str) else value
            if clean_key in _FEATURE_POINT_NUMERIC_COLUMNS:
                try:
                    coerced[clean_key] = float(clean_value)
                except (TypeError, ValueError):
                    coerced[clean_key] = clean_value
            else:
                coerced[clean_key] = clean_value
        return coerced

    @staticmethod
    def _prepare_spatial_provenance_args(
        args: dict[str, Any],
        episode_context: dict[str, Any],
    ) -> dict[str, Any]:
        resolved = dict(args)
        metadata = resolved.get("metadata") if isinstance(resolved.get("metadata"), dict) else {}
        source_file = str(metadata.get("source_file") or "").strip()
        source_excerpt = str(metadata.get("source_excerpt") or "").strip()
        explicit_source = str(resolved.get("coordinate_source") or "").strip()
        last_search = episode_context.get("last_coordinate_search")

        if explicit_source in _VALID_COORDINATE_SOURCES:
            coordinate_source = explicit_source
        elif source_file or source_excerpt:
            coordinate_source = "artifact"
        elif isinstance(last_search, dict) and last_search.get("coordinate_source") in {"geonames", "web"}:
            coordinate_source = str(last_search["coordinate_source"])
        else:
            coordinate_source = "creative_fallback"

        resolved["coordinate_source"] = coordinate_source
        resolved["source_file"] = source_file or None
        resolved["source_excerpt"] = source_excerpt or None
        return resolved

    def _exec_get_experiment_summary(
        self,
        containers: list[Container],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Return all phase data in one call for translate and rewrite agents."""
        phase_records = ctx.episode_context.get("phase_records", {})
        hypothesise = phase_records.get("hypothesise", {})
        code = phase_records.get("code", {})
        translate = phase_records.get("translate", {})
        evaluate = phase_records.get("evaluate", {})
        code_executed, code_truncated = _compact_agent_text(
            code.get("code_executed", ""), _SUMMARY_CODE_MAX_CHARS
        )
        result_summary, result_truncated = _compact_agent_text(
            code.get("result_summary", ""), _SUMMARY_RESULT_MAX_CHARS
        )
        artifact_files = code.get("artifact_files", [])
        if not isinstance(artifact_files, list):
            artifact_files = []
        feature_points, feature_points_truncated = self._load_feature_points(
            artifact_files, code.get("artifact_directory", "")
        )

        return CapabilityResult(
            "get_experiment_summary",
            output={
                "hypothesis": hypothesise.get("hypothesis", ""),
                "data_spec": hypothesise.get("data_spec", {}),
                "code_executed": code_executed,
                "code_executed_truncated": code_truncated,
                "result_summary": result_summary,
                "result_summary_truncated": result_truncated,
                "artifact_directory": code.get("artifact_directory", ""),
                "artifact_files": artifact_files,
                "artifact_count": len(artifact_files),
                "feature_points": feature_points,
                "feature_points_count": len(feature_points),
                "feature_points_truncated": feature_points_truncated,
                "feature_layer_name": translate.get("feature_layer_name", ""),
                "dtype": translate.get("dtype", "float"),
                "bic_delta": evaluate.get("bic_delta"),
                "admitted": evaluate.get("admitted", False),
                "mutual_info": evaluate.get("mutual_info", {}),
                "admission_path": evaluate.get("admission_path"),
                "proposal_evidence_tier": self._normalise_evidence_tier(hypothesise.get("evidence_tier")),
                "self_assessment": hypothesise.get("self_assessment", {}),
                "confidence": (hypothesise.get("self_assessment") or {}).get("confidence")
                if isinstance(hypothesise.get("self_assessment"), dict)
                else None,
                "evidence_strength": evaluate.get("evidence_strength"),
                "admission_tier": evaluate.get("admission_tier"),
                "crossbreed_parent_eligible": evaluate.get("crossbreed_parent_eligible"),
                "relative_mae_mean": evaluate.get("relative_mae_mean"),
            },
            success=True,
        )

    def _exec_submit_code(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Submit and execute code (coding agent)."""
        code = args.get("code", "")
        if not code:
            return CapabilityResult("submit_code", success=False, error="No code provided")
        
        # Execute in analysis container
        analysis = self._pick_container(containers, "analysis")
        if analysis is None:
            return CapabilityResult("submit_code", success=False, error="No analysis container")
        
        # Save inside the container, then copy back into the host artifact tree.
        episode_id = _safe_artifact_component(
            ctx.episode_context.get("episode_id") or ctx.episode_id
        )
        container_artifact_dir = f"/workspace/out/artifacts/{episode_id}"
        artifact_dir = str(self._execution_artifact_root(ctx) / "submit_code")
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        
        # Wrap user code with automatic artifact capture
        indented_code = '\n'.join("    " + line for line in code.split('\n'))
        
        # Use string concatenation to avoid f-string variable scope issues
        wrapped_code = '''
import os
import glob
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Create artifact directory and common output directories
artifact_dir = "''' + container_artifact_dir + '''"
os.makedirs(artifact_dir, exist_ok=True)
os.makedirs("/workspace/output", exist_ok=True)  # Fix common user code issue

# Store original locals to compare later
_original_locals = set(locals().keys())

try:
    # Execute user's analysis code
''' + indented_code + '''

except Exception as user_code_error:
    print(f"ERROR in user code: {user_code_error}")
    import traceback
    traceback.print_exc()

finally:
    # Always attempt artifact capture, even if user code failed
    print("\\n" + "="*50)
    print("ANALYSIS COMPLETE - CAPTURING ARTIFACTS")
    print("="*50)
    
    # Capture artifacts from final namespace
    _final_locals = locals().copy()
    _artifacts_saved = []
    
    for var_name, obj in _final_locals.items():
        if (not var_name.startswith('_') and 
            var_name not in _original_locals and 
            var_name not in ['artifact_dir', 'os', 'glob', 'pickle', 'pd', 'np', 'Path']):
            
            try:
                if isinstance(obj, pd.DataFrame) and not obj.empty:
                    filepath = f"{artifact_dir}/{var_name}_dataframe.csv"
                    obj.to_csv(filepath, index=False)
                    _artifacts_saved.append(filepath)
                    print(f"Saved DataFrame '{var_name}' -> {filepath}")
                    print(f"  Shape: {obj.shape}, Columns: {list(obj.columns)}")
                
                elif isinstance(obj, np.ndarray):
                    filepath = f"{artifact_dir}/{var_name}_array.npy" 
                    np.save(filepath, obj)
                    _artifacts_saved.append(filepath)
                    print(f"Saved numpy array '{var_name}' -> {filepath}")
                    print(f"  Shape: {obj.shape}, dtype: {obj.dtype}")
                
                elif isinstance(obj, (dict, list, tuple)) and len(str(obj)) < 10000:
                    filepath = f"{artifact_dir}/{var_name}_object.pkl"
                    with open(filepath, 'wb') as f:
                        pickle.dump(obj, f)
                    _artifacts_saved.append(filepath)
                    print(f"Saved object '{var_name}' -> {filepath}")
                    print(f"  Type: {type(obj)}, Size: {len(str(obj))} chars")
                
                elif isinstance(obj, (int, float, str, bool)):
                    # Save simple scalars as JSON-like format
                    filepath = f"{artifact_dir}/{var_name}_scalar.txt"
                    with open(filepath, 'w') as f:
                        f.write(var_name + ": " + str(obj) + "\\ntype: " + type(obj).__name__)
                    _artifacts_saved.append(filepath)
                    print(f"Saved scalar '{var_name}' -> {filepath}")
                    print(f"  Value: {obj}")
                
            except Exception as save_err:
                print(f"Failed to save '{var_name}': {save_err}")
    
    # List all artifacts in directory
    all_artifacts = glob.glob(f"{artifact_dir}/*")
    print(f"\\nARTIFACTS_DIRECTORY: {artifact_dir}")
    print(f"ARTIFACTS_SAVED: {all_artifacts}")
    print("="*50)
'''
        
        cmd = ["python", "-c", wrapped_code]
        try:
            result = exec_run_with_timeout(analysis, cmd, timeout_s=120)
            exit_code, raw = coerce_exec_result(result)
            stdout = raw.decode(errors="replace")
            
            # Copy container artifacts into the host artifact directory.
            artifact_files = []
            try:
                import io
                import tarfile

                archive_script = (
                    "import sys, tarfile\n"
                    f"path = {json.dumps(container_artifact_dir)}\n"
                    "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:\n"
                    "    tar.add(path, arcname='.')\n"
                )
                archive_result = exec_run_with_timeout(
                    analysis, ["python3", "-c", archive_script], timeout_s=30
                )
                archive_code, archive_raw = coerce_exec_result(archive_result)
                if archive_code != 0:
                    raise RuntimeError(archive_raw.decode(errors="replace"))
                tar_data = io.BytesIO(archive_raw)
                tar_data.seek(0)
                with tarfile.open(fileobj=tar_data) as tar:
                    tar.extractall(artifact_dir, filter="data")
                for root, _, files in os.walk(artifact_dir):
                    for file_name in files:
                        artifact_files.append(os.path.join(root, file_name))
            except Exception as copy_err:
                stdout += f"\nWarning: Could not extract artifacts from container: {copy_err}"
            
            # Store code and result with artifact information
            phase_records = ctx.episode_context.setdefault("phase_records", {})
            phase_records["code"] = {
                "code_executed": code,
                "result_summary": stdout,
                "artifact_directory": artifact_dir,
                "artifact_files": artifact_files,
                "success": exit_code == 0,
                "timestamp": time.time(),
            }
            
            return CapabilityResult(
                "submit_code",
                output={
                    "stdout": stdout,
                    "stderr": "",
                    "success": exit_code == 0,
                    "artifact_directory": artifact_dir,
                    "artifact_files": artifact_files,
                },
                success=True,
            )
        except Exception as e:
            return CapabilityResult("submit_code", success=False, error=str(e))
    
    def _exec_submit_rewrite(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Submit rewritten experiment record. Auto-generates graph node and appends BIC."""
        import pickle
        import json
        from pathlib import Path
        from datetime import datetime
        
        # `prompt`/`response` arrive as flat top-level args (see the
        # submit_rewrite capability schema for why it is not nested).
        training_pair = {
            "prompt": args.get("prompt", ""),
            "response": args.get("response", ""),
        }

        phase_records = ctx.episode_context.get("phase_records", {})
        hypothesise = phase_records.get("hypothesise", {})
        code = phase_records.get("code", {})
        translate = phase_records.get("translate", {})
        evaluate = phase_records.get("evaluate", {})

        # Auto-generate graph node from phase records
        graph_node = {
            "hypothesis": hypothesise.get("hypothesis", ""),
            "data_spec": hypothesise.get("data_spec", {}),
            "experiment_summary": code.get("result_summary", ""),
            "feature_layer_name": translate.get("feature_layer_name", ""),
            "outcome": {
                "bic_delta": evaluate.get("bic_delta"),
                "admitted": evaluate.get("admitted", False),
                "mutual_info": evaluate.get("mutual_info", {}),
            },
        }

        # Auto-append BIC result to training pair response
        bic_delta = evaluate.get("bic_delta")
        admitted = evaluate.get("admitted", False)
        if isinstance(training_pair.get("response"), str) and bic_delta is not None:
            verdict = "Admitted" if admitted else "Not admitted"
            training_pair["response"] += (
                f"\n\nResult: {bic_delta:.4f} BIC delta. {verdict}."
            )

        # Prepare training record for persistence
        episode_id = ctx.episode_context.get("episode_id", "")
        store_dir = ctx.episode_context.get("store_dir", "")
        
        # Extract data paths - need to get base data directory
        if store_dir:
            data_base_path = Path(store_dir).parent.parent  # from store/teniz_basin to data/kazakhstan/feature-hypothesis
        else:
            data_base_path = Path("/home/jen/Desktop/geonsl/NSL2-geology-task/data/feature-hypothesis")
        
        # Extract two-stage scoring results
        masking_test_passed = evaluate.get('masking_test_passed', True)
        masking_test_improvement = evaluate.get('masking_test_improvement', 0.0)
        masking_test_direction = evaluate.get('masking_test_direction', 'not_applicable')
        stage_1_tolerance_used = bool(evaluate.get('stage_1_tolerance_used', False))
        stage_1_mae_tolerance = evaluate.get('stage_1_mae_tolerance')
        stage_1_bic_rescue_threshold = evaluate.get('stage_1_bic_rescue_threshold')
        stage_completed = evaluate.get('stage_completed', 'stage_2_completed')
        admission_path = evaluate.get('admission_path', 'normal')
        proposal_evidence_tier = self._normalise_evidence_tier(hypothesise.get("evidence_tier"))
        self_assessment = hypothesise.get("self_assessment")
        if not isinstance(self_assessment, dict):
            self_assessment = {}
        confidence = self_assessment.get("confidence")
        
        training_record = {
            'prompt': training_pair.get('prompt', ''),
            'response': training_pair.get('response', ''),
            'bic_delta': bic_delta,
            'episode_id': episode_id,
            'timestamp': time.time(),
            'admitted': admitted,
            'layer_name': translate.get('feature_layer_name', ''),
            # Two-stage scoring results
            'masking_test_passed': masking_test_passed,
            'masking_test_improvement': masking_test_improvement,
            'masking_test_direction': masking_test_direction,
            'stage_1_tolerance_used': stage_1_tolerance_used,
            'stage_1_mae_tolerance': stage_1_mae_tolerance,
            'stage_1_bic_rescue_threshold': stage_1_bic_rescue_threshold,
            'stage_completed': stage_completed,
            'admission_path': admission_path,
            'proposal_evidence_tier': proposal_evidence_tier,
            'self_assessment': self_assessment,
            'confidence': confidence,
            'evidence_strength': evaluate.get('evidence_strength'),
            'admission_tier': evaluate.get('admission_tier'),
            'crossbreed_parent_eligible': evaluate.get('crossbreed_parent_eligible'),
            'relative_mae_mean': evaluate.get('relative_mae_mean'),
            'relative_mae_min': evaluate.get('relative_mae_min'),
            'relative_mae_max': evaluate.get('relative_mae_max'),
            'metadata': {
                'hypothesis': hypothesise.get('hypothesis', ''),
                'grid_bounds': ctx.episode_context.get('grid_spec', {}),
                'mutual_info': evaluate.get('mutual_info', {}),
                'experiment_summary': code.get('result_summary', ''),
                # Additional two-stage context
                'two_stage_scoring': {
                    'stage_1_passed': masking_test_passed,
                    'stage_1_improvement': masking_test_improvement,
                    'stage_1_direction': masking_test_direction,
                    'stage_1_tolerance_used': stage_1_tolerance_used,
                    'stage_1_mae_tolerance': stage_1_mae_tolerance,
                    'stage_1_bic_rescue_threshold': stage_1_bic_rescue_threshold,
                    'stage_completed': stage_completed,
                    'admission_path': admission_path,
                    'scoring_version': 'two_stage_v2'
                }
            }
        }
        
        # Save training data (ALL experiments)
        try:
            training_dir = data_base_path / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            training_file = training_dir / "training_pairs.pkl"

            # Load existing records or create new list
            existing_records = []
            if training_file.exists():
                with open(training_file, 'rb') as f:
                    existing_records = pickle.load(f)
            
            # Append new record
            existing_records.append(training_record)
            
            # Save back to file
            with open(training_file, 'wb') as f:
                pickle.dump(existing_records, f)
                
        except Exception as e:
            print(f"Warning: Failed to save training data: {e}")
        
        # Save to knowledge graph (ONLY experiments that passed BOTH stages)
        # Stage 1: predictive capacity test; Stage 2: BIC improvement.
        # See ``_should_persist_to_kg`` for the gate semantics, including
        # the stage_completed string allowlist.
        both_stages_passed = self._should_persist_to_kg(
            masking_test_passed=masking_test_passed,
            admitted=admitted,
            bic_delta=bic_delta,
            stage_completed=stage_completed,
            admission_path=admission_path,
        )
        
        # Prefer the kg_dir wired through populate() so dedup ledger and
        # experiments.jsonl always live next to each other. Fall back to the
        # legacy `data_base_path / knowledge / teniz_basin` derivation so
        # existing deployments keep working.
        kg_dir_ctx = ctx.episode_context.get("kg_dir", "")
        if kg_dir_ctx:
            knowledge_dir = Path(kg_dir_ctx)
        else:
            knowledge_dir = data_base_path / "knowledge" / "teniz_basin"

        # Pull queue-served parent IDs from the hypothesise phase record so
        # the kg node closes the TODO at the previous line numbers.
        parent_experiments = hypothesise.get("parent_experiments") or []
        parent_node_1 = parent_experiments[0] if len(parent_experiments) > 0 else None
        parent_node_2 = parent_experiments[1] if len(parent_experiments) > 1 else None

        duplicate_rejected = False
        rejected_quarantine = None
        if not both_stages_passed:
            rejected_quarantine = ctx.episode_context.get("rejected_candidate_quarantine")
            if not isinstance(rejected_quarantine, dict):
                rejected_quarantine = self._quarantine_rejected_candidate(
                    store_dir=store_dir,
                    episode_id=episode_id,
                    layer_name=translate.get('feature_layer_name', '') or evaluate.get('layer_name', ''),
                    evaluate=evaluate,
                    phase_records=phase_records,
                )
            if rejected_quarantine is not None:
                ctx.episode_context["rejected_candidate_quarantine"] = rejected_quarantine

        if both_stages_passed:
            try:
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                node_id = f"exp_{episode_id}" if episode_id else f"exp_{int(time.time())}"
                feature_layer_name = translate.get('feature_layer_name', '') or ''
                kg_record = {
                    "node_id": node_id,
                    "prompt": training_pair.get('prompt', ''),
                    "response": training_pair.get('response', ''),
                    "bic_delta": bic_delta,
                    "masking_test_passed": masking_test_passed,
                    "masking_test_improvement": masking_test_improvement,
                    "masking_test_direction": masking_test_direction,
                    "stage_1_tolerance_used": stage_1_tolerance_used,
                    "stage_1_mae_tolerance": stage_1_mae_tolerance,
                    "stage_1_bic_rescue_threshold": stage_1_bic_rescue_threshold,
                    "stage_completed": stage_completed,
                    "admission_path": admission_path,
                    "scoring_version": "two_stage_v2",
                    "proposal_evidence_tier": proposal_evidence_tier,
                    "self_assessment": self_assessment,
                    "confidence": confidence,
                    "relative_mae_mean": evaluate.get('relative_mae_mean'),
                    "relative_mae_min": evaluate.get('relative_mae_min'),
                    "relative_mae_max": evaluate.get('relative_mae_max'),
                    "bic_delta_raw": evaluate.get('bic_delta_raw'),
                    "n_effective_samples": evaluate.get('n_effective_samples'),
                    "candidate_nonzero_voxels": evaluate.get('candidate_nonzero_voxels'),
                    "candidate_fill_fraction": evaluate.get('candidate_fill_fraction'),
                    "artifact_links": {
                        "layer_file": f"store/teniz_basin/admitted/layers/{feature_layer_name}.npy" if feature_layer_name else None,
                        "spatial_ops": f"store/teniz_basin/scratch/{episode_id}/spatial.db:experiment_{episode_id}" if episode_id else None
                    },
                    "parent_node_1": parent_node_1,
                    "parent_node_2": parent_node_2,
                    "timestamp": datetime.now().isoformat(),
                    "mutual_info": evaluate.get('mutual_info', {}),
                    "layer_name": feature_layer_name,
                    "hypothesis": hypothesise.get('hypothesis', '')
                }

                # Atomic dedup + scratch→admitted promotion (both inside the
                # kg lock). Duplicates keep the episode's reward but never
                # enter the pool, and leave the scratch file in place for
                # the cleanup hook to reclaim.
                fingerprint_parents = [p for p in parent_experiments if isinstance(p, str) and p]
                scratch_dir = (
                    Path(store_dir) / "scratch" / episode_id
                    if store_dir and episode_id
                    else None
                )
                admitted_dir = (
                    Path(store_dir) / "admitted" if store_dir else None
                )
                admitted_to_kg = self._admit_with_dedup(
                    knowledge_dir,
                    kg_record,
                    parents=fingerprint_parents,
                    hypothesis=hypothesise.get('hypothesis', ''),
                    scratch_dir=scratch_dir,
                    admitted_dir=admitted_dir,
                    layer_name=feature_layer_name or None,
                )
                for key in (
                    "evidence_strength",
                    "admission_tier",
                    "crossbreed_parent_eligible",
                    "corroboration_count",
                    "artifact_backed_fraction",
                    "declared_nothing",
                    "emptiness_rejection_reason",
                    "declared_footprint_size",
                    "candidate_value_entropy",
                    "candidate_unique_nonzero_values",
                    "spatial_operation_provenance_count",
                    "coordinate_source_counts",
                    "novelty_guard_passed",
                    "novelty_rejection_reason",
                    "provenance_guard_passed",
                    "provenance_rejection_reason",
                ):
                    if key in kg_record:
                        evaluate[key] = kg_record[key]
                phase_records["evaluate"] = evaluate
                duplicate_rejected = not admitted_to_kg
                if (
                    not admitted_to_kg
                    and kg_record.get("admission_tier") == "guard_rejected"
                ):
                    guard_evaluate = dict(evaluate)
                    guard_evaluate.update({
                        key: value
                        for key, value in kg_record.items()
                        if key not in {"prompt", "response"}
                    })
                    guard_evaluate["admitted"] = False
                    guard_evaluate["stage_completed"] = "guard_rejected"
                    rejected_quarantine = self._quarantine_rejected_candidate(
                        store_dir=store_dir,
                        episode_id=episode_id,
                        layer_name=feature_layer_name,
                        evaluate=guard_evaluate,
                        phase_records=phase_records,
                    )
                if admitted_to_kg:
                    self._update_crossbreed_index(
                        knowledge_dir,
                        node_id,
                        feature_layer_name,
                        evaluate.get('mutual_info', {}),
                    )
                    self._update_pairwise_distance_index(
                        knowledge_dir,
                        node_id,
                        feature_layer_name,
                        evaluate.get('pairwise_distance', {}),
                    )

            except Exception as e:
                print(f"Warning: Failed to save knowledge graph data: {e}")

        ctx.episode_context["terminal_record"] = {
            "graph_node": graph_node,
            "training_pair": training_pair,
            "timestamp": time.time(),
        }
        if duplicate_rejected:
            # finalize_episode reads this to stamp `duplicate_rejected` into
            # the reward breakdown for telemetry.
            ctx.episode_context["duplicate_rejected"] = True

        # Emit a synthetic inference row carrying the rewriter's polished
        # (prompt, response). See FeatureHypothesisTask._record_rewrite_output_row
        # for the rationale.
        self._record_rewrite_output_row(ctx, training_pair)

        return CapabilityResult(
            "submit_rewrite",
            output={
                "recorded": True,
                "training_saved": True,
                "knowledge_saved": both_stages_passed and not duplicate_rejected,
                "duplicate_rejected": duplicate_rejected,
                "rejected_candidate_quarantined": rejected_quarantine is not None,
                "rejected_candidate_quarantine": rejected_quarantine,
                "two_stage_results": {
                    "stage_1_passed": masking_test_passed,
                    "stage_1_improvement": masking_test_improvement,
                    "stage_2_passed": admitted,
                    "bic_delta": bic_delta,
                    "final_admitted": both_stages_passed and not duplicate_rejected,
                }
            },
            success=True,
        )

    @staticmethod
    def _record_rewrite_output_row(
        ctx: CapabilityExecutionContext,
        training_pair: dict[str, str],
    ) -> None:
        """Synthesize one TrajectoryRecord for the rewriter's polished output.

        The rewrite phase's real inference rows are tool-call-only — their
        ``raw_response`` is empty — so the SFT export would otherwise have no
        visibility of the ``(prompt, response)`` the agent crafted. No-op
        when no recorder is wired (unit-test path).
        """
        recorder = ctx.recorder
        if recorder is None:
            return
        prompt = training_pair.get("prompt", "")
        response = training_pair.get("response", "")
        if not isinstance(prompt, str) or not isinstance(response, str):
            return
        if not prompt and not response:
            return

        from datetime import datetime as _dt
        from src.harness.recorder import TrajectoryRecord

        phase_records = ctx.episode_context.get("phase_records", {})
        terminal_record = ctx.episode_context.get("terminal_record", {})
        experiment_record = _to_jsonable(
            {
                "schema_version": 1,
                "source": "submit_rewrite",
                "phase_records": phase_records if isinstance(phase_records, dict) else {},
                "terminal_record": terminal_record if isinstance(terminal_record, dict) else {},
                "crossbreed_context": ctx.episode_context.get("crossbreed_context", {}),
                "workflow_kind": ctx.episode_context.get("workflow_kind"),
                "duplicate_rejected": bool(ctx.episode_context.get("duplicate_rejected", False)),
            }
        )

        record = TrajectoryRecord(
            episode_id=ctx.episode_id,
            phase="rewrite_output",
            messages=[{"role": "user", "content": prompt}],
            response=response,
            usage=None,
            timestamp=_dt.now().isoformat(),
            success=True,
            error_message=None,
            meta={
                "workflow_step": "rewrite",
                "actor_role": "rewriter_output",
                "synthesized": True,
                "client": "task_synth",
                "model": "task_synth",
                "episode_id": ctx.episode_id,
                "experiment_record": experiment_record,
            },
        )
        try:
            recorder.record_inference(record)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"feature_hypothesis_kazakhstan: failed to record rewrite_output row: {exc}"
            )

    def _update_crossbreed_index(
        self,
        knowledge_dir: Path,
        new_node_id: str,
        new_layer_name: str,
        new_mutual_info: dict[str, float]
    ) -> None:
        """Update legacy crossbreed MI index for record audit.

        The crossbreed queue no longer consumes this file (see
        `_update_pairwise_distance_index`); kept so existing telemetry / audit
        tooling that diff-tracks `crossbreed_index.jsonl` still sees writes.
        """
        import json
        from datetime import datetime

        try:
            experiments_file = knowledge_dir / _KG_EXPERIMENTS
            crossbreed_file = knowledge_dir / _KG_CROSSBREED_INDEX

            # Read existing experiments to calculate MI with each
            existing_experiments = []
            if experiments_file.exists():
                with open(experiments_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                exp = json.loads(line)
                                existing_experiments.append(exp)
                            except json.JSONDecodeError:
                                continue

            # Calculate mutual information between new experiment and all existing ones
            new_mi_records = []
            for existing_exp in existing_experiments:
                if existing_exp['node_id'] == new_node_id:
                    continue  # Skip self

                # Get mutual information score between layer names
                mi_score = 0.0
                new_layer = new_mutual_info
                existing_layer = existing_exp.get('mutual_info', {})

                # Look for cross-references in mutual info dictionaries
                if existing_exp.get('layer_name') in new_layer:
                    mi_score = new_layer[existing_exp['layer_name']]
                elif new_layer_name in existing_layer:
                    mi_score = existing_layer[new_layer_name]

                # Create pair record
                pair_id = f"{min(new_node_id, existing_exp['node_id'])}_{max(new_node_id, existing_exp['node_id'])}"
                mi_record = {
                    "pair_id": pair_id,
                    "node_1": new_node_id,
                    "node_2": existing_exp['node_id'],
                    "mutual_information": mi_score,
                    "calculated_at": datetime.now().isoformat()
                }
                new_mi_records.append(mi_record)

            # Append new MI records to crossbreed index
            if new_mi_records:
                with open(crossbreed_file, 'a') as f:
                    for record in new_mi_records:
                        f.write(json.dumps(record) + '\n')

        except Exception as e:
            print(f"Warning: Failed to update crossbreed index: {e}")

    def _update_pairwise_distance_index(
        self,
        knowledge_dir: Path,
        new_node_id: str,
        new_layer_name: str,
        new_pairwise_distance: dict[str, float],
    ) -> None:
        """Append pairwise-distance records for the new admit's layer.

        Replaces `_update_crossbreed_index` as the source for queue
        ranking. Pair ids are alphabetically sorted so the symmetric
        distance is written once per unordered pair (matches
        `_load_distance_index`'s lookup key). Existing experiments without
        a layer name match in `new_pairwise_distance` are written at
        distance=0.0 so the queue treats them neutrally rather than
        skipping the entry.
        """
        from datetime import datetime

        try:
            experiments_file = knowledge_dir / _KG_EXPERIMENTS
            distance_file = knowledge_dir / _KG_PAIRWISE_DISTANCE

            existing_experiments: list[dict[str, Any]] = []
            if experiments_file.exists():
                with experiments_file.open("r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            existing_experiments.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            new_records: list[dict[str, Any]] = []
            for existing_exp in existing_experiments:
                existing_id = existing_exp.get("node_id")
                if not isinstance(existing_id, str) or existing_id == new_node_id:
                    continue
                existing_layer = existing_exp.get("layer_name") or ""
                # The evaluate result keyed pairwise_distance by the *other*
                # layer name in the store at the time it was scored.
                dist = float(new_pairwise_distance.get(existing_layer, 0.0))
                pair_id = (
                    f"{min(new_node_id, existing_id)}_"
                    f"{max(new_node_id, existing_id)}"
                )
                new_records.append({
                    "pair_id": pair_id,
                    "node_1": new_node_id,
                    "node_2": existing_id,
                    "layer_1": new_layer_name,
                    "layer_2": existing_layer,
                    "pairwise_distance": dist,
                    "calculated_at": datetime.now().isoformat(),
                })

            if new_records:
                distance_file.parent.mkdir(parents=True, exist_ok=True)
                with distance_file.open("a") as fh:
                    for record in new_records:
                        fh.write(json.dumps(record) + "\n")
        except Exception as e:  # noqa: BLE001 — kg writes are best-effort
            print(f"Warning: Failed to update pairwise distance index: {e}")
    
    def _exec_execution_finalize(
        self,
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Finalize execution results and store in phase records."""
        execution_id = args.get("execution_id", "")
        success = args.get("success", False)
        summary = args.get("summary", "")
        
        if not execution_id:
            return CapabilityResult("execution_finalize", success=False, error="execution_id required")
        
        try:
            # Import and call execution_results directly
            import sys
            from pathlib import Path
            vfm_path = str(Path(__file__).parent.parent.parent / "voxel-features-mcp")
            if vfm_path not in sys.path:
                sys.path.append(vfm_path)
                
            from voxel_features.mcp.tools.execution_tools import execution_results
            
            result = execution_results(execution_id=execution_id)
            
            if not result.get("success", False):
                return CapabilityResult(
                    "execution_finalize",
                    success=False,
                    error=f"Failed to get execution results: {result.get('error', 'Unknown error')}"
                )
            
            result_data = result
            
            # Validate that artifacts were created if execution succeeded
            if success and result_data.get("artifacts_count", 0) == 0:
                return CapabilityResult(
                    "execution_finalize",
                    success=False,
                    error="Execution reported success but no artifacts were created"
                )
            
            # Store in phase records in the same format as old submit_code
            phase_records = ctx.episode_context.setdefault("phase_records", {})
            
            # Get any existing code data to preserve original code
            existing_code = phase_records.get("code", {})
            
            phase_records["code"] = {
                "code_executed": existing_code.get("code_executed", ""),  # Keep original if available
                "result_summary": result_data.get("stdout", ""),
                "artifact_directory": result_data.get("artifact_directory", ""),
                "artifact_files": result_data.get("artifact_files", []),
                "success": success and result_data.get("execution_success", False),
                "timestamp": time.time(),
                "execution_id": execution_id,
                "summary": summary,
            }
            
            return CapabilityResult(
                "execution_finalize",
                output={
                    "execution_id": execution_id,
                    "success": success,
                    "summary": summary,
                    "artifacts_count": result_data.get("artifacts_count", 0),
                    "artifact_files": result_data.get("artifact_files", []),
                },
                success=True,
            )
            
        except Exception as e:
            return CapabilityResult("execution_finalize", success=False, error=str(e))

    def _exec_search_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute a geological location search capability (web / geonames).

        Imports the search tools from the same voxel-features-mcp package the
        other _exec_* capabilities use, so the runtime import path stays
        consistent. Network errors surface as a non-fatal failed result.
        """
        try:
            import sys
            from pathlib import Path
            vfm_path = str(Path(__file__).parent.parent.parent / "voxel-features-mcp")
            if vfm_path not in sys.path:
                sys.path.append(vfm_path)

            from voxel_features.mcp.tools.search_tools import (
                web_search_geological,
                geonames_lookup,
            )

            if capability_name == "search_web_geological":
                result = web_search_geological(**args)
            elif capability_name == "search_geonames_lookup":
                result = geonames_lookup(**args)
            else:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error=f"Unknown search capability: {capability_name}",
                )

            if result.get("success"):
                search_record = {
                    "capability": capability_name,
                    "coordinate_source": "geonames"
                    if capability_name == "search_geonames_lookup"
                    else "web",
                    "timestamp": time.time(),
                }
                ctx.episode_context["last_coordinate_search"] = search_record
                phase_records = ctx.episode_context.setdefault("phase_records", {})
                translate_record = phase_records.setdefault("translate", {})
                translate_record["last_coordinate_search"] = search_record

            return CapabilityResult(
                capability_name,
                output=result,
                success=result.get("success", False),
                error=result.get("error"),
            )
        except Exception as e:
            return CapabilityResult(
                capability_name,
                success=False,
                error=f"Search capability execution failed: {str(e)}",
            )

    def _exec_execution_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute execution tool capability via direct import."""
        
        try:
            # Add voxel-features-mcp to path
            import sys
            from pathlib import Path
            vfm_path = str(Path(__file__).parent.parent.parent / "voxel-features-mcp")
            if vfm_path not in sys.path:
                sys.path.append(vfm_path)
            
            # Import execution tools directly
            from voxel_features.mcp.tools.execution_tools import (
                execution_submit, execution_status, execution_results, 
                execution_cancel, execution_reset_session
            )
            
            # Map capability name to function
            tool_functions = {
                "execution_submit": execution_submit,
                "execution_status": execution_status,
                "execution_results": execution_results,
                "execution_cancel": execution_cancel,
                "execution_reset_session": execution_reset_session,
            }
            
            if capability_name not in tool_functions:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error=f"Unknown execution capability: {capability_name}"
                )

            args = dict(args)
            if capability_name == "execution_submit":
                analysis = self._pick_container(containers, "analysis")
                if analysis is not None:
                    args["container"] = analysis
                else:
                    print(
                        "Warning: No analysis container available for "
                        "execution_submit, using fallback mode"
                    )
                args.setdefault(
                    "artifact_root", str(self._execution_artifact_root(ctx))
                )
            
            # Call the tool function directly
            tool_func = tool_functions[capability_name]
            result = tool_func(**args)
            
            # Special handling for execution_submit to store the original code
            if capability_name == "execution_submit" and result.get("success", False):
                code = args.get("code", "")
                if code:
                    # Store the code in phase records for later use
                    phase_records = ctx.episode_context.setdefault("phase_records", {})
                    existing_code = phase_records.get("code", {})
                    phase_records["code"] = {
                        **existing_code,
                        "code_executed": code,
                        "execution_submitted": True,
                        "submission_timestamp": time.time(),
                    }
            
            return CapabilityResult(
                capability_name,
                output=_to_jsonable(result),
                success=result.get("success", True)
            )
            
        except Exception as e:
            return CapabilityResult(
                capability_name,
                success=False,
                error=f"Execution capability failed: {str(e)}",
            )

    def _execution_artifact_root(self, ctx: CapabilityExecutionContext) -> Path:
        raw_base = (
            ctx.episode_context.get("artifact_dir")
            or ctx.episode_context.get("artifact_root")
        )
        if raw_base:
            base = Path(str(raw_base)).expanduser()
        else:
            train_data = ctx.episode_context.get("train_data_save_folder")
            base = (
                Path(str(train_data)).expanduser() / "artifacts"
                if train_data
                else self._artifact_dir
            )
        run_id = _safe_artifact_component(ctx.episode_context.get("run_id", "manual"))
        episode_id = _safe_artifact_component(
            ctx.episode_context.get("framework_episode_id") or ctx.episode_id
        )
        artifact_root = base.resolve() / run_id / episode_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        return artifact_root
    
    def _exec_spatial_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute spatial tool capability via voxel-features-mcp."""
        
        print(f"🔧 DEBUG: Starting spatial capability: {capability_name}")
        print(f"🔧 DEBUG: Args: {args}")
        
        # Set up environment for spatial operations
        import sys
        from pathlib import Path
        
        # Add voxel-features-mcp to path
        vfm_path = str(Path(__file__).parent.parent.parent / "voxel-features-mcp")
        sys.path.append(vfm_path)
        print(f"🔧 DEBUG: Added path: {vfm_path}")
        
        try:
            # Import required spatial tools
            print("🔧 DEBUG: Importing spatial modules...")
            from voxel_features.spatial import SpatialVoxelStore
            from voxel_features.store import GridSpec
            from voxel_features.mcp.tools.spatial_tools import (
                spatial_add_point, spatial_add_line, spatial_query_region,
                spatial_coord_to_voxel, spatial_get_operations_log
            )
            print("🔧 DEBUG: ✅ Imports successful")

            # Get store directory from episode context
            store_dir = ctx.episode_context.get("store_dir")
            episode_id = ctx.episode_context.get("episode_id", "")
            print(f"🔧 DEBUG: Episode context keys: {list(ctx.episode_context.keys())}")
            print(f"🔧 DEBUG: Store dir: {store_dir}")
            if not store_dir or not episode_id:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error="No store directory or episode_id available in episode context",
                )

            # Resolve the variation's grid from episode context. Falls back to
            # the Kazakhstan default if (somehow) absent — but never to Coe
            # Fairbairn, which was the bug in upstream feature_hypothesis.py.
            grid_dict = ctx.episode_context.get("grid_spec") or _KAZAKHSTAN_TENIZ_GRID
            grid = GridSpec.from_dict(grid_dict)

            # Per-episode scratch with the admitted pool as read-only
            # overlay — isolates this slot's in-flight mutations from
            # every other slot's admitted writes.
            scratch_dir = Path(store_dir) / "scratch" / episode_id
            admitted_dir = Path(store_dir) / "admitted"
            admitted_dir.mkdir(parents=True, exist_ok=True)

            print("🔧 DEBUG: Creating SpatialVoxelStore...")
            store = SpatialVoxelStore(
                scratch_dir, grid, read_only_overlay=admitted_dir
            )
            print(f"🔧 DEBUG: ✅ Store created, grid shape: {store.grid.shape}")
            print(f"🔧 DEBUG: Grid bounds: lon {store.grid.origin[0]:.3f}-{store.grid.maximum[0]:.3f}, lat {store.grid.origin[1]:.3f}-{store.grid.maximum[1]:.3f}, depth {store.grid.origin[2]:.1f}-{store.grid.maximum[2]:.1f}")

            args = dict(args)
            if capability_name in ["spatial_add_point", "spatial_add_line"]:
                args = self._prepare_spatial_provenance_args(args, ctx.episode_context)
             
            # Validate coordinates if this is a spatial operation with coordinates
            if capability_name in ["spatial_add_point", "spatial_add_line"]:
                if "longitude" in args and "latitude" in args:
                    lon, lat = args["longitude"], args["latitude"]
                    in_bounds = (store.grid.origin[0] <= lon <= store.grid.maximum[0] and 
                                store.grid.origin[1] <= lat <= store.grid.maximum[1])
                    print(f"🔧 DEBUG: Coordinate validation - lon={lon:.6f}, lat={lat:.6f}, in_bounds={in_bounds}")
                    
                    if not in_bounds:
                        return CapabilityResult(
                            capability_name,
                            success=False,
                            error=f"Coordinates ({lon:.6f}, {lat:.6f}) outside grid bounds",
                        )
            
            # Route to appropriate spatial tool function
            print(f"🔧 DEBUG: Routing to tool: {capability_name}")
            if capability_name == "spatial_add_point":
                print("🔧 DEBUG: Calling spatial_add_point...")
                result = spatial_add_point(store, **args)
            elif capability_name == "spatial_add_line":
                print("🔧 DEBUG: Calling spatial_add_line...")
                result = spatial_add_line(store, **args)
            elif capability_name == "spatial_query_region":
                print("🔧 DEBUG: Calling spatial_query_region...")
                result = spatial_query_region(store, **args)
            elif capability_name == "spatial_coord_to_voxel":
                print("🔧 DEBUG: Calling spatial_coord_to_voxel...")
                result = spatial_coord_to_voxel(store, **args)
            elif capability_name == "spatial_get_operations_log":
                print("🔧 DEBUG: Calling spatial_get_operations_log...")
                result = spatial_get_operations_log(store)
            else:
                print(f"🔧 DEBUG: ❌ Unknown capability: {capability_name}")
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error=f"Unknown spatial capability: {capability_name}",
                )
            
            print(f"🔧 DEBUG: ✅ Tool result: {result}")
            
            # Update translate phase with layer name for later evaluation
            if result.get("success") and result.get("layer_name"):
                phase_records = ctx.episode_context.setdefault("phase_records", {})
                translate_record = phase_records.setdefault("translate", {})
                if not translate_record.get("feature_layer_name"):
                    translate_record["feature_layer_name"] = result["layer_name"]
                    translate_record["timestamp"] = __import__('time').time()
                    print(f"🔧 DEBUG: Stored layer name '{result['layer_name']}' for later evaluation")
                operations = store.get_spatial_operations()
                layer_ops = [
                    op for op in operations
                    if op.get("feature_name") == result.get("layer_name")
                ]
                fallback_count = sum(
                    1 for op in layer_ops
                    if op.get("coordinate_source") == "creative_fallback"
                )
                translate_record["spatial_operation_provenance_count"] = len(layer_ops)
                translate_record["translate_fallback_used"] = fallback_count > 0
                translate_record["coordinate_source_counts"] = {
                    source: sum(1 for op in layer_ops if op.get("coordinate_source") == source)
                    for source in sorted({str(op.get("coordinate_source") or "unknown") for op in layer_ops})
                }
            
            # Return result
            return CapabilityResult(
                capability_name,
                output=result,
                success=result.get("success", False),
            )
            
        except Exception as e:
            print(f"🔧 DEBUG: ❌ Exception in spatial capability: {e}")
            import traceback
            traceback.print_exc()
            return CapabilityResult(
                capability_name,
                success=False,
                error=f"Spatial capability execution failed: {str(e)}",
            )
    
    def _exec_scoring_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute scoring tool capability via voxel-features-mcp."""
        
        print(f"🎯 DEBUG: Starting scoring capability: {capability_name}")
        print(f"🎯 DEBUG: Args: {args}")
        
        # Set up environment for scoring operations
        import sys
        from pathlib import Path
        
        # Add voxel-features-mcp to path
        vfm_path = str(Path(__file__).parent.parent.parent / "voxel-features-mcp")
        sys.path.append(vfm_path)
        print(f"🎯 DEBUG: Added path: {vfm_path}")
        
        try:
            # Import required scoring tools
            print("🎯 DEBUG: Importing scoring modules...")
            from voxel_features.spatial import SpatialVoxelStore
            from voxel_features.store import GridSpec
            from voxel_features.mcp.tools.scoring_tools import (
                scoring_create_feature_layer
            )
            print("🎯 DEBUG: ✅ Imports successful")

            # Get store directory from episode context
            store_dir = ctx.episode_context.get("store_dir")
            episode_id = ctx.episode_context.get("episode_id", "")
            print(f"🎯 DEBUG: Store dir: {store_dir}")
            if not store_dir or not episode_id:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error="No store directory or episode_id available in episode context",
                )

            grid_dict = ctx.episode_context.get("grid_spec") or _KAZAKHSTAN_TENIZ_GRID
            grid = GridSpec.from_dict(grid_dict)

            scratch_dir = Path(store_dir) / "scratch" / episode_id
            admitted_dir = Path(store_dir) / "admitted"
            admitted_dir.mkdir(parents=True, exist_ok=True)

            print("🎯 DEBUG: Creating SpatialVoxelStore...")
            store = SpatialVoxelStore(
                scratch_dir, grid, read_only_overlay=admitted_dir
            )
            print(f"🎯 DEBUG: ✅ Store created, grid shape: {store.grid.shape}")
            
            # Route to scoring.create_feature_layer MCP tool
            print(f"🎯 DEBUG: Routing to tool: {capability_name}")
            if capability_name == "scoring_create_feature_layer":
                args = dict(args)
                requested_name = str(args.get("name", "")).strip()
                phase_records = ctx.episode_context.setdefault("phase_records", {})
                translated_name = str(
                    phase_records.get("translate", {}).get("feature_layer_name", "")
                ).strip()

                # NAT can emit multiple tool calls in one assistant turn. If a
                # scoring call races ahead of spatial_add_*, give the scratch
                # store a short chance to observe the layer before failing.
                available_layers = set(store.layer_names)
                if requested_name not in available_layers and not (
                    translated_name and translated_name in available_layers
                ):
                    for _ in range(5):
                        time.sleep(0.1)
                        store = SpatialVoxelStore(
                            scratch_dir, grid, read_only_overlay=admitted_dir
                        )
                        available_layers = set(store.layer_names)
                        if requested_name in available_layers or (
                            translated_name and translated_name in available_layers
                        ):
                            break

                if requested_name not in available_layers:
                    if translated_name and translated_name in available_layers:
                        print(
                            "🎯 DEBUG: Requested scoring layer "
                            f"{requested_name!r} missing; using last translated "
                            f"layer {translated_name!r}"
                        )
                        args["name"] = translated_name
                    else:
                        layer_list = sorted(available_layers)
                        message = (
                            f"Spatial layer {requested_name!r} does not exist in "
                            "the episode scratch store. Call spatial_add_point "
                            "or spatial_add_line successfully before "
                            "scoring_create_feature_layer."
                        )
                        result = {
                            "success": False,
                            "error": message,
                            "requested_layer": requested_name,
                            "last_translated_layer": translated_name,
                            "available_layers": layer_list,
                        }
                        print(f"🎯 DEBUG: ❌ {message}")
                        return CapabilityResult(
                            capability_name,
                            output=result,
                            success=False,
                            error=message,
                        )

                print("🎯 DEBUG: Calling scoring_create_feature_layer MCP function...")
                result = scoring_create_feature_layer(store, **args)
                # BIC scoring returns numpy scalars (np.bool_/np.float64); coerce
                # to native types so the result — and the evaluate phase record
                # downstream consumers read — stays JSON-serializable.
                result = _to_jsonable(result)
            else:
                print(f"🎯 DEBUG: ❌ Unknown capability: {capability_name}")
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error=f"Unknown scoring capability: {capability_name}",
                )
            
            print(f"🎯 DEBUG: ✅ Tool result: {result}")
            
            # Store evaluation results in episode context for rewrite phase
            if result.get("success"):
                phase_records = ctx.episode_context.setdefault("phase_records", {})
                
                # Update translate phase record. Use the scoring tool's returned
                # layer_name (timestamped, e.g. copper_concentration_1780199863749)
                # — the authoritative name of the .npy actually written — not the
                # bare agent-supplied args["name"] (rabbit-hole-bias fix: the
                # greedy init + crossbreed parent lookup must match real files).
                layer_name = result.get("layer_name") or args.get("name", "")
                if layer_name:
                    translate_record = phase_records.setdefault("translate", {})
                    translate_record["feature_layer_name"] = layer_name
                    translate_record["timestamp"] = __import__('time').time()
                
                # Store evaluation results
                phase_records["evaluate"] = result
                quarantine_info = self._quarantine_rejected_candidate(
                    store_dir=store_dir,
                    episode_id=episode_id,
                    layer_name=str(layer_name or ""),
                    evaluate=result,
                    phase_records=phase_records,
                )
                if quarantine_info is not None:
                    result["rejected_candidate_quarantined"] = True
                    result["rejected_candidate_quarantine"] = quarantine_info
                    ctx.episode_context["rejected_candidate_quarantine"] = quarantine_info
                print(f"🎯 DEBUG: Stored evaluation data in phase records")
            
            # Return result
            return CapabilityResult(
                capability_name,
                output=result,
                success=result.get("success", False),
            )
            
        except Exception as e:
            print(f"🎯 DEBUG: ❌ Exception in scoring capability: {e}")
            import traceback
            traceback.print_exc()
            return CapabilityResult(
                capability_name,
                success=False,
                error=f"Scoring capability execution failed: {str(e)}",
            )
    
    # ------------------------------------------------------------------
    # State measurement and rewards
    # ------------------------------------------------------------------
    
    def measure_initial_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> FeatureHypothesisKazakhstanState:
        return FeatureHypothesisKazakhstanState(
            episode_id=episode_context.get("episode_id", ""),
            workflow_kind=episode_context.get("workflow_kind", "survey"),
            n_features=episode_context.get("n_features", 0),
        )
    
    def measure_final_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
    ) -> FeatureHypothesisKazakhstanState:
        phase_records = episode_context.get("phase_records", {})
        terminal_record = episode_context.get("terminal_record", {})
        
        # Extract state from phase records
        hypothesise = phase_records.get("hypothesise", {})
        code = phase_records.get("code", {})
        translate = phase_records.get("translate", {})
        evaluate = phase_records.get("evaluate", {})
        
        return FeatureHypothesisKazakhstanState(
            episode_id=episode_context.get("episode_id", ""),
            workflow_kind=episode_context.get("workflow_kind", "survey"),
            n_features=episode_context.get("n_features", 0),
            hypothesis=hypothesise.get("hypothesis", ""),
            data_spec=hypothesise.get("data_spec", {}),
            code_executed=code.get("code_executed", ""),
            result_summary=code.get("result_summary", ""),
            feature_layer_name=translate.get("feature_layer_name"),
            bic_before=evaluate.get("bic_before"),
            bic_after=evaluate.get("bic_after"),
            bic_delta=evaluate.get("bic_delta"),
            cv_mse_delta=evaluate.get("cv_mse_delta"),
            mutual_info=evaluate.get("mutual_info", {}),
            admitted=evaluate.get("admitted", False),
            # Two-stage scoring results
            masking_test_passed=evaluate.get("masking_test_passed", True),
            masking_test_improvement=evaluate.get("masking_test_improvement", 0.0),
            masking_test_direction=evaluate.get("masking_test_direction", "not_applicable"),
            stage_1_tolerance_used=bool(evaluate.get("stage_1_tolerance_used", False)),
            stage_1_mae_tolerance=evaluate.get("stage_1_mae_tolerance"),
            stage_1_bic_rescue_threshold=evaluate.get("stage_1_bic_rescue_threshold"),
            stage_completed=evaluate.get("stage_completed", "stage_2_completed"),
            admission_path=evaluate.get("admission_path", "normal"),
            proposal_evidence_tier=self._normalise_evidence_tier(hypothesise.get("evidence_tier")),
            confidence=(hypothesise.get("self_assessment") or {}).get("confidence")
            if isinstance(hypothesise.get("self_assessment"), dict)
            else None,
            evidence_strength=evaluate.get("evidence_strength"),
            admission_tier=evaluate.get("admission_tier"),
            crossbreed_parent_eligible=evaluate.get("crossbreed_parent_eligible"),
            prompt_response_pair=terminal_record.get("training_pair", {}),
        )
    
    def compute_reward(
        self,
        initial: FeatureHypothesisKazakhstanState,
        final: FeatureHypothesisKazakhstanState,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        """Compute reward based on two-stage scoring results."""
        
        # Extract two-stage scoring results
        bic_delta = final.bic_delta
        masking_test_passed = final.masking_test_passed
        masking_test_improvement = final.masking_test_improvement
        masking_test_direction = final.masking_test_direction
        stage_1_tolerance_used = final.stage_1_tolerance_used
        stage_1_mae_tolerance = final.stage_1_mae_tolerance
        stage_1_bic_rescue_threshold = final.stage_1_bic_rescue_threshold
        admitted = final.admitted
        stage_completed = final.stage_completed

        if bic_delta is None and final.admission_path == "first_layer_auto" and admitted:
            return TaskReward(
                value=1.0,
                success=True,
                breakdown={
                    "stage_1_passed": True,
                    "stage_2_passed": False,
                    "first_layer_auto": True,
                    "bic_delta": None,
                    "proposal_evidence_tier": final.proposal_evidence_tier,
                    "evidence_strength": final.evidence_strength,
                    "admission_tier": final.admission_tier,
                    "crossbreed_parent_eligible": final.crossbreed_parent_eligible,
                    "region": "kazakhstan",
                },
            )

        if bic_delta is None:
            # No feature layer created
            return TaskReward(
                value=0.0,
                success=False,
                breakdown={
                    "no_feature": True,
                    "stage_completed": stage_completed
                }
            )

        # Two-stage reward — bic_delta is per-sample normalized post-scoring fix.
        # auto_pass / first_layer cannot compute a before/after MAE delta, so they
        # take full Stage 1 credit (no baseline to compare against).
        if masking_test_passed and admitted:
            if masking_test_direction in ("auto_pass", "first_layer", "first_layer_auto"):
                stage1_reward = 1.0
            else:
                stage1_reward = min(1.0, max(0.0, masking_test_improvement / 1e-4))
            stage2_reward = min(1.0, max(0.0, -bic_delta / 1.0))
            value = 0.4 * stage1_reward + 0.6 * stage2_reward

            return TaskReward(
                value=value,
                success=True,
                breakdown={
                    "stage_1_passed": True,
                    "stage_1_improvement": masking_test_improvement,
                    "stage_1_tolerance_used": stage_1_tolerance_used,
                    "stage_1_mae_tolerance": stage_1_mae_tolerance,
                    "stage_1_bic_rescue_threshold": stage_1_bic_rescue_threshold,
                    "stage_2_passed": True,
                    "bic_delta": bic_delta,
                    "stage1_reward": stage1_reward,
                    "stage2_reward": stage2_reward,
                    "final_reward": value,
                    "both_stages_passed": True,
                    "proposal_evidence_tier": final.proposal_evidence_tier,
                    "evidence_strength": final.evidence_strength,
                    "admission_tier": final.admission_tier,
                    "crossbreed_parent_eligible": final.crossbreed_parent_eligible,
                    "region": "kazakhstan",
                },
            )
        elif masking_test_passed and not admitted:
            if masking_test_direction in ("auto_pass", "first_layer", "first_layer_auto"):
                stage1_reward = 1.0
            else:
                stage1_reward = min(1.0, max(0.0, masking_test_improvement / 1e-4))
            value = 0.3 * stage1_reward

            return TaskReward(
                value=value,
                success=False,
                breakdown={
                    "stage_1_passed": True,
                    "stage_1_improvement": masking_test_improvement,
                    "stage_1_tolerance_used": stage_1_tolerance_used,
                    "stage_1_mae_tolerance": stage_1_mae_tolerance,
                    "stage_1_bic_rescue_threshold": stage_1_bic_rescue_threshold,
                    "stage_2_passed": False,
                    "bic_delta": bic_delta,
                    "stage1_reward": stage1_reward,
                    "partial_success": True,
                    "proposal_evidence_tier": final.proposal_evidence_tier,
                    "evidence_strength": final.evidence_strength,
                    "admission_tier": final.admission_tier,
                    "crossbreed_parent_eligible": final.crossbreed_parent_eligible,
                    "region": "kazakhstan",
                },
            )
        else:
            return TaskReward(
                value=0.05,
                success=False,
                breakdown={
                    "stage_1_passed": False,
                    "stage_1_improvement": masking_test_improvement,
                    "stage_1_tolerance_used": stage_1_tolerance_used,
                    "stage_1_mae_tolerance": stage_1_mae_tolerance,
                    "stage_1_bic_rescue_threshold": stage_1_bic_rescue_threshold,
                    "stage_2_passed": admitted,
                    "bic_delta": bic_delta,
                    "no_predictive_value": True,
                    "proposal_evidence_tier": final.proposal_evidence_tier,
                    "evidence_strength": final.evidence_strength,
                    "admission_tier": final.admission_tier,
                    "crossbreed_parent_eligible": final.crossbreed_parent_eligible,
                    "region": "kazakhstan",
                },
            )

    def finalize_episode(
        self,
        containers: list[Container],
        initial: FeatureHypothesisKazakhstanState,
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
        finalization_context: Any | None = None,
    ) -> TaskReward:
        try:
            final = self.measure_final_state(
                containers, episode_context, artifacts, private_context=private_context
            )
            reward = self.compute_reward(initial, final, artifacts)
            breakdown = dict(reward.breakdown or {})
            # The framework's max_bootstrap_episodes guard inspects
            # task_breakdown["bootstrap_active"] (src/execution/generation.py).
            # For feature_hypothesis, "bootstrap" == the early phase before
            # the feature pool reaches min_features (workflow_kind="survey").
            breakdown["bootstrap_active"] = (
                final.workflow_kind == "survey"
                and not bool(episode_context.get("interweave_bootstrap"))
            )
            if episode_context.get("duplicate_rejected"):
                breakdown["duplicate_rejected"] = True
            kg_dir_ctx = episode_context.get("kg_dir")
            if isinstance(kg_dir_ctx, str) and kg_dir_ctx:
                try:
                    interweave_state = self._record_interweave_episode_result(
                        Path(kg_dir_ctx),
                        workflow_kind=final.workflow_kind,
                        produced_new_admit=bool(reward.success)
                        and not bool(episode_context.get("duplicate_rejected")),
                        interweave_bootstrap=bool(
                            episode_context.get("interweave_bootstrap")
                        ),
                    )
                    if interweave_state:
                        breakdown["interweave_failed_crossbreed_streak"] = int(
                            interweave_state.get("consecutive_failed_crossbreed", 0)
                        )
                        breakdown["interweave_survey_remaining"] = int(
                            interweave_state.get("interweave_survey_remaining", 0)
                        )
                        if episode_context.get("interweave_bootstrap"):
                            breakdown["interweave_bootstrap"] = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"feature_hypothesis: interweave state update failed: {exc}"
                    )
            if breakdown != (reward.breakdown or {}):
                reward = TaskReward(
                    value=reward.value,
                    success=reward.success,
                    breakdown=breakdown,
                )
            return reward
        finally:
            # Always release the bootstrap permit so a crashed or short-
            # circuited episode does not hold the in-flight slot until it
            # ages out via stale_after_s.
            permit_slot_id = episode_context.get("bootstrap_permit_slot_id")
            kg_dir_ctx = episode_context.get("kg_dir")
            if isinstance(permit_slot_id, str) and isinstance(kg_dir_ctx, str) and kg_dir_ctx:
                try:
                    self._release_bootstrap_permit(Path(kg_dir_ctx), permit_slot_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"feature_hypothesis: bootstrap permit release failed "
                        f"({permit_slot_id}): {exc}"
                    )
            # Reclaim the per-episode scratch dir. Runs unconditionally so
            # a crashed mid-episode leaves no orphans.
            try:
                self.cleanup_episode_resources(episode_context)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"feature_hypothesis: scratch cleanup failed: {exc}"
                )

    def cleanup_episode_resources(
        self, episode_context: dict[str, Any]
    ) -> None:
        """Remove the per-episode scratch dir.

        Idempotent. Designed to run from ``finalize_episode``'s ``finally``
        block — success or failure of the episode is irrelevant. The
        admitted pool is *never* touched here; promotion is the only path
        from scratch to admitted.
        """
        import shutil

        store_dir = episode_context.get("store_dir")
        episode_id = episode_context.get("episode_id")
        if not isinstance(store_dir, str) or not isinstance(episode_id, str):
            return
        # ``episode_id`` already carries an ``ep_<ts>_<hex>`` prefix from
        # ``populate``, so the scratch dir is simply ``scratch/<episode_id>``.
        scratch_dir = Path(store_dir) / "scratch" / episode_id
        shutil.rmtree(scratch_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _pick_container(
        self,
        containers: list[Container],
        role: str,
    ) -> Container | None:
        service = _ROLE_SERVICE.get(role)
        if service is None:
            return None
        for container in containers:
            if container_to_service(container) == service:
                return container
        return None

    def _count_features(self, variation: FeatureHypothesisKazakhstanVariation) -> int:
        """Count features in the admitted pool.

        Pre-isolation, layers lived directly under ``store_dir``; the
        legacy path is checked as a fallback so existing runs whose
        ``index.json`` was never migrated still report a non-zero count.
        """
        admitted_index = Path(variation.store_dir) / "admitted" / "index.json"
        legacy_index = Path(variation.store_dir) / "index.json"
        for path in (admitted_index, legacy_index):
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                return len(data.get("layers", {}))
            except Exception:
                continue
        return 0
    
    def _has_crossbreed_pairs(self, variation: FeatureHypothesisKazakhstanVariation) -> bool:
        """Check if there are crossbreed pairs available."""
        try:
            admitted_count = len(self._load_successful_experiments(Path(variation.kg_dir)))
            # Floor raised 2 → 5 so the full source-rotation list is explored
            # at least once before crossbreeding begins (file-rotation tuning).
            return admitted_count >= 5
        except Exception:
            return False

    def _all_sources_visited(self, kg_dir: str) -> bool:
        """Return True when every source in _KAZAKHSTAN_SOURCE_FILES has been
        visited at least once according to file_rotation_state.json.

        Gates crossbreeding so it cannot begin before the rotation has covered
        the whole dataset (rabbit-hole-bias fix, JenD86/file-rotation@72e3239).
        """
        state_path = Path(kg_dir) / "file_rotation_state.json"
        if not state_path.exists():
            return False
        try:
            with open(state_path) as f:
                counts = json.load(f).get("counts", {})
        except Exception:  # noqa: BLE001
            return False
        return all(counts.get(s["key"], 0) >= 1 for s in _KAZAKHSTAN_SOURCE_FILES)

    def _run_greedy_bic_initialization(
        self, variation: "FeatureHypothesisKazakhstanVariation"
    ) -> None:
        """Forward greedy BIC selection over all admitted bootstrap layers.

        Ported from JenD86/file-rotation@72e3239. Guarded by
        greedy_init_complete.json + _kg_lock so only one parallel episode runs it
        (O(N^2) evals). Round 1 picks the highest null-model BIC layer (most
        spatially variable → richest single foundation); rounds 2+ greedily add
        whichever remaining layer most reduces geological_coherence_score (most
        complementary). Non-selected layers stay in the store but are excluded
        from crossbreeding by the populate() flag check. Writes the selection to
        greedy_init_complete.json.
        """
        kg_dir = Path(variation.kg_dir)
        flag_path = kg_dir / "greedy_init_complete.json"
        if flag_path.exists():
            return

        with self._kg_lock(kg_dir):
            if flag_path.exists():
                return

            try:
                import numpy as _np
                from voxel_features.spatial import SpatialVoxelStore as _SVS
                from voxel_features.store import GridSpec as _GridSpec
                from voxel_features.scoring import (
                    geological_coherence_score as _geo_score,
                    _single_layer_null_bic as _null_bic,
                )
            except ImportError as exc:
                logger.warning(f"greedy_bic_init: import failed: {exc}")
                flag_path.write_text(json.dumps({"status": "skipped", "reason": str(exc)}))
                return

            admitted_dir = Path(variation.store_dir) / "admitted"
            try:
                grid = _GridSpec.from_dict(variation.grid_spec)
                store = _SVS(admitted_dir, grid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"greedy_bic_init: store open failed: {exc}")
                flag_path.write_text(json.dumps({"status": "skipped", "reason": str(exc)}))
                return

            layer_names = list(store.layer_names)
            n = len(layer_names)
            if n < 2:
                logger.info(f"greedy_bic_init: {n} layers — skipping selection")
                flag_path.write_text(json.dumps({
                    "status": "skipped", "n": n, "reason": "too_few_layers",
                    "selected": layer_names,
                }))
                return

            values = [store.get_layer_values(name).flatten() for name in layer_names]
            dtypes = [store.get_layer(name).dtype for name in layer_names]
            shape = tuple(grid.shape)
            logger.info(f"greedy_bic_init: forward greedy selection over {n} layers")

            # Round 1: highest null-model BIC = most spatially variable layer.
            null_bics: list[float] = []
            for i in range(n):
                try:
                    r = _null_bic(values[i], dtypes[i], grid, shape)
                    null_bics.append(r.get("bic", 0.0))
                except Exception:  # noqa: BLE001
                    null_bics.append(0.0)

            first_idx = int(_np.argmax(null_bics))
            selected: list[int] = [first_idx]
            remaining: list[int] = [i for i in range(n) if i != first_idx]
            bic_current = 0.0  # geo_coherence_score for a single layer = 0

            # Rounds 2+: add whichever layer most reduces geological_coherence BIC.
            while remaining:
                best_j: int | None = None
                best_delta = 0.0
                bic_next = bic_current
                for j in remaining:
                    cand_vals = [values[i] for i in selected] + [values[j]]
                    cand_dtypes = [dtypes[i] for i in selected] + [dtypes[j]]
                    try:
                        r = _geo_score(cand_vals, cand_dtypes, grid, shape)
                        delta = r.get("bic", 0.0) - bic_current
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                            bic_next = r.get("bic", 0.0)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            f"greedy_bic_init: scoring failed for {layer_names[j]}: {exc}"
                        )

                if best_j is None:
                    break
                selected.append(best_j)
                remaining.remove(best_j)
                bic_current = bic_next

            selected_names = [layer_names[i] for i in selected]
            not_selected = [layer_names[i] for i in range(n) if i not in selected]
            logger.info(
                f"greedy_bic_init: selected {len(selected_names)}/{n} layers; "
                f"final BIC {bic_current:.2f}; excluded: {not_selected}"
            )
            flag_path.write_text(json.dumps({
                "status": "complete",
                "n_total": n,
                "n_selected": len(selected_names),
                "selected": selected_names,
                "not_selected": not_selected,
                "final_bic": bic_current,
            }, indent=2))

    def _assign_rotation_source(
        self,
        episode_context: dict[str, Any],
        variation: "FeatureHypothesisKazakhstanVariation",
    ) -> None:
        """Assign a least-explored source (file rotation) + pre-read sample into
        ``episode_context`` for survey prompts."""
        rotation = self._pick_assigned_source(variation.kg_dir, _KAZAKHSTAN_SOURCE_FILES)
        episode_context["assigned_source"] = rotation["source"]
        episode_context["source_coverage"] = rotation["all_counts"]
        episode_context["source_sample"] = self._read_source_sample(
            rotation["source"], variation.dataset_dir
        )

    def _assigned_source_blocks(self, episode_context: dict[str, Any]) -> str:
        """ASSIGNED SOURCE + SAMPLE CONTENT blocks for an explore prompt, or ""
        when no source is assigned.

        Used by survey prompts so the SFT export extracts read evidence from the
        stable headers below (matched by ExperimentReasoningRows._parse_assigned_source).
        """
        assigned = episode_context.get("assigned_source", {})
        if not assigned:
            return ""
        glob_pattern = assigned.get("glob_pattern")
        if glob_pattern:
            patterns = [glob_pattern] if isinstance(glob_pattern, str) else glob_pattern
            glob_lines = "\n".join(
                f"    files += glob.glob(os.path.join(dataset_dir, '{assigned['path']}', '{p}'))"
                for p in patterns
            )
            assignment_block = (
                f"ASSIGNED SOURCE FOR THIS EPISODE\n"
                f"  Section: {assigned['key']}\n"
                f"  Details: {assigned['description']}\n\n"
                f"To list the files in this section, run in analysis_shell:\n"
                f"```python\n"
                f"import glob, os\n"
                f"dataset_dir = '/workspace/input'\n"
                f"files = []\n"
                f"{glob_lines}\n"
                f"files = sorted(files)\n"
                f"print(f'Found {{len(files)}} files:')\n"
                f"for f in files: print(f)\n"
                f"```\n"
            )
        else:
            assignment_block = (
                f"ASSIGNED SOURCE FOR THIS EPISODE\n"
                f"  Path   : /workspace/input/{assigned['path']}\n"
                f"  Details: {assigned['description']}\n"
            )
        sample = episode_context.get("source_sample", "")
        if sample:
            sample_block = (
                "\nSAMPLE CONTENT FROM YOUR ASSIGNED SOURCE\n"
                "─────────────────────────────────────────\n"
                + sample
                + "\n"
            )
        else:
            sample_block = ""
        return assignment_block + sample_block

    def _pick_assigned_source(
        self,
        kg_dir: str,
        source_files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Pick the least-explored source file and update the rotation state.

        Reads ``{kg_dir}/file_rotation_state.json``, increments the count for
        the chosen source, and writes back.  Ties are broken by list order so
        the assignment is stable and deterministic.
        """
        state_path = Path(kg_dir) / "file_rotation_state.json"
        counts: dict[str, int] = {}
        if state_path.exists():
            try:
                with open(state_path) as f:
                    counts = json.load(f).get("counts", {})
            except Exception:
                counts = {}

        # Least-explored source wins; list order breaks ties
        assigned = min(source_files, key=lambda s: counts.get(s["key"], 0))
        counts[assigned["key"]] = counts.get(assigned["key"], 0) + 1

        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_path, "w") as f:
                json.dump({"counts": counts}, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not write file_rotation_state.json: {exc}")

        return {"source": assigned, "all_counts": counts}

    def _read_source_sample(
        self,
        assigned: dict[str, Any],
        dataset_dir: str,
    ) -> str:
        """Read a compact sample from the assigned source for injection into the explore prompt.

        Returns a formatted string block capped at ~2500 chars total.
        Silently returns an empty string on any read error or missing path.

        Sampling strategy by entry type:
        - glob_pattern (md chunks): pick first, middle, last file; read ~800 chars each.
        - GeoJSON: property schema + first 3 features (properties only, no geometry).
        - CSV: header row + first 5 data rows.
        - Directory: list up to 10 files + read first 2 files ~800 chars each.
        """
        import csv as csv_mod
        import glob as glob_mod
        import json as json_mod

        base = Path(dataset_dir)
        path = assigned.get("path", "")
        glob_pattern = assigned.get("glob_pattern")
        full_path = base / path

        MAX_PER_FILE = 800
        MAX_TOTAL = 2500
        parts: list[str] = []

        try:
            if glob_pattern:
                # Section-level markdown chunks — pick first, middle, last
                patterns = [glob_pattern] if isinstance(glob_pattern, str) else glob_pattern
                files: list[str] = []
                for p in patterns:
                    files.extend(glob_mod.glob(str(base / path / p)))
                files = sorted(set(files))
                if not files:
                    return ""
                indices = sorted({0, len(files) // 2, len(files) - 1})
                selected = [files[i] for i in indices]
                for fpath in selected:
                    try:
                        text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                        snippet = text[:MAX_PER_FILE]
                        if len(text) > MAX_PER_FILE:
                            snippet += "\n[…truncated]"
                        parts.append(f"--- {Path(fpath).name} ---\n{snippet}")
                    except Exception:
                        pass

            elif path.endswith(".geojson"):
                # GeoJSON: schema + first 3 features (properties only, no geometry)
                with open(full_path, encoding="utf-8") as fh:
                    data = json_mod.load(fh)
                features = data.get("features", [])
                if features:
                    keys = list(features[0].get("properties", {}).keys())
                    parts.append(f"Property columns: {keys}")
                    for feat in features[:3]:
                        props = feat.get("properties", {})
                        geom_type = feat.get("geometry", {}).get("type", "?")
                        row = json_mod.dumps(props)
                        parts.append(f"  [{geom_type}] {row[:280]}")

            elif path.lower().endswith(".csv"):
                # CSV: header row + first 5 data rows (always include header)
                with open(full_path, encoding="utf-8", errors="replace") as fh:
                    reader = csv_mod.reader(fh)
                    rows: list[str] = []
                    for i, row in enumerate(reader):
                        rows.append(",".join(str(c) for c in row[:15]))
                        if i >= 5:
                            break
                parts.append("\n".join(rows))

            elif full_path.is_dir():
                # Directory: list files + read first 2
                all_files = sorted(p for p in full_path.iterdir() if p.is_file())
                file_list = [fp.name for fp in all_files[:10]]
                parts.append("Files: " + ", ".join(file_list))
                for fp in all_files[:2]:
                    try:
                        text = fp.read_text(encoding="utf-8", errors="replace")
                        snippet = text[:MAX_PER_FILE]
                        if len(text) > MAX_PER_FILE:
                            snippet += "\n[…truncated]"
                        parts.append(f"--- {fp.name} ---\n{snippet}")
                    except Exception:
                        pass

            else:
                # Single file fallback
                if full_path.is_file():
                    text = full_path.read_text(encoding="utf-8", errors="replace")
                    parts.append(text[:MAX_PER_FILE])

        except Exception:
            return ""

        combined = "\n\n".join(parts)
        if len(combined) > MAX_TOTAL:
            combined = combined[:MAX_TOTAL] + "\n[…truncated]"
        return combined

    def _generate_explore_prompt(self, episode_context: dict[str, Any]) -> str:
        """Generate the combined explore+hypothesise prompt for the survey workflow.

        Each episode is anchored to a specific source selected at populate() time.
        Sample content is pre-read and injected so the agent sees real data immediately.
        """
        # Assignment + pre-read sample blocks (shared with the crossbreed prompt
        # via _assigned_source_blocks). Fall back to a generic explore line when
        # no source is assigned.
        blocks = self._assigned_source_blocks(episode_context)
        if not blocks:
            blocks = (
                "Explore the Kazakhstan Teniz Basin dataset to identify a regional feature opportunity.\n"
            )

        prompt = (
            "Phase 1: Explore and Hypothesise\n\n"
            + blocks
            + "\n"
            "Use analysis_shell to read and explore your assigned source.\n"
            "When you have identified a promising geological pattern, record ONE falsifiable hypothesis:\n\n"
            "  record_phase(\n"
            "      phase='hypothesise',\n"
            "      hypothesis='...',\n"
            "      data_spec={\n"
            "          'files': ['/workspace/input/{your_assigned_path}', ...],\n"
            "          'analysis': '...',\n"
            "          'output': '...'\n"
            "      }\n"
            "      evidence_tier='weak' | 'mixed' | 'strong',\n"
            "      self_assessment={\n"
            "          'confidence': 0.0-1.0,\n"
            "          'evidence_basis': '...',\n"
            "          'known_limitations': [...],\n"
            "          'evidence_sources': [...]\n"
            "      }\n"
            "  )\n\n"
            "Declare evidence_tier='weak' when support is indirect, text-only, single-record, or contradicted by prospect metadata. Declaring weak is allowed and costs no reward; it only makes the system more cautious about crossbreed parentage.\n"
            "Your hypothesis MUST be grounded in what you found in the assigned source above.\n"
            "Do NOT introduce topics not present in the assigned source."
        )
        return prompt

    def _get_crossbreed_context(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
    ) -> dict[str, Any]:
        """Get crossbreed prompt and parent IDs using JSONL knowledge graph."""
        import json
        
        try:
            crossbreed_file = Path(variation.kg_dir) / _KG_CROSSBREED_INDEX
            experiments = self._load_successful_experiments(Path(variation.kg_dir))
            
            if len(experiments) < 2:
                return {}
            
            # Load mutual information index if available
            mi_index = {}
            if crossbreed_file.exists():
                with open(crossbreed_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                mi_record = json.loads(line)
                                pair_id = mi_record["pair_id"]
                                mi_index[pair_id] = mi_record["mutual_information"]
                            except (json.JSONDecodeError, KeyError):
                                continue
            
            # Find best crossbreed pair (high BIC improvement + low MI)
            best_pair = None
            best_score = float('-inf')
            
            for i, exp_a in enumerate(experiments):
                for exp_b in experiments[i+1:]:
                    # Calculate combined BIC improvement
                    bic_a = abs(exp_a.get("bic_delta", 0))
                    bic_b = abs(exp_b.get("bic_delta", 0))
                    combined_bic = bic_a + bic_b
                    
                    # Get mutual information (prefer low MI = orthogonal features)
                    pair_id = f"{min(exp_a['node_id'], exp_b['node_id'])}_{max(exp_a['node_id'], exp_b['node_id'])}"
                    mi_score = mi_index.get(pair_id, 0.0)
                    
                    # Combined score: high BIC improvement, low MI
                    pair_score = combined_bic - mi_score
                    
                    if pair_score > best_score:
                        best_score = pair_score
                        best_pair = (exp_a, exp_b)
            
            if not best_pair:
                return {}
            
            exp_a, exp_b = best_pair
            
            prompt = (
                f"These experiments both improved the world model:\n\n"
                f"Experiment 1: \"{exp_a.get('hypothesis', '')}\"\n"
                f"- Result: {exp_a.get('response', '').split('.')[0] if exp_a.get('response') else 'N/A'}\n"
                f"- Feature: {exp_a.get('layer_name', '')}\n"
                f"- BIC improvement: {abs(exp_a.get('bic_delta', 0)):.2f}\n\n"
                f"Experiment 2: \"{exp_b.get('hypothesis', '')}\"\n"
                f"- Result: {exp_b.get('response', '').split('.')[0] if exp_b.get('response') else 'N/A'}\n"
                f"- Feature: {exp_b.get('layer_name', '')}\n"
                f"- BIC improvement: {abs(exp_b.get('bic_delta', 0)):.2f}\n\n"
                f"Given that both patterns exist in the data, what new hypothesis "
                f"would you propose that combines or builds on these findings?"
            )
            
            return {
                "prompt": prompt,
                "parent_ids": [exp_a["node_id"], exp_b["node_id"]],
            }
        except Exception as e:
            print(f"Warning: JSONL crossbreed failed, using simple fallback: {e}")
            return self._get_crossbreed_context_simple(variation)
    
    def _get_crossbreed_context_simple(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
    ) -> dict[str, Any]:
        """Simple fallback crossbreed selection - just first two admitted."""
        try:
            experiments = self._load_successful_experiments(Path(variation.kg_dir))[:2]
            
            if len(experiments) < 2:
                return {}
            
            exp_a, exp_b = experiments[0], experiments[1]
            
            prompt = (
                f"These experiments both improved the world model:\n\n"
                f"Experiment 1: \"{exp_a.get('hypothesis', '')}\"\n"
                f"- Result: {exp_a.get('response', '').split('.')[0] if exp_a.get('response') else 'N/A'}\n"
                f"- Feature: {exp_a.get('layer_name', '')}\n\n"
                f"Experiment 2: \"{exp_b.get('hypothesis', '')}\"\n"
                f"- Result: {exp_b.get('response', '').split('.')[0] if exp_b.get('response') else 'N/A'}\n"
                f"- Feature: {exp_b.get('layer_name', '')}\n\n"
                f"Given that both patterns exist in the data, what new hypothesis "
                f"would you propose that combines or builds on these findings?"
            )
            
            return {
                "prompt": prompt,
                "parent_ids": [exp_a["node_id"], exp_b["node_id"]],
            }
        except Exception:
            return {}

    @staticmethod
    def _crossbreed_prompt(exp_a: dict[str, Any], exp_b: dict[str, Any]) -> str:
        """Render the crossbreed prompt for a specific ordered (A, B) pair."""
        return (
            f"These experiments both improved the world model:\n\n"
            f"Experiment 1: \"{exp_a.get('hypothesis', '')}\"\n"
            f"- Result: {exp_a.get('response', '').split('.')[0] if exp_a.get('response') else 'N/A'}\n"
            f"- Feature: {exp_a.get('layer_name', '')}\n"
            f"- BIC improvement: {abs(float(exp_a.get('bic_delta', 0.0))):.2f}\n\n"
            f"Experiment 2: \"{exp_b.get('hypothesis', '')}\"\n"
            f"- Result: {exp_b.get('response', '').split('.')[0] if exp_b.get('response') else 'N/A'}\n"
            f"- Feature: {exp_b.get('layer_name', '')}\n"
            f"- BIC improvement: {abs(float(exp_b.get('bic_delta', 0.0))):.2f}\n\n"
            f"Given that both patterns exist in the data, what new hypothesis "
            f"would you propose that combines or builds on these findings?"
        )

    def _build_crossbreed_context_for_pair(
        self,
        kg_dir: Path,
        parent_a_id: str,
        parent_b_id: str,
    ) -> dict[str, Any]:
        """Build crossbreed_context for a specific parent pair, popped from the queue.

        Falls back to an empty dict if either parent cannot be found in
        experiments.jsonl (e.g. a stale queue entry whose record was pruned).
        """
        experiments = self._load_successful_experiments(kg_dir)
        by_id = {exp.get("node_id"): exp for exp in experiments}
        exp_a = by_id.get(parent_a_id)
        exp_b = by_id.get(parent_b_id)
        if not (isinstance(exp_a, dict) and isinstance(exp_b, dict)):
            return {}
        return {
            "prompt": self._crossbreed_prompt(exp_a, exp_b),
            "parent_ids": [parent_a_id, parent_b_id],
        }

    # ------------------------------------------------------------------
    # Duplicate handling + bootstrap pacing
    # ------------------------------------------------------------------
    # All coordination is file-based so it survives across the parallel
    # worker threads that share a single TaskSpec instance. Mirrors the
    # `_pool_lock`/`_read_pool_index` pattern from `tasks/geology_graph.py`.

    @contextmanager
    def _kg_lock(self, kg_dir: Path | str) -> Iterator[None]:
        kg_path = Path(kg_dir)
        kg_path.mkdir(parents=True, exist_ok=True)
        lock_path = kg_path / _KG_LOCK
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write_json(path: Path, obj: Any) -> None:
        """Tmp-then-replace JSON writer. Unique tmp per pid+uuid prevents
        cross-process ENOENT races (see `geology_graph._write_pool_index`)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _atomic_write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        with tmp.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
        tmp.replace(path)

    @staticmethod
    def _read_json_or(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default)
        if not isinstance(data, dict):
            return dict(default)
        return data

    @staticmethod
    def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    # ``stage_completed`` strings produced by voxel-features-mcp scoring.
    # "stage_2_completed" was the legacy label; "mae_bic_completed" replaced
    # it when stage 2 was rewritten to use MAE+BIC. The kg gate must accept
    # both — without this allowlist, the post-rewrite scoring path silently
    # blocks every admission (observed mid-run: 30+ successful episodes with
    # bic_delta < -40 000 yet `experiments.jsonl` frozen at 3 rows).
    _STAGE_COMPLETED_ALLOWLIST: frozenset[str] = frozenset({
        "stage_2_completed",
        "mae_bic_completed",
    })
    _QUARANTINE_STAGE_ALLOWLIST: frozenset[str] = _STAGE_COMPLETED_ALLOWLIST | frozenset({
        "guard_rejected",
    })

    @classmethod
    def _should_persist_to_kg(
        cls,
        *,
        masking_test_passed: bool,
        admitted: bool,
        bic_delta: float | None,
        stage_completed: str,
        admission_path: str | None = None,
    ) -> bool:
        """Gate ``_admit_with_dedup`` so only fully-scored, two-stage-passing
        experiments enter the kg pool.

        All four conditions must hold:
          - ``masking_test_passed`` — stage 1 predictive-capacity check.
          - ``admitted`` — stage 2 scorer admitted the layer (bic_delta < 0).
          - ``bic_delta is not None and bic_delta < 0`` — defense in depth.
          - ``stage_completed`` in the allowlist — proves stage 2 actually ran
            (vs. partial / aborted scoring).
        """
        if not bool(masking_test_passed):
            return False
        if not bool(admitted):
            return False
        if admission_path == "first_layer_auto" and stage_completed == "first_layer_auto":
            return True
        if bic_delta is None or bic_delta >= 0:
            return False
        return stage_completed in cls._STAGE_COMPLETED_ALLOWLIST

    @classmethod
    def _should_quarantine_rejected_candidate(cls, evaluate: dict[str, Any]) -> bool:
        """True for scored candidates that failed canonical KG admission.

        This is evidence capture only. It deliberately mirrors the strict KG
        gate and never turns a rejected layer into a canonical admit.
        """
        bic_delta = evaluate.get("bic_delta")
        stage_completed = str(evaluate.get("stage_completed", ""))
        if bic_delta is None or stage_completed not in cls._QUARANTINE_STAGE_ALLOWLIST:
            return False
        return not cls._should_persist_to_kg(
            masking_test_passed=bool(evaluate.get("masking_test_passed", True)),
            admitted=bool(evaluate.get("admitted", False)),
            bic_delta=bic_delta,
            stage_completed=stage_completed,
        )

    @staticmethod
    def _rejection_stage(evaluate: dict[str, Any]) -> str:
        if evaluate.get("stage_completed") == "guard_rejected" or evaluate.get("admission_tier") == "guard_rejected":
            return "guard"
        if not bool(evaluate.get("masking_test_passed", True)):
            return "stage_1"
        if not bool(evaluate.get("admitted", False)):
            return "stage_2"
        return "kg_gate"

    def _quarantine_rejected_candidate(
        self,
        *,
        store_dir: str,
        episode_id: str,
        layer_name: str,
        evaluate: dict[str, Any],
        phase_records: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not store_dir or not episode_id or not layer_name:
            return None
        if not self._should_quarantine_rejected_candidate(evaluate):
            return None

        import shutil

        safe_episode = _safe_artifact_component(episode_id)
        safe_layer = _safe_artifact_component(layer_name)
        store_path = Path(store_dir)
        quarantine_dir = store_path / "rejected" / safe_episode
        metadata_path = quarantine_dir / "metadata.json"
        layer_dest = quarantine_dir / f"{safe_layer}.npy"
        layer_src = store_path / "scratch" / episode_id / "layers" / f"{layer_name}.npy"

        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                return {
                    "path": str(quarantine_dir),
                    "metadata_file": str(metadata_path),
                    "layer_file": existing.get("layer_file"),
                    "layer_file_copied": bool(existing.get("layer_file_copied", False)),
                    "rejection_stage": existing.get("rejection_stage"),
                }
            except json.JSONDecodeError:
                pass

        quarantine_dir.mkdir(parents=True, exist_ok=True)
        layer_file_copied = False
        if layer_src.exists():
            shutil.copy2(layer_src, layer_dest)
            layer_file_copied = True

        hypothesise = phase_records.get("hypothesise", {})
        code = phase_records.get("code", {})
        metadata = {
            "episode_id": episode_id,
            "layer_name": layer_name,
            "timestamp": time.time(),
            "rejection_stage": self._rejection_stage(evaluate),
            "layer_file": str(layer_dest) if layer_file_copied else None,
            "layer_file_copied": layer_file_copied,
            "source_layer_file": str(layer_src),
            "hypothesis": hypothesise.get("hypothesis", ""),
            "data_spec": hypothesise.get("data_spec", {}),
            "experiment_summary": code.get("result_summary", ""),
            "evaluate": _to_jsonable(evaluate),
        }
        self._atomic_write_json(metadata_path, metadata)
        return {
            "path": str(quarantine_dir),
            "metadata_file": str(metadata_path),
            "layer_file": str(layer_dest) if layer_file_copied else None,
            "layer_file_copied": layer_file_copied,
            "rejection_stage": metadata["rejection_stage"],
        }

    def _recent_admitted_hypotheses(
        self,
        kg_dir: Path | str,
        k: int,
    ) -> list[dict[str, str]]:
        """Return up to ``k`` most recently admitted hypotheses as
        ``[{layer_name, hypothesis}, ...]`` in admit order (oldest → newest).

        Reads ``experiments.jsonl`` directly: every row there is admitted by
        construction (see ``_admit_with_dedup``), so the join through
        ``admitted_index.json`` is redundant and the file's append order is
        the canonical admit order. Rows missing a non-blank ``hypothesis``
        are skipped.
        """
        if k <= 0:
            return []
        rows = self._read_jsonl_records(Path(kg_dir) / _KG_EXPERIMENTS)
        out: list[dict[str, str]] = []
        for row in rows[-k:]:
            hyp = str(row.get("hypothesis", "")).strip()
            if not hyp:
                continue
            out.append({
                "layer_name": str(row.get("layer_name") or ""),
                "hypothesis": hyp,
            })
        return out

    @staticmethod
    def _render_novelty_block(
        recent: list[dict[str, str]],
        max_chars: int = 280,
    ) -> str:
        """Render the 'avoid variants of these' block, or empty on first
        episodes.

        Per-entry hypothesis is truncated to ``max_chars`` to bound the
        per-episode input-token cost (block scales linearly with K). The
        instruction text is deliberately strong-but-positive: "take a new
        direction" rather than "anything-but-X", to discourage degenerate
        anti-imitation hypotheses.
        """
        if not recent:
            return ""
        lines = [
            "## Already discovered — DO NOT propose variants of these:",
        ]
        for i, entry in enumerate(recent, 1):
            layer = entry.get("layer_name") or "(unnamed)"
            hyp = entry.get("hypothesis") or ""
            if max_chars > 3 and len(hyp) > max_chars:
                hyp = hyp[: max_chars - 3].rstrip() + "..."
            lines.append(f"{i}. [{layer}] {hyp}")
        lines.append("")
        lines.append(
            "Your proposal MUST take a genuinely new direction — a different "
            "geological process, feature family, or spatial scale than any of "
            "the above. If your draft hypothesis overlaps with any item, "
            "discard it and pick a different angle from the dataset."
        )
        return "\n".join(lines)

    # Rule-based keyword buckets for mechanism-family classification. Used
    # to summarise the recent-admit pool by family in the survey-stage
    # novelty block so the agent sees *what class of hypothesis dominates*
    # rather than only individual examples. Keep narrow and high-precision:
    # the summary line is advisory, not a hard gate, and false positives
    # turn the diversity nudge into noise. Order matters — first-match wins
    # so that, e.g., "fold proximity at basin margin" is tagged structural
    # rather than basin_geometry (the fold is the verb).
    _MECHANISM_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("structural", (
            "fold", "anticlin", "synclin", "limb", "hinge",
            "axial", "plunge", "fault", "fracture", "shear", "lineament",
        )),
        # drillhole sits ahead of geochemical / lithological because
        # wireline-log signals (SP curve, gamma, assays-by-depth) are
        # almost always paired with redox or lithology terms in the same
        # sentence — we want the drill-hole-derived family tag to win on
        # overlap.
        ("drillhole", (
            "borehole", "drill hole", "drillhole", "wireline", "well log",
            "sp curve", "resistivity log", "per-borehole", "per borehole",
        )),
        ("geochemical", (
            "redox", "oxid", "reduc", "pyrite", "chalcopyrite", "bornite",
            "chalcocite", "hematite", "sulfide", "sulphide", "red bed",
            "red-bed", "bleach", "assay", "ppm", "spectral analysis",
            "mineralogy", "geochem",
        )),
        ("lithological", (
            "sandstone", "mudstone", "siltstone", "conglomerate", "shale",
            "limestone", "dolomite", "marl", "lithology", "host rock",
            "host suite", "vladimirov", "kayraktin", "kirey", "dzhaksykon",
            "facies", "stratigraph", "unconform",
        )),
        ("hydrologic", (
            "permeab", "porosity", "fluid", "brine", "aquifer", "groundwater",
            "reservoir", "flux channel", "conduit",
        )),
        ("basin_geometry", (
            "basin margin", "basin boundary", "tract boundary", "depocenter",
            "depocentre", "margin proxim", "distance to basin",
            "assessment tract", "aoi",
        )),
    )

    @staticmethod
    def _classify_mechanism(text: str) -> str:
        """Return a single mechanism-family tag for one hypothesis string.

        First-match wins per ``_MECHANISM_BUCKETS`` ordering — see the
        comment there. ``"other"`` if no bucket matches.
        """
        t = (text or "").lower()
        if not t:
            return "other"
        for name, keywords in FeatureHypothesisKazakhstanTask._MECHANISM_BUCKETS:
            for kw in keywords:
                if kw in t:
                    return name
        return "other"

    @staticmethod
    def _render_mechanism_summary(recent: list[dict[str, str]]) -> str:
        """Render a one-line "Recent admissions concentrate on ..." nudge.

        Counts mechanism-family tags across ``recent`` and surfaces the top
        1-2 by share. Empty string when ``recent`` is empty or every entry
        is ``other`` (no actionable summary).
        """
        if not recent:
            return ""
        counts: dict[str, int] = {}
        for entry in recent:
            tag = FeatureHypothesisKazakhstanTask._classify_mechanism(
                entry.get("hypothesis") or ""
            )
            counts[tag] = counts.get(tag, 0) + 1
        ranked = sorted(
            ((c, n) for c, n in counts.items() if c != "other"),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if not ranked:
            return ""
        total = sum(counts.values())
        top = ranked[: 2 if len(ranked) > 1 else 1]
        parts = [
            f"{name.replace('_', ' ')} ({n}/{total})" for name, n in top
        ]
        joined = " and ".join(parts) if len(parts) == 2 else parts[0]
        return f"Recent admissions concentrate on: {joined}."

    def _novelty_block_for(
        self,
        variation: "FeatureHypothesisKazakhstanVariation",
    ) -> str:
        """Compute the novelty-nudge prompt block for a given variation.

        NOT WIRED into any prompt. A 2026-05-31 crossbreed experiment injected
        this block and was reverted the same day: it backfired via
        negation-priming (listing the saturated families primed them —
        geochemical share rose, no diversity gain). An explicit "be a different
        family" instruction is the wrong lever; diversity is meant to emerge
        organically from file rotation.
        Retained (with `_render_novelty_block`, `_render_mechanism_summary`,
        `_classify_mechanism`, `novelty_*` knobs) for analysis use only.

        Returns empty when the knob is disabled, K=0, or no admissions
        exist — callers can prepend the result unconditionally. Block now
        appends a one-line mechanism-family summary so the agent sees
        *which classes* dominate the pool, not just individual examples.
        """
        if not getattr(variation, "novelty_nudge_enabled", False):
            return ""
        k = int(getattr(variation, "novelty_recent_k", 0) or 0)
        if k <= 0:
            return ""
        recent = self._recent_admitted_hypotheses(variation.kg_dir, k)
        if not recent:
            return ""
        max_chars = int(
            getattr(variation, "novelty_max_chars_per_hypothesis", 280) or 280
        )
        block = self._render_novelty_block(recent, max_chars=max_chars)
        summary = self._render_mechanism_summary(recent)
        if summary:
            block = block + "\n\n" + summary
        return block

    def _read_interweave_state(self, kg_dir: Path | str) -> dict[str, Any]:
        data = self._read_json_or(
            Path(kg_dir) / _KG_INTERWEAVE_STATE,
            {"consecutive_failed_crossbreed": 0, "interweave_bootstraps_claimed": 0},
        )
        try:
            data["consecutive_failed_crossbreed"] = max(
                0, int(data.get("consecutive_failed_crossbreed", 0) or 0)
            )
        except (TypeError, ValueError):
            data["consecutive_failed_crossbreed"] = 0
        try:
            data["interweave_bootstraps_claimed"] = max(
                0, int(data.get("interweave_bootstraps_claimed", 0) or 0)
            )
        except (TypeError, ValueError):
            data["interweave_bootstraps_claimed"] = 0
        try:
            data["interweave_bursts_claimed"] = max(
                0, int(data.get("interweave_bursts_claimed", 0) or 0)
            )
        except (TypeError, ValueError):
            data["interweave_bursts_claimed"] = 0
        try:
            data["interweave_survey_remaining"] = max(
                0, int(data.get("interweave_survey_remaining", 0) or 0)
            )
        except (TypeError, ValueError):
            data["interweave_survey_remaining"] = 0
        return data

    def _claim_interweave_bootstrap(
        self,
        kg_dir: Path | str,
        *,
        threshold: int,
        episode_id: str,
        enabled: bool = True,
        burst_episodes: int = 1,
    ) -> bool:
        """Atomically claim survey interweave after a crossbreed plateau.

        The counter resets when a survey burst starts. Each claim consumes one
        survey slot from the burst, including the first threshold-crossing slot.
        """
        if not enabled or threshold <= 0:
            return False
        burst_size = max(1, int(burst_episodes or 1))
        kg_path = Path(kg_dir)
        with self._kg_lock(kg_path):
            state = self._read_interweave_state(kg_path)
            remaining = int(state.get("interweave_survey_remaining", 0))
            if remaining > 0:
                state["interweave_survey_remaining"] = remaining - 1
                state["interweave_bootstraps_claimed"] = (
                    int(state.get("interweave_bootstraps_claimed", 0)) + 1
                )
                state["last_interweave_claimed_at"] = time.time()
                state["last_interweave_reason"] = "active_survey_burst"
                if episode_id:
                    state["last_interweave_episode_id"] = episode_id
                self._atomic_write_json(kg_path / _KG_INTERWEAVE_STATE, state)
                return True
            failures = int(state.get("consecutive_failed_crossbreed", 0))
            if failures < threshold:
                return False
            state["consecutive_failed_crossbreed"] = 0
            state["interweave_survey_remaining"] = burst_size - 1
            state["interweave_bursts_claimed"] = (
                int(state.get("interweave_bursts_claimed", 0)) + 1
            )
            state["interweave_bootstraps_claimed"] = (
                int(state.get("interweave_bootstraps_claimed", 0)) + 1
            )
            state["last_interweave_claimed_at"] = time.time()
            state["last_interweave_reason"] = "crossbreed_plateau"
            if episode_id:
                state["last_interweave_episode_id"] = episode_id
            self._atomic_write_json(kg_path / _KG_INTERWEAVE_STATE, state)
        return True

    def _record_interweave_episode_result(
        self,
        kg_dir: Path | str,
        *,
        workflow_kind: str,
        produced_new_admit: bool,
        interweave_bootstrap: bool = False,
    ) -> dict[str, Any]:
        """Update the failed-crossbreed streak used by survey interweaving."""
        if workflow_kind != "crossbreed" and not interweave_bootstrap:
            return {}
        kg_path = Path(kg_dir)
        with self._kg_lock(kg_path):
            state = self._read_interweave_state(kg_path)
            if workflow_kind == "crossbreed":
                if produced_new_admit:
                    state["consecutive_failed_crossbreed"] = 0
                    state["interweave_survey_remaining"] = 0
                elif int(state.get("interweave_survey_remaining", 0)) > 0:
                    state["consecutive_failed_crossbreed"] = int(
                        state.get("consecutive_failed_crossbreed", 0)
                    )
                else:
                    state["consecutive_failed_crossbreed"] = (
                        int(state.get("consecutive_failed_crossbreed", 0)) + 1
                    )
            elif interweave_bootstrap and produced_new_admit:
                state["consecutive_failed_crossbreed"] = 0
                state["interweave_survey_remaining"] = 0
            state["last_recorded_at"] = time.time()
            state["last_recorded_workflow_kind"] = workflow_kind
            state["last_recorded_produced_new_admit"] = bool(produced_new_admit)
            self._atomic_write_json(kg_path / _KG_INTERWEAVE_STATE, state)
            return dict(state)

    @staticmethod
    def _fingerprint(parent_experiments: list[str] | None, hypothesis: str) -> str:
        """SHA256 over ordered parents + whitespace-normalised hypothesis.

        Order-sensitive: `(A, B)` and `(B, A)` hash differently because the
        agent sees parents in order — the resulting hypotheses are legitimately
        distinct experiments. Hypothesis normalisation stops at whitespace;
        anything stronger (lower-case, stemming) risks false dedups.
        """
        parents = list(parent_experiments or [])
        normalized = re.sub(r"\s+", " ", (hypothesis or "")).strip()
        payload = "|".join(parents) + "::" + normalized
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _support_hash(values: Any) -> str:
        import numpy as np

        support = np.asarray(values).astype(bool)
        return "sha256:" + hashlib.sha256(support.tobytes()).hexdigest()

    @staticmethod
    def _tensor_hash(values: Any) -> str:
        import numpy as np

        rounded = np.round(np.asarray(values, dtype=float), 6)
        return "sha256:" + hashlib.sha256(rounded.tobytes()).hexdigest()

    _PARENTAGE_BASE_THRESHOLD = 0.50
    _PARENTAGE_WEAK_THRESHOLD_BONUS = 0.25
    _ARTIFACT_BACKED_SOURCES: frozenset[str] = frozenset({"artifact", "geonames", "web"})

    @staticmethod
    def _normalise_evidence_tier(value: Any) -> str:
        tier = str(value or "mixed").strip().lower()
        return tier if tier in {"weak", "mixed", "strong"} else "mixed"

    @classmethod
    def _parentage_threshold_for(cls, record: dict[str, Any]) -> float:
        threshold = cls._PARENTAGE_BASE_THRESHOLD
        if cls._normalise_evidence_tier(record.get("proposal_evidence_tier")) == "weak":
            threshold += cls._PARENTAGE_WEAK_THRESHOLD_BONUS
        return threshold

    @classmethod
    def _is_crossbreed_parent_eligible(cls, record: dict[str, Any]) -> bool:
        if not bool(record.get("novelty_guard_passed", True)):
            return False
        if not bool(record.get("provenance_guard_passed", True)):
            return False
        if bool(record.get("declared_nothing", False)):
            return False
        try:
            strength = float(record.get("evidence_strength", 0.0) or 0.0)
        except (TypeError, ValueError):
            strength = 0.0
        return strength >= cls._parentage_threshold_for(record)

    @classmethod
    def _stamp_candidate_triviality(cls, kg_record: dict, *, values: Any) -> None:
        import numpy as np

        candidate = np.asarray(values, dtype=float)
        nonzero_values = candidate[candidate != 0]
        if nonzero_values.size:
            unique_nonzero = np.unique(np.round(nonzero_values, 6))
            value_min = float(np.min(nonzero_values))
            value_max = float(np.max(nonzero_values))
            if unique_nonzero.size <= 1:
                entropy = 0.0
            else:
                _, counts = np.unique(np.round(nonzero_values, 6), return_counts=True)
                probs = counts / counts.sum()
                entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            support = candidate != 0
        else:
            unique_nonzero = np.array([])
            value_min = None
            value_max = None
            entropy = 0.0
            support = np.zeros(candidate.shape, dtype=bool)

        z_levels = int(np.any(support, axis=(0, 1)).sum()) if candidate.ndim == 3 else 0
        depth_levels_filled = bool(candidate.ndim == 3 and z_levels == candidate.shape[2] and z_levels > 0)
        op_count = int(kg_record.get("spatial_operation_provenance_count", 0) or 0)
        nonzero = int(kg_record.get("candidate_nonzero_voxels", np.count_nonzero(candidate)) or 0)
        declared_footprint_size = nonzero if nonzero > 0 else op_count
        declared_nothing = op_count == 0 and nonzero == 0

        kg_record.update({
            "candidate_unique_nonzero_values": int(unique_nonzero.size),
            "candidate_nonzero_value_min": value_min,
            "candidate_nonzero_value_max": value_max,
            "candidate_value_entropy": entropy,
            "single_spatial_operation": op_count == 1,
            "uniform_nonzero_value": bool(nonzero_values.size > 0 and unique_nonzero.size <= 1),
            "depth_levels_filled": depth_levels_filled,
            "declared_footprint_size": int(declared_footprint_size),
            "declared_nothing": declared_nothing,
            "emptiness_rejection_reason": "declared_nothing" if declared_nothing else "none",
        })

    @classmethod
    def _compute_evidence_strength(cls, record: dict[str, Any]) -> float:
        counts = record.get("coordinate_source_counts") or {}
        if not isinstance(counts, dict):
            counts = {}
        op_count = int(record.get("spatial_operation_provenance_count", 0) or 0)
        artifact_count = sum(
            int(counts.get(source, 0) or 0)
            for source in cls._ARTIFACT_BACKED_SOURCES
        )
        artifact_fraction = float(artifact_count / op_count) if op_count else 0.0
        op_score = min(op_count / 3.0, 1.0)
        entropy_score = min(float(record.get("candidate_value_entropy", 0.0) or 0.0), 1.0)
        corroboration_score = min(float(record.get("corroboration_count", 0) or 0) / 3.0, 1.0)
        null_regret_penalty = min(max(float(record.get("null_regret", 0.0) or 0.0), 0.0), 1.0)
        depth_penalty = 1.0 if record.get("depth_levels_filled") and record.get("single_spatial_operation") else 0.0
        strength = (
            0.45 * artifact_fraction
            + 0.25 * op_score
            + 0.20 * entropy_score
            + 0.10 * corroboration_score
            - 0.20 * null_regret_penalty
            - 0.10 * depth_penalty
        )
        strength = max(0.0, min(1.0, strength))
        record["artifact_backed_fraction"] = artifact_fraction
        record["evidence_strength"] = strength
        return strength

    @classmethod
    def _stamp_parentage(cls, kg_record: dict) -> None:
        kg_record.setdefault("corroboration_count", 0)
        kg_record.setdefault("proposal_evidence_tier", "mixed")
        cls._compute_evidence_strength(kg_record)
        parent_eligible = cls._is_crossbreed_parent_eligible(kg_record)
        kg_record["crossbreed_parent_eligible"] = parent_eligible
        kg_record["admission_tier"] = "kg_parent_eligible" if parent_eligible else "kg_evidence"

    def _run_corroboration_promotion(
        self,
        kg_dir: Path,
        kg_record: dict[str, Any],
        *,
        candidate_values: Any,
        admitted_dir: Path | str | None,
    ) -> None:
        """Promote older evidence-only records corroborated by a new admission.

        This intentionally uses a conservative, auditable signal: the new layer
        must be artifact-backed, overlap the older layer's nonzero support, and
        agree with its values better than the older layer's own mean-null.
        """
        if float(kg_record.get("artifact_backed_fraction", 0.0) or 0.0) <= 0.0:
            return
        if admitted_dir is None:
            return

        import numpy as np

        candidate = np.asarray(candidate_values, dtype=float)
        candidate_support = candidate != 0
        if not np.any(candidate_support):
            return

        layers_dir = Path(admitted_dir) / "layers"
        if not layers_dir.exists():
            return

        records_path = kg_dir / _KG_EXPERIMENTS
        records = self._read_jsonl_records(records_path)
        changed = False
        now = time.time()
        current_node = kg_record.get("node_id")

        for record in records:
            if record.get("node_id") == current_node:
                continue
            if record.get("admission_tier") != "kg_evidence":
                continue
            if record.get("crossbreed_parent_eligible") is True:
                continue
            layer_name = record.get("layer_name")
            if not isinstance(layer_name, str) or not layer_name:
                continue
            layer_path = layers_dir / f"{layer_name}.npy"
            if not layer_path.exists():
                continue
            try:
                existing = np.load(layer_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"feature_hypothesis: promotion skipped {layer_path}: {exc}")
                continue
            if existing.shape != candidate.shape:
                continue

            existing = np.asarray(existing, dtype=float)
            overlap = candidate_support & (existing != 0)
            if not np.any(overlap):
                continue

            target = existing[overlap]
            predictor = candidate[overlap]
            mae_null = float(np.mean(np.abs(target - float(np.mean(target)))))
            if mae_null <= 1e-10:
                continue
            relative_mae = float(np.mean(np.abs(target - predictor)) / mae_null)
            if relative_mae >= 1.0:
                continue

            record["corroboration_count"] = int(record.get("corroboration_count", 0) or 0) + 1
            record["last_corroborated_by"] = current_node
            record["last_corroboration_relative_mae"] = relative_mae
            self._compute_evidence_strength(record)
            parent_eligible = self._is_crossbreed_parent_eligible(record)
            record["crossbreed_parent_eligible"] = parent_eligible
            if parent_eligible:
                record["admission_tier"] = "kg_parent_eligible"
                record["promoted_to_parent_at"] = now
            changed = True

        if changed:
            self._atomic_write_jsonl(records_path, records)

    def _rescore_first_layer_auto_roots(
        self,
        kg_dir: Path,
        *,
        admitted_dir: Path | str | None,
    ) -> None:
        if admitted_dir is None:
            return

        try:
            import numpy as np
            from voxel_features.scoring import (
                _bic_with_common_effective_samples,
                _single_layer_null_bic,
                geological_coherence_score,
            )
            from voxel_features.spatial import SpatialVoxelStore
            from voxel_features.store import GridSpec
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"feature_hypothesis: first-layer rescore imports failed: {exc}")
            return

        admitted_path = Path(admitted_dir)
        index_path = admitted_path / "index.json"
        if not index_path.exists():
            return
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            grid = GridSpec.from_dict(index_data["grid"])
            store = SpatialVoxelStore(admitted_path, grid)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"feature_hypothesis: first-layer rescore store open failed: {exc}")
            return

        layer_names = list(store.layer_names)
        if len(layer_names) < 2:
            return

        records_path = kg_dir / _KG_EXPERIMENTS
        records = self._read_jsonl_records(records_path)
        changed = False
        shape = tuple(store.grid.shape)

        for record in records:
            if record.get("admission_path") != "first_layer_auto":
                continue
            if record.get("bic_delta") is not None:
                continue
            layer_name = record.get("layer_name")
            if not isinstance(layer_name, str) or layer_name not in layer_names:
                continue

            try:
                all_values = [store.get_layer_values(name).flatten() for name in layer_names]
                all_dtypes = [store.get_layer(name).dtype for name in layer_names]
                score_with = geological_coherence_score(all_values, all_dtypes, store.grid, shape)
                without_names = [name for name in layer_names if name != layer_name]
                without_values = [store.get_layer_values(name).flatten() for name in without_names]
                without_dtypes = [store.get_layer(name).dtype for name in without_names]
                if len(without_values) == 1:
                    score_without = _single_layer_null_bic(
                        without_values[0], without_dtypes[0], store.grid, shape
                    )
                else:
                    score_without = geological_coherence_score(
                        without_values, without_dtypes, store.grid, shape
                    )
                n_eff_with = int(score_with.get("n_effective_samples", 0) or 0)
                n_eff_without = int(score_without.get("n_effective_samples", 0) or 0)
                comparison_n = min(max(n_eff_with, n_eff_without, 1), 10_000)
                bic_with = _bic_with_common_effective_samples(
                    score_with, len(all_values), comparison_n
                )
                bic_without = _bic_with_common_effective_samples(
                    score_without, len(without_values), comparison_n
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"feature_hypothesis: first-layer rescore skipped {layer_name}: {exc}"
                )
                continue

            bic_delta_raw = float(bic_with - bic_without)
            bic_delta = bic_delta_raw / float(comparison_n)
            record["bic_delta"] = bic_delta
            record["bic_delta_raw"] = bic_delta_raw
            record["bic_comparison_n_effective_samples"] = comparison_n
            record["n_effective_samples"] = n_eff_with
            record["first_layer_rescored_at"] = time.time()
            changed = True

        if changed:
            self._atomic_write_jsonl(records_path, records)

    @classmethod
    def _stamp_candidate_novelty(
        cls,
        kg_record: dict,
        *,
        values: Any,
        admitted_dir: Path | str | None,
    ) -> bool:
        import numpy as np

        candidate = np.asarray(values, dtype=float)
        nonzero = int(np.count_nonzero(candidate))
        total = int(candidate.size)
        operation_support_hash = kg_record.get("spatial_operation_support_hash")
        if nonzero == 0 and isinstance(operation_support_hash, str) and operation_support_hash:
            candidate_support_hash = operation_support_hash
            candidate_tensor_hash = operation_support_hash
        else:
            candidate_support_hash = cls._support_hash(candidate)
            candidate_tensor_hash = cls._tensor_hash(candidate)

        nearest_layer_name = None
        nearest_tensor_distance = None
        nearest_support_match = False
        novelty_rejection_reason = "none"

        layers_dir = Path(admitted_dir) / "layers" if admitted_dir is not None else None
        if layers_dir is not None and layers_dir.exists():
            for admitted_npy in sorted(layers_dir.glob("*.npy")):
                try:
                    admitted = np.load(admitted_npy)
                except Exception as exc:
                    logger.warning(
                        f"feature_hypothesis: novelty scan skipped {admitted_npy}: {exc}"
                    )
                    continue
                if admitted.shape != candidate.shape:
                    continue
                distance = float(np.mean(np.abs(candidate - admitted)))
                support_match = cls._support_hash(admitted) == candidate_support_hash
                tensor_match = cls._tensor_hash(admitted) == candidate_tensor_hash
                if nearest_tensor_distance is None or distance < nearest_tensor_distance:
                    nearest_layer_name = admitted_npy.stem
                    nearest_tensor_distance = distance
                    nearest_support_match = support_match
                if tensor_match:
                    novelty_rejection_reason = "exact_tensor_duplicate"
                    nearest_layer_name = admitted_npy.stem
                    nearest_tensor_distance = distance
                    nearest_support_match = True
                    break
                if support_match and novelty_rejection_reason == "none":
                    novelty_rejection_reason = "support_duplicate"
                    nearest_layer_name = admitted_npy.stem
                    nearest_tensor_distance = distance
                    nearest_support_match = True

        kg_record.update({
            "candidate_support_hash": candidate_support_hash,
            "candidate_tensor_hash": candidate_tensor_hash,
            "support_hash": candidate_support_hash,
            "tensor_hash": candidate_tensor_hash,
            "candidate_nonzero_voxels": nonzero,
            "candidate_fill_fraction": (nonzero / total) if total else 0.0,
            "nearest_layer_name": nearest_layer_name,
            "nearest_tensor_distance": nearest_tensor_distance,
            "nearest_support_match": nearest_support_match,
            "novelty_guard_passed": novelty_rejection_reason == "none",
            "novelty_rejection_reason": novelty_rejection_reason,
        })
        return novelty_rejection_reason == "none"

    @staticmethod
    def _stamp_candidate_provenance(
        kg_record: dict,
        *,
        scratch_dir: Path | str | None,
        layer_name: str | None,
    ) -> bool:
        operations: list[dict[str, Any]] = []
        if scratch_dir is not None and isinstance(layer_name, str) and layer_name:
            spatial_db = Path(scratch_dir) / "spatial.db"
            if spatial_db.exists():
                import sqlite3

                with sqlite3.connect(str(spatial_db)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    try:
                        # Ops are logged under the BASE name the agent passed to spatial_add_*,
                        # but the admitted/scored layer carries a `_<ms-timestamp>` suffix added by
                        # scoring_create_feature_layer. Query both so the guard actually sees the ops.
                        import re as _re

                        _names = [layer_name]
                        _base = _re.sub(r"_\d{10,}$", "", layer_name)
                        if _base and _base != layer_name:
                            _names.append(_base)
                        cursor.execute(
                            """
                            SELECT operation_type, feature_name, coordinates, parameters,
                                   source_file, source_excerpt, coordinate_source
                            FROM spatial_operations
                            WHERE feature_name IN ({placeholders})
                            """.format(placeholders=",".join("?" for _ in _names)),
                            tuple(_names),
                        )
                    except sqlite3.OperationalError:
                        rows = []
                    else:
                        rows = cursor.fetchall()
                    operations = [dict(row) for row in rows]

        source_counts: dict[str, int] = {}
        for op in operations:
            source = str(op.get("coordinate_source") or "creative_fallback")
            source_counts[source] = source_counts.get(source, 0) + 1
        fallback_count = source_counts.get("creative_fallback", 0)
        all_creative_fallback = bool(operations) and fallback_count == len(operations)
        override_enabled = bool(kg_record.get("allow_creative_fallback_admission"))
        guard_passed = (not all_creative_fallback) or override_enabled
        operation_signatures = [
            "|".join(
                str(op.get(key) or "")
                for key in ("operation_type", "feature_name", "coordinates", "parameters")
            )
            for op in operations
        ]
        operation_hash = "sha256:" + hashlib.sha256(
            "\n".join(sorted(operation_signatures)).encode("utf-8")
        ).hexdigest()

        kg_record.update({
            "spatial_operation_provenance_count": len(operations),
            "coordinate_source_counts": source_counts,
            "spatial_operation_support_hash": operation_hash if operations else None,
            "spatial_operation_signatures": operation_signatures,
            "translate_fallback_used": fallback_count > 0,
            "provenance_guard_passed": guard_passed,
            "provenance_rejection_reason": "all_creative_fallback"
            if all_creative_fallback and not override_enabled
            else "none",
        })
        return guard_passed

    def _admit_with_dedup(
        self,
        kg_dir: Path | str,
        kg_record: dict,
        *,
        parents: list[str],
        hypothesis: str,
        scratch_dir: Path | str | None = None,
        admitted_dir: Path | str | None = None,
        layer_name: str | None = None,
    ) -> bool:
        """Append kg_record to experiments.jsonl iff (parents, hypothesis)
        is unseen. Returns True if newly admitted, False on duplicate.

        When ``scratch_dir`` / ``admitted_dir`` / ``layer_name`` are all
        supplied, the candidate's ``.npy`` is *promoted* from scratch into
        the admitted pool atomically inside the kg lock — only if the
        fingerprint is fresh. Duplicates leave the scratch file in place
        (the cleanup hook reclaims it after ``finalize_episode``).

        Duplicates leave the episode's reward intact: duplicates count as
        successes but do not flood the admitted pool.
        """
        kg_path = Path(kg_dir)
        fp = self._fingerprint(parents, hypothesis)
        candidate_values = None
        if (
            scratch_dir is not None
            and isinstance(layer_name, str)
            and layer_name
        ):
            scratch_npy = Path(scratch_dir) / "layers" / f"{layer_name}.npy"
            if scratch_npy.exists():
                import numpy as np

                candidate_values = np.load(scratch_npy)

        on_admit = None
        pre_admit = None
        if candidate_values is not None:
            def check_guards() -> bool:
                if self._allow_creative_fallback_admission:
                    kg_record["allow_creative_fallback_admission"] = True
                provenance_passed = self._stamp_candidate_provenance(
                    kg_record,
                    scratch_dir=scratch_dir,
                    layer_name=layer_name,
                )
                novelty_passed = self._stamp_candidate_novelty(
                    kg_record,
                    values=candidate_values,
                    admitted_dir=admitted_dir,
                )
                self._stamp_candidate_triviality(kg_record, values=candidate_values)
                emptiness_passed = not bool(kg_record.get("declared_nothing", False))
                if not (novelty_passed and provenance_passed and emptiness_passed):
                    kg_record["admission_tier"] = "guard_rejected"
                    kg_record["crossbreed_parent_eligible"] = False
                    return False
                self._stamp_parentage(kg_record)
                return True

            pre_admit = check_guards

        if (
            scratch_dir is not None
            and admitted_dir is not None
            and isinstance(layer_name, str)
            and layer_name
        ):
            def promote_layer() -> None:
                self._promote_scratch_layer(
                    Path(scratch_dir), Path(admitted_dir), layer_name
                )

            on_admit = promote_layer

        admitted = JsonDedupLedger(
            kg_path,
            ledger_filename=_KG_ADMITTED_INDEX,
            records_filename=_KG_EXPERIMENTS,
            lock_filename=_KG_LOCK,
        ).admit(kg_record, fingerprint=fp, pre_admit=pre_admit, on_admit=on_admit)
        if admitted and candidate_values is not None:
            self._run_corroboration_promotion(
                kg_path,
                kg_record,
                candidate_values=candidate_values,
                admitted_dir=admitted_dir,
            )
            self._rescore_first_layer_auto_roots(
                kg_path,
                admitted_dir=admitted_dir,
            )
        return admitted

    @staticmethod
    def _promote_scratch_layer(
        scratch_dir: Path,
        admitted_dir: Path,
        layer_name: str,
    ) -> None:
        """Move ``scratch/layers/<name>.npy`` into ``admitted/layers/`` and
        register it in ``admitted/index.json``. Called inside the kg lock.

        The npy is moved (not copied) so the scratch dir contains no stale
        copies after cleanup. The admitted ``index.json`` is updated via
        ``SpatialVoxelStore`` so the layer metadata format stays consistent
        with the rest of the codebase (e.g. content hashes, dtypes).
        """
        scratch_npy = scratch_dir / "layers" / f"{layer_name}.npy"
        if not scratch_npy.exists():
            logger.warning(
                f"feature_hypothesis: promote skipped — {scratch_npy} missing"
            )
            return

        # Need the scratch store's grid to seed admitted when it doesn't
        # yet exist. SpatialVoxelStore raises on missing index, so read
        # the JSON directly here.
        scratch_index = scratch_dir / "index.json"
        if not scratch_index.exists():
            logger.warning(
                f"feature_hypothesis: promote skipped — {scratch_index} missing"
            )
            return
        try:
            scratch_data = json.loads(scratch_index.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning(
                f"feature_hypothesis: promote skipped — bad scratch index: {exc}"
            )
            return

        from voxel_features.spatial import SpatialVoxelStore
        from voxel_features.store import GridSpec

        grid = GridSpec.from_dict(scratch_data["grid"])

        # Build / open the admitted store and add the layer there. Use
        # the npy values from scratch so the content hash matches.
        import numpy as np

        values = np.load(scratch_npy)
        admitted_dir.mkdir(parents=True, exist_ok=True)
        admitted_store = SpatialVoxelStore(admitted_dir, grid)
        if layer_name in admitted_store.layer_names:
            # Another slot promoted a same-named layer first. Skip the
            # second promotion — the kg ledger still appends the new
            # fingerprint, but the admitted pool keeps the first values.
            scratch_npy.unlink(missing_ok=True)
            return
        scratch_layer_meta = (
            scratch_data.get("layers", {}).get(layer_name, {})
        )
        admitted_store.add_layer(
            name=layer_name,
            values=values,
            dtype=scratch_layer_meta.get("dtype", "float"),
            metadata=scratch_layer_meta.get("metadata", {}),
            hypothesis_uri=scratch_layer_meta.get("hypothesis_uri"),
            experiment_id=scratch_layer_meta.get("experiment_id"),
        )
        scratch_npy.unlink(missing_ok=True)

    # ----- Bootstrap concurrency permit -------------------------------

    @staticmethod
    def _bootstrap_target_active(
        bootstrap_episodes_seen: int,
        configured_slots: int,
        window_size: int,
        min_fraction: float,
    ) -> int:
        """Active-slot target for bootstrap permit acquisition.

        The old bootstrap ramp is obsolete: every configured slot is available
        from episode zero. ``bootstrap_episodes_seen``, ``window_size``, and
        ``min_fraction`` are accepted for legacy callers but do not reduce the
        target.
        """
        return max(0, configured_slots)

    def _read_bootstrap_state(self, kg_dir: Path) -> dict[str, Any]:
        data = self._read_json_or(
            kg_dir / _KG_BOOTSTRAP_STATE,
            {"bootstrap_episodes_seen": 0, "in_flight": []},
        )
        data.setdefault("bootstrap_episodes_seen", 0)
        data.setdefault("in_flight", [])
        return data

    def _acquire_bootstrap_permit(
        self,
        kg_dir: Path | str,
        slot_id: str,
        configured_slots: int,
        window_size: int,
        min_fraction: float,
        timeout_s: float = 600.0,
        stale_after_s: float = 1800.0,
        poll_interval_s: float = 0.5,
    ) -> bool:
        """Block until in-flight slots < target, then claim a permit.

        Returns False on timeout. The caller MUST call
        `_release_bootstrap_permit(slot_id)` once the episode finishes,
        or the in-flight entry will linger until it ages out via
        ``stale_after_s``. The 0.5 s poll interval avoids burning lock
        contention while waiting for a configured-cap slot to free up.
        """
        kg_path = Path(kg_dir)
        state_path = kg_path / _KG_BOOTSTRAP_STATE
        deadline = time.monotonic() + max(timeout_s, 0.0)

        while True:
            with self._kg_lock(kg_path):
                state = self._read_bootstrap_state(kg_path)
                now = time.time()
                raw_in_flight = list(state.get("in_flight", []))
                # Reap stale entries: a crashed slot leaves its permit
                # behind; without this the run deadlocks.
                in_flight = [
                    entry
                    for entry in raw_in_flight
                    if isinstance(entry, dict)
                    and isinstance(entry.get("acquired_at"), (int, float))
                    and (now - float(entry["acquired_at"])) < stale_after_s
                ]
                target = self._bootstrap_target_active(
                    bootstrap_episodes_seen=int(state.get("bootstrap_episodes_seen", 0)),
                    configured_slots=configured_slots,
                    window_size=window_size,
                    min_fraction=min_fraction,
                )
                if len(in_flight) < target:
                    in_flight.append({"slot_id": slot_id, "acquired_at": now})
                    state["in_flight"] = in_flight
                    # `bootstrap_episodes_seen` is retained as a completed-
                    # episode diagnostic counter; it no longer changes the
                    # active-slot target.
                    self._atomic_write_json(state_path, state)
                    return True
                # Persist the reap so other slots benefit when they next
                # acquire — but only if we actually changed anything.
                if in_flight != raw_in_flight:
                    state["in_flight"] = in_flight
                    self._atomic_write_json(state_path, state)

            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_s)

    def _release_bootstrap_permit(self, kg_dir: Path | str, slot_id: str) -> None:
        kg_path = Path(kg_dir)
        state_path = kg_path / _KG_BOOTSTRAP_STATE
        if not state_path.exists():
            return
        with self._kg_lock(kg_path):
            state = self._read_bootstrap_state(kg_path)
            before = state.get("in_flight", [])
            after = [
                entry
                for entry in before
                if not (isinstance(entry, dict) and entry.get("slot_id") == slot_id)
            ]
            state["in_flight"] = after
            # Count completed bootstrap episodes for state/diagnostics. Acquires
            # that never complete (stale, reaped) do not bump the counter.
            if len(after) != len(before):
                state["bootstrap_episodes_seen"] = (
                    int(state.get("bootstrap_episodes_seen", 0)) + 1
                )
            self._atomic_write_json(state_path, state)

    # ----- Crossbreed queue -------------------------------------------

    @classmethod
    def _load_successful_experiments(cls, kg_dir: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in cls._read_jsonl_records(kg_dir / _KG_EXPERIMENTS):
            if rec.get("crossbreed_parent_eligible") is not True:
                continue
            try:
                bic_delta = float(rec.get("bic_delta"))
            except (TypeError, ValueError):
                continue
            if bic_delta < 0:
                out.append(rec)
        return out

    @classmethod
    def _load_distance_index(cls, kg_dir: Path) -> dict[str, float]:
        """Read pairwise_distance.jsonl into an alphabetically-keyed dict.

        Used by `_enumerate_pairs` to look up orthogonality scores by the
        unordered pair id `f"{min}_{max}"`. Returns an empty dict if the
        index file does not yet exist (queue then falls back to distance=0
        and the score becomes purely log1p(|bic|) sums).
        """
        distances: dict[str, float] = {}
        for rec in cls._read_jsonl_records(kg_dir / _KG_PAIRWISE_DISTANCE):
            pair_id = rec.get("pair_id")
            if not isinstance(pair_id, str):
                continue
            try:
                distances[pair_id] = float(rec.get("pairwise_distance", 0.0))
            except (TypeError, ValueError):
                continue
        return distances

    @staticmethod
    def _ordered_pair_id(a: str, b: str) -> str:
        # Pair ID is ORDERED — keeps (A,B) and (B,A) distinct in the queue.
        return f"{a}->{b}"

    def _enumerate_pairs(self, kg_dir: Path) -> list[dict[str, Any]]:
        experiments = self._load_successful_experiments(kg_dir)
        distance_index = self._load_distance_index(kg_dir)
        out: list[dict[str, Any]] = []
        for exp_a in experiments:
            for exp_b in experiments:
                if exp_a["node_id"] == exp_b["node_id"]:
                    continue
                bic_a = abs(float(exp_a.get("bic_delta", 0.0)))
                bic_b = abs(float(exp_b.get("bic_delta", 0.0)))
                # Distance is symmetric and uses the alphabetically-sorted
                # pair id (matches `_update_pairwise_distance_index`).
                dist_pair_id = (
                    f"{min(exp_a['node_id'], exp_b['node_id'])}_"
                    f"{max(exp_a['node_id'], exp_b['node_id'])}"
                )
                distance = distance_index.get(dist_pair_id, 0.0)
                # log1p shrinks BIC outliers (e.g. the |bic|=6.68 fold parent
                # that monopolised the queue under linear scoring); the λ·dist
                # term rewards orthogonal parents.
                score = (
                    math.log1p(bic_a)
                    + math.log1p(bic_b)
                    + _PAIR_DISTANCE_WEIGHT * distance
                )
                out.append({
                    "pair_id": self._ordered_pair_id(exp_a["node_id"], exp_b["node_id"]),
                    "parents": [exp_a["node_id"], exp_b["node_id"]],
                    "score": score,
                    "popped_at": None,
                    "attempt_count": 0,
                })
        out.sort(key=lambda entry: entry["score"], reverse=True)
        return out

    @classmethod
    def _read_queue(cls, kg_dir: Path) -> list[dict[str, Any]]:
        return cls._read_jsonl_records(kg_dir / _KG_QUEUE)

    @classmethod
    def _write_queue(cls, kg_dir: Path, entries: list[dict[str, Any]]) -> None:
        cls._atomic_write_jsonl(kg_dir / _KG_QUEUE, entries)

    @staticmethod
    def _experiments_changed_since_queue(kg_dir: Path) -> bool:
        """True if experiments.jsonl has been modified after the queue was last
        written (or the queue does not yet exist). Lets us skip the O(N²)
        re-enumeration when the experiment set is unchanged."""
        queue_path = kg_dir / _KG_QUEUE
        exp_path = kg_dir / _KG_EXPERIMENTS
        if not queue_path.exists():
            return True
        if not exp_path.exists():
            return False
        return exp_path.stat().st_mtime > queue_path.stat().st_mtime

    def _merge_new_pairs(
        self,
        kg_dir: Path,
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Union existing queue entries with freshly-enumerated pairs.

        Existing entries (including their popped_at state) are preserved;
        new pairs from newly-admitted experiments are inserted. Re-sorted
        by score desc.
        """
        merged = {entry.get("pair_id"): entry for entry in existing}
        for entry in self._enumerate_pairs(kg_dir):
            pid = entry["pair_id"]
            if pid in merged:
                # Refresh score in case MI/BIC moved; preserve popped state.
                merged[pid]["score"] = entry["score"]
            else:
                merged[pid] = entry
        return sorted(merged.values(), key=lambda e: e["score"], reverse=True)

    def _queue_refill(self, kg_dir: Path | str) -> None:
        """Re-enumerate ordered pairs and insert any missing ones into the queue.

        Existing entries (popped or not) are preserved so partial progress
        across episodes isn't lost. New experiments added between calls
        introduce new pairs that take their place in score order.
        """
        kg_path = Path(kg_dir)
        OrderedPairQueue(
            kg_path,
            queue_filename=_KG_QUEUE,
            lock_filename=_KG_LOCK,
        ).refill(lambda existing: self._merge_new_pairs(kg_path, existing))

    def _queue_pop_pair(self, kg_dir: Path | str) -> tuple[str, str] | None:
        """Pop the next pair under the kg lock.

        Selection scores each entry as
            score / (1 + α · attempt_count)              (unconsummated), or
            score · β / (1 + α · attempt_count)          (consummated).
        A pair is *consummated* when any admitted experiment in
        experiments.jsonl already lists its two parents (in either order)
        as parent_node_1, parent_node_2 — i.e. the joint info has been
        captured by at least one descendant.

        The chosen entry's ``attempt_count`` is bumped and ``popped_at``
        stamped (popped_at is retained for telemetry only; the score-decay
        replaces the earlier round-robin reset).

        Returns ``(parent_a, parent_b)`` or None when fewer than two admitted
        experiments exist.
        """
        kg_path = Path(kg_dir)
        queue = OrderedPairQueue(
            kg_path,
            queue_filename=_KG_QUEUE,
            lock_filename=_KG_LOCK,
        )

        def choose_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
            consummated = self._consummated_pairs(kg_path)
            parent_uses = self._parent_use_counts(entries)
            return max(
                entries,
                key=lambda entry: self._effective_pair_score(
                    entry, consummated, parent_uses
                ),
            )

        return queue.pop_pair(
            can_pop=lambda: len(self._load_successful_experiments(kg_path)) >= 2,
            should_refresh=lambda _entries: self._experiments_changed_since_queue(kg_path),
            merge_entries=lambda entries: self._merge_new_pairs(kg_path, entries),
            choose_entry=choose_entry,
        )

    @classmethod
    def _consummated_pairs(cls, kg_dir: Path) -> set[frozenset[str]]:
        """Unordered parent pairs that already have an admitted child.

        Treats (A,B) and (B,A) as the same consummation — the joint
        information has been captured regardless of which ordering produced
        the child.
        """
        out: set[frozenset[str]] = set()
        for rec in cls._read_jsonl_records(kg_dir / _KG_EXPERIMENTS):
            if rec.get("bic_delta", 0) >= 0:
                continue
            p1 = rec.get("parent_node_1")
            p2 = rec.get("parent_node_2")
            if (
                isinstance(p1, str) and p1
                and isinstance(p2, str) and p2
                and p1 != p2
            ):
                out.add(frozenset({p1, p2}))
        return out

    @staticmethod
    def _parent_use_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
        """Sum each parent's attempt_count across all queue rows it appears in.

        Derived on read from existing queue state — no extra persistent
        counter. This is the input to `_effective_pair_score`'s per-parent
        fatigue term (γ), which prevents a single dominant parent (e.g. a
        fold-derived layer) from monopolizing the top-N queue.
        """
        uses: dict[str, int] = {}
        for entry in entries:
            attempts = int(entry.get("attempt_count", 0))
            if attempts == 0:
                continue
            for parent in entry.get("parents") or []:
                if isinstance(parent, str) and parent:
                    uses[parent] = uses.get(parent, 0) + attempts
        return uses

    @staticmethod
    def _effective_pair_score(
        entry: dict[str, Any],
        consummated: set[frozenset[str]],
        parent_uses: dict[str, int] | None = None,
    ) -> float:
        score = float(entry.get("score", 0.0))
        attempts = int(entry.get("attempt_count", 0))
        decayed = score / (1.0 + _PAIR_ATTEMPT_DECAY * attempts)
        parents = entry.get("parents") or []
        if parent_uses:
            for parent in parents:
                if not isinstance(parent, str) or not parent:
                    continue
                uses = parent_uses.get(parent, 0)
                decayed /= 1.0 + _PARENT_USE_DECAY * uses
        if len(parents) == 2 and frozenset(parents) in consummated:
            decayed *= _CONSUMMATED_DISCOUNT
        return decayed
