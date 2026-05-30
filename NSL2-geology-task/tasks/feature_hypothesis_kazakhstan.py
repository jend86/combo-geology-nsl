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
from tasks.common.foundry_exec import coerce_exec_result, exec_run_with_timeout


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
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
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

_ANALYSIS_INPUT = "/workspace/input"
_ANALYSIS_OUT = "/workspace/out"

# Filenames inside `variation.kg_dir`. Centralised so the task code, tests,
# and any external tooling (e.g. analytics scripts) reference one source.
_KG_EXPERIMENTS = "experiments.jsonl"
_KG_CROSSBREED_INDEX = "crossbreed_index.jsonl"          # legacy MI index — read for back-compat only
_KG_PAIRWISE_DISTANCE = "pairwise_distance.jsonl"        # new orthogonality index (Jaccard/MAE)
_KG_ADMITTED_INDEX = "admitted_index.json"
_KG_BOOTSTRAP_STATE = "bootstrap_state.json"
_KG_QUEUE = "crossbreed_queue.jsonl"
_KG_LOCK = "kg.lock"

# Queue selection knobs. The effective score of an entry is
#   score / (1 + α · attempt_count) / Π (1 + γ · uses(parent_i)),
# with an extra · β multiplier when the pair is "consummated" (already has ≥1
# admitted crossbreed child). α decays repeatedly-tried *pairs*; γ decays
# repeatedly-tried *parents* (the lever that breaks the §8 monoculture where
# 32 fold-pairs share the top slope); β puts consummated pairs in a slow lane
# without banning them — LLM hypothesis generation is high-variance and the
# surrounding feature pool keeps growing, so a second attempt at the same
# pair is a genuinely different experiment.
_PAIR_ATTEMPT_DECAY = 0.5
_PARENT_USE_DECAY = 0.01     # γ — chosen conservatively for retroactive
                             # rollout (see redesign §5.5: with 360 historical
                             # uses on the fold parent, γ=0.1 gives a divisor
                             # of 37 and effectively exiles it). 0.01 still
                             # gives ~4.6× on resume — enough to demote fold
                             # below fresh non-fold pairs.
_PAIR_DISTANCE_WEIGHT = 2.0  # λ for the orthogonality term in the score prior
_CONSUMMATED_DISCOUNT = 0.25


# Kazakhstan Teniz Basin grid specification - Regional scale for basin analysis
_KAZAKHSTAN_TENIZ_GRID = {
    "origin": [66.5, 49.5, 0.0],      # 66°30'E, 49°30'N, 0m depth
    "maximum": [71.5, 52.5, 80.0],    # 71°30'E, 52°30'N, 80m depth
    "shape": [200, 200, 8],            # ~1.75km x 1.75km x 10m resolution, 320k total voxels
    "crs": "EPSG:4326",
}


_SYSTEM_PROMPT = """You are in mineral exploration mode for Kazakhstan geological analysis.

Your goal is to identify informative feature layers that would improve compression
of a voxel-based world model. You are rewarded when adding a feature layer improves BIC on a ridge regression of the overall world model.

## Grid

The voxel grid covers the Teniz Basin region of Kazakhstan:
- Longitude: 66.5° to 71.5°E
- Latitude: 49.5° to 52.5°N
- Depth: 0 to 80m
- Resolution: 200 × 200 × 8 voxels (~1.75km × 1.75km × 10m per voxel)

## Scoring

The workflow converts geographic analysis results into 3D feature layers through spatial operations:
1. **Analysis phase** identifies geological patterns in coordinate data
2. **Translation phase** converts findings to spatial commands (spatial_add_point, spatial_add_line)  
3. **Automatic voxel mapping** projects geographic coordinates onto the 200×200×8 grid
4. **BIC evaluation** tests if the new spatial feature improves joint prediction

Feature layers are evaluated by:
- BIC - n*ln(MSE) + k*ln(n) (joint ridge regression across all layers)

A layer is admitted if bic_delta < 0.

## Capabilities

**Phase 1-3 (Hypothesis & Analysis):**
- analysis_shell: Execute Python code in a sandbox with polars/duckdb/scipy
- hypothesis_create: Register a falsifiable hypothesis
- execution_submit/status/results/finalize: Async code execution with budget control

**Phase 4 (Spatial Translation):**
- spatial_add_point: Add point features at geographic coordinates with radius of effect
- spatial_add_line: Add linear features (faults, veins) between two 3D points
- spatial_query_region: Query existing spatial features in a geographic region
- spatial_coord_to_voxel: Convert geographic coordinates to voxel indices
- spatial_get_operations_log: Get history of spatial operations

**Phase 5 (Rewrite):**
- record_phase: Record workflow phase completion
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
    # Upper bound on concurrent bootstrap (= survey) episodes. Set this to
    # your `GenerationConfig.parallel_episodes` to choke bootstrap to the
    # full slot count; set it lower to leave headroom for ramp-up. If the
    # framework has fewer slots than this, the extras simply do nothing.
    bootstrap_concurrency_cap: int = 4
    bootstrap_window_size: int = 8  # episodes over which to ramp N/2 -> N
    bootstrap_min_concurrency_fraction: float = 0.5
    bootstrap_permit_timeout_s: float = 600.0
    bootstrap_permit_stale_after_s: float = 1800.0
    # Novelty nudge: surface the last K admitted hypotheses in the proposer
    # prompt as a "do not propose variants of these" block. Counters the
    # diversity collapse pattern (e.g. 3 unique fingerprints across 245
    # episodes) at the prompt layer, complementing the lexical dedup gate.
    # See docs/design/kazakhstan-variance-and-throughput-2026-05-24.md
    # (Approach B).
    novelty_nudge_enabled: bool = True
    novelty_recent_k: int = 8
    # Per-entry cap so the block stays bounded under long hypotheses; the
    # rendering uses an ellipsis when exceeded.
    novelty_max_chars_per_hypothesis: int = 280


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
    stage_completed: str = "stage_2_completed"
    
    # Crossbreeding context
    parent_experiments: list[str] = field(default_factory=list)

    # Training data
    prompt_response_pair: dict[str, str] = field(default_factory=dict)


class FeatureHypothesisKazakhstanProposerRows:
    """SFT transform: keep proposer-persona turns, drop pure executor turns.

    Twin of :class:`tasks.feature_hypothesis.FeatureHypothesisProposerRows`
    for the Kazakhstan variation. Kept as a sibling class (rather than
    re-using the Australian one) so the recipe hash recorded in
    ``export_recipe.json`` makes the source task unambiguous when SFT data
    from both variants ends up in the same downstream sweep.
    """

    DEFAULT_INCLUDED_WORKFLOW_STEPS: tuple[str, ...] = (
        "survey",
        "hypothesise",
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

        # Check existing features to decide workflow
        n_features = self._count_features(variation)
        workflow_kind = "crossbreed" if (
            variation.crossbreed_enabled and
            n_features >= variation.min_features and
            self._has_crossbreed_pairs(variation)
        ) else "survey"

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

        # Survey (= bootstrap): block here until a slot permit is free so
        # the early generation runs at lower concurrency. The framework
        # still allocates `parallel_episodes` slots; we choke them at the
        # task layer to preserve the pytorch-lightning boundary.
        if (
            workflow_kind == "survey"
            and variation.dedup_enabled
            and variation.bootstrap_window_size > 0
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
            budgets=BudgetConstraints(max_task_tool_calls=100, max_llm_turns=120),
            success=SuccessConstraints(terminal_capability_for_success="submit_rewrite"),
        )

    def training_data_transforms(self) -> tuple[FeatureHypothesisKazakhstanProposerRows, ...]:
        return (FeatureHypothesisKazakhstanProposerRows(),)

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------
    
    def _survey_workflow(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Standard workflow: Survey → Hypothesise → Code → Translate → Evaluate → Rewrite"""

        # Novelty + mechanism-summary nudge is injected at survey (not at
        # hypothesise) so the agent sees the diversity signal *before*
        # choosing which files to open. Both the survey and crossbreed
        # workflows run this survey step, so the nudge lives here for both
        # (see _crossbreed_workflow, which reuses this step verbatim).
        novelty_block = self._novelty_block_for(variation)
        survey_prompt = (
            "Phase 1: Survey\n\n"
            + (novelty_block + "\n\n" if novelty_block else "")
            + "Explore the dataset to identify feature opportunities. You MUST\n"
            "sample at least one source from each of three corpus classes\n"
            "before recording your candidates:\n\n"
            "  - vector   : converted_spatial_data/*.geojson\n"
            "  - tabular  : USGS/TZ_ssCu_Prospects.csv\n"
            "  - text     : USGS/chunks/*.md,\n"
            "               36572_Smolianova_1984/chunks/*.md, or\n"
            "               36572_Smolianova_1984/drill_holes_data/\n"
            "               *.description.md\n\n"
            "Use analysis_shell to read file headers, schemas, value\n"
            "distributions, and 1-2 representative text snippets (head -N\n"
            "of a chunk is enough — full reads are wasteful).\n\n"
            "Find 2-3 promising feature layer candidates.\n\n"
            "Close with:\n"
            "  record_phase(phase='survey', candidates=[...],\n"
            "               corpora_sampled=['vector','tabular','text'])\n"
            "where corpora_sampled lists the corpus classes you actually\n"
            "inspected (subset of ['vector','tabular','text'])."
        )
        hypothesise_prompt = (
            "Phase 2: Hypothesise\n\n"
            "Pick one candidate from the survey phase and state a falsifiable\n"
            "hypothesis.\n\n"
            "Include a data_spec with:\n"
            "- files: list of data sources to analyze\n"
            "- analysis: analytical approach\n"
            "- output: what the output should represent\n\n"
            "Close with:\n"
            "  record_phase(phase='hypothesise', hypothesis=..., data_spec=...)"
        )

        return Workflow(
            steps=(
                # HYPOTHESIS AGENT: Phase 1
                WorkflowStep(
                    name="survey",
                    is_entry=True,
                    prompt=survey_prompt,
                    inherit_all_capabilities=False,
                    capabilities=(
                        "analysis_shell",
                        "record_phase",
                    ),
                    terminator_capabilities=("record_phase",),
                    next_steps=("hypothesise",),
                ),

                # HYPOTHESIS AGENT: Phase 2
                WorkflowStep(
                    name="hypothesise",
                    prompt=hypothesise_prompt,
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
                        "   - Stores results in variables (these become artifacts automatically)\n"
                        "3. Submit code for async execution:\n"
                        "   execution_submit(code='your_code_here', timeout_s=300)\n\n"
                        "4. Monitor execution progress:\n"
                        "   execution_status(execution_id='...')  # Check status/progress\n"
                        "   execution_status(execution_id='...')  # Keep checking until 'completed'\n\n"
                        "5. Get results and validate artifacts:\n"
                        "   execution_results(execution_id='...')  # Get artifacts + output\n\n"
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
                        "2. Generate spatial commands based on analysis findings:\n"
                        "   Grid bounds: lon 66.5°-71.5°E, lat 49.5°-52.5°N, depth 0-80m\n"
                        "   Resolution: ~1.75km × 1.75km × 10m per voxel (200×200×8 total)\n\n"
                        "   **For drill hole data with coordinates:**\n"
                        "   spatial_add_point(name='string', longitude=float, latitude=float, depth_m=float, value=float, radius_m=float)\n\n"
                        "   **For geological structures (faults, veins):**\n"
                        "   spatial_add_line(name='string', start_longitude=float, start_latitude=float, start_depth_m=float, end_longitude=float, end_latitude=float, end_depth_m=float, value=float, width_m=float)\n\n"
                        "   **For statistical results without coordinates:**\n"
                        "   - Use geological knowledge: 'near Emily Well' → find well coordinates\n"
                        "   - Create spatial patterns: 'high copper zone' → center of drill holes\n"
                        "3. -Create exactly ONE coherent feature layer:\n"
                        "   - ALL spatial operations must use the SAME layer name\n"
                        " - Values must be floats or booleans: 'clay' → has_clay=True\n", 
                        "   - Example: spatial_add_point(name='mineralization_potential', ...) \n"
                        "            spatial_add_line(name='mineralization_potential', ...) \n"
                        "4. Validate coordinates using spatial_coord_to_voxel() to check grid bounds\n\n"
                        "5. **MANDATORY TO COMPLETE THIS PHASE**:\n"
                        "   🚨 When you are done YOU MUST CALL scoring_create_feature_layer(name='your_layer_name') 🚨\n"
                        "   Example workflow:\n"
                        "   1. spatial_add_point(name='name', ...)\n"
                        "   2. spatial_add_line(name='name', ...)\n"
                        "   3. scoring_create_feature_layer(name='name')  ← REQUIRED!\n"
                        "   \n"
                        "Focus on geological intelligence, not array mathematics!"
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
        """Crossbreed workflow: same Survey → Hypothesise → Code → Translate →
        Rewrite chain as the standard workflow, but the hypothesise step combines
        parent experiments instead of picking a fresh survey candidate.

        The survey step is retained (and stays the entry) so crossbreed episodes
        still ground themselves in the source files before hypothesising — survey
        is supposed to happen even in crossbreed mode. Only the hypothesise step's
        prompt differs; the novelty nudge stays on the survey step (built by
        _survey_workflow), so it is NOT re-injected here.
        """

        crossbreed_ctx = episode_context.get("crossbreed_context", {})
        parent_ids = crossbreed_ctx.get("parent_ids", [])

        base_workflow = self._survey_workflow(variation, episode_context)

        crossbreed_prompt = (
            "Phase 2: Hypothesise (Crossbreed Mode)\n\n"
            f"Parent experiments: {parent_ids}\n\n"
            f"{crossbreed_ctx.get('prompt', '')}\n\n"
            "You have just surveyed the dataset. Building on the parent findings\n"
            "above together with what you observed in the data, propose a\n"
            "hypothesis that combines or extends them.\n\n"
            "Include a data_spec as before.\n\n"
            "Close with:\n"
            "  record_phase(phase='hypothesise', hypothesis=..., data_spec=..., "
            f"parent_experiments={parent_ids})"
        )

        # Keep the full Survey → … → Rewrite chain; only swap the hypothesise
        # step's prompt for the crossbreed variant. Survey remains is_entry, so
        # the crossbreed hypothesise step is a normal (non-entry) successor.
        crossbreed_hypothesise = WorkflowStep(
            name="hypothesise",
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
            crossbreed_hypothesise if s.name == "hypothesise" else s
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
        phase_records[phase] = {
            "candidates": args.get("candidates"),
            "corpora_sampled": args.get("corpora_sampled"),
            "hypothesis": args.get("hypothesis"),
            "data_spec": args.get("data_spec"),
            "feature_layer_name": args.get("feature_layer_name"),
            "parent_experiments": args.get("parent_experiments"),
            "timestamp": time.time(),
        }
        
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
        
        # Auto-enhance data_spec for coding phase
        if phase == "hypothesise" and "data_spec" in output:
            output["data_spec"] = self._enhance_data_spec(output["data_spec"])
        
        return CapabilityResult(
            "phase_get",
            output=output,
            success=True,
        )
    
    def _enhance_data_spec(self, data_spec: dict[str, Any]) -> dict[str, Any]:
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
        
        return CapabilityResult(
            "get_experiment_summary",
            output={
                "hypothesis": hypothesise.get("hypothesis", ""),
                "data_spec": hypothesise.get("data_spec", {}),
                "code_executed": code.get("code_executed", ""),
                "result_summary": code.get("result_summary", ""),
                "artifact_directory": code.get("artifact_directory", ""),
                "artifact_files": code.get("artifact_files", []),
                "feature_layer_name": translate.get("feature_layer_name", ""),
                "dtype": translate.get("dtype", "float"),
                "bic_delta": evaluate.get("bic_delta"),
                "admitted": evaluate.get("admitted", False),
                "mutual_info": evaluate.get("mutual_info", {}),
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
        
        # Create artifact directory
        episode_id = ctx.episode_context.get("episode_id", "unknown")
        artifact_dir = f"/tmp/artifacts/{episode_id}"
        
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
artifact_dir = "''' + artifact_dir + '''"
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
            
            # Extract artifact information from stdout
            artifact_files = []
            if "ARTIFACTS_SAVED:" in stdout:
                import re
                artifacts_match = re.search(r"ARTIFACTS_SAVED: \[(.*?)\]", stdout)
                if artifacts_match:
                    artifacts_str = artifacts_match.group(1)
                    # Parse the list of file paths
                    artifact_files = [f.strip().strip("'\"") for f in artifacts_str.split(",") if f.strip()]
            
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
        stage_completed = evaluate.get('stage_completed', 'stage_2_completed')
        
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
            'stage_completed': stage_completed,
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
                    'stage_completed': stage_completed,
                    'stage_1_threshold': 0.0001,  # Lowered for sparse geological data
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
                    "stage_completed": stage_completed,
                    "scoring_version": "two_stage_v2",
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
                duplicate_rejected = not admitted_to_kg
                if admitted_to_kg:
                    self._update_crossbreed_index(
                        knowledge_dir, node_id, evaluate.get('mutual_info', {})
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
                elif 'layer_name' in locals() and locals()['layer_name'] in existing_layer:
                    mi_score = existing_layer[locals()['layer_name']]

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
            # every other slot's. See
            # docs/design/feature_hypothesis_voxel_store_isolation.md.
            scratch_dir = Path(store_dir) / "scratch" / episode_id
            admitted_dir = Path(store_dir) / "admitted"
            admitted_dir.mkdir(parents=True, exist_ok=True)

            print("🔧 DEBUG: Creating SpatialVoxelStore...")
            store = SpatialVoxelStore(
                scratch_dir, grid, read_only_overlay=admitted_dir
            )
            print(f"🔧 DEBUG: ✅ Store created, grid shape: {store.grid.shape}")
            print(f"🔧 DEBUG: Grid bounds: lon {store.grid.origin[0]:.3f}-{store.grid.maximum[0]:.3f}, lat {store.grid.origin[1]:.3f}-{store.grid.maximum[1]:.3f}, depth {store.grid.origin[2]:.1f}-{store.grid.maximum[2]:.1f}")
            
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
                
                # Update translate phase record
                layer_name = args.get("name", "")
                if layer_name:
                    translate_record = phase_records.setdefault("translate", {})
                    translate_record["feature_layer_name"] = layer_name
                    translate_record["timestamp"] = __import__('time').time()
                
                # Store evaluation results
                phase_records["evaluate"] = result
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
            stage_completed=evaluate.get("stage_completed", "stage_2_completed"),
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
        admitted = final.admitted
        stage_completed = final.stage_completed

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
            if masking_test_direction in ("auto_pass", "first_layer"):
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
                    "stage_2_passed": True,
                    "bic_delta": bic_delta,
                    "stage1_reward": stage1_reward,
                    "stage2_reward": stage2_reward,
                    "final_reward": value,
                    "both_stages_passed": True,
                    "region": "kazakhstan",
                },
            )
        elif masking_test_passed and not admitted:
            if masking_test_direction in ("auto_pass", "first_layer"):
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
                    "stage_2_passed": False,
                    "bic_delta": bic_delta,
                    "stage1_reward": stage1_reward,
                    "partial_success": True,
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
                    "stage_2_passed": admitted,
                    "bic_delta": bic_delta,
                    "no_predictive_value": True,
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
            breakdown["bootstrap_active"] = final.workflow_kind == "survey"
            if episode_context.get("duplicate_rejected"):
                breakdown["duplicate_rejected"] = True
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
        experiments_file = Path(variation.kg_dir) / _KG_EXPERIMENTS
        if not experiments_file.exists():
            return False
        try:
            # Count successful experiments in JSONL format
            admitted_count = 0
            with open(experiments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            exp = json.loads(line)
                            if exp.get("bic_delta", 0) < 0:  # Successful experiments only
                                admitted_count += 1
                        except json.JSONDecodeError:
                            continue
            return admitted_count >= 2
        except Exception:
            return False
    
    def _get_crossbreed_context(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
    ) -> dict[str, Any]:
        """Get crossbreed prompt and parent IDs using JSONL knowledge graph."""
        import json
        
        try:
            experiments_file = Path(variation.kg_dir) / _KG_EXPERIMENTS
            crossbreed_file = Path(variation.kg_dir) / _KG_CROSSBREED_INDEX

            if not experiments_file.exists():
                return {}
            
            # Load all successful experiments
            experiments = []
            with open(experiments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            exp = json.loads(line)
                            if exp.get("bic_delta", 0) < 0:  # Only successful experiments
                                experiments.append(exp)
                        except json.JSONDecodeError:
                            continue
            
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
        experiments_file = Path(variation.kg_dir) / _KG_EXPERIMENTS
        if not experiments_file.exists():
            return {}
        
        try:
            # Load first two successful experiments from JSONL
            experiments = []
            with open(experiments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            exp = json.loads(line)
                            if exp.get("bic_delta", 0) < 0:  # Only successful experiments
                                experiments.append(exp)
                                if len(experiments) >= 2:
                                    break  # Just need first two
                        except json.JSONDecodeError:
                            continue
            
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

    @classmethod
    def _should_persist_to_kg(
        cls,
        *,
        masking_test_passed: bool,
        admitted: bool,
        bic_delta: float | None,
        stage_completed: str,
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
        if bic_delta is None or bic_delta >= 0:
            return False
        return stage_completed in cls._STAGE_COMPLETED_ALLOWLIST

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

        Duplicates leave the episode's reward intact — the design intent
        is "duplicates count as successes but do not flood the pool".
        """
        kg_path = Path(kg_dir)
        fp = self._fingerprint(parents, hypothesis)

        with self._kg_lock(kg_path):
            ledger = self._read_json_or(kg_path / _KG_ADMITTED_INDEX, {"fingerprints": []})
            seen: list[str] = list(ledger.get("fingerprints", []))
            if fp in seen:
                return False
            if (
                scratch_dir is not None
                and admitted_dir is not None
                and isinstance(layer_name, str)
                and layer_name
            ):
                self._promote_scratch_layer(
                    Path(scratch_dir), Path(admitted_dir), layer_name
                )
            with (kg_path / _KG_EXPERIMENTS).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(kg_record) + "\n")
            seen.append(fp)
            self._atomic_write_json(kg_path / _KG_ADMITTED_INDEX, {"fingerprints": seen})
        return True

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
        from voxel_features.spatial import SpatialVoxelStore
        from voxel_features.store import GridSpec

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

    # ----- Bootstrap concurrency ramp ---------------------------------

    @staticmethod
    def _bootstrap_target_active(
        bootstrap_episodes_seen: int,
        configured_slots: int,
        window_size: int,
        min_fraction: float,
    ) -> int:
        """Active-slot target as the bootstrap window progresses.

        progress = seen / window  ∈ [0, 1]
        fraction = min_fraction + (1 - min_fraction) * progress  ∈ [min, 1]
        target   = ceil(configured_slots * fraction)

        window_size <= 0 means "no ramp" — every slot is allowed.
        """
        if window_size <= 0 or configured_slots <= 0:
            return max(0, configured_slots)
        progress = min(max(bootstrap_episodes_seen / window_size, 0.0), 1.0)
        fraction = min_fraction + (1.0 - min_fraction) * progress
        return max(1, math.ceil(configured_slots * fraction))

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
        ``stale_after_s``. The 0.5 s poll interval is deliberate: the ramp
        unit is "episodes", not subseconds, so faster polling just burns
        lock contention with no scheduling benefit.
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
                    # `bootstrap_episodes_seen` increments on RELEASE, not
                    # acquire — otherwise the ramp accelerates with raw
                    # parallelism and lets configured concurrency in before
                    # any episode has actually completed.
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
            # Each *completed* episode (release) advances the ramp. Acquires
            # that never complete (stale, reaped) do not bump the counter.
            if len(after) != len(before):
                state["bootstrap_episodes_seen"] = (
                    int(state.get("bootstrap_episodes_seen", 0)) + 1
                )
            self._atomic_write_json(state_path, state)

    # ----- Crossbreed queue -------------------------------------------

    @classmethod
    def _load_successful_experiments(cls, kg_dir: Path) -> list[dict[str, Any]]:
        return [
            rec
            for rec in cls._read_jsonl_records(kg_dir / _KG_EXPERIMENTS)
            if rec.get("bic_delta", 0) < 0
        ]

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
                # term rewards orthogonal parents. See redesign §2.2 and §6.
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
        with self._kg_lock(kg_path):
            self._write_queue(kg_path, self._merge_new_pairs(kg_path, self._read_queue(kg_path)))

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
        # Cheap pre-check outside the lock — if there are <2 experiments,
        # there is no way to pop. Inside the lock we may still re-check after
        # acquiring the queue, but this avoids the lock overhead in the
        # common steady-state survey-only case.
        if len(self._load_successful_experiments(kg_path)) < 2:
            return None

        with self._kg_lock(kg_path):
            entries = self._read_queue(kg_path)
            # Only re-enumerate (the O(N²) hot path) when experiments.jsonl
            # has grown since the last queue write. In the steady state this
            # skips the enumeration entirely.
            if not entries or self._experiments_changed_since_queue(kg_path):
                entries = self._merge_new_pairs(kg_path, entries)

            if not entries:
                return None

            consummated = self._consummated_pairs(kg_path)
            parent_uses = self._parent_use_counts(entries)
            chosen = max(
                entries,
                key=lambda entry: self._effective_pair_score(
                    entry, consummated, parent_uses
                ),
            )

            chosen["attempt_count"] = int(chosen.get("attempt_count", 0)) + 1
            chosen["popped_at"] = time.time()
            self._write_queue(kg_path, entries)

            parents = chosen.get("parents") or []
            if len(parents) != 2:
                return None
            return str(parents[0]), str(parents[1])

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
        fatigue term (γ), which breaks the §8 monoculture where a single
        dominant parent (e.g. fold) ends up in 32 of the top-N pairs.
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

