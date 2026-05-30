"""Feature hypothesis task.

Agents explore geological datasets, hypothesize about informative feature layers,
write code to test hypotheses, and have features evaluated via BIC on ridge CV.

Architecture:
- Hypothesis Agent: Survey → Hypothesise → (wait) → Translate
- Coding Agent: Code (stateless, isolated from raw data)
- Framework: Evaluate (automated BIC/MI scoring)
- Rewriting Agent: Rewrite (creates training pairs and knowledge graph nodes)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# Coe Fairbairn grid specification - High resolution for spatial features
_COE_FAIRBAIRN_GRID = {
    "origin": [117.832397, -27.441096, 0.0],
    "maximum": [117.973493, -27.300000, 80.0],
    "shape": [200, 200, 8],  # ~70m x 79m x 10m resolution, 320k total voxels
    "crs": "EPSG:4326",
}


_SYSTEM_PROMPT = """You are in mineral exploration mode.

Your goal is to identify informative feature layers that would improve compression 
of a voxel-based world model. You are rewarded when adding a feature layer improves BIC on a ridge regression of the overall world model.

## Grid

The voxel grid covers:
- Longitude: 117.832° to 117.973°
- Latitude: -27.441° to -27.300°
- Depth: 0 to 80m
- Resolution: 200 × 200 × 8 voxels (~70m × 79m × 10m per voxel)

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


_DATASET_OVERVIEW = """## Coe Fairbairn Dataset Overview

- geochemDrillhole.csv: 3D drillhole samples with 80+ element assays (Au, Cu, etc.) and 3D coordinates
- geochemSurface.csv: Surface samples with coodinates
- other csvs: Mining tenement boundaries and history
- description.md files: detailed descriptions of maps
- WAMEX reports: OCR'd exploration reports (JSON chunks)
"""


# Ordered list of distinct source files/groups for round-robin episode assignment.
# Each episode is assigned the least-explored entry so agents are forced to
# derive hypotheses from different data sources rather than free-roaming and
# fixating on whatever the context history primes them toward.
_COE_FAIRBAIRN_SOURCE_FILES = [
    {
        "key": "drillhole",
        "path": "amalgamated_csvs/geochemDrillhole.csv",
        "description": (
            "3D drillhole assay data — 80+ element columns (Au, Cu, Pb, Zn, …), "
            "longitude, latitude, maxdepth_drill. Primary source for subsurface geochemistry."
        ),
    },
    {
        "key": "surface",
        "path": "amalgamated_csvs/geochemSurface.csv",
        "description": (
            "Surface geochemistry samples — element assays at surface coordinates. "
            "Useful for mapping near-surface anomalies and gossans."
        ),
    },
    {
        "key": "tenements",
        "path": "amalgamated_csvs/ (tenement / lease boundary CSVs)",
        "description": (
            "Mining tenement boundaries, lease history, and tenure data. "
            "Spatial polygons and metadata for the exploration area."
        ),
    },
    {
        "key": "description_maps",
        "path": "description.md files",
        "description": (
            "Detailed natural-language descriptions of geological maps — "
            "lithology, structure, alteration zones, interpreted contacts."
        ),
    },
    {
        "key": "wamex_reports",
        "path": "WAMEX/ (OCR'd exploration report JSON chunks)",
        "description": (
            "Scanned WAMEX exploration reports chunked as JSON. "
            "Contains historical assay tables, geological logs, and interpretations."
        ),
    },
]


@dataclass
class FeatureHypothesisVariation(Variation):
    """Variation configuration for feature hypothesis task."""
    
    dataset_dir: str = ""
    store_dir: str = ""
    kg_dir: str = ""
    grid_spec: dict[str, Any] = field(default_factory=lambda: dict(_COE_FAIRBAIRN_GRID))
    min_features: int = 0  # minimum features before crossbreeding
    crossbreed_enabled: bool = True


@dataclass
class FeatureHypothesisState:
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


class FeatureHypothesisTask(TaskSpec[FeatureHypothesisState]):
    """Feature hypothesis discovery task."""
    
    name = "feature-hypothesis"
    description = "Discover informative feature layers through hypothesis-driven exploration."
    metric_name = "bic_improvement"
    metric_unit = "nats"
    higher_is_better = False  # Lower BIC is better
    agent_service_name = "agent"
    
    def __init__(self, task_config: dict[str, Any]) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        
        # Dataset paths
        default_dataset = repo_root.parent / "Coe Fairbairn"
        self._dataset_dir = Path(task_config.get("dataset_dir", default_dataset)).resolve()
        
        # Store paths - Australia regional structure
        default_store = repo_root / "data" / "australia" / "feature-hypothesis" / "store"
        self._store_dir = Path(task_config.get("store_dir", default_store)).resolve()
        
        default_kg = repo_root / "data" / "australia" / "feature-hypothesis" / "knowledge"
        self._kg_dir = Path(task_config.get("kg_dir", default_kg)).resolve()
        
        self._docker_compose_dir = task_config.get(
            "docker_compose_dir", "docker/feature-hypothesis-compose"
        )

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
            FeatureHypothesisVariation(
                name="coe_fairbairn",
                description="Coe Fairbairn goldfield dataset - discover features.",
                dataset_dir=str(self._dataset_dir),
                store_dir=str(self._store_dir / "coe_fairbairn"),
                kg_dir=str(self._kg_dir / "coe_fairbairn"),
                grid_spec=dict(_COE_FAIRBAIRN_GRID),
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
        if not isinstance(variation, FeatureHypothesisVariation):
            raise TypeError("FeatureHypothesisTask requires FeatureHypothesisVariation")
        
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
        
        # If crossbreeding, add context
        if workflow_kind == "crossbreed":
            episode_context["crossbreed_context"] = self._get_crossbreed_context(variation)
        
        # For survey episodes, assign a specific source file to explore.
        # Picks the least-explored source so coverage spreads evenly across
        # the dataset rather than letting the agent fixate on whatever the
        # context history happens to prime it toward.
        if workflow_kind == "survey":
            rotation = self._pick_assigned_source(variation.kg_dir, _COE_FAIRBAIRN_SOURCE_FILES)
            episode_context["assigned_source"] = rotation["source"]
            episode_context["source_coverage"] = rotation["all_counts"]
        
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
        assert isinstance(variation, FeatureHypothesisVariation)
        
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
        assert isinstance(variation, FeatureHypothesisVariation)
        
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
    
    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------
    
    def _survey_workflow(
        self,
        variation: FeatureHypothesisVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Standard workflow: Survey → Hypothesise → Code → Translate → Evaluate → Rewrite"""
        
        # Generate dynamic survey prompt with file assignment for this episode
        survey_prompt = self._generate_survey_prompt_with_context(episode_context)
        
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
                    prompt=(
                        "Phase 2: Hypothesise\n\n"
                        "Pick one candidate and state a falsifiable hypothesis.\n\n"
                        "Include a data_spec with:\n"
                        "- files: list of data sources to analyze\n"
                        "- analysis: analytical approach\n"
                        "- output: what the output should represent\n\n"
                        "Close with:\n"
                        "  record_phase(phase='hypothesise', hypothesis=..., data_spec=...)"
                    ),
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
                        "   Grid bounds: lon 117.832°-117.973°, lat -27.441°--27.300°, depth 0-80m\n"
                        "   Resolution: ~70m × 79m × 10m per voxel (200×200×8 total)\n\n"
                        "   **For drill hole data with coordinates:**\n"
                        "   spatial_add_point(name='string', longitude=float, latitude=float, depth_m=float, value=float, radius_m=float)\n\n"
                        "   **For geological structures (faults, veins):**\n"
                        "   spatial_add_line(name='string', start_longitude=float, start_latitude=float, start_depth_m=float, end_longitude=float, end_latitude=float, end_depth_m=float, value=float, width_m=float)\n\n"
                        "   **For statistical results without coordinates:**\n"
                        "   - Use geological knowledge: 'near Emily Well' → find well coordinates\n"
                        "   - Create spatial patterns: 'high copper zone' → center of drill holes\n"
                        "3. -Create exactly ONE coherent feature layer:\n"
                        "   - ALL spatial operations must use the SAME layer name\n"
                        " - Values must be floats or booleans: 'clay' → has_clay=True\n"
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
        variation: FeatureHypothesisVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Crossbreed workflow: starts with crossbreed prompt instead of survey."""
        
        # Same as survey but skip survey phase
        crossbreed_ctx = episode_context.get("crossbreed_context", {})
        parent_ids = crossbreed_ctx.get("parent_ids", [])
        
        base_workflow = self._survey_workflow(variation, episode_context)
        
        # Build replacement steps tuple: swap survey for crossbreed hypothesise as entry
        crossbreed_hypothesise = WorkflowStep(
            name="hypothesise",
            is_entry=True,
            prompt=(
                "Phase 2: Hypothesise (Crossbreed Mode)\n\n"
                f"Parent experiments: {parent_ids}\n\n"
                f"{crossbreed_ctx.get('prompt', '')}\n\n"
                "Propose a hypothesis that combines or builds on these findings.\n\n"
                "Include a data_spec as before.\n\n"
                "Close with:\n"
                "  record_phase(phase='hypothesise', hypothesis=..., data_spec=..., "
                f"parent_experiments={parent_ids})"
            ),
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
            if s.name != "survey"
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
        """Enhance data_spec with file-type-specific technical guidance for coding agent."""
        import os
        enhanced = data_spec.copy()
        files = enhanced.get("files", [])
        
        # Add file-specific guidance
        file_specs = []
        for file_path in files:
            if file_path.endswith('.csv'):
                if 'geochemDrillhole' in file_path:
                    filename = os.path.basename(file_path)  # Extract just filename from any path
                    full_path = f"/workspace/input/amalgamated_csvs/{filename}"
                    file_specs.append({
                        "file": file_path,
                        "full_path": full_path,
                        "type": "csv",
                        "columns": ["longitude", "latitude", "maxdepth_drill", "au_ppm", "cu_ppm", "ag_ppm", "as_ppm", "al_ppm", "ba_ppm", "ca_ppm", "co_ppm", "cr_ppm", "fe_ppm", "k_ppm", "mg_ppm", "mn_ppm", "na_ppm", "ni_ppm", "pb_ppm", "s_ppm", "ti_ppm", "zn_ppm"],
                        "spatial_cols": ["longitude", "latitude", "maxdepth_drill"],
                        "note": "3D drillhole geochemistry data - use pd.read_csv(full_path)"
                    })
                elif 'geochemSurface' in file_path:
                    filename = os.path.basename(file_path)  # Extract just filename from any path
                    full_path = f"/workspace/input/amalgamated_csvs/{filename}"
                    file_specs.append({
                        "file": file_path,
                        "full_path": full_path,
                        "type": "csv", 
                        "columns": ["longitude", "latitude", "au_ppm", "cu_ppm", "ag_ppm", "as_ppm"],
                        "spatial_cols": ["longitude", "latitude"],
                        "note": "2D surface geochemistry data - use pd.read_csv(full_path)"
                    })
                elif 'tenements' in file_path:
                    filename = os.path.basename(file_path)  # Extract just filename from any path  
                    full_path = f"/workspace/input/amalgamated_csvs/{filename}"
                    file_specs.append({
                        "file": file_path,
                        "full_path": full_path,
                        "type": "csv",
                        "columns": ["tenement", "wkt_geometry", "longitude", "latitude"],
                        "spatial_cols": ["longitude", "latitude"],
                        "note": "Tenement boundaries and ownership - use pd.read_csv(full_path)"
                    })
                else:
                    file_specs.append({"file": file_path, "type": "csv", "note": "Check columns with df.columns"})
            elif file_path.endswith('.json') or 'wamex' in file_path.lower():
                file_specs.append({
                    "file": file_path,
                    "type": "json",
                    "fields": ["target_commodity_wamex", "abstract_wamex", "project_wamex"],
                    "note": "Use keyword search for geological terms, commodities"
                })
            elif file_path.endswith('.md'):
                file_specs.append({
                    "file": file_path,
                    "type": "markdown",
                    "method": "grep/search",
                    "note": "Use grep for keywords, never read full file"
                })
            elif file_path.endswith('.geojson'):
                file_specs.append({
                    "file": file_path,
                    "type": "geojson",
                    "geometry": "polygons/points",
                    "properties": "varies by file",
                    "note": "Spatial data with attributes"
                })
        
        enhanced["file_specs"] = file_specs
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
            data_base_path = Path(store_dir).parent.parent  # from store/coe_fairbairn to data/australia/feature-hypothesis
        else:
            data_base_path = Path("/home/jen/Desktop/geonsl/NSL2-geology-task/data/australia/feature-hypothesis")
        
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
                    'scoring_version': 'two_stage_v1'
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
        # Stage 1: Must pass predictive capacity test
        # Stage 2: Must improve BIC (complexity assessment)
        both_stages_passed = (
            masking_test_passed and 
            admitted and 
            bic_delta is not None and 
            bic_delta < 0 and
            stage_completed == 'stage_2_completed'
        )
        
        if both_stages_passed:
            try:
                knowledge_dir = data_base_path / "knowledge" / "coe_fairbairn"
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                experiments_file = knowledge_dir / "experiments.jsonl"
                crossbreed_file = knowledge_dir / "crossbreed_index.jsonl"
                
                # Generate node ID
                node_id = f"exp_{episode_id}" if episode_id else f"exp_{int(time.time())}"
                
                # Get parent experiment IDs if this is a crossbreed episode
                parent_experiments = hypothesise.get("parent_experiments", [])
                parent_node_1 = parent_experiments[0] if len(parent_experiments) > 0 else None
                parent_node_2 = parent_experiments[1] if len(parent_experiments) > 1 else None
                
                # Create knowledge graph record (only for experiments that passed both stages)
                kg_record = {
                    "node_id": node_id,
                    "prompt": training_pair.get('prompt', ''),
                    "response": training_pair.get('response', ''),
                    "bic_delta": bic_delta,
                    # Two-stage scoring results
                    "masking_test_passed": masking_test_passed,
                    "masking_test_improvement": masking_test_improvement,
                    "masking_test_direction": masking_test_direction,
                    "stage_completed": stage_completed,
                    "scoring_version": "two_stage_v1",
                    "artifact_links": {
                        "layer_file": f"store/coe_fairbairn/layers/{translate.get('feature_layer_name', '')}.npy" if translate.get('feature_layer_name') else None,
                        "spatial_ops": f"store/coe_fairbairn/spatial.db:experiment_{episode_id}" if episode_id else None
                    },
                    "parent_node_1": parent_node_1,
                    "parent_node_2": parent_node_2,
                    "timestamp": datetime.now().isoformat(),
                    "mutual_info": evaluate.get('mutual_info', {}),
                    "layer_name": translate.get('feature_layer_name', ''),
                    "hypothesis": hypothesise.get('hypothesis', '')
                }
                
                # Append to experiments.jsonl
                with open(experiments_file, 'a') as f:
                    f.write(json.dumps(kg_record) + '\n')
                
                # Calculate mutual information with existing experiments and update crossbreed index
                self._update_crossbreed_index(knowledge_dir, node_id, translate.get('feature_layer_name', ''), evaluate.get('mutual_info', {}))
                
            except Exception as e:
                print(f"Warning: Failed to save knowledge graph data: {e}")

        ctx.episode_context["terminal_record"] = {
            "graph_node": graph_node,
            "training_pair": training_pair,
            "timestamp": time.time(),
        }

        return CapabilityResult(
            "submit_rewrite",
            output={
                "recorded": True, 
                "training_saved": True, 
                "knowledge_saved": both_stages_passed,
                "two_stage_results": {
                    "stage_1_passed": masking_test_passed,
                    "stage_1_improvement": masking_test_improvement,
                    "stage_2_passed": admitted,
                    "bic_delta": bic_delta,
                    "final_admitted": both_stages_passed
                }
            },
            success=True,
        )
    
    def _update_crossbreed_index(
        self,
        knowledge_dir: Path,
        new_node_id: str,
        new_layer_name: str,
        new_mutual_info: dict[str, float]
    ) -> None:
        """Update crossbreed index with mutual information scores for new experiment."""
        import json
        from datetime import datetime
        
        try:
            experiments_file = knowledge_dir / "experiments.jsonl"
            crossbreed_file = knowledge_dir / "crossbreed_index.jsonl"
            
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
            
            # Special handling for execution_submit - pass analysis container
            if capability_name == "execution_submit":
                analysis = self._pick_container(containers, "analysis")
                if analysis is not None:
                    args = {**args, "container": analysis}
                else:
                    # Log warning but continue - will use fallback mode
                    print(f"Warning: No analysis container available for execution_submit, using fallback mode")
            
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
            from voxel_features.store import COE_FAIRBAIRN_GRID
            from voxel_features.mcp.tools.spatial_tools import (
                spatial_add_point, spatial_add_line, spatial_query_region,
                spatial_coord_to_voxel, spatial_get_operations_log
            )
            print("🔧 DEBUG: ✅ Imports successful")
            
            # Get store directory from episode context
            store_dir = ctx.episode_context.get("store_dir")
            print(f"🔧 DEBUG: Episode context keys: {list(ctx.episode_context.keys())}")
            print(f"🔧 DEBUG: Store dir: {store_dir}")
            if not store_dir:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error="No store directory available in episode context",
                )
            
            # Create or get spatial store
            print("🔧 DEBUG: Creating SpatialVoxelStore...")
            store = SpatialVoxelStore(store_dir, COE_FAIRBAIRN_GRID)
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
            from voxel_features.store import COE_FAIRBAIRN_GRID
            from voxel_features.mcp.tools.scoring_tools import (
                scoring_create_feature_layer
            )
            print("🎯 DEBUG: ✅ Imports successful")
            
            # Get store directory from episode context
            store_dir = ctx.episode_context.get("store_dir")
            print(f"🎯 DEBUG: Store dir: {store_dir}")
            if not store_dir:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error="No store directory available in episode context",
                )
            
            # Create or get spatial store
            print("🎯 DEBUG: Creating SpatialVoxelStore...")
            store = SpatialVoxelStore(store_dir, COE_FAIRBAIRN_GRID)
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
    ) -> FeatureHypothesisState:
        return FeatureHypothesisState(
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
    ) -> FeatureHypothesisState:
        phase_records = episode_context.get("phase_records", {})
        terminal_record = episode_context.get("terminal_record", {})
        
        # Extract state from phase records
        hypothesise = phase_records.get("hypothesise", {})
        code = phase_records.get("code", {})
        translate = phase_records.get("translate", {})
        evaluate = phase_records.get("evaluate", {})
        
        return FeatureHypothesisState(
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
        initial: FeatureHypothesisState,
        final: FeatureHypothesisState,
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
        
        # Two-stage reward calculation
        if masking_test_passed and admitted:
            # Both stages passed - full success
            # Stage 1: real MAE delta (auto_pass layers get full credit, they have no baseline)
            # masking_test_improvement is now actual mae_before - mae_after delta
            # Scale: 0.02 absolute MAE improvement = max reward (tunable)
            if masking_test_direction in ("auto_pass", "first_layer"):
                stage1_reward = 1.0  # Insufficient layers for MAE gate; full credit
            else:
                stage1_reward = min(1.0, max(0.0, masking_test_improvement / 0.02))
            # Stage 2: per-sample normalized BIC delta
            # bic_delta is now normalized by n_effective_samples (~[-0.5, 0] for good layers)
            # Scale: 0.1/sample BIC improvement = max reward (tunable)
            stage2_reward = min(1.0, max(0.0, -bic_delta / 0.1))
            
            # Weighted combination: Stage 1 (40%) + Stage 2 (60%)
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
                },
            )
        elif masking_test_passed and not admitted:
            # Stage 1 passed but Stage 2 failed - partial success
            # Reward for predictive capacity even if complexity penalty too high
            if masking_test_direction in ("auto_pass", "first_layer"):
                stage1_reward = 1.0
            else:
                stage1_reward = min(1.0, max(0.0, masking_test_improvement / 0.02))
            value = 0.3 * stage1_reward  # Reduced reward for Stage 1 only
            
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
                },
            )
        else:
            # Stage 1 failed - no geological understanding
            return TaskReward(
                value=0.05,  # Very small reward for attempting
                success=False,
                breakdown={
                    "stage_1_passed": False,
                    "stage_1_improvement": masking_test_improvement,
                    "stage_2_passed": admitted,
                    "bic_delta": bic_delta,
                    "no_predictive_value": True,
                },
            )
    
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
    
    def _count_features(self, variation: FeatureHypothesisVariation) -> int:
        """Count existing features in the store."""
        store_index = Path(variation.store_dir) / "index.json"
        if not store_index.exists():
            return 0
        try:
            with open(store_index) as f:
                data = json.load(f)
            return len(data.get("layers", {}))
        except Exception:
            return 0
    
    def _generate_survey_prompt_with_context(self, episode_context: dict[str, Any]) -> str:
        """Generate survey prompt with file assignment and coverage map.

        Each episode is anchored to a specific source file selected at populate()
        time.  The prompt shows a neutral file-coverage map (counts only, no
        hypothesis text) so the agent is never primed with past concepts.
        """
        assigned = episode_context.get("assigned_source", {})
        coverage = episode_context.get("source_coverage", {})

        # --- Build the coverage map (file names + counts, no concept words) ---
        coverage_lines = []
        for src in _COE_FAIRBAIRN_SOURCE_FILES:
            count = coverage.get(src["key"], 0)
            marker = "  ← assigned this episode" if src["key"] == assigned.get("key") else ""
            coverage_lines.append(f"  {src['path']}: {count} episode(s){marker}")
        coverage_block = "\n".join(coverage_lines)

        # --- Mandatory assignment block ---
        if assigned:
            assignment_block = (
                f"YOUR ASSIGNED SOURCE FILE FOR THIS EPISODE\n"
                f"  Path   : {assigned['path']}\n"
                f"  Details: {assigned['description']}\n\n"
                f"You MUST derive your hypothesis candidate from this file.\n"
                f"Do not freely browse other files during the survey phase.\n"
                f"Open and examine the assigned file first with analysis_shell.\n"
            )
        else:
            assignment_block = (
                "Explore the dataset to identify feature opportunities.\n"
            )

        prompt = (
            "Phase 1: Survey\n\n"
            + assignment_block
            + "\n"
            "SOURCE COVERAGE (episodes completed per file — for situational awareness only):\n"
            + coverage_block
            + "\n\n"
            "Use analysis_shell to read and explore your assigned file, then identify\n"
            "ONE promising feature layer candidate grounded in what you find there.\n\n"
            "Close with:\n"
            "  record_phase(phase='survey', candidates=[...])"
        )
        return prompt

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

    def _old_generate_survey_prompt_with_context(self) -> str:
        """RETIRED — kept for reference only. Replaced by _generate_survey_prompt_with_context(episode_context)."""
        base_prompt = (
            "Phase 1: Survey\n\n"
            "Explore the dataset to identify feature opportunities.\n\n"
            "Use analysis_shell to:\n"
            "- Read file headers and schemas\n"
            "- Identify interesting patterns\n\n"
        )
        
        # Try to get recent experiments for context
        try:
            import sys
            from pathlib import Path
            vfm_path = str(Path(__file__).parent.parent.parent / "voxel-features-mcp")
            if vfm_path not in sys.path:
                sys.path.append(vfm_path)
            
            from voxel_features.knowledge_graph import KnowledgeGraph
            from voxel_features.store import COE_FAIRBAIRN_GRID
            
            kg = KnowledgeGraph(COE_FAIRBAIRN_GRID)
            all_experiments = kg.list_all()
            
            if all_experiments:
                # Get 5 most recent experiments
                recent_experiments = sorted(
                    all_experiments,
                    key=lambda exp: exp.timestamp,
                    reverse=True
                )[:5]
                
                context_prompt = "\n**AVOID REPEATING RECENT EXPERIMENTS:**\n"
                context_prompt += "Recent hypotheses already tested (don't repeat these patterns):\n"
                
                for i, exp in enumerate(recent_experiments, 1):
                    status = "✅ ADMITTED" if exp.admitted else "❌ REJECTED"
                    context_prompt += f"{i}. {exp.hypothesis} [{status}]\n"
                    context_prompt += f"   Result: {exp.result_summary[:100]}...\n"
                
                context_prompt += "\nFocus on NEW geological patterns not covered above.\n\n"
                base_prompt += context_prompt
            
        except Exception as e:
            # If we can't get recent experiments, continue with base prompt
            print(f"Warning: Could not load recent experiments context: {e}")
        
        base_prompt += (
            "Find 2-3 promising feature layer candidates.\n\n"
            "Close with:\n"
            "  record_phase(phase='survey', candidates=[...])"
        )
        
        return base_prompt
    
    def _has_crossbreed_pairs(self, variation: FeatureHypothesisVariation) -> bool:
        """Check if there are crossbreed pairs available."""
        experiments_file = Path(variation.kg_dir) / "experiments.jsonl"
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
            return admitted_count >= 5
        except Exception:
            return False
    
    def _get_crossbreed_context(
        self,
        variation: FeatureHypothesisVariation,
    ) -> dict[str, Any]:
        """Get crossbreed prompt and parent IDs using JSONL knowledge graph."""
        import json
        
        try:
            experiments_file = Path(variation.kg_dir) / "experiments.jsonl"
            crossbreed_file = Path(variation.kg_dir) / "crossbreed_index.jsonl"
            
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
            
            # Get already used crossbreed pairs
            used_pairs = self._get_used_crossbreed_pairs(variation)
            
            # Find best unused crossbreed pair (maximum combined BIC improvement)
            best_pair = None
            best_score = float('-inf')
            
            for i, exp_a in enumerate(experiments):
                for exp_b in experiments[i+1:]:
                    # Skip if this pair was already crossbred
                    if self._is_pair_already_used(exp_a['node_id'], exp_b['node_id'], used_pairs):
                        continue
                    
                    # Calculate combined BIC improvement
                    bic_a = abs(exp_a.get("bic_delta", 0))
                    bic_b = abs(exp_b.get("bic_delta", 0))
                    combined_bic = bic_a + bic_b
                    
                    if combined_bic > best_score:
                        best_score = combined_bic
                        best_pair = (exp_a, exp_b)
            
            # If no unused pairs found, fall back to highest BIC pair with warning
            if not best_pair:
                print("Warning: All high-BIC pairs already used, selecting highest BIC pair")
                for i, exp_a in enumerate(experiments):
                    for exp_b in experiments[i+1:]:
                        bic_a = abs(exp_a.get("bic_delta", 0))
                        bic_b = abs(exp_b.get("bic_delta", 0))
                        combined_bic = bic_a + bic_b
                        
                        if combined_bic > best_score:
                            best_score = combined_bic
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
    
    def _get_used_crossbreed_pairs(self, variation: FeatureHypothesisVariation) -> set[tuple[str, str]]:
        """Extract already used crossbreed pairs from knowledge graph."""
        import json
        used_pairs = set()
        
        try:
            experiments_file = Path(variation.kg_dir) / "experiments.jsonl"
            if not experiments_file.exists():
                return used_pairs
            
            with open(experiments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            exp = json.loads(line)
                            parent_1 = exp.get("parent_node_1")
                            parent_2 = exp.get("parent_node_2")
                            if parent_1 and parent_2:
                                # Normalize pair order for consistent checking
                                pair = tuple(sorted([parent_1, parent_2]))
                                used_pairs.add(pair)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Could not load used crossbreed pairs: {e}")
        
        return used_pairs
    
    def _is_pair_already_used(self, node_a: str, node_b: str, used_pairs: set[tuple[str, str]]) -> bool:
        """Check if a pair of experiments has already been crossbred."""
        # Normalize pair order for consistent checking
        pair = tuple(sorted([node_a, node_b]))
        return pair in used_pairs
    
    def _get_crossbreed_context_simple(
        self,
        variation: FeatureHypothesisVariation,
    ) -> dict[str, Any]:
        """Simple fallback crossbreed selection - just first two admitted."""
        experiments_file = Path(variation.kg_dir) / "experiments.jsonl"
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

