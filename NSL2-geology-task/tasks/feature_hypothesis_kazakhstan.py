"""Feature hypothesis task for Kazakhstan geological dataset.

Agents explore Kazakhstan geological datasets, hypothesize about informative feature layers,
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

This dataset contains comprehensive geological data for the Teniz Basin region of Kazakhstan:

**USGS Data (English):**
- Scientific assessment report chunks and figure descriptions
- Technical reports on sediment-hosted copper systems

**Russian Survey Data (Smolianova 1984):**  
- 300+ detailed geological text chunks covering stratigraphy, tectonics, magmatism
- 60+ drill hole logs with lithology and depth data
- Comprehensive geophysical interpretations and mineral evaluations

**Spatial Data (GeoJSON) — path: /workspace/input/converted_spatial_data/:**
- converted_spatial_data/copper_prospects.geojson: 113 sediment-hosted copper prospects with coordinates, tonnage, grades
- converted_spatial_data/anticlines_synclines.geojson: 33 geological fold structures
- converted_spatial_data/assessment_tract.geojson: Teniz Basin boundary and assessment tract data (49,714 km²)
- converted_spatial_data/copper_prospects_aoi.geojson: Area of interest boundary

**Scale Note:** Each voxel covers ~1.75km × 1.75km, suitable for regional geological features like basin structures, regional mineral trends, and large-scale geological formations.
"""


@dataclass
class FeatureHypothesisKazakhstanVariation(Variation):
    """Variation configuration for Kazakhstan feature hypothesis task."""
    
    dataset_dir: str = ""
    store_dir: str = ""
    kg_dir: str = ""
    grid_spec: dict[str, Any] = field(default_factory=lambda: dict(_KAZAKHSTAN_TENIZ_GRID))
    min_features: int = 0  # minimum features before crossbreeding
    crossbreed_enabled: bool = True


@dataclass
class FeatureHypothesisKazakhstanState:
    """Episode state for Kazakhstan feature hypothesis task."""
    
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


class FeatureHypothesisKazakhstanTask(TaskSpec[FeatureHypothesisKazakhstanState]):
    """Feature hypothesis discovery task for Kazakhstan geological data."""
    
    name = "feature-hypothesis-kazakhstan"
    description = "Discover informative feature layers from Kazakhstan geological data through hypothesis-driven exploration."
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
        
        # If crossbreeding, add context
        if workflow_kind == "crossbreed":
            episode_context["crossbreed_context"] = self._get_crossbreed_context(variation)
        
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
    
    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------
    
    def _survey_workflow(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Standard workflow: Survey → Hypothesise → Code → Translate → Evaluate → Rewrite"""
        
        # Generate dynamic survey prompt with recent experiments context
        survey_prompt = self._generate_survey_prompt_with_context()
        
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
                        "   Grid bounds: lon 66.5°-71.5°E, lat 49.5°-52.5°N, depth 0-80m\n"
                        "   Resolution: ~1.75km × 1.75km × 10m per voxel (200×200×8 total)\n\n"
                        "   **For prospect/drill data with coordinates:**\n"
                        "   spatial_add_point(name='string', longitude=float, latitude=float, depth_m=float, value=float, radius_m=float)\n\n"
                        "   **For geological structures (faults, anticlines, basins):**\n"
                        "   spatial_add_line(name='string', start_longitude=float, start_latitude=float, start_depth_m=float, end_longitude=float, end_latitude=float, end_depth_m=float, value=float, width_m=float)\n\n"
                        "   **For statistical results without coordinates:**\n"
                        "   - Use geological knowledge: regional trends, basin centers, prospect clusters\n"
                        "   - Create spatial patterns: geological formations, mineral belts\n"
                        "3. Create exactly ONE coherent feature layer:\n"
                        "   - ALL spatial operations must use the SAME layer name\n"
                        "   - Values must be floats or booleans: 'copper_potential' → 0.75\n"
                        "   - Example: spatial_add_point(name='sediment_copper_potential', ...) \n"
                        "            spatial_add_line(name='sediment_copper_potential', ...) \n"
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
                        "being tested. What patterns in the Kazakhstan data suggested this hypothesis?\n"
                        "  response: What analysis was performed, what was found, and why "
                        "the result is or isn't informative for Kazakhstan mineral exploration.\n\n"
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
    
    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------
    
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
        
        # Normal execution for Kazakhstan analysis
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
        
        # Auto-enhance data_spec for coding phase with Kazakhstan data
        if phase == "hypothesise" and "data_spec" in output:
            output["data_spec"] = self._enhance_data_spec_kazakhstan(output["data_spec"])
        
        return CapabilityResult(
            "phase_get",
            output=output,
            success=True,
        )
    
    def _enhance_data_spec_kazakhstan(self, data_spec: dict[str, Any]) -> dict[str, Any]:
        """Enhance data_spec with Kazakhstan-specific file guidance and correct paths."""
        import os
        enhanced = data_spec.copy()
        files = enhanced.get("files", [])
        
        # Add Kazakhstan-specific file guidance with correct paths
        file_specs = []
        
        # Add the known Kazakhstan GeoJSON files with correct paths
        kazakhstan_geojson_files = [
            {
                "file": "converted_spatial_data/copper_prospects.geojson",
                "full_path": "/workspace/input/converted_spatial_data/copper_prospects.geojson",
                "type": "geojson",
                "geometry": "points",
                "properties": ["name", "tonnage", "grade", "coordinates", "deposit_type"],
                "note": "113 copper prospects with economic data - use geopandas.read_file(full_path)",
                "count": 113,
                "description": "Sediment-hosted copper prospects in Teniz Basin"
            },
            {
                "file": "converted_spatial_data/anticlines_synclines.geojson", 
                "full_path": "/workspace/input/converted_spatial_data/anticlines_synclines.geojson",
                "type": "geojson",
                "geometry": "linestrings/polygons",
                "properties": ["structure_type", "geological_age", "fold_axis"],
                "note": "33 geological fold structures - use geopandas.read_file(full_path)",
                "count": 33,
                "description": "Regional anticline and syncline structures"
            },
            {
                "file": "converted_spatial_data/assessment_tract.geojson",
                "full_path": "/workspace/input/converted_spatial_data/assessment_tract.geojson", 
                "type": "geojson",
                "geometry": "polygon",
                "properties": ["area_km2", "tract_name", "assessment_type"],
                "note": "Teniz Basin boundary (49,714 km²) - use geopandas.read_file(full_path)",
                "area_km2": 49714,
                "description": "USGS assessment tract boundary"
            },
            {
                "file": "converted_spatial_data/copper_prospects_aoi.geojson",
                "full_path": "/workspace/input/converted_spatial_data/copper_prospects_aoi.geojson",
                "type": "geojson", 
                "geometry": "polygons",
                "properties": ["area_of_interest", "prospect_density"],
                "note": "Copper prospect areas of interest - use geopandas.read_file(full_path)",
                "description": "AOI polygons for copper exploration"
            }
        ]
        
        # Add Russian geological survey data
        russian_survey_files = [
            {
                "file": "36572_Smolianova_1984",
                "full_path": "/workspace/input/36572_Smolianova_1984",
                "type": "directory",
                "language": "Russian/English",
                "content": "geological survey texts and drill hole data",
                "note": "579 Russian geological survey files - use os.listdir() to explore",
                "file_count": 579,
                "description": "Comprehensive geological survey (Smolianova 1984)"
            }
        ]
        
        # Add USGS data
        usgs_files = [
            {
                "file": "USGS", 
                "full_path": "/workspace/input/USGS",
                "type": "directory",
                "content": "USGS assessment data and reports",
                "note": "USGS English-language data - use os.listdir() to explore",
                "description": "USGS Teniz Basin assessment data"
            }
        ]
        
        # Combine all file specifications
        all_files = kazakhstan_geojson_files + russian_survey_files + usgs_files
        
        # Add any additional files from the original data_spec
        for file_path in files:
            if not any(spec["file"] in file_path for spec in all_files):
                file_specs.append({"file": file_path, "type": "unknown", "note": "Additional file - check format"})
        
        enhanced["file_specs"] = all_files
        enhanced["kazakhstan_data_structure"] = {
            "geojson_files": 4,
            "copper_prospects": 113,
            "geological_structures": 33,
            "assessment_area_km2": 49714,
            "russian_survey_files": 579,
            "data_languages": ["English", "Russian"],
            "coordinate_system": "EPSG:4326 (WGS84)",
            "grid_bounds": "66.5°-71.5°E × 49.5°-52.5°N"
        }
        
        return enhanced
    
    def _exec_get_experiment_summary(
        self,
        containers: list[Container],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Return all phase data for Kazakhstan experiment."""
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
                "region": "kazakhstan",
                "grid_coverage": "116160_km2",
            },
            success=True,
        )
    
    def _exec_execution_finalize(
        self,
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Finalize execution results for Kazakhstan analysis."""
        execution_id = args.get("execution_id", "")
        success = args.get("success", False)
        summary = args.get("summary", "")
        
        if not execution_id:
            return CapabilityResult("execution_finalize", success=False, error="execution_id required")
        
        # For now, simple implementation - can be enhanced with actual execution system
        phase_records = ctx.episode_context.setdefault("phase_records", {})
        phase_records["code"] = {
            "code_executed": "Kazakhstan analysis executed",
            "result_summary": summary,
            "success": success,
            "timestamp": time.time(),
            "execution_id": execution_id,
        }
        
        return CapabilityResult(
            "execution_finalize",
            output={
                "execution_id": execution_id,
                "success": success,
                "summary": summary,
            },
            success=True,
        )
    
    def _exec_execution_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute execution tool capability via direct import for Kazakhstan."""
        
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
                    error=f"Unknown Kazakhstan execution capability: {capability_name}"
                )
            
            # Special handling for execution_submit - pass analysis container
            if capability_name == "execution_submit":
                analysis = self._pick_container(containers, "analysis")
                if analysis is not None:
                    args = {**args, "container": analysis}
                else:
                    # Log warning but continue - will use fallback mode
                    print(f"Warning: No analysis container available for Kazakhstan execution_submit, using fallback mode")
            
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
                        "region": "kazakhstan",
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
                error=f"Kazakhstan execution capability failed: {str(e)}",
            )
    
    def _exec_submit_rewrite(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Submit Kazakhstan rewritten experiment record. Auto-generates graph node and appends BIC."""
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
            "region": "kazakhstan",
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
                f"\n\nKazakhstan Result: {bic_delta:.4f} BIC delta. {verdict}."
            )

        # Prepare training record for persistence
        episode_id = ctx.episode_context.get("episode_id", "")
        store_dir = ctx.episode_context.get("store_dir", "")
        
        # Extract Kazakhstan data paths - need to get base data directory
        if store_dir:
            data_base_path = Path(store_dir).parent.parent  # from store/teniz_basin to data/kazakhstan/feature-hypothesis
        else:
            data_base_path = Path("/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/NSL2-geology-task/data/kazakhstan/feature-hypothesis")
        
        # Extract scoring results (Kazakhstan doesn't have two-stage scoring yet, so simplified)
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
            'region': 'kazakhstan',
            # Simplified scoring results for Kazakhstan
            'masking_test_passed': masking_test_passed,
            'masking_test_improvement': masking_test_improvement,
            'masking_test_direction': masking_test_direction,
            'stage_completed': stage_completed,
            'metadata': {
                'hypothesis': hypothesise.get('hypothesis', ''),
                'grid_bounds': ctx.episode_context.get('grid_spec', _KAZAKHSTAN_TENIZ_GRID),
                'mutual_info': evaluate.get('mutual_info', {}),
                'experiment_summary': code.get('result_summary', ''),
                'grid_coverage': '116160_km2',
                'voxel_resolution': '1.75km_x_1.66km_x_10m',
                'dataset': 'kazakhstan_teniz_basin',
                'scoring_version': 'bic_only_v1'  # Kazakhstan uses simplified scoring for now
            }
        }
        
        # Save training data (ALL Kazakhstan experiments)
        try:
            training_dir = data_base_path / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            training_file = training_dir / "training_pairs.pkl"

            # Load existing records or create new list
            existing_records = []
            if training_file.exists():
                with open(training_file, 'rb') as f:
                    existing_records = pickle.load(f)
            
            # Append new Kazakhstan record
            existing_records.append(training_record)
            
            # Save back to file
            with open(training_file, 'wb') as f:
                pickle.dump(existing_records, f)
                
        except Exception as e:
            print(f"Warning: Failed to save Kazakhstan training data: {e}")
        
        # Save to knowledge graph (successful Kazakhstan experiments)
        # For now, use simple admission criterion - can be enhanced later
        if admitted and bic_delta is not None and bic_delta < 0:
            try:
                knowledge_dir = data_base_path / "knowledge" / "teniz_basin"
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                experiments_file = knowledge_dir / "experiments.jsonl"
                
                # Generate node ID
                node_id = f"kz_exp_{episode_id}" if episode_id else f"kz_exp_{int(time.time())}"
                
                # Get parent experiment IDs if this is a crossbreed episode
                parent_experiments = hypothesise.get("parent_experiments", [])
                parent_node_1 = parent_experiments[0] if len(parent_experiments) > 0 else None
                parent_node_2 = parent_experiments[1] if len(parent_experiments) > 1 else None
                
                # Create Kazakhstan knowledge graph record
                kg_record = {
                    "node_id": node_id,
                    "prompt": training_pair.get('prompt', ''),
                    "response": training_pair.get('response', ''),
                    "bic_delta": bic_delta,
                    "region": "kazakhstan",
                    "grid_coverage": "116160_km2",
                    "voxel_resolution": "1.75km_x_1.66km_x_10m",
                    "scoring_version": "bic_only_v1",
                    "artifact_links": {
                        "layer_file": f"store/teniz_basin/layers/{translate.get('feature_layer_name', '')}.npy" if translate.get('feature_layer_name') else None,
                        "spatial_ops": f"store/teniz_basin/spatial.db:experiment_{episode_id}" if episode_id else None
                    },
                    "parent_node_1": parent_node_1,
                    "parent_node_2": parent_node_2,
                    "timestamp": datetime.now().isoformat(),
                    "mutual_info": evaluate.get('mutual_info', {}),
                    "layer_name": translate.get('feature_layer_name', ''),
                    "hypothesis": hypothesise.get('hypothesis', ''),
                    "dataset": "kazakhstan_teniz_basin"
                }
                
                # Append to experiments.jsonl
                with open(experiments_file, 'a') as f:
                    f.write(json.dumps(kg_record) + '\n')
                
                # TODO: Implement Kazakhstan crossbreed index calculation
                # self._update_crossbreed_index_kazakhstan(knowledge_dir, node_id, translate.get('feature_layer_name', ''), evaluate.get('mutual_info', {}))
                
            except Exception as e:
                print(f"Warning: Failed to save Kazakhstan knowledge graph data: {e}")

        ctx.episode_context["terminal_record"] = {
            "graph_node": graph_node,
            "training_pair": training_pair,
            "timestamp": time.time(),
            "region": "kazakhstan",
        }

        return CapabilityResult(
            "submit_rewrite",
            output={
                "recorded": True, 
                "training_saved": True, 
                "knowledge_saved": (admitted and bic_delta is not None and bic_delta < 0),
                "region": "kazakhstan",
                "grid_coverage": "116160_km2",
                "bic_delta": bic_delta,
                "admitted": admitted,
            },
            success=True,
        )
    
    def _exec_spatial_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute spatial tool capability via voxel-features-mcp with Kazakhstan grid."""
        
        print(f"🔧 DEBUG: Starting Kazakhstan spatial capability: {capability_name}")
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
            print(f"🔧 DEBUG: Episode context keys: {list(ctx.episode_context.keys())}")
            print(f"🔧 DEBUG: Store dir: {store_dir}")
            if not store_dir:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error="No store directory available in episode context",
                )
            
            # Create or get spatial store with Kazakhstan grid
            print("🔧 DEBUG: Creating SpatialVoxelStore with Kazakhstan grid...")
            # Create GridSpec object from Kazakhstan grid dictionary
            kazakhstan_grid_spec = GridSpec(
                origin=tuple(_KAZAKHSTAN_TENIZ_GRID["origin"]),
                maximum=tuple(_KAZAKHSTAN_TENIZ_GRID["maximum"]), 
                shape=tuple(_KAZAKHSTAN_TENIZ_GRID["shape"]),
                crs=_KAZAKHSTAN_TENIZ_GRID["crs"]
            )
            store = SpatialVoxelStore(store_dir, kazakhstan_grid_spec)
            print(f"🔧 DEBUG: ✅ Kazakhstan store created, grid shape: {store.grid.shape}")
            print(f"🔧 DEBUG: Grid bounds: lon {store.grid.origin[0]:.3f}-{store.grid.maximum[0]:.3f}, lat {store.grid.origin[1]:.3f}-{store.grid.maximum[1]:.3f}, depth {store.grid.origin[2]:.1f}-{store.grid.maximum[2]:.1f}")
            
            # Validate coordinates if this is a spatial operation with coordinates
            if capability_name in ["spatial_add_point", "spatial_add_line"]:
                if "longitude" in args and "latitude" in args:
                    lon, lat = args["longitude"], args["latitude"]
                    in_bounds = (store.grid.origin[0] <= lon <= store.grid.maximum[0] and 
                                store.grid.origin[1] <= lat <= store.grid.maximum[1])
                    print(f"🔧 DEBUG: Kazakhstan coordinate validation - lon={lon:.6f}, lat={lat:.6f}, in_bounds={in_bounds}")
                    
                    if not in_bounds:
                        return CapabilityResult(
                            capability_name,
                            success=False,
                            error=f"Coordinates ({lon:.6f}, {lat:.6f}) outside Kazakhstan grid bounds",
                        )
            
            # Route to appropriate spatial tool function
            print(f"🔧 DEBUG: Routing to Kazakhstan tool: {capability_name}")
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
                    error=f"Unknown Kazakhstan spatial capability: {capability_name}",
                )
            
            print(f"🔧 DEBUG: ✅ Kazakhstan tool result: {result}")
            
            # Update translate phase with layer name for later evaluation
            if result.get("success") and result.get("layer_name"):
                phase_records = ctx.episode_context.setdefault("phase_records", {})
                translate_record = phase_records.setdefault("translate", {})
                if not translate_record.get("feature_layer_name"):
                    translate_record["feature_layer_name"] = result["layer_name"]
                    translate_record["timestamp"] = __import__('time').time()
                    print(f"🔧 DEBUG: Stored Kazakhstan layer name '{result['layer_name']}' for later evaluation")
            
            # Return result
            return CapabilityResult(
                capability_name,
                output=result,
                success=result.get("success", False),
            )
            
        except Exception as e:
            print(f"🔧 DEBUG: ❌ Exception in Kazakhstan spatial capability: {e}")
            import traceback
            traceback.print_exc()
            return CapabilityResult(
                capability_name,
                success=False,
                error=f"Kazakhstan spatial capability execution failed: {str(e)}",
            )
    
    def _exec_scoring_capability(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
        capability_name: str,
    ) -> CapabilityResult:
        """Execute scoring tool capability via voxel-features-mcp with Kazakhstan grid."""
        
        print(f"🎯 DEBUG: Starting Kazakhstan scoring capability: {capability_name}")
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
            print(f"🎯 DEBUG: Kazakhstan store dir: {store_dir}")
            if not store_dir:
                return CapabilityResult(
                    capability_name,
                    success=False,
                    error="No store directory available in episode context",
                )
            
            # Create or get spatial store with Kazakhstan grid
            print("🎯 DEBUG: Creating SpatialVoxelStore with Kazakhstan grid...")
            # Create GridSpec object from Kazakhstan grid dictionary
            kazakhstan_grid_spec = GridSpec(
                origin=tuple(_KAZAKHSTAN_TENIZ_GRID["origin"]),
                maximum=tuple(_KAZAKHSTAN_TENIZ_GRID["maximum"]), 
                shape=tuple(_KAZAKHSTAN_TENIZ_GRID["shape"]),
                crs=_KAZAKHSTAN_TENIZ_GRID["crs"]
            )
            store = SpatialVoxelStore(store_dir, kazakhstan_grid_spec)
            print(f"🎯 DEBUG: ✅ Kazakhstan store created, grid shape: {store.grid.shape}")
            
            # Route to scoring.create_feature_layer MCP tool
            print(f"🎯 DEBUG: Routing to Kazakhstan tool: {capability_name}")
            if capability_name == "scoring_create_feature_layer":
                print("🎯 DEBUG: Calling scoring_create_feature_layer MCP function with Kazakhstan grid...")
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
                    error=f"Unknown Kazakhstan scoring capability: {capability_name}",
                )
            
            print(f"🎯 DEBUG: ✅ Kazakhstan tool result: {result}")
            
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
                print(f"🎯 DEBUG: Stored Kazakhstan evaluation data in phase records")
            
            # Return result
            return CapabilityResult(
                capability_name,
                output=result,
                success=result.get("success", False),
            )
            
        except Exception as e:
            print(f"🎯 DEBUG: ❌ Exception in Kazakhstan scoring capability: {e}")
            import traceback
            traceback.print_exc()
            return CapabilityResult(
                capability_name,
                success=False,
                error=f"Kazakhstan scoring capability execution failed: {str(e)}",
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
        )
    
    def compute_reward(
        self,
        initial: FeatureHypothesisKazakhstanState,
        final: FeatureHypothesisKazakhstanState,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        """Compute reward based on BIC results for Kazakhstan regional analysis."""
        
        bic_delta = final.bic_delta
        admitted = final.admitted
        
        if bic_delta is None:
            # No feature layer created
            return TaskReward(
                value=0.0, 
                success=False, 
                breakdown={"no_feature": True, "region": "kazakhstan"}
            )
        
        # Regional-scale reward calculation
        if admitted:
            # Feature was admitted (improved regional compression)
            # Scale reward based on BIC improvement for regional features
            # Regional features may have different BIC scales than deposit-scale
            reward_value = min(1.0, max(0.0, -bic_delta / 20.0))
            
            return TaskReward(
                value=reward_value,
                success=True,
                breakdown={
                    "bic_delta": bic_delta,
                    "admitted": True,
                    "region": "kazakhstan",
                    "scale": "basin_regional",
                    "reward": reward_value,
                },
            )
        else:
            # Feature was rejected (worse compression)
            return TaskReward(
                value=0.05,  # Small reward for attempting
                success=False,
                breakdown={
                    "bic_delta": bic_delta,
                    "admitted": False,
                    "region": "kazakhstan",
                    "scale": "basin_regional",
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
    
    def _count_features(self, variation: FeatureHypothesisKazakhstanVariation) -> int:
        """Count existing features in the Kazakhstan store."""
        store_index = Path(variation.store_dir) / "index.json"
        if not store_index.exists():
            return 0
        try:
            with open(store_index) as f:
                data = json.load(f)
            return len(data.get("layers", {}))
        except Exception:
            return 0
    
    def _generate_survey_prompt_with_context(self) -> str:
        """Generate survey prompt with recent Kazakhstan experiments context."""
        base_prompt = (
            "Phase 1: Survey\n\n"
            "Explore the Kazakhstan Teniz Basin dataset to identify regional feature opportunities.\n\n"
            "Use analysis_shell to:\n"
            "- Read file headers and schemas\n"
            "- Identify interesting regional patterns\n"
            "- Focus on basin-scale geological features\n\n"
        )
        
        # For Kazakhstan, we'll start simple without experiment history
        # This can be enhanced later with Kazakhstan-specific experiment tracking
        
        base_prompt += (
            "Find 2-3 promising regional feature layer candidates.\n\n"
            "Focus on:\n"
            "- Sediment-hosted copper systems\n"
            "- Basin-scale structural geology\n"
            "- Regional geological trends\n\n"
            "Close with:\n"
            "  record_phase(phase='survey', candidates=[...])"
        )
        
        return base_prompt
    
    def _has_crossbreed_pairs(self, variation: FeatureHypothesisKazakhstanVariation) -> bool:
        """Check if there are crossbreed pairs available for Kazakhstan."""
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
            return admitted_count >= 2
        except Exception:
            return False
    
    def _get_crossbreed_context(
        self,
        variation: FeatureHypothesisKazakhstanVariation,
    ) -> dict[str, Any]:
        """Get crossbreed prompt and parent IDs for Kazakhstan experiments."""
        import json
        
        try:
            experiments_file = Path(variation.kg_dir) / "experiments.jsonl"
            
            if not experiments_file.exists():
                return {}
            
            # Load successful Kazakhstan experiments
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
                print("Warning: All Kazakhstan high-BIC pairs already used, selecting highest BIC pair")
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
            
            print(f"Selected Kazakhstan crossbreed pair: {exp_a['node_id']} (BIC: {abs(exp_a.get('bic_delta', 0)):.2f}) + {exp_b['node_id']} (BIC: {abs(exp_b.get('bic_delta', 0)):.2f})")
            
            crossbreed_prompt = (
                f"Combine insights from these successful Kazakhstan experiments:\n\n"
                f"Experiment A: {exp_a.get('hypothesis', 'Unknown')}\n"
                f"Result A: BIC delta {exp_a.get('bic_delta', 'N/A')}\n\n"
                f"Experiment B: {exp_b.get('hypothesis', 'Unknown')}\n"
                f"Result B: BIC delta {exp_b.get('bic_delta', 'N/A')}\n\n"
                f"Create a new hypothesis that combines or builds on these findings "
                f"for Kazakhstan basin-scale analysis."
            )
            
            return {
                "prompt": crossbreed_prompt,
                "parent_ids": [exp_a["node_id"], exp_b["node_id"]],
            }
            
        except Exception as e:
            print(f"Warning: Could not load Kazakhstan crossbreed context: {e}")
            return {}
    
    def _get_used_crossbreed_pairs(self, variation: FeatureHypothesisKazakhstanVariation) -> set[tuple[str, str]]:
        """Extract already used crossbreed pairs from Kazakhstan knowledge graph."""
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
            print(f"Warning: Could not load used Kazakhstan crossbreed pairs: {e}")
        
        return used_pairs
    
    def _is_pair_already_used(self, node_a: str, node_b: str, used_pairs: set[tuple[str, str]]) -> bool:
        """Check if a pair of Kazakhstan experiments has already been crossbred."""
        # Normalize pair order for consistent checking
        pair = tuple(sorted([node_a, node_b]))
        return pair in used_pairs
