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


_ROLE_SERVICE = {
    "agent": "agent",
    "vfm": "vfm",  # voxel-features-mcp
    "analysis": "analysis",
}

_ANALYSIS_INPUT = "/workspace/input"
_ANALYSIS_OUT = "/workspace/out"


# Coe Fairbairn grid specification
_COE_FAIRBAIRN_GRID = {
    "origin": [117.832397, -27.441096, 0.0],
    "maximum": [117.973493, -27.300000, 80.0],
    "shape": [25, 25, 5],
    "crs": "EPSG:4326",
}


_SYSTEM_PROMPT = """You are in mineral exploration mode.

Your goal is to identify informative feature layers that would improve compression 
of a voxel-based world model. You are rewarded when adding a feature layer improves BIC on a ridge regression of the overall world model.

## Dataset

You have access to the Coe Fairbairn dataset:
- geochemDrillhole.csv: 3D drillhole samples with 80+ element assays (Au, Cu, etc.)
- geochemSurface.csv: Surface samples
- other csvs: Mining tenement boundaries and history
- description.md files: detailed descriptions of maps
- WAMEX reports: OCR'd exploration reports (JSON chunks)

## Grid

The voxel grid covers:
- Longitude: 117.832° to 117.973°
- Latitude: -27.441° to -27.300°
- Depth: 0 to 80m
- Resolution: 25 × 25 × 5 voxels

## Scoring

Feature layers are evaluated by:
- Joint prediction score (each layer ~ all others via ridge regression)
- BIC - n*ln(MSE) + k*ln(n) 

A layer is admitted if bic_delta < 0.

## Capabilities

- analysis_shell: Execute Python code in a sandbox with polars/duckdb/scipy
- record_phase: Record workflow phase completion
- hypothesis_create: Register a falsifiable hypothesis
- submit_code: Submit code for execution (coding agent only)
- create_feature_layer: Create a 3D feature layer in the voxel store
"""


_DATASET_OVERVIEW = """## Coe Fairbairn Dataset Overview

### geochemDrillhole.csv (primary)
- 1299 drillhole samples with 3D coordinates
- Key columns: longitude, latitude, maxdepth_drill
- Assays: au_ppm, cu_ppm, ag_ppm, as_ppm, plus ~80 REE and trace elements

### geochemSurface.csv
- Surface geochemistry samples
- Similar element suite

### tenements.csv  
- Tenement boundaries and ownership history
- Can be used for spatial features (proximity to boundaries, etc.)

### WAMEX Reports
- OCR'd exploration reports in JSON chunks
- Historical geological interpretations
- Lithology descriptions, structural geology notes
"""


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
        
        # Store paths
        default_store = repo_root / "tasks" / "feature_hypothesis" / "store"
        self._store_dir = Path(task_config.get("store_dir", default_store)).resolve()
        
        default_kg = repo_root / "tasks" / "feature_hypothesis" / "knowledge"
        self._kg_dir = Path(task_config.get("kg_dir", default_kg)).resolve()
        
        self._docker_compose_dir = task_config.get(
            "docker_compose_dir", "docker/feature-hypothesis-compose"
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
            system=system_instruction,
            mission=mission,
            environment_context=env_context,
        )
    
    def build_workflow(
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
        
        return Workflow(
            initial_step="survey",
            steps={
                # HYPOTHESIS AGENT: Phase 1
                "survey": WorkflowStep(
                    name="survey",
                    prompt=(
                        "Phase 1: Survey\n\n"
                        "Explore the dataset to identify feature opportunities.\n\n"
                        "Use analysis_shell to:\n"
                        "- Read file headers and schemas\n"
                        "- Check data distributions\n"
                        "- Identify interesting patterns\n\n"
                        "Find 2-3 promising feature layer candidates.\n\n"
                        "Close with:\n"
                        "  record_phase(phase='survey', candidates=[...])"
                    ),
                    capabilities=(
                        "analysis_shell",
                        "record_phase",
                    ),
                    terminator_capabilities=("record_phase",),
                    next_steps=("hypothesise",),
                ),
                
                # HYPOTHESIS AGENT: Phase 2
                "hypothesise": WorkflowStep(
                    name="hypothesise",
                    prompt=(
                        "Phase 2: Hypothesise\n\n"
                        "Pick one candidate and state a falsifiable hypothesis.\n\n"
                        "Include a data_spec with:\n"
                        "- files: which files to use\n"
                        "- columns: which columns are relevant\n"
                        "- analysis: what analysis to perform\n"
                        "- output: what output format to produce\n\n"
                        "Example:\n"
                        "  hypothesis: 'High Au correlates with elevated Y in drillholes'\n"
                        "  data_spec: {\n"
                        "    files: ['geochemDrillhole.csv'],\n"
                        "    columns: ['au_ppm', 'y_ppm', 'longitude', 'latitude', 'maxdepth_drill'],\n"
                        "    analysis: 'OLS regression with spatial output',\n"
                        "    output: '3D array of predicted values'\n"
                        "  }\n\n"
                        "Close with:\n"
                        "  record_phase(phase='hypothesise', hypothesis=..., data_spec=...)"
                    ),
                    capabilities=(
                        "analysis_shell",
                        "record_phase",
                    ),
                    terminator_capabilities=("record_phase",),
                    next_steps=("code",),
                ),
                
                # CODING AGENT: Phase 3 (isolated, stateless)
                "code": WorkflowStep(
                    name="code",
                    prompt=(
                        "Phase 3: Code\n\n"
                        "Write Python code to test the hypothesis.\n\n"
                        "You have access to: polars, duckdb, scipy, numpy.\n"
                        "Data is at: /workspace/input/amalgamated_csvs/\n\n"
                        "The code should:\n"
                        "1. Load the specified data\n"
                        "2. Perform the analysis\n"
                        "3. Output a result summary\n"
                        "4. Optionally output a 3D array for the feature layer\n\n"
                        "Close with:\n"
                        "  submit_code(code=..., expected_output='...')"
                    ),
                    context_mode="isolated",  # Fresh context for coding agent
                    capabilities=(
                        "submit_code",
                        "phase_get",  # Can retrieve hypothesis from previous phase
                    ),
                    terminator_capabilities=("submit_code",),
                    next_steps=("translate",),
                ),
                
                # HYPOTHESIS AGENT: Phase 4 (resumed)
                "translate": WorkflowStep(
                    name="translate",
                    prompt=(
                        "Phase 4: Translate\n\n"
                        "Convert the analysis output to a voxel feature layer.\n\n"
                        "Consider:\n"
                        "- For surface-only data: how to handle subsurface cells?\n"
                        "- For partial coverage: how to handle missing areas?\n"
                        "- What dtype: float, categorical, boolean?\n\n"
                        "Call create_feature_layer with the 3D array.\n\n"
                        "Close with:\n"
                        "  record_phase(phase='translate', feature_layer_name=...)"
                    ),
                    capabilities=(
                        "create_feature_layer",
                        "phase_get",
                        "record_phase",
                    ),
                    terminator_capabilities=("record_phase",),
                    next_steps=("evaluate",),
                ),
                
                # FRAMEWORK: Phase 5 (automated)
                "evaluate": WorkflowStep(
                    name="evaluate",
                    prompt="[AUTOMATED] Framework computes BIC/MI scores.",
                    capabilities=(),  # No agent capabilities
                    terminator_capabilities=(),
                    next_steps=("rewrite",),
                    max_tool_calls=0,  # Auto-advance
                ),
                
                # REWRITING AGENT: Phase 6
                "rewrite": WorkflowStep(
                    name="rewrite",
                    prompt=(
                        "Phase 6: Rewrite\n\n"
                        "An experiment was conducted. Write it up as:\n"
                        "1. A knowledge graph node (for future crossbreeding)\n"
                        "2. A training prompt/response pair\n\n"
                        "You will see the hypothesis, code, result, and Ridge CV and BIC scores.\n\n"
                        "Close with:\n"
                        "  submit_rewrite(graph_node=..., training_pair=...)"
                    ),
                    context_mode="isolated",  # Fresh context for rewriting agent
                    capabilities=(
                        "phase_get",
                        "submit_rewrite",
                    ),
                    terminator_capabilities=("submit_rewrite",),
                    next_steps=(),  # Terminal
                ),
            },
        )
    
    def _crossbreed_workflow(
        self,
        variation: FeatureHypothesisVariation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        """Crossbreed workflow: starts with crossbreed prompt instead of survey."""
        
        # Same as survey but skip survey phase
        base_workflow = self._survey_workflow(variation, episode_context)
        
        # Modify to start at hypothesise with crossbreed context
        crossbreed_ctx = episode_context.get("crossbreed_context", {})
        parent_ids = crossbreed_ctx.get("parent_ids", [])
        
        base_workflow.initial_step = "hypothesise"
        base_workflow.steps["hypothesise"] = WorkflowStep(
            name="hypothesise",
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
            capabilities=(
                "analysis_shell",
                "record_phase",
            ),
            terminator_capabilities=("record_phase",),
            next_steps=("code",),
        )
        
        return base_workflow
    
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
                name="submit_code",
                description="Submit code for execution (coding agent only).",
                schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "expected_output": {"type": "string"},
                    },
                    "required": ["code"],
                },
            ),
            Capability(
                name="create_feature_layer",
                description="Create a 3D feature layer in the voxel store.",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "values": {"type": "array", "description": "3D array (25x25x5)"},
                        "dtype": {"type": "string", "enum": ["float", "categorical", "boolean"]},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "values"],
                },
            ),
            Capability(
                name="submit_rewrite",
                description="Submit the rewritten experiment record (rewriting agent only).",
                schema={
                    "type": "object",
                    "properties": {
                        "graph_node": {"type": "object"},
                        "training_pair": {"type": "object"},
                    },
                    "required": ["graph_node", "training_pair"],
                },
            ),
        ]
    
    def execute_capability(
        self,
        containers: list[Container],
        invocation: CapabilityInvocation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Execute a capability invocation."""
        
        name = invocation.name
        args = invocation.arguments or {}
        
        if name == "analysis_shell":
            return self._exec_analysis_shell(containers, args, ctx)
        elif name == "record_phase":
            return self._exec_record_phase(args, ctx)
        elif name == "phase_get":
            return self._exec_phase_get(args, ctx)
        elif name == "submit_code":
            return self._exec_submit_code(containers, args, ctx)
        elif name == "create_feature_layer":
            return self._exec_create_feature_layer(containers, args, ctx)
        elif name == "submit_rewrite":
            return self._exec_submit_rewrite(containers, args, ctx)
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
        
        # Execute code
        cmd = ["python", "-c", code]
        try:
            result = exec_run_with_timeout(analysis, cmd, timeout=60)
            output = coerce_exec_result(result)
            return CapabilityResult(
                "analysis_shell",
                output={"stdout": output.get("stdout", ""), "stderr": output.get("stderr", "")},
                success=output.get("exit_code", 1) == 0,
                error=output.get("stderr") if output.get("exit_code", 1) != 0 else None,
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
        
        return CapabilityResult(
            "phase_get",
            output=phase_records[phase],
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
        
        cmd = ["python", "-c", code]
        try:
            result = exec_run_with_timeout(analysis, cmd, timeout=120)
            output = coerce_exec_result(result)
            
            # Store code and result
            phase_records = ctx.episode_context.setdefault("phase_records", {})
            phase_records["code"] = {
                "code_executed": code,
                "result_summary": output.get("stdout", ""),
                "success": output.get("exit_code", 1) == 0,
                "timestamp": time.time(),
            }
            
            return CapabilityResult(
                "submit_code",
                output={
                    "stdout": output.get("stdout", ""),
                    "stderr": output.get("stderr", ""),
                    "success": output.get("exit_code", 1) == 0,
                },
                success=True,
            )
        except Exception as e:
            return CapabilityResult("submit_code", success=False, error=str(e))
    
    def _exec_create_feature_layer(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Create a feature layer in the voxel store."""
        name = args.get("name", "")
        values = args.get("values", [])
        dtype = args.get("dtype", "float")
        metadata = args.get("metadata", {})
        
        if not name or not values:
            return CapabilityResult(
                "create_feature_layer",
                success=False,
                error="name and values are required",
            )
        
        # Store for later evaluation
        phase_records = ctx.episode_context.setdefault("phase_records", {})
        phase_records["translate"] = {
            "feature_layer_name": name,
            "feature_values": values,
            "dtype": dtype,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        
        return CapabilityResult(
            "create_feature_layer",
            output={"name": name, "dtype": dtype, "staged": True},
            success=True,
        )
    
    def _exec_submit_rewrite(
        self,
        containers: list[Container],
        args: dict[str, Any],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Submit rewritten experiment record."""
        graph_node = args.get("graph_node", {})
        training_pair = args.get("training_pair", {})
        
        # Store terminal record
        ctx.episode_context["terminal_record"] = {
            "graph_node": graph_node,
            "training_pair": training_pair,
            "timestamp": time.time(),
        }
        
        return CapabilityResult(
            "submit_rewrite",
            output={"recorded": True},
            success=True,
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
            prompt_response_pair=terminal_record.get("training_pair", {}),
        )
    
    def compute_reward(
        self,
        initial: FeatureHypothesisState,
        final: FeatureHypothesisState,
    ) -> TaskReward:
        """Compute reward based on BIC improvement."""
        
        # Reward is based on BIC delta (negative is better)
        bic_delta = final.bic_delta
        
        if bic_delta is None:
            # No feature layer created
            return TaskReward(value=0.0, success=False, breakdown={"no_feature": True})
        
        if final.admitted:
            # Layer was admitted - BIC improved
            # Normalize to 0-1 range (typical BIC improvements are 0-1000)
            value = min(1.0, max(0.0, -bic_delta / 1000.0))
            return TaskReward(
                value=value,
                success=True,
                breakdown={
                    "bic_delta": bic_delta,
                    "admitted": True,
                },
            )
        else:
            # Layer was rejected
            return TaskReward(
                value=0.1,  # Small reward for attempting
                success=False,
                breakdown={
                    "bic_delta": bic_delta,
                    "admitted": False,
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
    
    def _has_crossbreed_pairs(self, variation: FeatureHypothesisVariation) -> bool:
        """Check if there are crossbreed pairs available."""
        kg_index = Path(variation.kg_dir) / "experiments.json"
        if not kg_index.exists():
            return False
        try:
            with open(kg_index) as f:
                data = json.load(f)
            admitted = [exp for exp in data.values() if exp.get("admitted")]
            return len(admitted) >= 2
        except Exception:
            return False
    
    def _get_crossbreed_context(
        self,
        variation: FeatureHypothesisVariation,
    ) -> dict[str, Any]:
        """Get crossbreed prompt and parent IDs using KnowledgeGraph."""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "voxel-features-mcp"))
        
        try:
            from voxel_features.knowledge_graph import KnowledgeGraph
            
            # Load knowledge graph
            kg = KnowledgeGraph(variation.kg_dir)
            kg.load()
            
            # Get best crossbreed pairs (prefers high performance + low MI)
            pairs = kg.get_crossbreed_pairs(
                max_pairs=1,  # Just need one pair
                prefer_orthogonal=True,  # Prefer low mutual information
            )
            
            if not pairs:
                return {}
            
            exp_a, exp_b = pairs[0]
            
            prompt = (
                f"These experiments both improved the world model:\n\n"
                f"Experiment 1: \"{exp_a.hypothesis}\"\n"
                f"- Result: {exp_a.result_summary}\n"
                f"- Feature: {exp_a.feature_layer_name}\n"
                f"- BIC improvement: {abs(exp_a.bic_delta or 0):.2f}\n\n"
                f"Experiment 2: \"{exp_b.hypothesis}\"\n"
                f"- Result: {exp_b.result_summary}\n"
                f"- Feature: {exp_b.feature_layer_name}\n"
                f"- BIC improvement: {abs(exp_b.bic_delta or 0):.2f}\n\n"
                f"Given that both patterns exist in the data, what new hypothesis "
                f"would you propose that combines or builds on these findings?"
            )
            
            return {
                "prompt": prompt,
                "parent_ids": [exp_a.id, exp_b.id],
            }
        except Exception as e:
            # Fallback to simple selection if KnowledgeGraph fails
            print(f"Warning: KnowledgeGraph failed, using simple selection: {e}")
            return self._get_crossbreed_context_simple(variation)
    
    def _get_crossbreed_context_simple(
        self,
        variation: FeatureHypothesisVariation,
    ) -> dict[str, Any]:
        """Simple fallback crossbreed selection - just first two admitted."""
        kg_index = Path(variation.kg_dir) / "experiments.json"
        if not kg_index.exists():
            return {}
        
        try:
            with open(kg_index) as f:
                data = json.load(f)
            
            admitted = [
                (exp_id, exp) 
                for exp_id, exp in data.items() 
                if exp.get("admitted")
            ]
            
            if len(admitted) < 2:
                return {}
            
            # Simple: just take first two
            exp_a_id, exp_a = admitted[0]
            exp_b_id, exp_b = admitted[1]
            
            prompt = (
                f"These experiments both improved the world model:\n\n"
                f"Experiment 1: \"{exp_a.get('hypothesis', '')}\"\n"
                f"- Result: {exp_a.get('result_summary', '')}\n"
                f"- Feature: {exp_a.get('feature_layer_name', '')}\n\n"
                f"Experiment 2: \"{exp_b.get('hypothesis', '')}\"\n"
                f"- Result: {exp_b.get('result_summary', '')}\n"
                f"- Feature: {exp_b.get('feature_layer_name', '')}\n\n"
                f"Given that both patterns exist in the data, what new hypothesis "
                f"would you propose that combines or builds on these findings?"
            )
            
            return {
                "prompt": prompt,
                "parent_ids": [exp_a_id, exp_b_id],
            }
        except Exception:
            return {}
