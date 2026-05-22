"""Pydantic models for all workspace-managed resources."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


class FieldRunSpec(BaseModel):
    """Full cache key for a voxel field computation.

    Two engine.run calls may share a cached field_uri only when ALL fields here match.
    """

    graph_content_hash: str
    grid: dict[str, Any] | None = None
    grid_origin: tuple[float, float, float] | None = None
    grid_maximum: tuple[float, float, float] | None = None
    grid_shape: tuple[int, int, int] | None = None
    bandwidth: float | dict[str, float] | None = None
    subgrid_factor: int = 1
    min_membership: float = 0.05
    epsg: int | None = None
    engine_name: str = "loopstructural"
    engine_version: str = "unknown"
    options: dict[str, Any] = Field(default_factory=dict)
    prior_field_hash: str | None = None
    drop_threshold: float | None = None

    @model_validator(mode="after")
    def _require_grid(self) -> FieldRunSpec:
        if self.grid is None and (
            self.grid_origin is None or self.grid_maximum is None or self.grid_shape is None
        ):
            raise ValueError("FieldRunSpec requires either grid or grid_origin/grid_maximum/grid_shape")
        return self

    def cache_key(self) -> str:
        raw = json.dumps(self.model_dump(mode="json", exclude_none=False), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def content_hash(self) -> str:
        return self.cache_key()


# ---------------------------------------------------------------------------
# Resource record base + subtypes
# ---------------------------------------------------------------------------

class ResourceRecord(BaseModel):
    """Base record stored in the workspace resource registry."""

    uri: str
    kind: str
    content_hash: str
    created_at: datetime = Field(default_factory=_now)
    size_bytes: int = 0
    pinned_by: list[str] = Field(default_factory=list)  # list of holder IDs
    tags: dict[str, str] = Field(default_factory=dict)


class GraphRecord(ResourceRecord):
    kind: Literal["graph"] = "graph"
    message: str | None = None
    node_count: int = 0
    edge_count: int = 0
    unit_catalog: list[str] = Field(default_factory=list)


class ScratchRecord(ResourceRecord):
    kind: Literal["scratch"] = "scratch"
    base_graph_uri: str
    head_rev: int = 0  # monotonically increasing rev counter
    head_rev_uri: str = ""  # URI of current head scratch revision snapshot
    committed: bool = False


class ScratchRevRecord(ResourceRecord):
    """Immutable snapshot of one scratch revision."""

    kind: Literal["scratch_rev"] = "scratch_rev"
    scratch_uri: str
    rev: int


class FieldRecord(ResourceRecord):
    kind: Literal["field"] = "field"
    field_run_spec_hash: str
    graph_uri: str
    grid_origin: tuple[float, float, float] | None = None
    grid_maximum: tuple[float, float, float] | None = None
    grid_shape: tuple[int, int, int] | None = None
    unit_catalog: list[str] = Field(default_factory=list)
    epsg: int | None = None


class DataRecord(ResourceRecord):
    kind: Literal["data"] = "data"
    source_path_rel: str = ""  # relative to workspace data root
    media_type: str = "application/octet-stream"
    detected_schema: dict[str, Any] = Field(default_factory=dict)
    crs: str | None = None
    preview_text: str = ""
    parent_uri: str | None = None  # set for derived/transformed data


class HypothesisRecord(ResourceRecord):
    kind: Literal["hypothesis"] = "hypothesis"
    statement: str
    graph_refs: list[str] = Field(default_factory=list)
    data_refs: list[str] = Field(default_factory=list)
    rationale: str | None = None


class GraphRef(BaseModel):
    role: str
    uri: str


class DataRef(BaseModel):
    role: str
    uri: str


class FieldRef(BaseModel):
    role: str
    uri: str


class Budget(BaseModel):
    time_s: int = 300
    memory_mb: int = 2048
    storage_mb: int | None = None
    gpu_count: int = 0


class SuccessCriteria(BaseModel):
    """Machine-comparable success criteria for an experiment."""

    criteria: list[dict[str, Any]] = Field(default_factory=list)
    # Each criterion: {id, type, threshold?, direction?, metric?, description}


class ExperimentSpec(BaseModel):
    id: str
    hypothesis_uri: str | None = None
    graph_refs: list[str] = Field(default_factory=list)
    data_refs: list[str] = Field(default_factory=list)
    field_refs: list[str] = Field(default_factory=list)
    procedure_uri: str
    procedure_params: dict[str, Any] = Field(default_factory=dict)
    success_criteria: SuccessCriteria
    budget: Budget = Field(default_factory=Budget)

    @field_validator("graph_refs", "data_refs", "field_refs", mode="before")
    @classmethod
    def _normalise_refs(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        refs: list[Any] = []
        for item in value:
            if isinstance(item, str):
                refs.append(item)
            elif isinstance(item, dict) and "uri" in item:
                refs.append(item["uri"])
            elif hasattr(item, "uri"):
                refs.append(item.uri)
            else:
                refs.append(item)
        return refs


ExperimentStatus = Literal[
    "queued", "claimed", "running", "completed", "failed",
    "cancelled", "refused", "needs_revision",
]


class ExperimentRecord(ResourceRecord):
    kind: Literal["experiment"] = "experiment"
    spec: ExperimentSpec
    status: ExperimentStatus = "queued"
    graph_ref_resolution: list[dict[str, Any]] = Field(default_factory=list)
    claim_id: str | None = None
    lease_expires_at: datetime | None = None
    result_uri: str | None = None
    retry_count: int = 0
    max_retries: int = 3


class CriterionOutcome(BaseModel):
    criterion_id: str
    status: Literal["passed", "failed", "inconclusive", "not_applicable"]
    evidence_refs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    rationale: str | None = None


class ExperimentResult(BaseModel):
    id: str
    spec_id: str
    status: Literal["success", "failed", "partial", "cancelled", "refused"]
    criterion_outcomes: list[CriterionOutcome] = Field(default_factory=list)
    artefact_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    action_log_uri: str = ""


class ResultRecord(ResourceRecord):
    kind: Literal["result"] = "result"
    experiment_uri: str
    result: ExperimentResult


class ExperimentReview(BaseModel):
    experiment_uri: str
    status: Literal["accepted", "rejected", "needs_revision"]
    criterion_assessments: list[dict[str, Any]] = Field(default_factory=list)
    candidate_refs: list[str] = Field(default_factory=list)
    notes: str | None = None


class ReviewRecord(ResourceRecord):
    kind: Literal["review"] = "review"
    experiment_uri: str
    review: ExperimentReview


JobStatus = Literal[
    "queued", "pending", "running", "completed", "failed", "cancelled", "cancelling"
]


class JobRecord(ResourceRecord):
    kind: Literal["job"] = "job"
    job_type: str
    status: JobStatus = "pending"
    progress: float = 0.0  # 0.0 – 1.0
    current_step: str = ""
    input_uris: list[str] = Field(default_factory=list)
    result_refs: list[str] = Field(default_factory=list)
    error: str | None = None


class CandidateRecord(ResourceRecord):
    kind: Literal["candidate"] = "candidate"
    graph_uri: str
    reference_pair: tuple[str, str]
    evidence_refs: list[str] = Field(default_factory=list)
    score_refs: list[str] = Field(default_factory=list)


class ScoreRecord(ResourceRecord):
    kind: Literal["score"] = "score"
    candidate_graph_uri: str
    reference_a_graph_uri: str
    reference_b_graph_uri: str
    candidate_field_uri: str
    reference_a_field_uri: str
    reference_b_field_uri: str
    score_value: float | None = None
    breakdown: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel):
    """Append-only provenance entry stamped by the workspace server."""

    id: str
    agent_id: str
    role: str
    capability: str
    tool: str
    timestamp: datetime = Field(default_factory=_now)
    input_uris: list[str] = Field(default_factory=list)
    output_uris: list[str] = Field(default_factory=list)
    job_uri: str | None = None
    experiment_uri: str | None = None
    candidate_uri: str | None = None
    cache_hit: bool | None = None
    budget_used: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Field cache index entry
# ---------------------------------------------------------------------------

class FieldCacheEntry(BaseModel):
    spec_hash: str
    field_uri: str
    created_at: datetime = Field(default_factory=_now)
