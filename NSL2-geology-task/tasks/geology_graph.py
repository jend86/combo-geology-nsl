"""Geology graph-refinement task.

Agents inspect a small geology dataset, propose/refine graph-to-voxel world
models, and are rewarded by task-side graph refinement scoring. The scoring
criterion is never exposed as an agent capability.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shlex
import tarfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from src.training_data.transforms import EpisodeTrainingRows, TrainingDataExportContext
from tasks.common.foundry_exec import coerce_exec_result, exec_run_with_timeout


_ROLE_SERVICE = {
    "agent": "agent",
    "g2v": "g2v",
    "analysis": "analysis",
}

_G2V_WORKSPACE = "/var/lib/g2v/workspace"
_ANALYSIS_INPUT = "/workspace/input"
_ANALYSIS_OUT = "/workspace/out"
_MAX_PROMOTE_BYTES = 64 * 1024 * 1024

_REGULAR_PHASE_TOOLS: dict[str, tuple[str, ...]] = {
    "mcp_explore_call": (
        "graph_query",
        "graph_subgraph",
        "graph_diff",
        "graph_provenance",
        "workspace_get",
        "data_ingest",
    ),
    "mcp_hypothesise_call": (
        "hypothesis_create",
        "hypothesis_list",
        "hypothesis_get",
        "graph_query",
        "graph_branch",
        "graph_apply_patch",
        "graph_commit",
        "data_ingest",
    ),
    "mcp_execute_call": (
        "hypothesis_list",
        "hypothesis_get",
        "engine_run_preview",
        "engine_run",
        "job_status",
        "voxel_sample",
        "voxel_stats",
        "data_ingest",
    ),
    "mcp_refine_call": (
        "hypothesis_get",
        "graph_branch",
        "graph_apply_patch",
        "graph_commit",
        "graph_diff",
        "graph_query",
        "graph_subgraph",
        "engine_run_preview",
        "engine_run",
        "job_status",
        "data_ingest",
    ),
    "mcp_submit_call": ("candidate_submit",),
    "mcp_submit_call_seed": (
        "graph_ingest",
        "data_ingest",
        "engine_run_preview",
        "engine_run",
        "job_status",
    ),
}

_ALL_PHASE_RECORDS = ("explore", "explore_data", "hypothesise", "execute", "refine")


_PHASE_CAPABILITY_DESCRIPTIONS: dict[str, str] = {
    "mcp_explore_call": (
        "Read-only graph and data exploration. Inner tools: "
        "graph_query (selector-based node/edge lookup), "
        "graph_subgraph (BFS subgraph around seed_nodes), "
        "graph_diff (a vs b structural delta), "
        "graph_provenance (per-node provenance trail), "
        "workspace_get (resource record by URI), "
        "data_ingest (register a data blob -> data_uri). "
        "All take args as a dict; responses are JSON objects."
    ),
    "mcp_hypothesise_call": (
        "Hypothesis registration and low-level graph drafting. Inner tools: "
        "hypothesis_create (returns {hypothesis_uri, statement}), "
        "hypothesis_list, hypothesis_get, "
        "graph_query, graph_branch (returns {scratch_uri}), "
        "graph_apply_patch (operations on a scratch_uri), "
        "graph_commit (scratch_uri -> {graph_uri}), "
        "data_ingest. Use hypothesis_create here, defer graph mutation to refine."
    ),
    "mcp_execute_call": (
        "Read-only experiment running. Inner tools: "
        "hypothesis_list / hypothesis_get, "
        "engine_run_preview (sync build of a voxel field within preview budget -> "
        "{field_uri, ...} or {job_uri}), "
        "engine_run (async job -> {job_uri}), "
        "job_status (poll a job_uri), "
        "voxel_sample (sample field at points), "
        "voxel_stats (summary stats over a field region), "
        "data_ingest. Engine tools require a committed graph_uri; do not call them "
        "in bootstrap before a graph exists."
    ),
    "mcp_refine_call": (
        "Low-level graph mutation. Prefer the dedicated refine_commit capability, "
        "which atomically branches + patches + commits + previews. Inner tools "
        "here: hypothesis_get, graph_branch, graph_apply_patch, graph_commit, "
        "graph_diff, graph_query, graph_subgraph, engine_run_preview, engine_run, "
        "job_status, data_ingest."
    ),
    "mcp_submit_call": (
        "Final candidate registration. Inner tool: candidate_submit(graph_uri). "
        "Prefer candidate_submit_and_report, which also records self-assessment."
    ),
    "mcp_submit_call_seed": (
        "Bootstrap-only g2v access for ingesting and previewing a seed. Inner tools: "
        "graph_ingest, data_ingest, engine_run_preview, engine_run, job_status. "
        "Prefer seed_graph_submit, which ingests and records the terminal in one call."
    ),
}

_SEED_GRAPH_TEMPLATE = {
    "nodes": [
        {
            "kind": "stratigraphic_unit",
            "id": "u_perm_lower",
            "unit_id": "permian_lower_division",
            "series_id": "permian",
            "topology": "layer",
            "provenance": {
                "source": "agent",
                "reference": "knowledge_base/chunks/36572_055_V_STRATIGRAPHY_Permian_Lower_Division.md",
                "confidence": 0.7,
                "timestamp": "2026-01-01T00:00:00Z",
            },
        },
        {
            "kind": "stratigraphic_unit",
            "id": "u_kumansai",
            "unit_id": "ordovician_kumansai_suite_O2km",
            "series_id": "ordovician",
            "topology": "layer",
            "provenance": {
                "source": "agent",
                "reference": "knowledge_base/chunks/36572_020_V_STRATIGRAPHY_Ordovician_Kumansai_Suite_O2km.md",
                "confidence": 0.7,
                "timestamp": "2026-01-01T00:00:00Z",
            },
        },
    ],
    "edges": [
        {
            "kind": "overlies",
            "source": "u_perm_lower",
            "target": "u_kumansai",
            "provenance": {
                "source": "agent",
                "reference": "knowledge_base narrative — Permian overlies Ordovician basement",
                "confidence": 0.8,
                "timestamp": "2026-01-01T00:00:00Z",
            },
        }
    ],
    "metadata": {"seed_origin": "bootstrap_template"},
}

_SEED_GRAPH_TEMPLATE_JSON = json.dumps(_SEED_GRAPH_TEMPLATE, indent=2)


# Per-variation pools of grounded episode-focus prompts. One entry is selected
# per episode using a hash of (variation_name, episode_workspace_id) so the same
# episode reproduces the same nudge from logs alone. Variations with an empty
# pool produce no focus line - the prompt then matches the pre-pool behavior.
_VARIATION_OBJECTIVES: dict[str, tuple[str, ...]] = {
    "smolianova_copper": (
        "Find which Carboniferous suite hosts the highest-grade Cu intervals "
        "in outputs/boreholes_copper_master_36572.csv (candidates: "
        "carboniferous_tersakkansk_suite_C3tr, lower_carboniferous_spassk_suite_C2sp, "
        "upper_carboniferous_vladimirov_suite, carboniferous_kiiminskaya_suite). "
        "Use that unit as the anchor of the graph and ground its overlies/contact "
        "relations against knowledge_base/chunks/36572_101_VIII_USEFUL_MINERALS_*.md "
        "or 36572_102_VIII_USEFUL_MINERALS_DIRECT_PROJECTION_OF_ORES_K-23.md.",
        "Cross-check outputs/indicators_geo_36572.csv rows where element='Cu' "
        "against the stratigraphic_unit column. The unit with the most Cu "
        "indicators (Tersakkansk or Spassk are strong candidates) is a host "
        "worth representing as a sample node connected via `at` to a Location.",
        "Open outputs/agent_identified_prospects_36572.geojson and pick one "
        "high-prospectivity Cu feature (e.g. Spasskoye, Vladimirovsky, "
        "Taberkolskoye). Anchor the graph on that prospect's inferred host "
        "unit and cite the matching VIII. USEFUL MINERALS chunk for provenance.",
        "Test the falsifiable claim that reduced (chalcopyrite/bornite) vs "
        "oxidized (malachite/azurite) Cu mineralisation correlates with depth "
        "in outputs/boreholes_copper_master_36572.csv. One polars/duckdb "
        "aggregation on max_cu_pct grouped by depth band is enough evidence.",
        "Treat synthesis/copper-report-1984.md as a starting hypothesis - it "
        "argues for sediment-hosted Cu in Carboniferous-Permian redbeds at the "
        "redox boundary. Verify its named units against the primary VIII. USEFUL "
        "MINERALS chunks before they enter the graph; the synthesis is secondary.",
    ),
    "smolianova_geojson": (
        "Anchor the graph on a `location` node taken from "
        "spatial/map_sheets_36572.geojson, then attach one observed_at or `at` "
        "edge from a stratigraphic_unit or sample described on that sheet.",
        "Pick one sheet from spatial/by_sheet/ and use its README to motivate "
        "a contact node between two units the sheet describes as in-contact; "
        "cite the README path in provenance.reference.",
    ),
    "smolianova_basic": (
        "Pick two named units that the knowledge_base STRATIGRAPHY chunks "
        "describe as in-contact and capture the overlies relation with a "
        "provenance.reference pointing at the specific chunk file.",
    ),
    "smolianova_mixed_modality": (
        "Use one knowledge_base chunk, one outputs/*.csv aggregation, and one "
        "spatial/*.geojson feature together: the resulting graph should cite "
        "all three modalities across its node/edge provenance.",
    ),
}


def _select_objective(variation_name: str, pool: tuple[str, ...], seed: str) -> str | None:
    """Pick one objective deterministically from ``pool`` for an episode.

    Hashing (variation_name, seed) keeps reproducibility from logs and avoids
    cross-variation collisions when the same workspace id surfaces twice."""
    if not pool:
        return None
    digest = hashlib.sha256(f"{variation_name}:{seed}".encode()).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


_GRAPH_SCHEMA = """\
Graph schema (every node and edge needs `provenance`):

  Node kinds (snake_case):
    stratigraphic_unit  unit_id, series_id, topology ("layer" | "embedded"),
                        anchor_inside=[x,y,z] (embedded only),
                        lithology? (Categorical), age_ma? (UncertaintyValue),
                        bulk_volume_bounds? (Interval)
    contact             position=[x,y,z], between=[unit_id_a, unit_id_b],
                        polarity? in {-1, 1}
    orientation         position=[x,y,z], dip (Orientation), for_unit
    fault               surface_points=[contact_id, ...]
    observation_point   position=[x,y,z], notes
    location            position=[x,y,z], name?
    sample              analyte, unit_of_measure, value (UncertaintyValue),
                        position? (else linked to a Location via `at`)
    series              series_id?, name?

  Edge kinds (closed vocabulary; `overlies` is the most common):
    overlies            source = upper unit, target = lower unit
    in_contact_with     undirected contact between two units
    offset_by           node -> fault
    member_of_series    unit -> series
    observed_at         node -> observation_point
    within              node -> container_node
    at                  sample -> location

  Provenance object (required on every node and edge):
    {"source": "agent" | <tool>, "confidence": 0..1, "timestamp": ISO-8601,
     "reference"?: <file path or citation>, "agent"?: <id>}

  Optional `p_exists` in [0, 1] (default 1.0) for existence uncertainty.

  UncertaintyValue envelopes (use these wherever a field expects an
  UncertaintyValue - lithology, age_ma, sample.value, dip components):
    {"kind": "Point",       "value": 0.42}
    {"kind": "Gaussian",    "mean": 0.42, "std": 0.05}
    {"kind": "Interval",    "lo": 0.30,   "hi": 0.55}
    {"kind": "Categorical", "probs": {"sandstone": 0.6, "shale": 0.4}}
  Plain strings or numbers in these fields are rejected by validation.
"""


_PATCH_GRAMMAR = """\
Patch operations (the `operations` list for refine_commit and
graph_apply_patch). 1-3 ops per call is the sweet spot.

  {"op": "add_node",    "node": <full node JSON>}
  {"op": "add_edge",    "edge": <full edge JSON>}
  {"op": "update_node", "node_id": <existing id>, "patch": {<fields to set>}}
  {"op": "remove_node", "node_id": <id>}          # cascades incident edges
  {"op": "remove_edge", "edge_id": <id>}

`update_node` requires `node_id` to exist in the reference graph. List
candidate ids first:

  mcp_explore_call(tool="graph_query",
                   args={"graph_uri": "<ref_a>",
                         "selector": {"kind": "stratigraphic_unit"},
                         "limit": 50})

Fully-typed add_node example (drop straight into operations):
  {
    "op": "add_node",
    "node": {
      "kind": "stratigraphic_unit",
      "id": "u_perm_lower",
      "unit_id": "permian_lower_division",
      "series_id": "permian",
      "topology": "layer",
      "lithology": {"kind": "Categorical",
                    "probs": {"sandstone": 0.6, "shale": 0.4}},
      "age_ma":    {"kind": "Interval", "lo": 252, "hi": 299},
      "provenance": {
        "source": "agent",
        "reference": "knowledge_base/chunks/36572_055_V_STRATIGRAPHY_Permian_Lower_Division.md",
        "confidence": 0.7,
        "timestamp": "2026-05-17T00:00:00Z"
      }
    }
  }

Fully-typed add_edge example:
  {
    "op": "add_edge",
    "edge": {
      "kind": "overlies",
      "source": "u_perm_lower",
      "target": "u_kumansai",
      "provenance": {"source": "agent",
                     "reference": "stratigraphic narrative",
                     "confidence": 0.8,
                     "timestamp": "2026-05-17T00:00:00Z"}
    }
  }
"""


_TOOL_SURFACE = """\
Tool surface. Each capability is a grouped router - call it as
capabilities__<capability>(tool="<inner>", args={...}). Returns are JSON.

  mcp_explore_call       graph_query(graph_uri, selector, limit) -> {nodes, edges}
                         graph_subgraph(graph_uri, seed_nodes, radius)
                         graph_diff(graph_uri_a, graph_uri_b)
                         graph_provenance(graph_uri, node_id)
                         workspace_get(uri)
                         data_ingest(filename, content_text|base64) -> {data_uri}

  mcp_hypothesise_call   hypothesis_create(statement, rationale?, graph_refs?)
                            -> {hypothesis_uri, statement}
                         hypothesis_get(hypothesis_uri) / hypothesis_list

  mcp_execute_call       engine_run_preview(graph_ref, field_spec) -> {field_uri} | {job_uri}
                         engine_run(...), job_status(job_uri)
                         voxel_sample(field_uri, points), voxel_stats(field_uri)
                         hypothesis_get(hypothesis_uri)

  refine_commit          (atomic) reference_graph_uri, operations, message
                            -> {candidate_graph_uri, candidate_field_uri, ...}
                         Prefer this over branch+patch+commit by hand.

  candidate_submit_and_report
                         candidate_graph_uri, reference_pair,
                         predicted_score_bits, gate_failures
                            -> {scored, ...}            # terminates submit

  seed_graph_submit      (bootstrap) filename, content_text,
                         predicted_passed_gates, gate_failures
                            -> {seed_graph_uri, ...}    # terminates submit_seed

  analysis_shell         command (python, rg, jq, ... in g2v:analysis)
  promote_analysis_artifact   move /workspace/out file into g2v imports
  record_phase           close the current step with per-phase keys
  phase_get              read back a prior step's recorded payload
"""


_ANALYSIS_ENV = f"""\
Analysis environment (analysis_shell runs in container `g2v:analysis`):

  I/O:  {{input}}   read-only dataset mount
        {{out}}     writable tmpfs (~512 MB)
        promote_analysis_artifact(src_path=...) bridges files into g2v imports.

  Shell:   rg, jq, file, less, head, sed, awk
  Python:  pandas, polars, numpy, scipy, pyarrow, duckdb, geopandas,
           shapely, pyproj, fiona, rasterio  (`python -c "..."` or python <<'PY')

  One rich call beats many head calls. Useful patterns:
    rg -l "Kumansai" {{input}}/knowledge_base
    python -c "import polars as pl; print(pl.read_csv('{{input}}/outputs/boreholes_copper_master_36572.csv').describe())"
    python -c "import duckdb; print(duckdb.read_csv('{{input}}/outputs/boreholes_copper_master_36572.csv').aggregate('preferred_sheet, COUNT(*)').df())"
    python -c "import geopandas as gpd; print(gpd.read_file('{{input}}/spatial/map_sheets_36572.geojson').head())"

  Defaults: timeout_s=30, max_output_bytes=20000 (bump as needed).
""".format(input=_ANALYSIS_INPUT, out=_ANALYSIS_OUT)


_DATASET_OVERVIEW_FALLBACK = (
    "Dataset overview unavailable - inspect "
    f"{_ANALYSIS_INPUT} via analysis_shell (start with README.md)."
)


def _load_dataset_overview(dataset_dir: Path) -> str:
    """Read the dataset README into a compact prompt block.

    Capped at ~1.5 KB. PROVENANCE.md is intentionally skipped - it
    documents ETL lineage, not what the agent should do with the data.
    """
    readme = dataset_dir / "README.md"
    if not readme.is_file():
        return _DATASET_OVERVIEW_FALLBACK
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _DATASET_OVERVIEW_FALLBACK
    kept: list[str] = []
    used = 0
    budget = 1500
    for line in text.splitlines()[:60]:
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    snippet = "\n".join(kept).rstrip()
    return snippet or _DATASET_OVERVIEW_FALLBACK


class _G2VWorkerClient:
    """Synchronous JSONL client over a Docker exec-start socket."""

    def __init__(self, container: Container, episode_workspace_id: str) -> None:
        self._container = container
        self._lock = threading.Lock()
        self._stderr_tail = ""
        self._stdout_buffer = b""
        cmd = [
            "python",
            "-m",
            "tasks.common.g2v_worker",
            "--workspace",
            _G2V_WORKSPACE,
            "--imports-subdir",
            f"imports/{episode_workspace_id}",
            "--line-protocol",
            "stdio",
        ]
        api = container.client.api
        self._exec_id = api.exec_create(
            container.id,
            cmd,
            stdin=True,
            stdout=True,
            stderr=True,
            tty=False,
        )["Id"]
        self._socket = api.exec_start(self._exec_id, socket=True, tty=False)
        self._raw_socket = getattr(self._socket, "_sock", self._socket)
        settimeout = getattr(self._raw_socket, "settimeout", None)
        if callable(settimeout):
            settimeout(120.0)

    def close(self) -> None:
        close = getattr(self._socket, "close", None)
        if callable(close):
            close()

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        request = json.dumps({"tool": tool, "args": args}, sort_keys=True).encode() + b"\n"
        with self._lock:
            try:
                self._sendall(request)
                line = self._read_stdout_line()
                return json.loads(line.decode())
            except BaseException as exc:  # noqa: BLE001 - sentinel for task scoring
                return {
                    "error": "g2v_worker_crash",
                    "type": type(exc).__name__,
                    "detail": str(exc),
                    "stderr_tail": self._stderr_tail[-2000:],
                }

    def _sendall(self, data: bytes) -> None:
        sendall = getattr(self._raw_socket, "sendall", None)
        if callable(sendall):
            sendall(data)
            return
        write = getattr(self._socket, "write", None)
        if callable(write):
            write(data)
            flush = getattr(self._socket, "flush", None)
            if callable(flush):
                flush()
            return
        raise RuntimeError("Docker exec socket is not writable")

    def _recv_exact(self, n: int) -> bytes:
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            recv = getattr(self._raw_socket, "recv", None)
            if callable(recv):
                chunk = recv(remaining)
            else:
                chunk = self._socket.read(remaining)
            if not chunk:
                raise EOFError("g2v worker stream closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_frame(self) -> tuple[int, bytes]:
        header = self._recv_exact(8)
        if header[0] in {1, 2} and header[1:4] == b"\x00\x00\x00":
            size = int.from_bytes(header[4:8], "big")
            return header[0], self._recv_exact(size) if size else b""
        return 1, header

    def _read_stdout_line(self) -> bytes:
        while b"\n" not in self._stdout_buffer:
            stream, payload = self._read_frame()
            if stream == 2:
                self._stderr_tail = (self._stderr_tail + payload.decode(errors="replace"))[-4000:]
                continue
            self._stdout_buffer += payload
        line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        return line


@dataclass
class GeologyGraphVariation(Variation):
    dataset_dir: str = ""
    pool_dir: str = ""
    min_pool_size: int = 2
    x_warmup_episodes: int = 4
    t_steady: float = 120.0
    t_initial: float | None = None
    anneal_window: int = 8
    pool_capacity: int = 16
    coverage_threshold: float = 0.95
    epsilon: float = 0.01
    effective_sample_size: float = 32.0
    dedup_epsilon: float = 1e-6
    max_promote_bytes: int = _MAX_PROMOTE_BYTES
    field_spec: dict[str, Any] = field(
        default_factory=lambda: {
            "grid_origin": [0.0, 0.0, 0.0],
            "grid_maximum": [10.0, 10.0, 10.0],
            "grid_shape": [8, 8, 8],
            "min_membership": 0.01,
        }
    )
    objective_pool: tuple[str, ...] = ()

    @property
    def criterion_config(self) -> dict[str, Any]:
        return {
            "effective_sample_size": self.effective_sample_size,
            "epsilon": self.epsilon,
            "coverage_threshold": self.coverage_threshold,
            "dedup_epsilon": self.dedup_epsilon,
        }

    @property
    def initial_threshold(self) -> float:
        return float(self.t_initial if self.t_initial is not None else 2.0 * self.t_steady)


@dataclass
class GeologyGraphState:
    workflow_kind: str
    pool_snapshot: dict[str, Any]
    dataset_snapshot: dict[str, str]
    phase_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    terminal_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    score_bits: float = math.inf
    structural_bits: float = math.inf
    fit_bits: float = math.inf
    physics_bits: float = math.inf
    passed_gates: bool = False
    gate_failures: list[str] = field(default_factory=list)
    admission_threshold: float = math.inf
    t_steady: float = 120.0
    x_warmup_episodes: int = 0
    episode_count: int = 0
    admission_count: int = 0
    calibration_error_bits: float | None = None
    budget_exhausted_step: str | None = None
    phase_budget_used: dict[str, int] = field(default_factory=dict)
    cheating_detected: str | None = None
    dataset_drift_paths: list[str] = field(default_factory=list)
    scorer_result: dict[str, Any] = field(default_factory=dict)


class GeologyProposerRows:
    included_workflow_steps = (
        "explore",
        "explore_data",
        "hypothesise",
        "refine",
        "submit",
        "submit_seed",
    )

    @property
    def name(self) -> str:
        return "GeologyProposerRows[v1]"

    def config(self) -> dict[str, Any]:
        return {"included_workflow_steps": list(self.included_workflow_steps)}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        allowed = set(self.included_workflow_steps)
        transformed: list[EpisodeTrainingRows] = []
        for episode in episodes:
            rows: list[dict[str, Any]] = []
            for row in episode.rows:
                workflow_step = row.get("workflow_step")
                if workflow_step is None:
                    raise ValueError("geology export row is missing workflow_step")
                if workflow_step in allowed:
                    rows.append(row)
            transformed.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=episode.episode_score,
                    rows=rows,
                )
            )
        return transformed


class GeologyGraphTask(TaskSpec[GeologyGraphState]):
    name = "geology-graph"
    description = "Refine geological world-model graphs through hypothesise-experiment-refine cycles."
    metric_name = "refinement_reward"
    metric_unit = "fraction"
    higher_is_better = True
    agent_service_name = "agent"

    def __init__(self, task_config: dict[str, Any]) -> None:
        self._docker_compose_dir = task_config.get(
            "docker_compose_dir", "docker/geology-graph-compose"
        )
        repo_root = Path(__file__).resolve().parent.parent
        default_dataset = repo_root / "data" / "geology" / "36572_smolianova_1984"
        self._dataset_dir = Path(task_config.get("dataset_dir", default_dataset)).resolve()
        default_pool_root = repo_root / "tasks" / "geology_graph" / "pools"
        self._pool_root = Path(task_config.get("pool_root", default_pool_root)).resolve()
        self._t_steady = float(task_config.get("t_steady", 120.0))
        variation_names = task_config.get("variation_names")
        if variation_names is None:
            self._variation_names: set[str] | None = None
        elif isinstance(variation_names, str):
            self._variation_names = {variation_names}
        else:
            self._variation_names = {str(name) for name in variation_names}

    @property
    def docker_compose_dir(self) -> str:
        return self._docker_compose_dir

    def training_data_transforms(self) -> tuple[GeologyProposerRows, ...]:
        return (GeologyProposerRows(),)

    def list_variations(self) -> list[Variation]:
        dataset = str(self._dataset_dir)
        base = [
            ("smolianova_basic", 2, 4, "Small Smolianova 1984 text/CSV/GeoJSON geology dataset."),
            ("smolianova_geojson", 3, 6, "Smolianova dataset with stronger spatial-layer emphasis."),
            ("smolianova_copper", 2, 8, "Copper evidence and borehole-table emphasis."),
            ("smolianova_mixed_modality", 3, 12, "Mixed text, table, and GeoJSON stress variation."),
        ]
        variations = [
            GeologyGraphVariation(
                name=name,
                description=description,
                dataset_dir=dataset,
                pool_dir=str(self._pool_root / name),
                min_pool_size=min_pool_size,
                x_warmup_episodes=warmup,
                t_steady=self._t_steady,
                anneal_window=max(4, warmup),
                objective_pool=_VARIATION_OBJECTIVES.get(name, ()),
            )
            for name, min_pool_size, warmup, description in base
        ]
        if self._variation_names is None:
            return variations
        known = {variation.name for variation in variations}
        missing = sorted(self._variation_names - known)
        if missing:
            raise ValueError(
                f"Unknown geology variation_names {missing}; known variations: {sorted(known)}"
            )
        return [variation for variation in variations if variation.name in self._variation_names]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def populate(
        self,
        containers: list[Container],
        variation: Variation,
    ) -> PopulationOutcome:
        if not isinstance(variation, GeologyGraphVariation):
            raise TypeError("GeologyGraphTask requires GeologyGraphVariation")
        with self._pool_lock(variation):
            self._ensure_pool(variation)
            pool_state = self._read_pool_index(variation)
            pool_state["episode_count"] = int(pool_state.get("episode_count", 0)) + 1
            self._write_pool_index(variation, pool_state)

        pool_graphs = list(pool_state.get("graphs", []))
        graph_ids = [f"g2v://graph/{item['hash']}" for item in pool_graphs]
        workflow_kind = "bootstrap" if len(graph_ids) < variation.min_pool_size else "regular"
        episode_workspace_id = f"episode_{int(time.time() * 1000)}_{os.getpid()}"

        g2v = self._maybe_pick_container(containers, "g2v")
        analysis = self._maybe_pick_container(containers, "analysis")
        worker: _G2VWorkerClient | None = None
        if g2v is not None:
            self._prepare_g2v_workspace(g2v, episode_workspace_id)
            try:
                worker = _G2VWorkerClient(g2v, episode_workspace_id)
            except Exception as exc:
                logger.warning(f"failed to start long-lived g2v worker; falling back to one-shot exec: {exc}")
            for item in pool_graphs:
                graph_path = Path(variation.pool_dir) / item["path"]
                if graph_path.exists():
                    args = {
                        "filename": graph_path.name,
                        "content_text": graph_path.read_text(encoding="utf-8"),
                        "message": "registered pool graph for episode",
                        "tags": {"pool_hash": item["hash"]},
                    }
                    if worker is not None:
                        worker.call("graph_ingest", args)
                    else:
                        self._g2v_exec_call(g2v, episode_workspace_id, "graph_ingest", args)

        assigned_references = self._select_reference_pair(graph_ids, variation.name)
        dataset_snapshot = self._snapshot_dataset(analysis, Path(variation.dataset_dir))
        dataset_overview = _load_dataset_overview(Path(variation.dataset_dir))
        pool_snapshot = {
            "graph_ids": graph_ids,
            "graph_hashes": {f"g2v://graph/{item['hash']}": item["hash"] for item in pool_graphs},
            "min_pool_size": variation.min_pool_size,
            "episode_count": int(pool_state.get("episode_count", 0)),
            "admission_count": int(pool_state.get("admission_count", 0)),
        }
        episode_context: dict[str, Any] = {
            "variation_name": variation.name,
            "workflow_kind": workflow_kind,
            "dataset_dir": variation.dataset_dir,
            "pool_dir": variation.pool_dir,
            "pool_snapshot": pool_snapshot,
            "dataset_snapshot": dataset_snapshot,
            "dataset_overview": dataset_overview,
            "assigned_references": assigned_references,
            "phase_records": {},
            "terminal_records": {},
            "episode_workspace_id": episode_workspace_id,
            "g2v_workspace": _G2V_WORKSPACE,
            "g2v_imports_dir": f"{_G2V_WORKSPACE}/imports/{episode_workspace_id}",
            "field_spec": dict(variation.field_spec),
            "selected_objective": _select_objective(
                variation.name, variation.objective_pool, episode_workspace_id
            ),
        }
        if worker is not None:
            episode_context["_g2v_worker"] = worker
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
        if not results:
            results = [
                PopulationResult(
                    container_id="host",
                    variation_name=variation.name,
                    description=variation.description,
                    success=True,
                    details={"service": "host"},
                )
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
        return bool(episode_context.get("dataset_snapshot") is not None)

    # ------------------------------------------------------------------
    # Prompt and workflow
    # ------------------------------------------------------------------

    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> TaskPromptSpec:
        assert isinstance(variation, GeologyGraphVariation)
        workflow_kind = episode_context.get("workflow_kind", "bootstrap")
        assigned = episode_context.get("assigned_references") or {}
        pool_ids = (episode_context.get("pool_snapshot") or {}).get("graph_ids", [])
        dataset_overview = episode_context.get("dataset_overview") or _DATASET_OVERVIEW_FALLBACK
        if workflow_kind == "bootstrap":
            mission = (
                "Bootstrap: the pool is empty. Submit one structurally valid seed graph "
                "(stratigraphic units + an overlies edge) using units you read in the "
                "dataset. No field exists yet, so engine / voxel tools do not apply."
            )
        else:
            mission = (
                "Refine: two reference graphs A and B are in the pool. Find a structural "
                "disagreement between them, justify a small patch from the data, commit "
                "the candidate, and submit it. An external criterion scores it against "
                "A and B."
            )
        focus = episode_context.get("selected_objective")
        focus_line = f"\nEpisode focus: {focus}" if focus else ""
        environment = (
            f"Workflow: {workflow_kind}\n"
            f"{mission}\n"
            f"Dataset (read-only): {_ANALYSIS_INPUT}\n"
            f"Host path: {variation.dataset_dir}\n"
            f"G2V imports: {episode_context.get('g2v_imports_dir', '<created during populate>')}\n"
            f"Pool graphs: {json.dumps(pool_ids)}\n"
            f"Assigned references: {json.dumps(assigned)}"
            f"{focus_line}"
        )
        system_instruction = (
            "You are a geology agent that maintains a probabilistic graph world-model "
            "of a regional geological survey. Each episode you either seed the pool "
            "with a small valid graph (bootstrap) or refine one reference graph into a "
            "stronger candidate scored against two references (refine).\n"
            "\n"
            "=== Dataset ===\n"
            f"{dataset_overview}\n"
            "\n"
            f"=== {_GRAPH_SCHEMA.rstrip()}\n"
            "\n"
            f"=== {_PATCH_GRAMMAR.rstrip()}\n"
            "\n"
            f"=== {_TOOL_SURFACE.rstrip()}\n"
            "\n"
            f"=== {_ANALYSIS_ENV.rstrip()}"
        )
        return TaskPromptSpec(
            system_instruction=system_instruction,
            environment_context=environment,
            capabilities=self._capabilities(),
        )

    def workflow(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> Workflow | None:
        assert isinstance(variation, GeologyGraphVariation)
        snapshot = episode_context.get("pool_snapshot") or {}
        pool_size = len(snapshot.get("graph_ids", []))
        if pool_size < variation.min_pool_size:
            return self._bootstrap_workflow(variation, episode_context)
        return self._regular_workflow(variation, episode_context)

    def episode_constraints(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> EpisodeConstraints:
        workflow_kind = episode_context.get("workflow_kind")
        if workflow_kind == "regular" or len((episode_context.get("pool_snapshot") or {}).get("graph_ids", [])) >= getattr(variation, "min_pool_size", 2):
            return EpisodeConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=200, max_llm_turns=224),
                success=SuccessConstraints(terminal_capability_for_success="candidate_submit_and_report"),
                step_overrides={
                    "explore": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=40, max_llm_turns=48)),
                    "hypothesise": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=64, max_llm_turns=32)),
                    "execute": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=130, max_llm_turns=72)),
                    "refine": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=180, max_llm_turns=52)),
                    "submit": StepConstraints(
                        budgets=BudgetConstraints(max_task_tool_calls=200, max_llm_turns=20),
                        success=SuccessConstraints(terminal_capability_for_success="candidate_submit_and_report"),
                    ),
                },
            )
        return EpisodeConstraints(
            budgets=BudgetConstraints(max_task_tool_calls=150, max_llm_turns=148),
            success=SuccessConstraints(terminal_capability_for_success="seed_graph_submit"),
            step_overrides={
                "explore_data": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=28, max_llm_turns=28)),
                "hypothesise": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=48, max_llm_turns=20)),
                "execute": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=108, max_llm_turns=56)),
                "submit_seed": StepConstraints(
                    budgets=BudgetConstraints(max_task_tool_calls=150, max_llm_turns=44),
                    success=SuccessConstraints(terminal_capability_for_success="seed_graph_submit"),
                ),
            },
        )

    def _regular_workflow(self, variation: GeologyGraphVariation, episode_context: dict[str, Any]) -> Workflow:
        del variation
        assigned = episode_context.get("assigned_references") or {}
        ref_a = assigned.get("reference_a", "<assigned-reference-a>")
        ref_b = assigned.get("reference_b", "<assigned-reference-b>")
        focus = episode_context.get("selected_objective")
        focus_block = f"\nEpisode focus: {focus}\n" if focus else ""
        return Workflow(
            steps=(
                WorkflowStep(
                    name="explore",
                    description="Inspect assigned pool references and dataset evidence.",
                    prompt=(
                        "Step 1 of 5 - explore.\n"
                        f"References: A = {ref_a}, B = {ref_b}.\n"
                        f"{focus_block}"
                        "\n"
                        "Goal: spot how A and B differ (unit ordering, extra/missing contact, "
                        "different lithology, fault). The disagreement is the seed of your "
                        "hypothesis.\n"
                        "\n"
                        "Useful moves (pick what fits):\n"
                        f"  mcp_explore_call(tool='graph_diff', args={{'graph_uri_a': '{ref_a}', 'graph_uri_b': '{ref_b}'}})\n"
                        f"  mcp_explore_call(tool='graph_query', args={{'graph_uri': '{ref_a}', 'selector': {{'kind': 'stratigraphic_unit'}}, 'limit': 50}})\n"
                        "    Note the returned node `id` values - those are what you'll patch by in refine.\n"
                        f"  mcp_explore_call(tool='graph_subgraph', args={{'graph_uri': '{ref_a}', 'seed_nodes': [], 'radius': 2}})\n"
                        "  analysis_shell to grep the dataset for any unit that appears in only one of A or B.\n"
                        "\n"
                        "Close with:\n"
                        "  record_phase(phase='explore',\n"
                        f"               reference_a='{ref_a}', reference_b='{ref_b}',\n"
                        "               rationale='<2 sentences: the disagreement and which side the data supports>',\n"
                        "               candidate_node_ids=['<id1>', '<id2>'])   # ids you may patch in refine"
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("mcp_explore_call", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    is_entry=True,
                    next_steps=("hypothesise",),
                ),
                WorkflowStep(
                    name="hypothesise",
                    description="Register a falsifiable geological hypothesis.",
                    prompt=(
                        "Step 2 of 5 - hypothesise.\n"
                        "\n"
                        "Goal: register one falsifiable claim that, if true, justifies a specific patch in refine.\n"
                        "\n"
                        "Good example:\n"
                        "  statement: 'Contact between permian_lower_division and ordovician_kumansai_suite_O2km has polarity -1 (Permian above Ordovician) on map sheet M-42.'\n"
                        "  rationale: 'Permian_Lower_Division.md describes the unit resting on Ordovician basement; Kumansai is the named Ordovician unit there.'\n"
                        "\n"
                        "Call:\n"
                        "  mcp_hypothesise_call(tool='hypothesis_create',\n"
                        f"                       args={{'statement': ..., 'rationale': ..., 'graph_refs': ['{ref_a}']}})\n"
                        "Response carries hypothesis_uri.\n"
                        "\n"
                        "Close with:\n"
                        "  record_phase(phase='hypothesise', hypothesis_uri=<from response>,\n"
                        "               statement=<echo>, rationale=<echo>)"
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("mcp_hypothesise_call", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    next_steps=("execute",),
                ),
                WorkflowStep(
                    name="execute",
                    description="Run a focused experiment against the hypothesis.",
                    prompt=(
                        "Step 3 of 5 - execute (isolated context).\n"
                        "\n"
                        "Goal: one piece of evidence that supports or weakens the hypothesis.\n"
                        "\n"
                        "Recover:\n"
                        "  phase_get(phase='hypothesise')   # {hypothesis_uri, statement, rationale}\n"
                        "\n"
                        "Pick one check:\n"
                        f"  rg -l '<unit>' {_ANALYSIS_INPUT}/knowledge_base\n"
                        f"  python -c \"import polars as pl; print(pl.read_csv('{_ANALYSIS_INPUT}/outputs/<file>.csv').filter(...))\"\n"
                        f"  mcp_execute_call(tool='engine_run_preview', args={{'graph_ref': '{ref_a}', 'field_spec': {{...}}}})\n"
                        "    then voxel_sample / voxel_stats - only when the hypothesis is about spatial occupancy.\n"
                        "\n"
                        "Close with:\n"
                        "  record_phase(phase='execute', status='ok',\n"
                        "               experiment_result={'summary': '<what you measured>',\n"
                        "                                  'evidence_path': '<file>'})\n"
                        "  status='unparameterisable' if no numeric check fits; 'engine_failed' if a tool errored."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("mcp_execute_call", "phase_get", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    next_steps=("refine",),
                    context_mode="isolated",
                ),
                WorkflowStep(
                    name="refine",
                    description="Commit a patched candidate graph based on the experiment.",
                    prompt=(
                        "Step 4 of 5 - refine (isolated context).\n"
                        "\n"
                        "Goal: produce a candidate_graph_uri by patching reference A.\n"
                        "\n"
                        "Recover:\n"
                        "  phase_get(phase='explore')       # {reference_a, reference_b, rationale, candidate_node_ids?}\n"
                        "  phase_get(phase='hypothesise')   # {hypothesis_uri, statement, rationale}\n"
                        "  phase_get(phase='execute')       # {experiment_result, status}\n"
                        "\n"
                        "If you intend update_node or remove_node, list real ids first:\n"
                        f"  mcp_explore_call(tool='graph_query',\n"
                        f"                   args={{'graph_uri': '{ref_a}', 'selector': {{'kind': 'stratigraphic_unit'}}, 'limit': 50}})\n"
                        "  Patch by the returned `id`, not by a unit_id you have in mind.\n"
                        "\n"
                        "Build operations from the system-prompt patch grammar (lithology / age_ma\n"
                        "use UncertaintyValue envelopes - not plain strings/numbers).\n"
                        "\n"
                        "Atomic call:\n"
                        "  refine_commit(\n"
                        f"    reference_graph_uri='{ref_a}',\n"
                        "    operations=[<1-3 ops>],\n"
                        "    message='<one line: hypothesis + evidence>')\n"
                        "Response carries candidate_graph_uri and candidate_field_uri.\n"
                        "\n"
                        "refine_commit auto-records the phase for you; an explicit\n"
                        "record_phase(phase='refine', candidate_graph_uri=..., candidate_field_uri=...)\n"
                        "still works if you prefer."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("refine_commit", "phase_get", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    next_steps=("submit",),
                    context_mode="isolated",
                ),
                WorkflowStep(
                    name="submit",
                    description="Submit candidate and self-assess expected score.",
                    prompt=(
                        "Step 5 of 5 - submit.\n"
                        "\n"
                        "Recover:\n"
                        "  phase_get(phase='refine')   # {candidate_graph_uri, candidate_field_uri}\n"
                        "\n"
                        "One call:\n"
                        "  candidate_submit_and_report(\n"
                        "    candidate_graph_uri=<from refine>,\n"
                        f"    reference_pair=['{ref_a}', '{ref_b}'],\n"
                        "    predicted_score_bits=<float; lower = more confident; typical 0..15>,\n"
                        "    gate_failures=[])   # add 'coverage' | 'structural' | 'fit' if you expect any"
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("candidate_submit_and_report", "phase_get"),
                    terminator_capabilities=("candidate_submit_and_report",),
                ),
            )
        )

    def _bootstrap_workflow(self, variation: GeologyGraphVariation, episode_context: dict[str, Any]) -> Workflow:
        del variation
        focus = episode_context.get("selected_objective")
        focus_block = f"\nEpisode focus: {focus}\n" if focus else ""
        return Workflow(
            steps=(
                WorkflowStep(
                    name="explore_data",
                    description="Inspect the raw dataset before proposing a seed graph.",
                    prompt=(
                        "Step 1 of 4 - explore_data (bootstrap).\n"
                        f"{focus_block}"
                        "\n"
                        "Goal: pick 2-4 real lithostratigraphic units from the dataset and one ordering between them.\n"
                        "\n"
                        "Good first moves:\n"
                        f"  analysis_shell(command='ls {_ANALYSIS_INPUT} && head -n 60 {_ANALYSIS_INPUT}/README.md')\n"
                        f"  analysis_shell(command='ls {_ANALYSIS_INPUT}/knowledge_base/chunks | rg STRATIGRAPHY | head')\n"
                        f"  analysis_shell(command='head -n 60 {_ANALYSIS_INPUT}/knowledge_base/chunks/<one stratigraphy file>')\n"
                        "  For tables/geojson, one python -c invocation with polars / duckdb / geopandas beats many head calls (see Analysis environment).\n"
                        "\n"
                        "Close with:\n"
                        "  record_phase(phase='explore_data',\n"
                        "               dataset_summary='<one line: domain + 2-4 unit names, e.g. \"Kazakhstan M-42 Cu survey; Proterozoic Group, Kumansai O2km, Permian Lower Division\">',\n"
                        "               cited_files=['<file>', '<file>'])"
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("mcp_explore_call", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    is_entry=True,
                    next_steps=("hypothesise",),
                ),
                WorkflowStep(
                    name="hypothesise",
                    description="Register a hypothesis to guide the seed graph.",
                    prompt=(
                        "Step 2 of 4 - hypothesise (bootstrap).\n"
                        "\n"
                        "Goal: register one falsifiable ordering or contact claim naming two of the units you found.\n"
                        "\n"
                        "Good example:\n"
                        "  statement: 'permian_lower_division overlies ordovician_kumansai_suite_O2km on map sheet M-42.'\n"
                        "  rationale: 'Permian_Lower_Division.md describes the unit resting on Ordovician basement; Kumansai is the named Ordovician unit there.'\n"
                        "\n"
                        "Call:\n"
                        "  mcp_hypothesise_call(tool='hypothesis_create',\n"
                        "                       args={'statement': ..., 'rationale': ...})\n"
                        "Response carries hypothesis_uri.\n"
                        "\n"
                        "Close with:\n"
                        "  record_phase(phase='hypothesise', hypothesis_uri=<from response>,\n"
                        "               statement=<echo>, rationale=<echo>)"
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("mcp_hypothesise_call", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    next_steps=("execute",),
                ),
                WorkflowStep(
                    name="execute",
                    description="Cross-check the hypothesis against the dataset.",
                    prompt=(
                        "Step 3 of 4 - execute (bootstrap, isolated context).\n"
                        "\n"
                        "Goal: cross-check with one additional read.\n"
                        "\n"
                        "Recover:\n"
                        "  phase_get(phase='hypothesise')   # {hypothesis_uri, statement, rationale}\n"
                        "\n"
                        "Pick one check:\n"
                        f"  rg -l '<unit>' {_ANALYSIS_INPUT}/knowledge_base\n"
                        "  analysis_shell(command='head -n 60 <another stratigraphy chunk>')\n"
                        "\n"
                        "Close with:\n"
                        "  record_phase(phase='execute', status='ok',\n"
                        "               experiment_result={'summary': '<what the second source said>',\n"
                        "                                  'evidence_path': '<file>'})\n"
                        "  status='unparameterisable' if no relevant text exists."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("mcp_execute_call", "phase_get", "analysis_shell", "promote_analysis_artifact", "record_phase"),
                    terminator_capabilities=("record_phase",),
                    context_mode="isolated",
                    next_steps=("submit_seed",),
                ),
                WorkflowStep(
                    name="submit_seed",
                    description="Ingest and submit a structurally valid seed graph.",
                    prompt=(
                        "Step 4 of 4 - submit_seed (bootstrap, isolated context).\n"
                        "\n"
                        "Goal: submit one schema-valid seed graph using real units.\n"
                        "\n"
                        "Optional:\n"
                        "  phase_get(phase='hypothesise')   # to recall the units / rationale\n"
                        "\n"
                        "Build the JSON. Edit unit_id, series_id, and provenance.reference to match the units you actually read about; the template below is for shape only:\n"
                        "\n"
                        f"{_SEED_GRAPH_TEMPLATE_JSON}\n"
                        "\n"
                        "Submit with:\n"
                        "  seed_graph_submit(filename='seed.json',\n"
                        "                    content_text=<full JSON document as a string>,\n"
                        "                    predicted_passed_gates=true,\n"
                        "                    gate_failures=[])"
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("seed_graph_submit", "phase_get", "analysis_shell"),
                    terminator_capabilities=("seed_graph_submit",),
                    context_mode="isolated",
                ),
            )
        )

    # ------------------------------------------------------------------
    # Capability declarations and execution
    # ------------------------------------------------------------------

    def _capabilities(self) -> list[Capability]:
        caps = [
            self._phase_capability(name, tools, _PHASE_CAPABILITY_DESCRIPTIONS.get(name))
            for name, tools in _REGULAR_PHASE_TOOLS.items()
        ]
        caps.extend(
            [
                Capability(
                    name="analysis_shell",
                    description=(
                        "Run a bounded shell command in the scorer-free analysis container "
                        "(see the 'Analysis environment' section of the system prompt for the "
                        "full shell + Python library surface). Defaults: timeout_s=30, "
                        "max_output_bytes=20000. Prefer one rich invocation (rg, "
                        "duckdb/polars/geopandas via python -c) over many shallow head/cat "
                        "calls - it is cheaper and produces grounded evidence faster."
                    ),
                    runs_code=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_s": {"type": "integer", "default": 30},
                            "max_output_bytes": {"type": "integer", "default": 20000},
                        },
                        "required": ["command"],
                    },
                ),
                Capability(
                    name="promote_analysis_artifact",
                    description="Copy a file from /workspace/out in analysis into the g2v imports directory.",
                    schema={
                        "type": "object",
                        "properties": {
                            "src_path": {"type": "string"},
                            "name": {"type": "string"},
                        },
                        "required": ["src_path"],
                    },
                ),
                Capability(
                    name="record_phase",
                    description=(
                        "Close the current workflow step by recording its hand-off payload. "
                        "phase must equal the current step name. Per-phase expected keys: "
                        "explore_data -> dataset_summary, cited_files; "
                        "explore -> reference_a, reference_b, rationale; "
                        "hypothesise -> hypothesis_uri, statement, rationale; "
                        "execute -> status ('ok'|'unparameterisable'|'engine_failed'), "
                        "experiment_result; "
                        "refine -> candidate_graph_uri, candidate_field_uri. "
                        "Extra keys are stored verbatim and visible to later phase_get calls."
                    ),
                    schema={
                        "type": "object",
                        "properties": {"phase": {"type": "string", "enum": list(_ALL_PHASE_RECORDS)}},
                        "required": ["phase"],
                        "additionalProperties": True,
                    },
                ),
                Capability(
                    name="phase_get",
                    description=(
                        "Read back a prior step's recorded payload. Returns "
                        "{phase, payload, found}. payload contains whatever record_phase "
                        "stored, plus phase_recorded=true. If the agent skipped "
                        "record_phase, auto-record promotes the most recent "
                        "hypothesis_uri / candidate_graph_uri / candidate_field_uri returned "
                        "by g2v tools - so phase_get still yields a usable URI in that case. "
                        "Call at most once per phase per step; the result is cached for the "
                        "rest of the episode."
                    ),
                    schema={
                        "type": "object",
                        "properties": {"phase": {"type": "string", "enum": list(_ALL_PHASE_RECORDS)}},
                        "required": ["phase"],
                    },
                ),
                Capability(
                    name="refine_commit",
                    description=(
                        "Regular-workflow atomic refine. Branches reference_graph_uri, applies "
                        "operations as a single transactional patch, commits to a new immutable "
                        "graph, then runs engine_run_preview unless run_preview=false. "
                        "Operations follow the g2v patch grammar: "
                        "{'op': 'add_node'|'add_edge'|'remove_node'|'remove_edge'|'update_node', "
                        "<op-specific fields>}. message is the commit log line. "
                        "Returns {candidate_graph_uri, candidate_field_uri?, scratch_uri, "
                        "head_rev_uri, validation_report, engine_run_preview, refine_commit}. "
                        "On success the refine phase auto-records candidate_graph_uri."
                    ),
                    schema={
                        "type": "object",
                        "properties": {
                            "reference_graph_uri": {"type": "string"},
                            "operations": {"type": "array", "items": {"type": "object"}},
                            "message": {"type": "string"},
                            "run_preview": {"type": "boolean", "default": True},
                        },
                        "required": ["reference_graph_uri", "operations", "message"],
                    },
                ),
                Capability(
                    name="candidate_submit_and_report",
                    description=(
                        "Regular-workflow atomic submit. Registers candidate_graph_uri against "
                        "the assigned reference_pair and stores the agent's calibrated "
                        "self-assessment (predicted_score_bits, gate_failures) in one call. "
                        "predicted_score_bits is a non-negative float; lower means more "
                        "confident the candidate beats the pool. gate_failures is a list of "
                        "named gates you expect to fail (e.g. 'coverage', 'structural'); empty "
                        "for a confident submission. Terminates the submit step."
                    ),
                    publishes_metric=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "candidate_graph_uri": {"type": "string"},
                            "reference_pair": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "predicted_score_bits": {"type": ["number", "null"]},
                            "gate_failures": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["candidate_graph_uri", "reference_pair", "predicted_score_bits"],
                    },
                ),
                Capability(
                    name="seed_graph_submit",
                    description=(
                        "Bootstrap-only atomic seed admission. content_text must be the full "
                        "seed-graph JSON document as a string (NOT a placeholder, NOT a "
                        "filesystem path). The task ingests it through g2v, validates that the "
                        "returned seed_graph_uri matches g2v://graph/<hash>, and records both "
                        "the seed terminal and the bootstrap self-assessment "
                        "(predicted_passed_gates, gate_failures) in one call. Schema rules: "
                        "node kinds are snake_case and from the closed vocabulary "
                        "(stratigraphic_unit, contact, orientation, fault, observation_point, "
                        "location, sample, series); edge kinds from {overlies, in_contact_with, "
                        "offset_by, member_of_series, observed_at, within, at}; every node and "
                        "edge needs a provenance object. Terminates the submit_seed step."
                    ),
                    publishes_metric=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Use seed.json unless there is a concrete reason not to."},
                            "content_text": {"type": "string", "description": "Complete graph JSON, not placeholder text."},
                            "predicted_passed_gates": {"type": "boolean"},
                            "gate_failures": {"type": "array", "items": {"type": "string"}},
                            "seed_field_uri": {"type": ["string", "null"]},
                        },
                        "required": ["filename", "content_text", "predicted_passed_gates"],
                    },
                ),
                Capability(
                    name="seed_submit",
                    description="Bootstrap-only terminal marker for the seed graph URI.",
                    schema={
                        "type": "object",
                        "properties": {
                            "seed_graph_uri": {"type": "string"},
                            "seed_field_uri": {"type": ["string", "null"]},
                        },
                        "required": ["seed_graph_uri"],
                    },
                ),
                Capability(
                    name="report_metric",
                    description="Terminal self-assessment. This is logged but never used for reward scoring.",
                    publishes_metric=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "predicted_score_bits": {"type": ["number", "null"]},
                            "predicted_passed_gates": {"type": ["boolean", "null"]},
                            "gate_failures": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                ),
            ]
        )
        return caps

    @staticmethod
    def _phase_capability(
        name: str,
        tools: tuple[str, ...],
        description: str | None = None,
    ) -> Capability:
        if description is None:
            description = (
                f"Call an allowed g2v tool for phase {name}. "
                f"Inner tools: {', '.join(tools)}."
            )
        return Capability(
            name=name,
            description=description,
            schema={
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": list(tools)},
                    "args": {"type": "object", "additionalProperties": True},
                },
                "required": ["tool", "args"],
            },
        )

    def parse_response(
        self,
        raw_response: str,
        *,
        invoked_capability: str | None = None,
    ) -> list[CapabilityInvocation]:
        """Parse simple JSON/tool-tag responses for orchestrator-shaped harnesses."""
        if invoked_capability:
            payload = _parse_json_object(raw_response)
            return [CapabilityInvocation(invoked_capability, payload)] if payload is not None else []
        invocations: list[CapabilityInvocation] = []
        for match in re.finditer(r"<tool\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</tool>", raw_response, re.DOTALL):
            payload = _parse_json_object(match.group(2)) or {}
            invocations.append(CapabilityInvocation(match.group(1), payload))
        return invocations

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: Variation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        if not isinstance(variation, GeologyGraphVariation):
            return CapabilityResult(invocation.name, success=False, error="invalid variation type")
        self._maybe_auto_record_previous_step(ctx)
        if invocation.name in _REGULAR_PHASE_TOOLS:
            return self._execute_mcp_phase(invocation, containers, variation, ctx)
        if invocation.name == "analysis_shell":
            return self._analysis_shell(invocation, containers)
        if invocation.name == "promote_analysis_artifact":
            return self._promote_analysis_artifact(invocation, containers, variation, ctx)
        if invocation.name == "record_phase":
            return self._record_phase(invocation, ctx)
        if invocation.name == "phase_get":
            return self._phase_get(invocation, ctx)
        if invocation.name == "refine_commit":
            return self._refine_commit(invocation, containers, variation, ctx)
        if invocation.name == "candidate_submit_and_report":
            return self._candidate_submit_and_report(invocation, containers, variation, ctx)
        if invocation.name == "seed_graph_submit":
            return self._seed_graph_submit(invocation, containers, variation, ctx)
        if invocation.name == "seed_submit":
            payload = dict(invocation.input)
            graph_uri = payload.get("seed_graph_uri")
            if not _is_g2v_graph_uri(graph_uri):
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error=(
                        "seed_graph_uri must be a 'g2v://graph/<hash>' URI returned by "
                        "graph_ingest; received "
                        f"{graph_uri!r}"
                    ),
                )
            field_uri = payload.get("seed_field_uri")
            if field_uri not in (None, "") and not _is_g2v_field_uri(field_uri):
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error=(
                        "seed_field_uri must be null or a 'g2v://field/<hash>' URI; received "
                        f"{field_uri!r}"
                    ),
                )
            ctx.episode_context.setdefault("terminal_records", {})["seed_submit"] = payload
            return CapabilityResult(name=invocation.name, output=payload, success=True)
        if invocation.name == "report_metric":
            payload = dict(invocation.input)
            payload.setdefault("gate_failures", [])
            ctx.episode_context.setdefault("terminal_records", {})["report_metric"] = payload
            return CapabilityResult(name=invocation.name, output=payload, success=True)
        return super().execute_capability(invocation, containers, variation, ctx)

    def _execute_mcp_phase(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: GeologyGraphVariation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        tool = invocation.input.get("tool")
        args = invocation.input.get("args") or {}
        allowed = _REGULAR_PHASE_TOOLS[invocation.name]
        if tool not in allowed:
            return CapabilityResult(
                name=invocation.name,
                output={"tool": tool, "criterion_probe": tool in {"ic_score", "ic_score_from_graphs"}},
                success=False,
                error=f"tool {tool!r} is not allowed for {invocation.name}",
            )
        if not isinstance(args, dict):
            return CapabilityResult(name=invocation.name, success=False, error="args must be an object")
        if tool == "candidate_submit":
            graph_uri = args.get("graph_uri") or args.get("candidate_graph_uri")
            if not _is_g2v_graph_uri(graph_uri):
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error=(
                        "candidate_submit requires graph_uri to be a 'g2v://graph/<hash>' URI; "
                        f"received {graph_uri!r}"
                    ),
                )
        return self._execute_g2v_tool(invocation.name, str(tool), args, containers, ctx)

    def _execute_g2v_tool(
        self,
        capability_name: str,
        tool: str,
        args: dict[str, Any],
        containers: list[Container],
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        g2v = self._maybe_pick_container(containers, "g2v")
        if g2v is None:
            shim = ctx.episode_context.get("_g2v_shim")
            if shim is not None:
                output = shim.dispatch(tool, args)
                result = CapabilityResult(name=capability_name, output=output, success="error" not in output, error=output.get("detail") or output.get("error"))
                self._auto_promote_uris(tool, result, ctx)
                return result
            return CapabilityResult(name=capability_name, output={}, success=False, error="no g2v container available")
        worker = ctx.episode_context.get("_g2v_worker")
        if isinstance(worker, _G2VWorkerClient):
            output = worker.call(tool, args)
        else:
            output = self._g2v_exec_call(g2v, ctx.episode_context.get("episode_workspace_id", ctx.episode_id), tool, args)
        result = CapabilityResult(name=capability_name, output=output, success="error" not in output, error=output.get("detail") or output.get("error"))
        self._auto_promote_uris(tool, result, ctx)
        return result

    @staticmethod
    def _auto_promote_uris(
        tool: str,
        result: CapabilityResult,
        ctx: CapabilityExecutionContext,
    ) -> None:
        """Stash URIs returned by g2v tools so auto-record can splice them in.

        The agent often calls hypothesis_create / refine_commit and then advances
        the workflow step without explicitly recording the returned URI. Without
        promotion, the subsequent phase_get only shows {phase_recorded: true},
        which has caused agents to enter retry loops. Promotion makes the URI
        available to the auto-record fallback even when the agent forgets.
        """
        if not result.success:
            return
        output = result.output or {}
        promoted = ctx.episode_context.setdefault("_auto_promoted", {})
        for key in (
            "hypothesis_uri",
            "candidate_graph_uri",
            "candidate_field_uri",
            "field_uri",
            "scratch_uri",
            "seed_graph_uri",
            "experiment_uri",
            "job_uri",
        ):
            value = output.get(key)
            if isinstance(value, str) and value:
                promoted[key] = value
        if tool == "hypothesis_create":
            stmt = output.get("statement")
            if isinstance(stmt, str):
                promoted["statement"] = stmt

    def _refine_commit(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: GeologyGraphVariation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        payload = dict(invocation.input)
        reference_uri = payload.get("reference_graph_uri") or payload.get("graph_uri")
        if not _is_g2v_graph_uri(reference_uri):
            return CapabilityResult(
                invocation.name,
                success=False,
                error=f"reference_graph_uri must be a 'g2v://graph/<hash>' URI; received {reference_uri!r}",
            )
        operations = payload.get("operations")
        if not isinstance(operations, list) or not all(isinstance(op, dict) for op in operations):
            return CapabilityResult(invocation.name, success=False, error="operations must be a list of objects")
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            return CapabilityResult(invocation.name, success=False, error="message must be a non-empty string")
        run_preview = payload.get("run_preview", True)
        if not isinstance(run_preview, bool):
            return CapabilityResult(invocation.name, success=False, error="run_preview must be boolean")

        commit_args = {"graph_uri": reference_uri, "operations": operations, "message": message}
        commit = self._execute_g2v_tool(invocation.name, "refine_commit", commit_args, containers, ctx)
        if not commit.success:
            return commit
        candidate_graph_uri = commit.output.get("candidate_graph_uri") or commit.output.get("graph_uri")
        if not _is_g2v_graph_uri(candidate_graph_uri):
            return CapabilityResult(
                invocation.name,
                output={"refine_commit": dict(commit.output)},
                success=False,
                error=(
                    "refine_commit expected g2v to return a 'g2v://graph/<hash>' URI; "
                    f"received {candidate_graph_uri!r}"
                ),
            )

        preview_payload: dict[str, Any] | None = None
        candidate_field_uri: str | None = None
        if run_preview:
            field_spec = ctx.episode_context.get("field_spec") or dict(variation.field_spec)
            if not isinstance(field_spec, dict):
                return CapabilityResult(invocation.name, success=False, error="field_spec must be an object")
            preview = self._execute_g2v_tool(
                invocation.name,
                "engine_run_preview",
                {"graph_ref": candidate_graph_uri, "field_spec": field_spec},
                containers,
                ctx,
            )
            preview_payload = dict(preview.output)
            if not preview.success:
                return CapabilityResult(
                    invocation.name,
                    output={"refine_commit": dict(commit.output), "engine_run_preview": preview_payload},
                    success=False,
                    error=preview.error or "engine_run_preview failed",
                )
            field_uri = preview.output.get("field_uri")
            if field_uri not in (None, ""):
                if not _is_g2v_field_uri(field_uri):
                    return CapabilityResult(
                        invocation.name,
                        output={"refine_commit": dict(commit.output), "engine_run_preview": preview_payload},
                        success=False,
                        error=f"engine_run_preview returned invalid field_uri {field_uri!r}",
                    )
                candidate_field_uri = field_uri

        refine_record = {
            "phase": "refine",
            "candidate_graph_uri": candidate_graph_uri,
            "candidate_field_uri": candidate_field_uri,
            "phase_recorded": True,
            "recorded_by": "refine_commit",
        }
        ctx.episode_context.setdefault("phase_records", {})["refine"] = refine_record
        ctx.episode_context["_last_refine_commit"] = refine_record
        return CapabilityResult(
            name=invocation.name,
            output={
                "candidate_graph_uri": candidate_graph_uri,
                "scratch_uri": commit.output.get("scratch_uri"),
                "head_rev_uri": commit.output.get("head_rev_uri"),
                "validation_report": commit.output.get("validation_report"),
                "candidate_field_uri": candidate_field_uri,
                "engine_run_preview": preview_payload,
                "refine_commit": dict(commit.output),
            },
            success=True,
        )

    def _candidate_submit_and_report(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: GeologyGraphVariation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        del variation
        payload = dict(invocation.input)
        graph_uri = payload.get("candidate_graph_uri") or payload.get("graph_uri")
        if not _is_g2v_graph_uri(graph_uri):
            refine_record = (ctx.episode_context.get("phase_records") or {}).get("refine") or ctx.episode_context.get("_last_refine_commit") or {}
            fallback_uri = refine_record.get("candidate_graph_uri") if isinstance(refine_record, dict) else None
            if _is_g2v_graph_uri(fallback_uri):
                graph_uri = fallback_uri
            else:
                return CapabilityResult(
                    invocation.name,
                    success=False,
                    error=f"candidate_graph_uri must be a 'g2v://graph/<hash>' URI; received {graph_uri!r}",
                )

        reference_pair_raw = payload.get("reference_pair")
        reference_pair = list(reference_pair_raw) if isinstance(reference_pair_raw, (list, tuple)) else []
        if len(reference_pair) != 2 or not all(_is_g2v_graph_uri(uri) for uri in reference_pair):
            assigned = ctx.episode_context.get("assigned_references") or {}
            fallback_pair = [assigned.get("reference_a"), assigned.get("reference_b")]
            if len(fallback_pair) == 2 and all(_is_g2v_graph_uri(uri) for uri in fallback_pair):
                reference_pair = [str(fallback_pair[0]), str(fallback_pair[1])]
            else:
                return CapabilityResult(
                    invocation.name,
                    success=False,
                    error="reference_pair must contain exactly two 'g2v://graph/<hash>' URIs",
                )

        predicted = payload.get("predicted_score_bits")
        if predicted is not None:
            if isinstance(predicted, bool) or not isinstance(predicted, (int, float)) or not math.isfinite(float(predicted)):
                return CapabilityResult(invocation.name, success=False, error="predicted_score_bits must be a finite number or null")
            predicted = float(predicted)
        gate_failures_raw = payload.get("gate_failures") or []
        if not isinstance(gate_failures_raw, list):
            return CapabilityResult(invocation.name, success=False, error="gate_failures must be a list")
        gate_failures = [str(item) for item in gate_failures_raw]

        submit = self._execute_g2v_tool(
            invocation.name,
            "candidate_submit",
            {"graph_uri": graph_uri, "reference_pair": reference_pair},
            containers,
            ctx,
        )
        if not submit.success:
            return submit

        candidate_record = {
            "candidate_graph_uri": graph_uri,
            "graph_uri": graph_uri,
            "reference_pair": reference_pair,
        }
        candidate_uri = submit.output.get("candidate_uri")
        if isinstance(candidate_uri, str) and candidate_uri:
            candidate_record["candidate_uri"] = candidate_uri
        metric_record = {"predicted_score_bits": predicted, "gate_failures": gate_failures}
        terminal_records = ctx.episode_context.setdefault("terminal_records", {})
        terminal_records["candidate_submit"] = candidate_record
        terminal_records["report_metric"] = metric_record
        return CapabilityResult(
            name=invocation.name,
            output={
                "candidate_submit": candidate_record,
                "report_metric": metric_record,
                "candidate_submit_result": dict(submit.output),
            },
            success=True,
        )

    def _seed_graph_submit(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: GeologyGraphVariation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        payload = dict(invocation.input)
        filename = payload.get("filename")
        content_text = payload.get("content_text")
        if not isinstance(filename, str) or not _valid_filename(filename):
            return CapabilityResult(invocation.name, success=False, error="seed_graph_submit requires a valid filename")
        if not isinstance(content_text, str) or not content_text.strip():
            return CapabilityResult(invocation.name, success=False, error="seed_graph_submit requires complete graph JSON in content_text")
        predicted = payload.get("predicted_passed_gates")
        if not isinstance(predicted, bool):
            return CapabilityResult(invocation.name, success=False, error="predicted_passed_gates must be boolean")
        gate_failures_raw = payload.get("gate_failures") or []
        if not isinstance(gate_failures_raw, list):
            return CapabilityResult(invocation.name, success=False, error="gate_failures must be a list")
        gate_failures = [str(item) for item in gate_failures_raw]
        field_uri = payload.get("seed_field_uri")
        if field_uri == "":
            field_uri = None
        if field_uri is not None and not _is_g2v_field_uri(field_uri):
            return CapabilityResult(
                invocation.name,
                success=False,
                error=f"seed_field_uri must be null or a 'g2v://field/<hash>' URI; received {field_uri!r}",
            )

        submit_args = {
            "filename": filename,
            "content_text": content_text,
            "message": "bootstrap seed graph",
            "tags": {"submission_kind": "bootstrap_seed"},
        }
        ingest = self._execute_g2v_tool(invocation.name, "seed_graph_submit", submit_args, containers, ctx)
        if not ingest.success:
            return ingest
        graph_uri = ingest.output.get("seed_graph_uri") or ingest.output.get("graph_uri")
        if not _is_g2v_graph_uri(graph_uri):
            return CapabilityResult(
                invocation.name,
                output={"graph_ingest": dict(ingest.output)},
                success=False,
                error=(
                    "seed_graph_submit expected g2v to return a 'g2v://graph/<hash>' URI; "
                    f"received {graph_uri!r}"
                ),
            )

        seed_record = {"seed_graph_uri": graph_uri, "seed_field_uri": field_uri}
        metric_record = {"predicted_passed_gates": predicted, "gate_failures": gate_failures}
        terminal_records = ctx.episode_context.setdefault("terminal_records", {})
        terminal_records["seed_submit"] = seed_record
        terminal_records["report_metric"] = metric_record
        return CapabilityResult(
            name=invocation.name,
            output={
                "graph_uri": graph_uri,
                "seed_graph_uri": graph_uri,
                "seed_submit": seed_record,
                "report_metric": metric_record,
                "graph_ingest": dict(ingest.output),
            },
            success=True,
        )

    def _analysis_shell(self, invocation: CapabilityInvocation, containers: list[Container]) -> CapabilityResult:
        command = invocation.input.get("command")
        if not isinstance(command, str) or not command.strip():
            return CapabilityResult(invocation.name, success=False, error="analysis_shell requires command")
        timeout_s = int(invocation.input.get("timeout_s") or 30)
        max_output = int(invocation.input.get("max_output_bytes") or 20000)
        analysis = self._maybe_pick_container(containers, "analysis")
        if analysis is None:
            return CapabilityResult(invocation.name, success=False, error="no analysis container available")
        try:
            raw = exec_run_with_timeout(
                analysis,
                ["sh", "-lc", command],
                timeout_s=timeout_s,
                workdir="/workspace",
            )
            exit_code, output = coerce_exec_result(raw)
        except TaskEnvironmentError as exc:
            return CapabilityResult(
                invocation.name,
                output={"exit_code": -1, "stdout": "", "stderr": str(exc), "timed_out": True},
                success=False,
                error=str(exc),
            )
        text = output.decode(errors="replace")
        stdout = text[:max_output]
        truncated = len(text) > max_output
        return CapabilityResult(
            name=invocation.name,
            output={
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": "",
                "timed_out": False,
                "truncated": truncated,
            },
            success=exit_code == 0,
            error=None if exit_code == 0 else f"analysis_shell exit_code={exit_code}",
        )

    def _promote_analysis_artifact(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: GeologyGraphVariation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        src_path = invocation.input.get("src_path")
        if not isinstance(src_path, str) or not src_path.startswith(f"{_ANALYSIS_OUT}/") or ".." in src_path:
            return CapabilityResult(invocation.name, success=False, error="src_path must be under /workspace/out")
        name = invocation.input.get("name") or Path(src_path).name
        if not isinstance(name, str) or not _valid_filename(name):
            return CapabilityResult(invocation.name, success=False, error="invalid target filename")
        analysis = self._maybe_pick_container(containers, "analysis")
        g2v = self._maybe_pick_container(containers, "g2v")
        if analysis is None or g2v is None:
            return CapabilityResult(invocation.name, success=False, error="analysis and g2v containers are required")
        resolved = self._resolve_analysis_path(analysis, src_path)
        if not resolved.startswith(f"{_ANALYSIS_OUT}/"):
            return CapabilityResult(invocation.name, success=False, error="src_path resolves outside /workspace/out")
        try:
            chunks, _stat = analysis.get_archive(src_path)
            data = b"".join(chunks)
        except Exception as exc:
            return CapabilityResult(invocation.name, success=False, error=f"get_archive failed: {exc}")
        file_bytes = _extract_single_file_from_tar(data)
        if len(file_bytes) > variation.max_promote_bytes:
            return CapabilityResult(invocation.name, success=False, error="promoted artifact exceeds size cap")
        sha = hashlib.sha256(file_bytes).hexdigest()
        dest_dir = f"{_G2V_WORKSPACE}/imports/{ctx.episode_context.get('episode_workspace_id', ctx.episode_id)}"
        self._put_file(g2v, dest_dir, name, file_bytes)
        g2v_path = f"{dest_dir}/{name}"
        return CapabilityResult(
            name=invocation.name,
            output={"g2v_path": g2v_path, "size_bytes": len(file_bytes), "sha256": sha},
            success=True,
        )

    @staticmethod
    def _resolve_analysis_path(container: Container, src_path: str) -> str:
        raw = exec_run_with_timeout(container, ["readlink", "-f", "--", src_path], timeout_s=5)
        exit_code, output = coerce_exec_result(raw)
        if exit_code != 0:
            raise TaskEnvironmentError(f"readlink failed for {src_path}")
        return output.decode(errors="replace").strip()

    @staticmethod
    def _put_file(container: Container, dest_dir: str, name: str, content: bytes) -> None:
        exec_run_with_timeout(container, ["mkdir", "-p", dest_dir], timeout_s=5)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(content))
        stream.seek(0)
        container.put_archive(dest_dir, stream.getvalue())

    def _maybe_auto_record_previous_step(self, ctx: CapabilityExecutionContext) -> None:
        """Auto-record any prior phase the agent didn't close with record_phase.

        ms-agent advances workflow steps after a single tool call, so a small model
        often makes the step's main capability call but never the trailing
        record_phase. We detect the step transition by observing ``ctx.workflow_step``
        change, and stamp a sentinel record so downstream phase_get / validators see
        the step as closed (auto-recorded). The agent's explicit record_phase, if
        called within the step, takes precedence — it ran before this transition.
        """
        current_step = ctx.workflow_step
        episode_context = ctx.episode_context
        previous = episode_context.get("_last_workflow_step")
        if previous and previous != current_step and previous in _ALL_PHASE_RECORDS:
            records = episode_context.setdefault("phase_records", {})
            if previous not in records:
                auto = {"phase": previous, "auto_recorded": True, "phase_recorded": True}
                auto.update(self._auto_recorded_fields(previous, episode_context))
                records[previous] = auto
        if current_step is not None:
            episode_context["_last_workflow_step"] = current_step

    def _finalize_auto_record(self, episode_context: dict[str, Any]) -> None:
        """Auto-record the final workflow step at episode end.

        ``_maybe_auto_record_previous_step`` only fires on a transition; for the
        last step there is no following capability call. Called from
        ``finalize_episode``.
        """
        last = episode_context.get("_last_workflow_step")
        if last and last in _ALL_PHASE_RECORDS:
            records = episode_context.setdefault("phase_records", {})
            if last not in records:
                auto = {"phase": last, "auto_recorded": True, "phase_recorded": True}
                auto.update(self._auto_recorded_fields(last, episode_context))
                records[last] = auto

    @staticmethod
    def _auto_recorded_fields(phase: str, episode_context: dict[str, Any]) -> dict[str, Any]:
        """Promote tool-returned URIs into an auto-record payload by phase."""
        promoted = episode_context.get("_auto_promoted") or {}
        if phase == "hypothesise":
            return {
                k: promoted[k]
                for k in ("hypothesis_uri", "statement")
                if k in promoted
            }
        if phase == "refine":
            return {
                k: promoted[k]
                for k in ("candidate_graph_uri", "candidate_field_uri")
                if k in promoted
            }
        return {}

    def _record_phase(self, invocation: CapabilityInvocation, ctx: CapabilityExecutionContext) -> CapabilityResult:
        phase = invocation.input.get("phase")
        if phase not in _ALL_PHASE_RECORDS:
            return CapabilityResult(invocation.name, success=False, error="invalid phase")
        if ctx.workflow_step is not None and phase != ctx.workflow_step:
            return CapabilityResult(
                invocation.name,
                output={"phase": phase, "workflow_step": ctx.workflow_step},
                success=False,
                error="record_phase phase does not match trusted workflow step",
            )
        payload = dict(invocation.input)
        if phase == "refine" and not _is_g2v_graph_uri(payload.get("candidate_graph_uri")):
            prior = ctx.episode_context.get("_last_refine_commit") or (ctx.episode_context.get("phase_records") or {}).get("refine") or {}
            if isinstance(prior, dict) and _is_g2v_graph_uri(prior.get("candidate_graph_uri")):
                payload["candidate_graph_uri"] = prior["candidate_graph_uri"]
                payload.setdefault("candidate_field_uri", prior.get("candidate_field_uri"))
        payload["phase_recorded"] = True
        ctx.episode_context.setdefault("phase_records", {})[phase] = payload
        return CapabilityResult(name=invocation.name, output=payload, success=True)

    def _phase_get(self, invocation: CapabilityInvocation, ctx: CapabilityExecutionContext) -> CapabilityResult:
        phase = invocation.input.get("phase")
        if phase not in _ALL_PHASE_RECORDS:
            return CapabilityResult(invocation.name, success=False, error="invalid phase")
        records = ctx.episode_context.setdefault("phase_records", {})
        found = phase in records
        return CapabilityResult(
            name=invocation.name,
            output={"phase": phase, "payload": dict(records.get(phase, {})), "found": found},
            success=True,
        )

    # ------------------------------------------------------------------
    # Measurement and reward
    # ------------------------------------------------------------------

    def measure_initial_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> GeologyGraphState:
        variation = self._variation_from_context(episode_context)
        pool_snapshot = dict(episode_context.get("pool_snapshot") or {})
        return GeologyGraphState(
            workflow_kind=str(episode_context.get("workflow_kind") or "bootstrap"),
            pool_snapshot=pool_snapshot,
            dataset_snapshot=dict(episode_context.get("dataset_snapshot") or {}),
            admission_threshold=self._admission_threshold(variation, pool_snapshot),
            t_steady=variation.t_steady,
            x_warmup_episodes=variation.x_warmup_episodes,
            episode_count=int(pool_snapshot.get("episode_count", 0)),
            admission_count=int(pool_snapshot.get("admission_count", 0)),
        )

    def measure_final_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
    ) -> GeologyGraphState:
        variation = self._variation_from_context(episode_context)
        phase_artifacts = self._phase_artifacts(artifacts, episode_context)
        terminal_artifacts = self._terminal_artifacts(artifacts)
        state = self.measure_initial_state(containers, episode_context, private_context=private_context)
        state.phase_artifacts = phase_artifacts
        state.terminal_artifacts = terminal_artifacts
        state.admission_threshold = self._admission_threshold(variation, state.pool_snapshot)
        if state.workflow_kind == "regular":
            self._score_regular(containers, episode_context, variation, state)
        else:
            self._validate_bootstrap(containers, episode_context, variation, state)
        state.calibration_error_bits = self._calibration_error(state)
        return state

    def compute_reward(
        self,
        initial: GeologyGraphState,
        final: GeologyGraphState,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        gate_failures = list(final.gate_failures)
        report_metric = final.terminal_artifacts.get("report_metric")
        report_input = (report_metric or {}).get("input", {})
        candidate_submit = final.terminal_artifacts.get("candidate_submit")
        if final.workflow_kind == "bootstrap":
            success = bool(final.passed_gates)
            value = 0.5 if success else 0.0
        else:
            success = bool(final.passed_gates) and final.score_bits <= final.admission_threshold
            if not final.passed_gates:
                value = 0.0
            elif final.score_bits <= final.t_steady:
                value = 1.0
            else:
                value = max(0.0, 1.0 - (final.score_bits - final.t_steady) / final.t_steady)

        execute_status = (final.phase_artifacts.get("execute") or {}).get("status")
        if final.workflow_kind == "regular" and execute_status in {"unparameterisable", "engine_failed"} and not success:
            value = max(value, 0.1)

        if final.budget_exhausted_step is not None:
            value, success = 0.0, False
        if report_metric is None:
            value, success = 0.0, False
            gate_failures.append("missing_report_metric")
        if final.workflow_kind == "regular" and candidate_submit is None:
            value, success = 0.0, False
            gate_failures.append("missing_candidate_submit")
        if final.workflow_kind == "bootstrap" and final.terminal_artifacts.get("seed_submit") is None:
            value, success = 0.0, False
            gate_failures.append("missing_seed_submit")
        if final.workflow_kind == "bootstrap" and report_input.get("predicted_passed_gates") is None:
            value, success = 0.0, False
            gate_failures.append("missing_predicted_passed_gates")
        if final.workflow_kind == "regular" and report_input.get("predicted_score_bits") is None:
            value, success = 0.0, False
            gate_failures.append("missing_predicted_score_bits")
        if final.cheating_detected == "dataset_tampered":
            value, success = 0.0, False
        if "g2v_worker_crash" in gate_failures:
            value, success = 0.0, False
        breakdown = {
            "workflow_kind": final.workflow_kind,
            "score_bits": final.score_bits,
            "structural_bits": final.structural_bits,
            "fit_bits": final.fit_bits,
            "gate_failures": gate_failures,
            "admission_threshold": final.admission_threshold,
            "admitted": bool(success),
            "bootstrap_active": final.workflow_kind == "bootstrap",
            "warmup_active": final.workflow_kind == "regular" and final.episode_count < final.x_warmup_episodes,
            "experiment_status": execute_status,
            "agent_predicted_score_bits": report_input.get("predicted_score_bits"),
            "agent_predicted_passed_gates": report_input.get("predicted_passed_gates"),
            "agent_predicted_gate_failures": list(report_input.get("gate_failures") or []),
            "calibration_error_bits": final.calibration_error_bits,
            "budget_exhausted_step": final.budget_exhausted_step,
            "phase_budget_used": final.phase_budget_used,
            "cheating_detected": final.cheating_detected,
            "dataset_drift_paths": list(final.dataset_drift_paths),
        }
        return TaskReward(value=float(max(0.0, min(1.0, value))), success=bool(success), breakdown=breakdown)

    def finalize_episode(
        self,
        containers: list[Container],
        initial: GeologyGraphState,
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
        finalization_context: FinalizationContext | None = None,
    ) -> TaskReward:
        try:
            variation = self._variation_from_context(episode_context)
            self._finalize_auto_record(episode_context)
            final = self.measure_final_state(containers, episode_context, artifacts, private_context=private_context)
            if finalization_context is not None:
                if finalization_context.budget_exhaustion is not None:
                    final.budget_exhausted_step = finalization_context.budget_exhaustion.step or finalization_context.last_workflow_step
                final.phase_budget_used = {"task_tool_calls": finalization_context.tool_calls_count}
            drift = self._dataset_drift(containers, episode_context)
            if drift:
                final.cheating_detected = "dataset_tampered"
                final.dataset_drift_paths = drift[:8]
            reward = self.compute_reward(initial, final, artifacts)
            if reward.success:
                try:
                    self._admit_candidate(containers, variation, final)
                except Exception as exc:
                    logger.warning(f"geology pool admission failed: {exc}")
                    breakdown = dict(reward.breakdown)
                    breakdown["admission_error"] = str(exc)
                    return TaskReward(value=reward.value, success=False, breakdown=breakdown)
            return reward
        finally:
            self._close_g2v_worker(episode_context)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_regular(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        variation: GeologyGraphVariation,
        state: GeologyGraphState,
    ) -> None:
        required = ["explore", "hypothesise", "execute", "refine"]
        missing = [phase for phase in required if phase not in state.phase_artifacts]
        if missing:
            state.gate_failures.extend([f"missing_phase_record:{phase}" for phase in missing])
        candidate_graph_uri = (state.phase_artifacts.get("refine") or {}).get("candidate_graph_uri")
        if not isinstance(candidate_graph_uri, str) or not candidate_graph_uri:
            state.gate_failures.append("missing_candidate_graph_uri")
            state.passed_gates = False
            return
        assigned = episode_context.get("assigned_references") or {}
        ref_a = assigned.get("reference_a")
        ref_b = assigned.get("reference_b")
        if not isinstance(ref_a, str) or not isinstance(ref_b, str):
            state.gate_failures.append("missing_assigned_references")
            state.passed_gates = False
            return
        candidate_field_uri = (state.phase_artifacts.get("refine") or {}).get("candidate_field_uri") or ""
        scorer_result = episode_context.get("scorer_result")
        if not isinstance(scorer_result, dict):
            scorer_result = self._run_g2v_scorer(
                containers,
                variation,
                candidate_graph_uri=candidate_graph_uri,
                candidate_field_uri=str(candidate_field_uri or ""),
                ref_a_uri=ref_a,
                ref_b_uri=ref_b,
            )
        state.scorer_result = dict(scorer_result)
        state.score_bits = float(scorer_result.get("score_bits", math.inf))
        state.structural_bits = float(scorer_result.get("structural_bits", math.inf))
        state.fit_bits = float(scorer_result.get("fit_bits", math.inf))
        state.physics_bits = float(scorer_result.get("physics_bits", math.inf))
        state.passed_gates = bool(scorer_result.get("passed_gates", False))
        state.gate_failures.extend(str(x) for x in scorer_result.get("gate_failures", []) or [])

    def _validate_bootstrap(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        variation: GeologyGraphVariation,
        state: GeologyGraphState,
    ) -> None:
        for phase in ("explore_data", "hypothesise", "execute"):
            if phase not in state.phase_artifacts:
                state.gate_failures.append(f"missing_phase_record:{phase}")
        seed = state.terminal_artifacts.get("seed_submit")
        seed_input = (seed or {}).get("input", {})
        seed_graph_uri = seed_input.get("seed_graph_uri")
        if not isinstance(seed_graph_uri, str) or not seed_graph_uri:
            state.gate_failures.append("missing_seed_graph_uri")
            state.passed_gates = False
            return
        validator_result = episode_context.get("seed_validator_result")
        if not isinstance(validator_result, dict):
            validator_result = self._run_seed_validator(
                containers,
                variation,
                seed_graph_uri=seed_graph_uri,
                seed_field_uri=str(seed_input.get("seed_field_uri") or ""),
            )
        state.scorer_result = dict(validator_result)
        state.passed_gates = bool(validator_result.get("passed_gates", False))
        state.gate_failures.extend(str(x) for x in validator_result.get("gate_failures", []) or [])

    def _run_g2v_scorer(self, containers: list[Container], variation: GeologyGraphVariation, **kwargs: str) -> dict[str, Any]:
        g2v = self._maybe_pick_container(containers, "g2v")
        if g2v is None:
            return {"passed_gates": False, "score_bits": math.inf, "gate_failures": ["scorer_unavailable"]}
        cmd = [
            "python",
            "-m",
            "tasks.common.g2v_scorer",
            "--workspace",
            _G2V_WORKSPACE,
            "--candidate-graph-uri",
            kwargs["candidate_graph_uri"],
            "--candidate-field-uri",
            kwargs.get("candidate_field_uri", ""),
            "--ref-a-uri",
            kwargs["ref_a_uri"],
            "--ref-b-uri",
            kwargs["ref_b_uri"],
            "--config",
            json.dumps(variation.criterion_config),
            "--field-spec",
            json.dumps(variation.field_spec),
            "--output-format",
            "json",
        ]
        for uri in (self._current_pool_graph_uris(variation)):
            cmd.extend(["--pool-graph-uri", uri])
        return self._exec_json(g2v, cmd, fallback_failure="scorer_crash")

    def _run_seed_validator(self, containers: list[Container], variation: GeologyGraphVariation, **kwargs: str) -> dict[str, Any]:
        g2v = self._maybe_pick_container(containers, "g2v")
        if g2v is None:
            return {"passed_gates": False, "gate_failures": ["seed_validator_unavailable"]}
        cmd = [
            "python",
            "-m",
            "tasks.common.g2v_seed_validator",
            "--workspace",
            _G2V_WORKSPACE,
            "--seed-graph-uri",
            kwargs["seed_graph_uri"],
            "--seed-field-uri",
            kwargs.get("seed_field_uri", ""),
            "--config",
            json.dumps(variation.criterion_config),
            "--field-spec",
            json.dumps(variation.field_spec),
            "--output-format",
            "json",
        ]
        for uri in self._current_pool_graph_uris(variation):
            cmd.extend(["--pool-graph-uri", uri])
        return self._exec_json(g2v, cmd, fallback_failure="seed_validator_crash")

    @staticmethod
    def _exec_json(container: Container, cmd: list[str], fallback_failure: str) -> dict[str, Any]:
        try:
            raw = exec_run_with_timeout(container, cmd, timeout_s=180)
            exit_code, output = coerce_exec_result(raw)
        except Exception as exc:
            return {"passed_gates": False, "score_bits": math.inf, "gate_failures": [fallback_failure], "error": str(exc)}
        text = output.decode(errors="replace")
        if exit_code != 0:
            return {"passed_gates": False, "score_bits": math.inf, "gate_failures": [fallback_failure], "stderr": text[-4000:]}
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"passed_gates": False, "score_bits": math.inf, "gate_failures": [fallback_failure], "stderr": text[-4000:]}

    # ------------------------------------------------------------------
    # Pool and dataset helpers
    # ------------------------------------------------------------------

    def _variation_from_context(self, episode_context: dict[str, Any]) -> GeologyGraphVariation:
        name = episode_context.get("variation_name")
        for variation in self.list_variations():
            if variation.name == name:
                assert isinstance(variation, GeologyGraphVariation)
                return variation
        variations = self.list_variations()
        assert isinstance(variations[0], GeologyGraphVariation)
        return variations[0]

    @staticmethod
    def _pick_container(containers: list[Container], role: str) -> Container:
        service_name = _ROLE_SERVICE[role]
        by_service = {container_to_service(container): container for container in containers}
        if service_name in by_service:
            return by_service[service_name]
        raise TaskEnvironmentError(f"No container found for role={role!r}; available={sorted(by_service)}")

    def _maybe_pick_container(self, containers: list[Container], role: str) -> Container | None:
        try:
            return self._pick_container(containers, role)
        except Exception:
            return None

    @staticmethod
    def _close_g2v_worker(episode_context: dict[str, Any]) -> None:
        worker = episode_context.pop("_g2v_worker", None)
        if isinstance(worker, _G2VWorkerClient):
            worker.close()

    @contextmanager
    def _pool_lock(self, variation: GeologyGraphVariation) -> Iterator[None]:
        lock_path = Path(variation.pool_dir) / "index.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _ensure_pool(self, variation: GeologyGraphVariation) -> None:
        root = Path(variation.pool_dir)
        (root / "graphs").mkdir(parents=True, exist_ok=True)
        (root / "fields").mkdir(parents=True, exist_ok=True)
        if not (root / "index.json").exists():
            self._write_pool_index(variation, {"graphs": [], "locked_hashes": [], "episode_count": 0, "admission_count": 0})

    def _read_pool_index(self, variation: GeologyGraphVariation) -> dict[str, Any]:
        path = Path(variation.pool_dir) / "index.json"
        if not path.exists():
            return {"graphs": [], "locked_hashes": [], "episode_count": 0, "admission_count": 0}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_pool_index(variation: GeologyGraphVariation, state: dict[str, Any]) -> None:
        path = Path(variation.pool_dir) / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique tmp per writer: shared "index.json.tmp" otherwise races between
        # concurrent populates — first os.replace moves the tmp, later replaces
        # see ENOENT.
        tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _current_pool_graph_uris(self, variation: GeologyGraphVariation) -> list[str]:
        state = self._read_pool_index(variation)
        return [f"g2v://graph/{item['hash']}" for item in state.get("graphs", [])]

    @staticmethod
    def _select_reference_pair(graph_ids: list[str], variation_name: str) -> dict[str, str]:
        if len(graph_ids) < 2:
            return {}
        best: tuple[str, str] | None = None
        best_distance = -1
        for i, left in enumerate(graph_ids):
            for right in graph_ids[i + 1 :]:
                distance = sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
                tie = hashlib.sha256(f"{variation_name}:{left}:{right}".encode()).hexdigest()
                score = distance * 10_000 + int(tie[:6], 16)
                if score > best_distance:
                    best_distance = score
                    best = (left, right)
        assert best is not None
        return {"reference_a": best[0], "reference_b": best[1]}

    def _admission_threshold(self, variation: GeologyGraphVariation, pool_snapshot: dict[str, Any]) -> float:
        pool_size = len(pool_snapshot.get("graph_ids", []))
        if pool_size < variation.min_pool_size:
            return math.inf
        episode_count = int(pool_snapshot.get("episode_count", 0))
        if episode_count < variation.x_warmup_episodes:
            return math.inf
        admission_count = int(pool_snapshot.get("admission_count", 0))
        if variation.anneal_window <= 0:
            return variation.t_steady
        fraction = min(admission_count / variation.anneal_window, 1.0)
        return variation.initial_threshold + (variation.t_steady - variation.initial_threshold) * fraction

    def _admit_candidate(self, containers: list[Container], variation: GeologyGraphVariation, final: GeologyGraphState) -> None:
        if final.workflow_kind == "regular":
            graph_uri = (final.phase_artifacts.get("refine") or {}).get("candidate_graph_uri")
            graph_hash = final.scorer_result.get("candidate_graph_hash")
        else:
            seed = final.terminal_artifacts.get("seed_submit") or {}
            graph_uri = (seed.get("input") or {}).get("seed_graph_uri")
            graph_hash = final.scorer_result.get("seed_graph_hash")
        if not isinstance(graph_uri, str) or not isinstance(graph_hash, str):
            raise TaskEnvironmentError("cannot admit candidate without graph uri/hash")
        with self._pool_lock(variation):
            state = self._read_pool_index(variation)
            if any(item.get("hash") == graph_hash for item in state.get("graphs", [])):
                return
            graph_rel = f"graphs/{graph_hash}.graph.json"
            graph_dest = Path(variation.pool_dir) / graph_rel
            g2v = self._maybe_pick_container(containers, "g2v")
            if g2v is not None:
                self._copy_g2v_graph_to_host(g2v, graph_hash, graph_dest)
            elif not graph_dest.exists():
                graph_dest.write_text(json.dumps({"uri": graph_uri, "hash": graph_hash}), encoding="utf-8")
            item = {"hash": graph_hash, "uri": graph_uri, "path": graph_rel, "admitted_at": datetime.now(UTC).isoformat()}
            graphs = [*state.get("graphs", []), item]
            state["graphs"] = graphs[-variation.pool_capacity :]
            state["admission_count"] = int(state.get("admission_count", 0)) + 1
            self._write_pool_index(variation, state)

    @staticmethod
    def _copy_g2v_graph_to_host(container: Container, graph_hash: str, dest: Path) -> None:
        # `/var/lib/g2v/workspace` is a tmpfs in docker-compose. Docker's
        # archive/cp API (container.get_archive, docker cp) cannot enumerate
        # files on tmpfs mounts on some daemon configurations (notably WSL2)
        # and returns 404 even when the file is present. Read via `exec cat`
        # instead, which goes through the running container's POSIX layer.
        src = f"{_G2V_WORKSPACE}/graphs/{graph_hash}.json"
        raw = exec_run_with_timeout(container, ["cat", src], timeout_s=20)
        exit_code, output = coerce_exec_result(raw)
        if exit_code != 0:
            raise TaskEnvironmentError(
                f"failed to read g2v graph {graph_hash} (exit={exit_code}): "
                f"{output.decode(errors='replace')[:400]}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(output)

    def _prepare_g2v_workspace(self, container: Container, episode_workspace_id: str) -> None:
        cmd = (
            f"rm -rf {shlex.quote(_G2V_WORKSPACE)}/* && "
            f"mkdir -p {shlex.quote(_G2V_WORKSPACE)}/imports/{shlex.quote(episode_workspace_id)}"
        )
        raw = exec_run_with_timeout(container, ["sh", "-lc", cmd], timeout_s=20)
        exit_code, output = coerce_exec_result(raw)
        if exit_code != 0:
            raise TaskEnvironmentError(f"failed to prepare g2v workspace: {output.decode(errors='replace')}")

    def _g2v_exec_call(self, container: Container, episode_workspace_id: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        request = json.dumps({"tool": tool, "args": args})
        encoded = base64.b64encode(request.encode()).decode()
        cmd = (
            f"printf '%s' {shlex.quote(encoded)} | base64 -d | "
            "python -m tasks.common.g2v_worker "
            f"--workspace {shlex.quote(_G2V_WORKSPACE)} "
            f"--imports-subdir {shlex.quote('imports/' + episode_workspace_id)}"
        )
        return self._exec_json(container, ["sh", "-lc", cmd], fallback_failure="g2v_worker_crash")

    def _snapshot_dataset(self, analysis: Container | None, dataset_dir: Path) -> dict[str, str]:
        if analysis is None:
            return _hash_tree(dataset_dir)
        cmd = "python - <<'PY'\nimport hashlib, json, pathlib\nroot=pathlib.Path('/workspace/input')\nout={}\nfor p in sorted(x for x in root.rglob('*') if x.is_file()):\n    out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()\nprint(json.dumps(out, sort_keys=True))\nPY"
        payload = self._exec_json(analysis, ["sh", "-lc", cmd], fallback_failure="dataset_snapshot_failed")
        if "gate_failures" in payload:
            return {}
        return {str(k): str(v) for k, v in payload.items()}

    def _dataset_drift(self, containers: list[Container], episode_context: dict[str, Any]) -> list[str]:
        original = episode_context.get("dataset_snapshot") or {}
        variation = self._variation_from_context(episode_context)
        current = self._snapshot_dataset(self._maybe_pick_container(containers, "analysis"), Path(variation.dataset_dir))
        paths = sorted(set(original) | set(current))
        return [path for path in paths if original.get(path) != current.get(path)]

    @staticmethod
    def _phase_artifacts(
        artifacts: EpisodeArtifacts,
        episode_context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if episode_context is not None:
            for phase, payload in (episode_context.get("phase_records") or {}).items():
                if isinstance(phase, str) and isinstance(payload, dict):
                    out[phase] = dict(payload)
        for inv, res in zip(artifacts.capability_invocations, artifacts.capability_results):
            if inv.name != "record_phase" or not res.success:
                continue
            phase = inv.input.get("phase")
            if isinstance(phase, str):
                out[phase] = dict(inv.input)
        return out

    @staticmethod
    def _terminal_artifacts(artifacts: EpisodeArtifacts) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for inv, res in zip(artifacts.capability_invocations, artifacts.capability_results):
            if inv.name == "mcp_submit_call" and inv.input.get("tool") == "candidate_submit":
                key = "candidate_submit"
            elif inv.name == "candidate_submit_and_report":
                if res.success:
                    candidate_record = res.output.get("candidate_submit")
                    metric_record = res.output.get("report_metric")
                    if isinstance(candidate_record, dict):
                        out["candidate_submit"] = {"input": dict(candidate_record), "output": dict(candidate_record)}
                    if isinstance(metric_record, dict):
                        out["report_metric"] = {"input": dict(metric_record), "output": dict(metric_record)}
                continue
            elif inv.name == "seed_graph_submit":
                if res.success:
                    seed_record = res.output.get("seed_submit")
                    metric_record = res.output.get("report_metric")
                    if isinstance(seed_record, dict):
                        out["seed_submit"] = {"input": dict(seed_record), "output": dict(seed_record)}
                    if isinstance(metric_record, dict):
                        out["report_metric"] = {"input": dict(metric_record), "output": dict(metric_record)}
                continue
            elif inv.name in {"report_metric", "seed_submit"}:
                key = inv.name
            else:
                continue
            if res.success:
                out[key] = {"input": dict(inv.input), "output": dict(res.output)}
        return out

    @staticmethod
    def _calibration_error(state: GeologyGraphState) -> float | None:
        report = state.terminal_artifacts.get("report_metric")
        if not report:
            return None
        predicted = (report.get("input") or {}).get("predicted_score_bits")
        if predicted is None or not math.isfinite(state.score_bits):
            return None
        try:
            return abs(float(predicted) - float(state.score_bits))
        except (TypeError, ValueError):
            return None


_G2V_GRAPH_URI_RE = re.compile(r"^g2v://graph/[0-9a-f]{16,}$")
_G2V_FIELD_URI_RE = re.compile(r"^g2v://field/[0-9a-f]{16,}$")


def _is_g2v_graph_uri(value: Any) -> bool:
    return isinstance(value, str) and bool(_G2V_GRAPH_URI_RE.match(value))


def _is_g2v_field_uri(value: Any) -> bool:
    return isinstance(value, str) and bool(_G2V_FIELD_URI_RE.match(value))


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _valid_filename(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name)) and "/" not in name and not name.startswith(".")


def _extract_single_file_from_tar(data: bytes) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError("expected archive to contain exactly one regular file")
        extracted = tar.extractfile(members[0])
        if extracted is None:
            raise ValueError("failed to extract archive file")
        return extracted.read()


def _hash_tree(root: Path) -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


__all__ = ["GeologyGraphState", "GeologyGraphTask", "GeologyGraphVariation"]
