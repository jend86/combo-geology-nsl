Represent geological knowledge as a graph that is:

- Natural for an LLM agent to read, write, and revise in language: the entity-graph representation
- Quantitative enough to be rendered into geometric/numerical form for downstream analysis: the voxel representation
- Able to encapsulate uncertainty in the data/claims (even "hard measurements" are subject to transcription error, etc.)
- Capable of holding multiple competing readings of the same evidence simultaneously, rather than forcing a collapse into one (adopted; see P1 below).

The agent's job is to maintain a world model, which is a probabilistic description of their interpretation of the geological data.

# Design principles

Three principles anchor the schema and govern what belongs in the graph vs. elsewhere.

**P1. Probabilistic and competing claims are first-class.**
Multiple units with overlapping geometry, two contacts disagreeing about a boundary, an embedded body inside another — these are normal cases for an LLM-authored model, not authoring errors. The graph records what the agent claims; reconciling overlap into a field (per-cell probability distributions, mixed-boundary interpolation) is the engine's job. This is the natural extension of `p_exists` + MCUE: "two units occupy the same space" is a probability question for the field, not a topology question for the schema.

**P2. The graph is an interface, not a state machine.**
The graph is a language for an LLM agent (queryable, traversable, easy to author and edit) and a deterministic input to the engine. It is not a place for cross-iteration history, supersession, lineage across runs, scenarios, or revision tracking — those are workspace-level concerns (files, version control, an outer session/project layer). Per-node static provenance ("this Sample was reported by agent X on date Y") is fine; "node N version 2 supersedes version 1" is not. In-session mutation (adding/editing nodes during one working session) is fully supported; cross-iteration revision lives outside the schema.

**P3. Some graph information is agent-facing, not engine-facing — and spatial relations are derived, not declared.**
The graph carries both an engine-facing subset (positions, contact pairs, `OVERLIES` order, `p_exists`) and an agent-facing surface (labels, annotations, query helpers). A planned query engine will answer questions like _"which units are spatially nested within unit X?"_ by reading positions, not by traversing declarative containment edges. We deliberately avoid edges that assert spatial relations the geometry already determines: such edges can disagree with the geometry, and we would resolve in favour of the geometry every time. The graph asserts what was authored; the engine and query layers derive what follows.

**P4. The agent should be concerned more with correctness of the graph rather than completeness**
In our internal representation and transforms, we want to adopt as many sensible defaults as possible. We don't want agents to have to be overly pedantic be faulted for not being sufficiently exhaustive. We do not create "fill in the blank" tasks for the agents. Data from the graph is used for overrides.

# Graph

The main representation of the world model.

We reiterate that splitting observation from interpretation is an explicit non-goal. Every entry should be able to bear uncertainty and provenance.

## Nodes

Nodes are typed geological entities (straitographic units, contacts, observation points, etc.)

Each node should have:

- Type appropriate params -> Whatever is necessary to be geometrically definable.
- Per-parameter uncertainty: Some parameters might be more constrained than others (e.g. for the same entity, some might have measurements, while others might have been inferred from qualitative descriptions).
- Existence confidence. Combined with the above, helps to jointly expresss "does this exist on this map?" and "given it exists, what features does it have?"
- Provenance

## Edges

Edges are a typed relation between nodes, and will be constructured from a curated closed vocabulary set. This is explicitly consumption-only; agents do not get to create new edge relations.

Edges hence have:

- Types (from a closed vocab set)
- Uncertainty
- Provenance

## History

Out of scope for the graph schema (P2). Cross-iteration revision — versions, supersession, scenario branches, lineage across runs — lives at the workspace level (separate JSON files, version control, an outer session/project abstraction), not inside the graph. Per-node static provenance stays on the node; multi-graph history sits above `Graph`.

# Downstream refinement criterion

The repo exposes a self-supervised graph-refinement criterion for downstream systems that maintain pools of alternative graph interpretations. The intended use case is described in `docs/design/06-self-supervised-graph-refinement-criterion.md`: a downstream constructor may use richer external data `D` to propose a candidate graph `C`, but the admission criterion itself scores only `C` against two reference graphs `A` and `B`.

The API lives in `graph_to_voxel.refinement`. It does not construct candidates, consult `D`, run a full pool loop, or claim that an accepted candidate is true. It provides the D-free admission score and supporting gates/diagnostics that downstream orchestration can use.

## One-shot scoring

Call `score_refinement` with the candidate graph and voxel field plus the two reference graphs and voxel fields:

```python
from graph_to_voxel.refinement import RefinementCriterionConfig, score_refinement

config = RefinementCriterionConfig(
    epsilon=0.01,
    effective_sample_size=1_000.0,
    coverage_threshold=0.95,
)

result = score_refinement(
    candidate_graph=C,
    candidate_field=field_C,
    reference_a_graph=A,
    reference_a_field=field_A,
    reference_b_graph=B,
    reference_b_field=field_B,
    config=config,
    pool=existing_pool,
)

if result.passed_gates:
    print(result.score_bits)
    print(result.structural_bits, result.fit_bits)
else:
    print([failure.name for failure in result.gate_failures])
```

`field_A`, `field_B`, and `field_C` are ordinary `VoxelField` instances, usually produced by `build_voxel_field(...)` or by reducing an `Ensemble`. The criterion aligns unit catalogs by raw `unit_id`, renormalises per-voxel unit probabilities before KL, uses the intersection of domain masks for fit, and applies a hard coverage gate to stop candidates from shrinking their domain to avoid hard regions.

The returned `RefinementScore` exposes:

- `score_bits`: total admission score, lower is better.
- `structural_bits`: pair-relative structural edit cost `L_delta(C | A, B)`.
- `fit_bits`: capped two-source fit `0.5 * (L_kappa(A | C) + L_kappa(B | C))`.
- `fit_a` and `fit_b`: per-reference `FitLoss` records with coverage, voxel count, ESS, cap, and union catalog.
- `gate_failures`: hard failures such as coverage, dedup, schema validity, stratigraphic consistency, domain closure, or voxel stratigraphic order.
- `diagnostics`: extra telemetry, currently including reverse-KL diagnostics and structural-added/residual components.
- `flags`: lightweight telemetry such as added structural elements.

## Configuration

`RefinementCriterionConfig` controls the score:

- `epsilon`: smoothing strength for KL comparisons. Default `0.01`.
- `kappa_bits`: per-voxel per-source cap. If omitted, defaults to `log2(1 / epsilon)`.
- `effective_sample_size`: ESS temperature for spatial autocorrelation correction. Hold this fixed across a downstream pool.
- `coverage_threshold`: required candidate coverage of each reference support. Default `0.95`.
- `dedup_epsilon`: structural-distance threshold for no-op/copy rejection when `pool` is supplied.
- `run_physics_gates`: enables existing schema, stratigraphic, domain, and voxel-order hard gates.
- `structural`: a `StructuralCostConfig` with placeholder bit costs for added, modified, deleted-consensus, split/merge, and residual absolute complexity terms.

The default structural costs are intentionally placeholders. Downstream deployments should calibrate them from a reference catalog of validated graph revisions if scores are used for real admission decisions.

## Pool Admission Helpers

The package also exposes small helpers for downstream orchestration:

- `annealed_threshold(admission_count, initial_threshold, steady_state_threshold, anneal_horizon)` implements the linear threshold schedule from the design.
- `reservoir_retain(items, capacity, rng=..., locked_indices=...)` keeps a random non-fitness-ranked subset when a pool exceeds capacity.
- `reject_dedup(candidate, pool, config=...)` rejects structural near-copies.
- `structural_distance(...)` and `structural_edit_cost(...)` expose the graph-level primitives used by scoring.
- `confidence_weighted_capped_forward_kl(...)`, `reverse_kl(...)`, and `coverage_ratio(...)` expose the voxel-level primitives for diagnostics and custom workflows.

A minimal downstream pool loop can therefore look like:

```python
from graph_to_voxel.refinement import annealed_threshold, reservoir_retain, score_refinement

score = score_refinement(
    candidate_graph=C,
    candidate_field=field_C,
    reference_a_graph=A,
    reference_a_field=field_A,
    reference_b_graph=B,
    reference_b_field=field_B,
    config=config,
    pool=pool,
)

threshold = annealed_threshold(
    admission_count=n_admitted,
    initial_threshold=float("inf"),
    steady_state_threshold=target_bits,
    anneal_horizon=pool_capacity,
)

if score.passed_gates and score.score_bits <= threshold:
    pool.append(C)
    n_admitted += 1
    pool = reservoir_retain(pool, pool_capacity, rng=rng)
```

The implemented scope is the MVP scorer and pool primitives. Perturbation stability, soft physics penalties, high-entropy holding-pen attribution, polarity-vs-order validation, and cross-cutting acyclicity are still design-level extensions rather than runtime APIs.
