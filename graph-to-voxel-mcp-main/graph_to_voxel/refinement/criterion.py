from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence, TypeVar

import numpy as np

from graph_to_voxel.analyses.checks import check_domain_closure, check_voxel_stratigraphic_order
from graph_to_voxel.engine import VoxelField
from graph_to_voxel.graph import EntityGraph, GraphValidationError, RealisationInfeasible
from graph_to_voxel.schema.edges import GraphEdge
from graph_to_voxel.schema.nodes import (
    Contact,
    Fault,
    Location,
    NodeValue,
    Orientation,
    Sample,
    Series,
    StratigraphicUnit,
)
from graph_to_voxel.schema.uncertainty import nominal_value
from graph_to_voxel.voxel import stratigraphic_constrain


@dataclass(frozen=True, slots=True)
class StructuralCostConfig:
    """Placeholder edit-cost calibration in bits.

    These defaults intentionally match the design doc's v1 placeholder stance:
    roughly log2(20) bits per new structural element, plus a small residual
    absolute-complexity pressure.
    """

    residual_absolute_weight: float = 0.1
    absolute_element_bits: float = math.log2(20.0)
    added_element_bits: float = math.log2(20.0)
    modified_distance_bits: float = 1.0
    consensus_deleted_bits: float = math.log2(20.0)
    split_merge_bits: float = math.log2(20.0)
    parameter_scale: float = 1.0
    match_threshold: float = 10.0

    def __post_init__(self) -> None:
        if self.residual_absolute_weight < 0.0:
            raise ValueError("residual_absolute_weight must be non-negative")
        if self.parameter_scale <= 0.0:
            raise ValueError("parameter_scale must be positive")
        if self.match_threshold < 0.0:
            raise ValueError("match_threshold must be non-negative")


@dataclass(frozen=True, slots=True)
class RefinementCriterionConfig:
    epsilon: float = 0.01
    kappa_bits: float | None = None
    effective_sample_size: float = 1.0
    coverage_threshold: float = 0.95
    dedup_epsilon: float = 1e-6
    run_physics_gates: bool = True
    structural: StructuralCostConfig = field(default_factory=StructuralCostConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if self.kappa_bits is not None and self.kappa_bits <= 0.0:
            raise ValueError("kappa_bits must be positive")
        if self.effective_sample_size <= 0.0:
            raise ValueError("effective_sample_size must be positive")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in [0, 1]")
        if self.dedup_epsilon < 0.0:
            raise ValueError("dedup_epsilon must be non-negative")

    @property
    def effective_kappa_bits(self) -> float:
        return self.kappa_bits if self.kappa_bits is not None else math.log2(1.0 / self.epsilon)


@dataclass(frozen=True, slots=True)
class GateFailure:
    name: str
    details: str


@dataclass(frozen=True, slots=True)
class FitLoss:
    loss_bits: float
    raw_capped_sum_bits: float
    n_voxels: int
    coverage: float
    effective_sample_size: float
    kappa_bits: float
    union_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuralEditCost:
    loss_bits: float
    absolute_bits: float
    residual_absolute_bits: float
    added_bits: float
    modification_bits: float
    consensus_deleted_bits: float
    split_merge_bits: float
    n_elements: int
    n_added: int
    n_modified: int
    n_deleted_consensus: int
    n_split_merge: int


@dataclass(frozen=True, slots=True)
class RefinementScore:
    score_bits: float
    structural_bits: float
    fit_bits: float
    physics_bits: float
    structural: StructuralEditCost | None
    fit_a: FitLoss | None
    fit_b: FitLoss | None
    gate_failures: tuple[GateFailure, ...]
    diagnostics: dict[str, float]
    flags: tuple[str, ...] = ()

    @property
    def passed_gates(self) -> bool:
        return not self.gate_failures


@dataclass(frozen=True, slots=True)
class _AlignedFields:
    unit_ids: tuple[str, ...]
    reference_probs: np.ndarray
    candidate_probs: np.ndarray
    reference_mask: np.ndarray
    candidate_mask: np.ndarray
    intersection_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _StructuralElement:
    kind: str
    role: tuple[str, ...]
    vector: tuple[float, ...]
    attrs: tuple[str, ...]
    label: str

    @property
    def match_key(self) -> tuple[str, tuple[str, ...]]:
        return self.kind, self.role


T = TypeVar("T")


def coverage_ratio(reference: VoxelField, candidate: VoxelField) -> float:
    """Return |candidate.mask ∩ reference.mask| / |reference.mask|."""

    _require_same_shape(reference, candidate)
    reference_mask = np.asarray(reference.domain_mask, dtype=bool)
    denom = int(np.count_nonzero(reference_mask))
    if denom == 0:
        return 1.0
    covered = np.count_nonzero(reference_mask & np.asarray(candidate.domain_mask, dtype=bool))
    return float(covered / denom)


def confidence_weighted_capped_forward_kl(
    reference: VoxelField,
    candidate: VoxelField,
    *,
    config: RefinementCriterionConfig | None = None,
) -> FitLoss:
    """Compute L_kappa(reference | candidate) from design §3.1-§3.3."""

    cfg = config or RefinementCriterionConfig()
    aligned = _align_fields(reference, candidate)
    n_voxels = int(np.count_nonzero(aligned.intersection_mask))
    coverage = coverage_ratio(reference, candidate)
    if n_voxels == 0:
        loss = math.inf if np.any(aligned.reference_mask) else 0.0
        return FitLoss(
            loss_bits=loss,
            raw_capped_sum_bits=loss,
            n_voxels=0,
            coverage=coverage,
            effective_sample_size=cfg.effective_sample_size,
            kappa_bits=cfg.effective_kappa_bits,
            union_unit_ids=aligned.unit_ids,
        )

    p_reference = _smooth(aligned.reference_probs, cfg.epsilon)
    p_candidate = _smooth(aligned.candidate_probs, cfg.epsilon)
    kl = _kl_bits(p_reference, p_candidate)
    confidence = 1.0 - _normalised_entropy(p_reference)
    capped = np.minimum(confidence * kl, cfg.effective_kappa_bits)
    raw_sum = float(np.sum(capped[aligned.intersection_mask], dtype=np.float64))
    loss = raw_sum * cfg.effective_sample_size / n_voxels
    return FitLoss(
        loss_bits=float(loss),
        raw_capped_sum_bits=raw_sum,
        n_voxels=n_voxels,
        coverage=coverage,
        effective_sample_size=cfg.effective_sample_size,
        kappa_bits=cfg.effective_kappa_bits,
        union_unit_ids=aligned.unit_ids,
    )


def reverse_kl(
    candidate: VoxelField,
    reference: VoxelField,
    *,
    config: RefinementCriterionConfig | None = None,
) -> FitLoss:
    """Compute ESS-scaled reverse KL, KL(candidate || reference), as telemetry."""

    cfg = config or RefinementCriterionConfig()
    aligned = _align_fields(reference, candidate)
    n_voxels = int(np.count_nonzero(aligned.intersection_mask))
    coverage = coverage_ratio(reference, candidate)
    if n_voxels == 0:
        loss = math.inf if np.any(aligned.reference_mask) else 0.0
        return FitLoss(
            loss_bits=loss,
            raw_capped_sum_bits=loss,
            n_voxels=0,
            coverage=coverage,
            effective_sample_size=cfg.effective_sample_size,
            kappa_bits=math.inf,
            union_unit_ids=aligned.unit_ids,
        )
    p_reference = _smooth(aligned.reference_probs, cfg.epsilon)
    p_candidate = _smooth(aligned.candidate_probs, cfg.epsilon)
    kl = _kl_bits(p_candidate, p_reference)
    raw_sum = float(np.sum(kl[aligned.intersection_mask], dtype=np.float64))
    loss = raw_sum * cfg.effective_sample_size / n_voxels
    return FitLoss(
        loss_bits=float(loss),
        raw_capped_sum_bits=raw_sum,
        n_voxels=n_voxels,
        coverage=coverage,
        effective_sample_size=cfg.effective_sample_size,
        kappa_bits=math.inf,
        union_unit_ids=aligned.unit_ids,
    )


def structural_edit_cost(
    candidate: EntityGraph,
    reference_a: EntityGraph,
    reference_b: EntityGraph,
    *,
    config: RefinementCriterionConfig | None = None,
) -> StructuralEditCost:
    """Compute the MVP pair-relative L_delta(C | A, B)."""

    cfg = config or RefinementCriterionConfig()
    costs = cfg.structural
    candidate_elements = _structural_elements(candidate)
    reference_a_elements = _structural_elements(reference_a)
    reference_b_elements = _structural_elements(reference_b)
    reference_elements = [*reference_a_elements, *reference_b_elements]

    added_count = 0
    modified_count = 0
    modification_distance = 0.0
    for element in candidate_elements:
        distance = _best_distance(element, reference_elements, costs)
        if math.isinf(distance) or distance > costs.match_threshold:
            added_count += 1
        else:
            modified_count += 1
            modification_distance += distance

    consensus_deleted = 0
    for a_element, b_element, _distance in _greedy_matches(
        reference_a_elements,
        reference_b_elements,
        costs,
        threshold=costs.match_threshold,
    ):
        if (
            _best_distance(a_element, candidate_elements, costs) > costs.match_threshold
            and _best_distance(b_element, candidate_elements, costs) > costs.match_threshold
        ):
            consensus_deleted += 1

    split_merge_count = _split_merge_count(candidate_elements, reference_a_elements, reference_b_elements)
    absolute_bits = len(candidate_elements) * costs.absolute_element_bits
    residual_absolute_bits = costs.residual_absolute_weight * absolute_bits
    added_bits = added_count * costs.added_element_bits
    modification_bits = modification_distance * costs.modified_distance_bits
    consensus_deleted_bits = consensus_deleted * costs.consensus_deleted_bits
    split_merge_bits = split_merge_count * costs.split_merge_bits
    loss_bits = residual_absolute_bits + added_bits + modification_bits
    loss_bits += consensus_deleted_bits + split_merge_bits
    return StructuralEditCost(
        loss_bits=float(loss_bits),
        absolute_bits=float(absolute_bits),
        residual_absolute_bits=float(residual_absolute_bits),
        added_bits=float(added_bits),
        modification_bits=float(modification_bits),
        consensus_deleted_bits=float(consensus_deleted_bits),
        split_merge_bits=float(split_merge_bits),
        n_elements=len(candidate_elements),
        n_added=added_count,
        n_modified=modified_count,
        n_deleted_consensus=consensus_deleted,
        n_split_merge=split_merge_count,
    )


def structural_distance(
    graph_a: EntityGraph,
    graph_b: EntityGraph,
    *,
    config: RefinementCriterionConfig | None = None,
) -> float:
    """Greedy finite-element structural distance used by the dedup gate."""

    cfg = config or RefinementCriterionConfig()
    costs = cfg.structural
    elements_a = _structural_elements(graph_a)
    elements_b = _structural_elements(graph_b)
    matches = _greedy_matches(elements_a, elements_b, costs, threshold=costs.match_threshold)
    matched_a = {id(a_element) for a_element, _b_element, _distance in matches}
    matched_b = {id(b_element) for _a_element, b_element, _distance in matches}
    distance = sum(match_distance for _a, _b, match_distance in matches)
    distance += len([element for element in elements_a if id(element) not in matched_a])
    distance += len([element for element in elements_b if id(element) not in matched_b])
    return float(distance / max(len(elements_a), len(elements_b), 1))


def reject_dedup(
    candidate: EntityGraph,
    pool: Iterable[EntityGraph],
    *,
    config: RefinementCriterionConfig | None = None,
) -> bool:
    cfg = config or RefinementCriterionConfig()
    return any(
        structural_distance(candidate, existing, config=cfg) < cfg.dedup_epsilon
        for existing in pool
    )


def score_refinement(
    *,
    candidate_graph: EntityGraph,
    candidate_field: VoxelField,
    reference_a_graph: EntityGraph,
    reference_a_field: VoxelField,
    reference_b_graph: EntityGraph,
    reference_b_field: VoxelField,
    config: RefinementCriterionConfig | None = None,
    pool: Iterable[EntityGraph] | None = None,
) -> RefinementScore:
    """Score C against references A and B using the design's MVP scorer."""

    cfg = config or RefinementCriterionConfig()
    gate_failures = list(
        _hard_gate_failures(
            candidate_graph=candidate_graph,
            candidate_field=candidate_field,
            reference_a_field=reference_a_field,
            reference_b_field=reference_b_field,
            config=cfg,
            pool=pool,
        )
    )
    diagnostics = {
        "coverage_a": coverage_ratio(reference_a_field, candidate_field),
        "coverage_b": coverage_ratio(reference_b_field, candidate_field),
    }
    if gate_failures:
        return RefinementScore(
            score_bits=math.inf,
            structural_bits=math.inf,
            fit_bits=math.inf,
            physics_bits=math.inf,
            structural=None,
            fit_a=None,
            fit_b=None,
            gate_failures=tuple(gate_failures),
            diagnostics=diagnostics,
        )

    structural = structural_edit_cost(
        candidate_graph,
        reference_a_graph,
        reference_b_graph,
        config=cfg,
    )
    fit_a = confidence_weighted_capped_forward_kl(
        reference_a_field,
        candidate_field,
        config=cfg,
    )
    fit_b = confidence_weighted_capped_forward_kl(
        reference_b_field,
        candidate_field,
        config=cfg,
    )
    fit_bits = 0.5 * (fit_a.loss_bits + fit_b.loss_bits)
    score_bits = structural.loss_bits + fit_bits
    reverse_a = reverse_kl(candidate_field, reference_a_field, config=cfg)
    reverse_b = reverse_kl(candidate_field, reference_b_field, config=cfg)
    diagnostics.update(
        {
            "reverse_kl_a_bits": reverse_a.loss_bits,
            "reverse_kl_b_bits": reverse_b.loss_bits,
            "structural_added_bits": structural.added_bits,
            "structural_residual_absolute_bits": structural.residual_absolute_bits,
        }
    )
    flags = ("added_structural_elements",) if structural.n_added else ()
    return RefinementScore(
        score_bits=float(score_bits),
        structural_bits=structural.loss_bits,
        fit_bits=float(fit_bits),
        physics_bits=0.0,
        structural=structural,
        fit_a=fit_a,
        fit_b=fit_b,
        gate_failures=(),
        diagnostics=diagnostics,
        flags=flags,
    )


def annealed_threshold(
    admission_count: int,
    *,
    initial_threshold: float,
    steady_state_threshold: float,
    anneal_horizon: int,
) -> float:
    """Linear admission-count threshold schedule from design §6.4."""

    if admission_count < 0:
        raise ValueError("admission_count must be non-negative")
    if anneal_horizon <= 0:
        return float(steady_state_threshold)
    if math.isinf(initial_threshold):
        return math.inf if admission_count < anneal_horizon else float(steady_state_threshold)
    fraction = min(admission_count / anneal_horizon, 1.0)
    return float(initial_threshold + (steady_state_threshold - initial_threshold) * fraction)


def reservoir_retain(
    items: Sequence[T],
    capacity: int,
    *,
    rng: np.random.Generator | None = None,
    locked_indices: set[int] | None = None,
) -> list[T]:
    """Return a random non-fitness-retained subset no larger than capacity."""

    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    retained = list(items)
    locked = set(locked_indices or set())
    random = rng or np.random.default_rng()
    while len(retained) > capacity:
        removable = [idx for idx in range(len(retained)) if idx not in locked]
        if not removable:
            break
        remove_idx = int(random.choice(removable))
        retained.pop(remove_idx)
        locked = {idx - 1 if idx > remove_idx else idx for idx in locked if idx != remove_idx}
    return retained


def _hard_gate_failures(
    *,
    candidate_graph: EntityGraph,
    candidate_field: VoxelField,
    reference_a_field: VoxelField,
    reference_b_field: VoxelField,
    config: RefinementCriterionConfig,
    pool: Iterable[EntityGraph] | None,
) -> Iterable[GateFailure]:
    cov_a = coverage_ratio(reference_a_field, candidate_field)
    cov_b = coverage_ratio(reference_b_field, candidate_field)
    if cov_a < config.coverage_threshold:
        yield GateFailure(
            "coverage_a",
            f"candidate covers {cov_a:.3f} of reference A, below {config.coverage_threshold:.3f}",
        )
    if cov_b < config.coverage_threshold:
        yield GateFailure(
            "coverage_b",
            f"candidate covers {cov_b:.3f} of reference B, below {config.coverage_threshold:.3f}",
        )
    if pool is not None and reject_dedup(candidate_graph, pool, config=config):
        yield GateFailure("dedup", "candidate is structurally within dedup_epsilon of a pool member")
    if not config.run_physics_gates:
        return
    try:
        candidate_graph.validate()
    except GraphValidationError as exc:
        yield GateFailure("schema_validity", str(exc))
    try:
        stratigraphic_constrain(candidate_graph)
    except RealisationInfeasible as exc:
        yield GateFailure("stratigraphic_consistency", str(exc))
    domain_result = check_domain_closure(candidate_field)
    if domain_result.severity == "fail":
        yield GateFailure(domain_result.name, domain_result.details)
    try:
        order_result = check_voxel_stratigraphic_order(candidate_graph, candidate_field)
    except Exception as exc:
        yield GateFailure("voxel_stratigraphic_order", str(exc))
    else:
        if order_result.severity == "fail":
            yield GateFailure(order_result.name, order_result.details)


def _align_fields(reference: VoxelField, candidate: VoxelField) -> _AlignedFields:
    _require_same_shape(reference, candidate)
    unit_ids = tuple(dict.fromkeys([*reference.unit_ids, *candidate.unit_ids]))
    reference_probs, reference_mask = _normalised_field_probs(reference)
    candidate_probs, candidate_mask = _normalised_field_probs(candidate)
    aligned_reference = _project_probs(reference_probs, reference.unit_ids, unit_ids)
    aligned_candidate = _project_probs(candidate_probs, candidate.unit_ids, unit_ids)
    intersection = reference_mask & candidate_mask
    return _AlignedFields(
        unit_ids=unit_ids,
        reference_probs=aligned_reference,
        candidate_probs=aligned_candidate,
        reference_mask=reference_mask,
        candidate_mask=candidate_mask,
        intersection_mask=intersection,
    )


def _normalised_field_probs(field: VoxelField) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(field.unit_probs, dtype=np.float64).copy()
    sums = probs.sum(axis=0, dtype=np.float64)
    mask = np.asarray(field.domain_mask, dtype=bool) & (sums > 0.0)
    probs[:, mask] /= sums[mask]
    probs[:, ~mask] = 0.0
    return probs, mask


def _project_probs(
    probs: np.ndarray,
    unit_ids: list[str],
    target_unit_ids: tuple[str, ...],
) -> np.ndarray:
    projected = np.zeros((len(target_unit_ids), *probs.shape[1:]), dtype=np.float64)
    target_index = {unit_id: idx for idx, unit_id in enumerate(target_unit_ids)}
    for local_idx, unit_id in enumerate(unit_ids):
        projected[target_index[unit_id]] = probs[local_idx]
    return projected


def _smooth(probs: np.ndarray, epsilon: float) -> np.ndarray:
    return (1.0 - epsilon) * probs + epsilon / probs.shape[0]


def _kl_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(p * (np.log2(p) - np.log2(q)), axis=0)


def _normalised_entropy(probs: np.ndarray) -> np.ndarray:
    if probs.shape[0] <= 1:
        return np.zeros(probs.shape[1:], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0.0, np.log2(probs), 0.0)
    entropy = -np.sum(probs * logp, axis=0)
    return entropy / math.log2(probs.shape[0])


def _require_same_shape(reference: VoxelField, candidate: VoxelField) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"VoxelField shapes differ: {reference.shape} != {candidate.shape}")


def _structural_elements(graph: EntityGraph) -> list[_StructuralElement]:
    elements = [_node_element(node) for node in graph.nodes()]
    elements.extend(_edge_element(edge, graph) for edge in graph.get_edges())
    return elements


def _node_element(node: NodeValue) -> _StructuralElement:
    if isinstance(node, StratigraphicUnit):
        vector = tuple(float(value) for value in node.anchor_inside or ())
        return _StructuralElement(
            kind="unit",
            role=(node.unit_id,),
            vector=vector,
            attrs=(node.series_id, node.topology),
            label=node.id,
        )
    if isinstance(node, Contact):
        return _StructuralElement(
            kind="contact",
            role=tuple(sorted(node.between)),
            vector=_position_vector(node),
            attrs=(f"polarity={node.polarity}",),
            label=node.id,
        )
    if isinstance(node, Orientation):
        return _StructuralElement(
            kind="orientation",
            role=(node.for_unit, node.feature or ""),
            vector=(*_position_vector(node), *_orientation_vector(node.dip)),
            attrs=(),
            label=node.id,
        )
    if isinstance(node, Fault):
        kinematic = node.kinematic.nominal() if node.kinematic is not None else ""
        return _StructuralElement(
            kind="fault",
            role=(str(node.chronology_rank), str(kinematic)),
            vector=(float(len(node.surface_points)),),
            attrs=tuple(sorted(node.surface_points)),
            label=node.id,
        )
    if isinstance(node, Series):
        return _StructuralElement(
            kind="series",
            role=(node.series_id or node.id,),
            vector=(),
            attrs=(node.name or "",),
            label=node.id,
        )
    if isinstance(node, Location):
        return _StructuralElement(
            kind="location",
            role=(node.name or node.id,),
            vector=_position_vector(node),
            attrs=(),
            label=node.id,
        )
    if isinstance(node, Sample):
        return _StructuralElement(
            kind="sample",
            role=(node.analyte, node.unit_of_measure),
            vector=_position_vector(node) if node.position is not None else (),
            attrs=(str(nominal_value(node.value)),),
            label=node.id,
        )
    return _StructuralElement(
        kind=type(node).__name__,
        role=(node.id,),
        vector=(),
        attrs=(),
        label=node.id,
    )


def _edge_element(edge: GraphEdge, graph: EntityGraph) -> _StructuralElement:
    return _StructuralElement(
        kind="edge",
        role=(edge.kind.value, _semantic_endpoint(edge.source, graph), _semantic_endpoint(edge.target, graph)),
        vector=(float(edge.p_exists),),
        attrs=(),
        label=edge.id or f"{edge.kind.value}:{edge.source}->{edge.target}",
    )


def _semantic_endpoint(node_id: str, graph: EntityGraph) -> str:
    try:
        node = graph.get_node(node_id)
    except KeyError:
        return node_id
    if isinstance(node, StratigraphicUnit):
        return node.unit_id
    if isinstance(node, Series):
        return node.series_id or node.id
    return node_id


def _position_vector(node: Contact | Orientation | Location | Sample) -> tuple[float, float, float]:
    values = []
    for component in node.position or ():
        nominal = nominal_value(component)
        if isinstance(nominal, tuple):
            values.append(float(nominal[0]))
        else:
            values.append(float(nominal))
    if len(values) != 3:
        return ()  # type: ignore[return-value]
    return tuple(values)  # type: ignore[return-value]


def _orientation_vector(orientation: object) -> tuple[float, float]:
    nominal = nominal_value(orientation)  # type: ignore[arg-type]
    if isinstance(nominal, tuple):
        return float(nominal[0]), float(nominal[1])
    return float(nominal), 0.0


def _element_distance(
    left: _StructuralElement,
    right: _StructuralElement,
    costs: StructuralCostConfig,
) -> float:
    if left.match_key != right.match_key:
        return math.inf
    distance = 0.0
    if left.vector or right.vector:
        if len(left.vector) != len(right.vector):
            return math.inf
        left_vec = np.asarray(left.vector, dtype=float)
        right_vec = np.asarray(right.vector, dtype=float)
        distance += float(np.linalg.norm(left_vec - right_vec) / costs.parameter_scale)
    distance += _attr_distance(left.attrs, right.attrs)
    return distance


def _attr_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if len(left) != len(right):
        return float(max(len(left), len(right)))
    return float(sum(a != b for a, b in zip(left, right, strict=True)))


def _best_distance(
    element: _StructuralElement,
    candidates: Sequence[_StructuralElement],
    costs: StructuralCostConfig,
) -> float:
    if not candidates:
        return math.inf
    return min(_element_distance(element, candidate, costs) for candidate in candidates)


def _greedy_matches(
    left: Sequence[_StructuralElement],
    right: Sequence[_StructuralElement],
    costs: StructuralCostConfig,
    *,
    threshold: float,
) -> list[tuple[_StructuralElement, _StructuralElement, float]]:
    candidates: list[tuple[float, int, int]] = []
    for left_idx, left_element in enumerate(left):
        for right_idx, right_element in enumerate(right):
            distance = _element_distance(left_element, right_element, costs)
            if distance <= threshold:
                candidates.append((distance, left_idx, right_idx))
    candidates.sort(key=lambda item: item[0])
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    matches: list[tuple[_StructuralElement, _StructuralElement, float]] = []
    for distance, left_idx, right_idx in candidates:
        if left_idx in matched_left or right_idx in matched_right:
            continue
        matched_left.add(left_idx)
        matched_right.add(right_idx)
        matches.append((left[left_idx], right[right_idx], distance))
    return matches


def _split_merge_count(
    candidate: Sequence[_StructuralElement],
    reference_a: Sequence[_StructuralElement],
    reference_b: Sequence[_StructuralElement],
) -> int:
    candidate_counts = Counter(element.match_key for element in candidate)
    a_counts = Counter(element.match_key for element in reference_a)
    b_counts = Counter(element.match_key for element in reference_b)
    count = 0
    for key in set(candidate_counts) | set(a_counts) | set(b_counts):
        source_count = max(a_counts[key], b_counts[key])
        candidate_count = candidate_counts[key]
        if source_count > 1 and candidate_count == 1:
            count += 1
        elif source_count == 1 and candidate_count > 1:
            count += candidate_count - 1
    return count
