"""LoopStructural-style engine adapter: graph -> probabilistic VoxelField.

The v1.7 adapter keeps the public LoopStructural-facing name but owns the soft
composition layer directly. Where a full LoopStructural installation is not
available, it evaluates signed-distance-like fields from graph constraints:
planar fields for conformable layers and explicit anchor-based envelopes for
embedded bodies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, cKDTree

from graph_to_voxel.engine.voxel_field import GridSpec, VoxelField
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.schema.edges import EdgeKind
from graph_to_voxel.schema.nodes import Contact, Fault, Orientation, StratigraphicUnit
from graph_to_voxel.schema.uncertainty import nominal_value


class InsufficientUnitDataError(ValueError):
    """Raised when a unit cannot produce an unambiguous scalar field."""

    def __init__(self, unit_id: str, reason: str) -> None:
        self.unit_id = unit_id
        self.reason = reason
        super().__init__(f"unit {unit_id!r} has insufficient data: {reason}")


class BandwidthMismatchWarning(UserWarning):
    """Warns when grid-derived and contact-density-derived bandwidths diverge."""


class CompositionAmbiguousWarning(UserWarning):
    """Warns when composition mode falls back to chronological erosion."""


class MemoryBudgetWarning(UserWarning):
    """Warns when the projected evaluation memory budget is large."""


class FaultsIgnoredWarning(UserWarning):
    """Warns when Fault nodes are present but not rendered (deferred to v2)."""


class AutoAnchoredWarning(UserWarning):
    """Warns when an embedded body anchor is inferred from closed contacts."""


class TopologyMismatchWarning(UserWarning):
    """Warns when declared unit topology disagrees with contact geometry."""


class TopologyMismatchError(ValueError):
    """Raised when declared embedded topology is incompatible with contact geometry."""

    def __init__(self, unit_id: str, declared: str, observed: str) -> None:
        self.unit_id = unit_id
        self.declared = declared
        self.observed = observed
        super().__init__(
            f"unit {unit_id!r} declares topology={declared!r} but contact geometry is {observed!r}"
        )


class PolarityIgnoredOnEmbeddedWarning(UserWarning):
    """Warns when Contact.polarity is authored for an embedded body."""


@dataclass(slots=True)
class StructuralFeature:
    name: str
    contacts: list[dict[str, Any]] = field(default_factory=list)
    orientations: list[dict[str, Any]] = field(default_factory=list)
    anchors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class StructuralFrame:
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    features: list[StructuralFeature]

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for feature in self.features:
            rows.extend(feature.contacts)
            rows.extend(feature.orientations)
            rows.extend(feature.anchors)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class PlaneConstraint:
    younger: str
    older: str
    plane: tuple[float, float, float] | None
    points: np.ndarray
    skip_for_distance: bool = False


def prepare_loopstructural(graph: Graph) -> StructuralFrame:
    """Extract per-unit feature inputs for the multi-scalar architecture."""
    graph.validate()
    _validate_topology(graph)
    features: list[StructuralFeature] = []
    all_points: list[tuple[float, float, float]] = []
    for unit_id in graph.unit_catalog():
        feature_name = _feature_name(unit_id)
        contacts: list[dict[str, Any]] = []
        orientations: list[dict[str, Any]] = []
        anchors: list[dict[str, Any]] = []
        unit_node = graph.unit_node_by_unit_id()[unit_id]
        anchor = unit_node.anchor_inside
        if anchor is not None:
            x, y, z = anchor
            anchors.append({"X": x, "Y": y, "Z": z, "feature_name": feature_name, "val": 1.0})
            all_points.append((x, y, z))
        for node in graph.nodes():
            if isinstance(node, Contact) and unit_id in node.between:
                x, y, z = [_as_float(nominal_value(component)) for component in node.position]
                contacts.append(
                    {
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "feature_name": feature_name,
                        "val": 0.0,
                        "polarity": node.polarity,
                        "interface": "__".join(sorted(node.between)),
                    }
                )
                all_points.append((x, y, z))
            elif isinstance(node, Orientation) and node.for_unit == unit_id:
                x, y, z = [_as_float(nominal_value(component)) for component in node.position]
                dip, azimuth = node.dip.nominal()
                dip_r = np.deg2rad(dip)
                az_r = np.deg2rad(azimuth)
                orientations.append(
                    {
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "feature_name": feature_name,
                        "val": 0.0,
                        "gx": float(np.sin(dip_r) * np.cos(az_r)),
                        "gy": float(np.sin(dip_r) * np.sin(az_r)),
                        "gz": float(np.cos(dip_r)),
                    }
                )
                all_points.append((x, y, z))
        features.append(StructuralFeature(feature_name, contacts, orientations, anchors))
    bounds = _bounds_for_points(all_points)
    return StructuralFrame(bounds=bounds, features=features)


def build_voxel_field(
    graph: Graph,
    grid: GridSpec,
    *,
    bandwidth: float | dict[str, float] | None = None,
    subgrid_factor: int = 1,
    min_membership: float = 0.05,
    batch_size: int = 1_000_000,
    zslab_size: int | None = None,
    epsg: int | None = None,
    prior: VoxelField | None = None,
    drop_threshold: float = 0.5,
) -> VoxelField:
    """Build a probabilistic VoxelField from a realised graph and regular grid."""
    graph.validate()
    _validate_topology(graph)
    fault_ids = sorted(node.id for node in graph.nodes() if isinstance(node, Fault))
    offset_edges = graph.get_edges(EdgeKind.OFFSET_BY)
    if fault_ids or offset_edges:
        warnings.warn(
            f"graph contains {len(fault_ids)} Fault node(s) and {len(offset_edges)} OFFSET_BY edge(s); "
            "fault rendering is deferred to v2 and will have no effect on the output",
            FaultsIgnoredWarning,
            stacklevel=2,
        )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if zslab_size is not None and zslab_size < 1:
        raise ValueError("zslab_size must be >= 1 when provided")

    unit_ids = graph.unit_catalog()
    if not unit_ids:
        raise ValueError("graph must contain at least one StratigraphicUnit")

    _warn_if_memory_budget_large(grid, len(unit_ids), subgrid_factor)
    h = _resolve_bandwidth(bandwidth, unit_ids, grid, graph)
    sample_points, weights = grid.sample_points(subgrid_factor=subgrid_factor)
    n_samples = len(sample_points)
    k_units = len(unit_ids)

    planes = _plane_constraints(graph)
    distances = np.empty((k_units, n_samples), dtype=np.float32)
    for unit_idx, unit_id in enumerate(unit_ids):
        distances[unit_idx] = _distance_field_for_unit(unit_id, graph, grid, sample_points, planes)

    h_arr = np.asarray([h[unit_id] for unit_id in unit_ids], dtype=np.float32)[:, None]
    support_membership = _sigmoid(distances / np.maximum(h_arr, np.finfo(np.float32).eps))
    weak_voxel = support_membership.max(axis=0) < min_membership

    composition_mode = _composition_modes(graph, grid, planes)
    exclusive = _apply_composition(support_membership, unit_ids, planes, composition_mode, graph)
    exclusive = np.clip(exclusive, 0.0, 1.0)
    background = np.maximum(0.0, 1.0 - exclusive.sum(axis=0, keepdims=True))
    total = exclusive.sum(axis=0, keepdims=True) + background
    probs = (exclusive / np.where(total > 1e-9, total, 1.0)).astype(np.float32)
    background_prob = (
        background.squeeze(0) / np.where(total.squeeze(0) > 1e-9, total.squeeze(0), 1.0)
    ).astype(np.float32)

    if subgrid_factor > 1:
        s3 = subgrid_factor**3
        weights_reshaped = weights.reshape((1, 1, 1, 1, s3))
        probs = (probs.reshape(k_units, *grid.shape, s3) * weights_reshaped).sum(axis=-1)
        support_membership = (
            support_membership.reshape(k_units, *grid.shape, s3) * weights_reshaped
        ).sum(axis=-1)
        scalar_field = (distances.reshape(k_units, *grid.shape, s3) * weights_reshaped).sum(axis=-1)
        background_prob = (background_prob.reshape(*grid.shape, s3) * weights.reshape((1, 1, 1, s3))).sum(axis=-1)
        weak_voxel = weak_voxel.reshape(*grid.shape, s3).all(axis=-1)
    else:
        probs = probs.reshape(k_units, *grid.shape)
        support_membership = support_membership.reshape(k_units, *grid.shape)
        scalar_field = distances.reshape(k_units, *grid.shape)
        background_prob = background_prob.reshape(grid.shape)
        weak_voxel = weak_voxel.reshape(grid.shape)

    domain_mask = ~weak_voxel
    most_likely = probs.argmax(axis=0).astype(np.int16)
    most_likely[~domain_mask] = -1
    entropy = _entropy_from_probs(probs, domain_mask)
    x, y, z = grid.coordinates()
    attrs = {
        "engine": "loopstructural-v2-multi-scalar",
        "unit_probs_kind": "mixing",
        "bandwidth_m": {unit_id: float(value) for unit_id, value in h.items()},
        "subgrid_factor": int(subgrid_factor),
        "min_membership": float(min_membership),
        "batch_size": int(batch_size),
        "zslab_size": zslab_size,
        "background_prob_max": float(background_prob.max(initial=0.0)),
        "composition_mode": composition_mode,
        "erode_bias_cells": _erode_bias_cells(composition_mode, h, grid),
        **({"faults_ignored": fault_ids} if fault_ids or offset_edges else {}),
        **(
            {"offset_by_edges_ignored": [(edge.source, edge.target) for edge in offset_edges]}
            if fault_ids or offset_edges
            else {}
        ),
    }
    field = VoxelField(
        most_likely_unit=most_likely,
        unit_probs=probs,
        support_membership=support_membership,
        entropy=entropy,
        domain_mask=domain_mask,
        scalar_field=scalar_field.astype(np.float32),
        x=x,
        y=y,
        z=z,
        unit_ids=unit_ids,
        feature_names=[_feature_name(unit_id) for unit_id in unit_ids],
        epsg=epsg,
        attrs=attrs,
    )
    if prior is not None:
        _warn_on_voxel_count_drop(field, prior, drop_threshold)
    return field


# Alias — tests and callers import this name.
build_loopstructural = build_voxel_field


def _plane_constraints(graph: Graph) -> list[PlaneConstraint]:
    constraints: list[PlaneConstraint] = []
    unit_nodes = graph.unit_node_by_unit_id()
    for edge in graph.get_edges(EdgeKind.OVERLIES):
        younger = graph._unit_id_for_node_id(edge.source)
        older = graph._unit_id_for_node_id(edge.target)
        younger_unit = unit_nodes.get(younger)
        points = np.asarray(_contacts_between(graph, (younger, older)), dtype=float)
        if younger_unit is not None and younger_unit.topology == "embedded":
            constraints.append(PlaneConstraint(younger, older, None, points, skip_for_distance=True))
            continue
        if len(points) == 0:
            continue
        if len(points) < 3:
            raise InsufficientUnitDataError(younger, f"boundary {younger}->{older} has fewer than 3 contacts")
        constraints.append(PlaneConstraint(younger, older, _fit_plane(points), points))
    return constraints


def _distance_field_for_unit(
    unit_id: str,
    graph: Graph,
    grid: GridSpec,
    points: np.ndarray,
    planes: list[PlaneConstraint],
) -> np.ndarray:
    unit = graph.unit_node_by_unit_id()[unit_id]
    if unit.topology == "embedded":
        if _has_authored_polarity(graph, unit_id):
            warnings.warn(
                f"Contact.polarity on embedded unit {unit_id!r} is ignored; anchor_inside defines the sign",
                PolarityIgnoredOnEmbeddedWarning,
                stacklevel=3,
            )
        anchor = unit.anchor_inside
        if anchor is None:
            raise InsufficientUnitDataError(unit_id, "embedded_without_anchor_or_closure")
        contacts = np.asarray(_contacts_for_embedded_unit(graph, unit_id), dtype=float)
        if len(contacts) < 4:
            raise InsufficientUnitDataError(unit_id, "closed body requires at least 4 contacts")
        return _closed_body_distance(points, anchor, contacts, grid).astype(np.float32)

    constraints: list[np.ndarray] = []
    for plane in planes:
        if plane.skip_for_distance or plane.plane is None:
            continue
        signed = _signed_distance_to_plane(points, plane.plane)
        if unit_id == plane.younger:
            constraints.append(_constraint_multiplier_for_unit(unit_id, plane, graph) * signed)
        elif unit_id == plane.older:
            constraints.append(_constraint_multiplier_for_unit(unit_id, plane, graph) * signed)
    if not constraints:
        if len(graph.unit_catalog()) == 1 or any(
            plane.older == unit_id and plane.skip_for_distance for plane in planes
        ):
            return np.full(len(points), grid.diagonal_length, dtype=np.float32)
        return np.full(len(points), -grid.diagonal_length, dtype=np.float32)
    return np.minimum.reduce(constraints).astype(np.float32)


def _closed_body_distance(
    points: np.ndarray,
    anchor: tuple[float, float, float],
    contacts: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    anchor_arr = np.asarray(anchor, dtype=float)
    spacing_floor = max(min(grid.spacing), np.finfo(float).eps)
    radii = np.maximum(np.max(np.abs(contacts - anchor_arr), axis=0), spacing_floor)
    scaled = (points - anchor_arr) / radii
    ellipsoid_radius = np.linalg.norm(scaled, axis=1)
    approx_distance = (1.0 - ellipsoid_radius) * float(np.mean(radii))

    tree = cKDTree(contacts)
    nearest_distance, _ = tree.query(points, workers=-1)
    return np.sign(approx_distance) * np.minimum(np.abs(approx_distance), nearest_distance)


def _apply_composition(
    support: np.ndarray,
    unit_ids: list[str],
    planes: list[PlaneConstraint],
    composition_mode: dict[str, str],
    graph: Graph,
) -> np.ndarray:
    exclusive = support.copy()
    unit_index = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
    unit_nodes = graph.unit_node_by_unit_id()
    explicit_edges = {
        (
            graph._unit_id_for_node_id(edge.source),
            graph._unit_id_for_node_id(edge.target),
        )
        for edge in graph.get_edges(EdgeKind.OVERLIES)
    }
    erode_pairs = [
        tuple(key.split("__", 1))
        for key, mode in composition_mode.items()
        if mode == "erode"
    ]
    siblings_by_older: dict[str, list[str]] = {}
    for younger, older in erode_pairs:
        if younger not in unit_index or older not in unit_index:
            continue
        siblings_by_older.setdefault(older, []).append(younger)
    handled_pairs: set[tuple[str, str]] = set()
    for older, siblings in siblings_by_older.items():
        siblings = sorted(set(siblings), key=lambda unit_id: (_chronology_rank(unit_nodes[unit_id]), unit_id))
        if len(siblings) <= 1:
            continue
        if _has_explicit_sibling_chronology(siblings, explicit_edges):
            continue
        older_idx = unit_index[older]
        ranks = [_chronology_rank(unit_nodes[unit_id]) for unit_id in siblings]
        if len(set(ranks)) == 1:
            sibling_support = np.stack([support[unit_index[y]] for y in siblings], axis=0)
            union = 1.0 - np.prod(1.0 - sibling_support, axis=0)
            total = sibling_support.sum(axis=0)
            weights = np.divide(
                sibling_support,
                total,
                out=np.full_like(sibling_support, 1.0 / len(siblings)),
                where=total > 1e-9,
            )
            for row_idx, sibling in enumerate(siblings):
                exclusive[unit_index[sibling]] = weights[row_idx] * union
                handled_pairs.add((sibling, older))
            exclusive[older_idx] *= 1.0 - union
            continue
        ordered = sorted(siblings, key=lambda unit_id: (-_chronology_rank(unit_nodes[unit_id]), unit_id))
        for idx, younger in enumerate(ordered):
            younger_idx = unit_index[younger]
            exclusive[older_idx] *= 1.0 - support[younger_idx]
            handled_pairs.add((younger, older))
            for lower in ordered[idx + 1:]:
                exclusive[unit_index[lower]] *= 1.0 - support[younger_idx]
    for younger, older in sorted(
        erode_pairs,
        key=lambda pair: (-_chronology_rank(unit_nodes[pair[0]]), pair[0], pair[1]),
    ):
        if (younger, older) in handled_pairs:
            continue
        if younger not in unit_index or older not in unit_index:
            continue
        exclusive[unit_index[older]] *= 1.0 - support[unit_index[younger]]
    return exclusive


def _composition_modes(
    graph: Graph,
    grid: GridSpec,
    planes: list[PlaneConstraint],
) -> dict[str, str]:
    by_pair = {_composition_key(plane.younger, plane.older): plane for plane in planes}
    modes: dict[str, str] = {}
    for edge in graph.get_edges(EdgeKind.OVERLIES):
        younger = graph._unit_id_for_node_id(edge.source)
        older = graph._unit_id_for_node_id(edge.target)
        key = _composition_key(younger, older)
        override = edge.metadata.get("composition")
        if override in {"erode", "softmax"}:
            modes[key] = str(override)
            continue
        plane = by_pair.get(key)
        if plane is None:
            modes[key] = "erode"
            warnings.warn(
                f"composition mode for {younger}->{older} is ambiguous: missing contacts; using erode",
                CompositionAmbiguousWarning,
                stacklevel=3,
            )
            continue
        if plane.skip_for_distance or plane.plane is None:
            modes[key] = "erode"
            continue
        modes[key] = "softmax" if _is_quasi_planar(plane.points, plane.plane, grid) else "erode"
    return modes


def _resolve_bandwidth(
    bandwidth: float | dict[str, float] | None,
    unit_ids: list[str],
    grid: GridSpec,
    graph: Graph,
) -> dict[str, float]:
    default = 0.5 * min(grid.spacing)
    if bandwidth is None:
        resolved = {unit_id: default for unit_id in unit_ids}
    elif isinstance(bandwidth, dict):
        resolved = {unit_id: float(bandwidth.get(unit_id, default)) for unit_id in unit_ids}
    else:
        resolved = {unit_id: float(bandwidth) for unit_id in unit_ids}
    for unit_id, value in resolved.items():
        if value <= 0.0:
            raise ValueError(f"bandwidth for unit {unit_id!r} must be > 0")
        contacts = np.asarray(_contacts_for_unit(graph, unit_id), dtype=float)
        if len(contacts) < 4:
            continue
        nn = cKDTree(contacts).query(contacts, k=2)[0][:, 1]
        h_data = 0.5 * float(np.median(nn))
        if h_data > 0.0 and max(value, h_data) / min(value, h_data) > 4.0:
            warnings.warn(
                f"bandwidth for unit {unit_id!r} differs from contact-density sentinel "
                f"(h_grid={value:g}, h_data={h_data:g}); consider a per-unit override",
                BandwidthMismatchWarning,
                stacklevel=3,
            )
    return resolved


def _warn_if_memory_budget_large(grid: GridSpec, n_units: int, subgrid_factor: int) -> None:
    projected = int(np.prod(grid.shape)) * (subgrid_factor**3) * max(n_units, 1) * 4 * 4
    if projected <= 4 * 1024**3:
        return
    warnings.warn(
        f"projected probabilistic field high-water mark is {projected / 1024**3:.1f} GiB; "
        "reduce subgrid_factor or render z slabs",
        MemoryBudgetWarning,
        stacklevel=3,
    )


def _contacts_between(graph: Graph, pair: tuple[str, str]) -> list[tuple[float, float, float]]:
    wanted = set(pair)
    result: list[tuple[float, float, float]] = []
    for node in graph.nodes():
        if not isinstance(node, Contact):
            continue
        if set(node.between) != wanted:
            continue
        result.append(tuple(_as_float(nominal_value(component)) for component in node.position))
    return result


def _contacts_for_unit(graph: Graph, unit_id: str) -> list[tuple[float, float, float]]:
    result: list[tuple[float, float, float]] = []
    for node in graph.nodes():
        if isinstance(node, Contact) and unit_id in node.between:
            result.append(tuple(_as_float(nominal_value(component)) for component in node.position))  # type: ignore[misc]
    return result


def _validate_topology(graph: Graph) -> None:
    for unit_id, unit in graph.unit_node_by_unit_id().items():
        raw_contacts = _contacts_for_embedded_unit(graph, unit_id) if unit.topology == "embedded" else _contacts_for_unit(graph, unit_id)
        contacts = np.asarray(raw_contacts, dtype=float)
        closure_anchor = _closed_envelope_anchor(contacts)
        if unit.topology == "embedded":
            if unit.anchor_inside is None:
                if closure_anchor is None:
                    raise InsufficientUnitDataError(unit_id, "embedded_without_anchor_or_closure")
                unit.anchor_inside = closure_anchor
                warnings.warn(
                    f"embedded unit {unit_id!r} did not specify anchor_inside; inferred {closure_anchor}",
                    AutoAnchoredWarning,
                    stacklevel=3,
                )
            elif closure_anchor is None:
                raise TopologyMismatchError(unit_id, declared="embedded", observed="open")
            continue
        if closure_anchor is not None:
            warnings.warn(
                f"unit {unit_id!r} declares topology='layer' but contacts form a closed envelope",
                TopologyMismatchWarning,
                stacklevel=3,
            )


def _closed_envelope_anchor(contacts: np.ndarray) -> tuple[float, float, float] | None:
    if len(contacts) < 4:
        return None
    try:
        hull = ConvexHull(contacts)
    except QhullError:
        return None
    centroid = hull.points[hull.vertices].mean(axis=0)
    slack = hull.equations[:, :3] @ centroid + hull.equations[:, 3]
    if not np.all(slack <= 1e-10):
        return None
    return tuple(float(value) for value in centroid)


def _contacts_for_embedded_unit(graph: Graph, unit_id: str) -> list[tuple[float, float, float]]:
    contacts: list[tuple[float, float, float]] = []
    for edge in graph.get_edges(EdgeKind.OVERLIES):
        if graph._unit_id_for_node_id(edge.source) != unit_id:
            continue
        older = graph._unit_id_for_node_id(edge.target)
        contacts.extend(_contacts_between(graph, (unit_id, older)))
    return contacts or _contacts_for_unit(graph, unit_id)


def _has_authored_polarity(graph: Graph, unit_id: str) -> bool:
    return any(
        isinstance(node, Contact) and unit_id in node.between and node.polarity is not None
        for node in graph.nodes()
    )


def _constraint_multiplier_for_unit(unit_id: str, plane: PlaneConstraint, graph: Graph) -> float:
    default = 1.0 if unit_id == plane.younger else -1.0
    unit = graph.unit_node_by_unit_id()[unit_id]
    if unit.topology != "layer":
        return default
    polarity = _polarity_for_unit_on_pair(graph, unit_id, (plane.younger, plane.older))
    if polarity is None:
        return default
    reference_raw_sign = 1 if default > 0.0 else -1
    return default if polarity == reference_raw_sign else -default


def _polarity_for_unit_on_pair(graph: Graph, unit_id: str, pair: tuple[str, str]) -> int | None:
    wanted = set(pair)
    for node in graph.nodes():
        if not isinstance(node, Contact) or node.polarity is None or set(node.between) != wanted:
            continue
        if unit_id == node.between[0]:
            return int(node.polarity)
        if unit_id == node.between[1]:
            return -int(node.polarity)
    return None


def _has_explicit_sibling_chronology(siblings: list[str], explicit_edges: set[tuple[str, str]]) -> bool:
    sibling_set = set(siblings)
    return any(source in sibling_set and target in sibling_set for source, target in explicit_edges)


def _chronology_rank(unit: StratigraphicUnit) -> int:
    return int(unit.metadata.get("chronology_rank", 0))


def _fit_plane(points: np.ndarray | list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Least-squares fit of z = a*x + b*y + c. Returns (a, b, c)."""
    xyz = np.asarray(points, dtype=float)
    if len(xyz) < 3:
        raise ValueError("at least 3 points are required to fit a plane")
    matrix = np.c_[xyz[:, 0], xyz[:, 1], np.ones(len(xyz))]
    a, b, c = np.linalg.lstsq(matrix, xyz[:, 2], rcond=None)[0]
    return float(a), float(b), float(c)


def _signed_distance_to_plane(points: np.ndarray, plane: tuple[float, float, float]) -> np.ndarray:
    a, b, c = plane
    denominator = float(np.sqrt(a * a + b * b + 1.0))
    return (points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)) / denominator


def _is_quasi_planar(points: np.ndarray, plane: tuple[float, float, float], grid: GridSpec) -> bool:
    if len(points) < 3:
        return False
    residual = np.abs(_signed_distance_to_plane(points, plane))
    return float(np.sqrt(np.mean(residual**2))) < 0.2 * min(grid.spacing)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))).astype(np.float32)


def _entropy_from_probs(probs: np.ndarray, domain_mask: np.ndarray) -> np.ndarray:
    probs64 = probs.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs64 > 0.0, np.log2(probs64), 0.0)
    entropy = -(probs64 * logp).sum(axis=0)
    if probs.shape[0] > 1:
        entropy = entropy / np.log2(probs.shape[0])
    else:
        entropy = np.zeros_like(entropy)
    return np.where(domain_mask, entropy, 0.0).astype(np.float32)


def _erode_bias_cells(
    composition_mode: dict[str, str],
    bandwidth: dict[str, float],
    grid: GridSpec,
) -> dict[str, float]:
    cell = min(grid.spacing)
    result: dict[str, float] = {}
    for key, mode in composition_mode.items():
        younger = key.split("__", 1)[0]
        result[key] = 0.0 if mode == "softmax" else 0.48 * bandwidth.get(younger, cell) / cell
    return result


def _composition_key(younger: str, older: str) -> str:
    return f"{younger}__{older}"


def _feature_name(unit_id: str) -> str:
    return f"f_{unit_id}"


def _coerce_point(value: object) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("anchor_inside must be a 3-value coordinate")
    return float(value[0]), float(value[1]), float(value[2])


def _bounds_for_points(points: list[tuple[float, float, float]]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if not points:
        return ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    xyz = np.asarray(points, dtype=float)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    pad = np.maximum((hi - lo) * 0.05, 1.0)
    lo = lo - pad
    hi = hi + pad
    return tuple((float(lo[idx]), float(hi[idx])) for idx in range(3))  # type: ignore[return-value]


def _as_float(value: object) -> float:
    if isinstance(value, tuple):
        raise TypeError("orientation tuples cannot be used as scalar coordinates")
    return float(value)


def _warn_on_voxel_count_drop(
    field: VoxelField,
    prior: VoxelField,
    drop_threshold: float,
) -> None:
    for unit_id in field.unit_ids:
        if unit_id not in prior.unit_ids:
            continue
        before_idx = prior.unit_ids.index(unit_id)
        after_idx = field.unit_ids.index(unit_id)
        before = int(np.count_nonzero(prior.most_likely_unit == before_idx))
        if before == 0:
            continue
        after = int(np.count_nonzero(field.most_likely_unit == after_idx))
        drop_fraction = (before - after) / before
        if drop_fraction <= drop_threshold:
            continue
        warnings.warn(
            f"unit {unit_id!r} has a voxel-count drop of {drop_fraction:.0%} on rebuild "
            f"({before} -> {after}). This often indicates a material geometry change; "
            "inspect support_membership and composition_mode for embedded-body effects.",
            UserWarning,
            stacklevel=3,
        )
