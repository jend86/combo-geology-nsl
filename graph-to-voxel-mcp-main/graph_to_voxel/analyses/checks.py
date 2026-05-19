from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.graph import EntityGraph, GraphValidationError


Severity = Literal["pass", "warn", "fail"]


@dataclass(slots=True)
class CheckResult:
    name: str
    severity: Severity
    details: str
    locations: list[tuple[float, float, float]]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["locations"] = [list(location) for location in self.locations]
        return data


def check_graph_stratigraphic_order(graph: EntityGraph) -> CheckResult:
    try:
        graph.validate()
    except GraphValidationError as exc:
        return CheckResult("graph_stratigraphic_order", "fail", str(exc), [])
    return CheckResult(
        "graph_stratigraphic_order",
        "pass",
        "OVERLIES relations are acyclic and ontology validation passed",
        [],
    )


def check_voxel_stratigraphic_order(graph: EntityGraph, field: VoxelField) -> CheckResult:
    order = graph.stratigraphic_order()
    unit_nodes = graph.unit_node_by_unit_id()
    layered_order = [unit_id for unit_id in order if unit_nodes[unit_id].topology == "layer"]
    rank = {unit_id: idx for idx, unit_id in enumerate(layered_order)}
    index_to_rank = {
        idx: rank[unit_id]
        for idx, unit_id in enumerate(field.unit_ids)
        if unit_id in rank
    }
    locations: list[tuple[float, float, float]] = []
    for ix in range(field.shape[0]):
        for iy in range(field.shape[1]):
            sequence = [unit for unit in field.most_likely_unit[ix, iy, ::-1] if unit >= 0]
            ranks = [index_to_rank.get(int(unit)) for unit in sequence if int(unit) in index_to_rank]
            if any(prev > nxt for prev, nxt in zip(ranks, ranks[1:], strict=False)):
                locations.append((float(field.x[ix]), float(field.y[iy]), float(field.z[0])))
    if locations:
        return CheckResult(
            "voxel_stratigraphic_order",
            "fail",
            f"{len(locations)} vertical column(s) violate declared unit order",
            locations,
        )
    return CheckResult("voxel_stratigraphic_order", "pass", "all vertical columns respect unit order", [])


def check_bulk_volume(graph: EntityGraph, field: VoxelField) -> CheckResult:
    unit_nodes = graph.unit_node_by_unit_id()
    failures = []
    locations: list[tuple[float, float, float]] = []
    for idx, unit_id in enumerate(field.unit_ids):
        bounds = unit_nodes.get(unit_id).bulk_volume_bounds if unit_id in unit_nodes else None
        if bounds is None:
            continue
        volume = float(np.count_nonzero(field.most_likely_unit == idx) * field.cell_volume)
        if volume < bounds.lo or volume > bounds.hi:
            failures.append(f"{unit_id}: {volume:.3g} outside [{bounds.lo:.3g}, {bounds.hi:.3g}]")
            unit_locations = np.argwhere(field.most_likely_unit == idx)
            if len(unit_locations):
                ix, iy, iz = unit_locations[0]
                locations.append((float(field.x[ix]), float(field.y[iy]), float(field.z[iz])))
    if failures:
        return CheckResult("bulk_volume", "warn", "; ".join(failures), locations)
    return CheckResult("bulk_volume", "pass", "all declared bulk-volume bounds are satisfied", [])


def check_domain_closure(field: VoxelField) -> CheckResult:
    failed = np.argwhere(field.domain_mask & (field.most_likely_unit < 0))
    if len(failed):
        locations = [_index_location(field, tuple(index)) for index in failed[:100]]
        return CheckResult(
            "domain_closure",
            "fail",
            f"{len(failed)} voxel(s) inside domain_mask are unassigned",
            locations,
        )
    return CheckResult("domain_closure", "pass", "all voxels inside domain_mask are assigned", [])


def check_orphan_blobs(graph: EntityGraph, field: VoxelField) -> CheckResult:
    unit_nodes = graph.unit_node_by_unit_id()
    failures = []
    locations: list[tuple[float, float, float]] = []
    for idx, unit_id in enumerate(field.unit_ids):
        node = unit_nodes.get(unit_id)
        if node is None or node.metadata.get("connectivity") != "single":
            continue
        labels, n_components = ndimage.label(field.most_likely_unit == idx)
        if n_components > 1:
            failures.append(f"{unit_id}: {n_components} connected components")
            for component_id in range(1, min(n_components, 3) + 1):
                component = np.argwhere(labels == component_id)
                if len(component):
                    locations.append(_index_location(field, tuple(component[0])))
    if failures:
        return CheckResult("orphan_blob", "warn", "; ".join(failures), locations)
    return CheckResult("orphan_blob", "pass", "single-connectivity units are connected", [])


def check_existence_presence(
    field: VoxelField,
    graph: EntityGraph,
    tolerance: float = 0.05,
) -> CheckResult:
    unit_nodes = graph.unit_node_by_unit_id()
    total = max(int(np.count_nonzero(field.domain_mask)), 1)
    failures = []
    locations: list[tuple[float, float, float]] = []
    for idx, unit_id in enumerate(field.unit_ids):
        node = unit_nodes.get(unit_id)
        if node is None or node.p_exists >= 0.5:
            continue
        voxel_fraction = float(np.count_nonzero((field.most_likely_unit == idx) & field.domain_mask) / total)
        if voxel_fraction > node.p_exists + tolerance:
            failures.append(
                f"{unit_id}: voxel_fraction={voxel_fraction:.3f} exceeds p_exists={node.p_exists:.3f}"
            )
            unit_locations = np.argwhere(field.most_likely_unit == idx)
            if len(unit_locations):
                locations.append(_index_location(field, tuple(unit_locations[0])))
    if failures:
        return CheckResult("existence_presence", "fail", "; ".join(failures), locations)
    return CheckResult(
        "existence_presence",
        "pass",
        "low-confidence units do not exceed declared existence confidence by voxel mass",
        [],
    )


def run_all_checks(graph: EntityGraph, field: VoxelField) -> list[CheckResult]:
    return [
        check_graph_stratigraphic_order(graph),
        check_voxel_stratigraphic_order(graph, field),
        check_bulk_volume(graph, field),
        check_domain_closure(field),
        check_orphan_blobs(graph, field),
        check_existence_presence(field, graph),
    ]


def _index_location(field: VoxelField, index: tuple[int, int, int]) -> tuple[float, float, float]:
    ix, iy, iz = index
    return float(field.x[ix]), float(field.y[iy]), float(field.z[iz])
