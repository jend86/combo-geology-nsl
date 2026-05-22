"""Sense-check analyses for voxel fields (design doc §5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.schema.nodes import StratigraphicUnit


@dataclass
class CheckResult:
    name: str
    severity: Literal["pass", "warn", "fail"]
    details: str
    locations: list[tuple[float, float, float]] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "details": self.details,
            "locations": self.locations,
        }


def check_domain_closure(field: VoxelField) -> CheckResult:
    """Check #4: every voxel inside domain_mask is assigned a unit (no -1)."""
    inside = field.domain_mask
    unassigned = np.sum(field.most_likely_unit[inside] < 0)
    if unassigned == 0:
        return CheckResult("domain_closure", "pass", "All domain voxels assigned")
    return CheckResult(
        "domain_closure", "fail",
        f"{unassigned} voxels inside domain_mask are unassigned (-1)",
    )


def check_existence_presence(
    field: VoxelField,
    graph: Graph,
    tolerance: float = 0.05,
) -> CheckResult:
    """Check #6: units with p_exists < 0.5 must not over-contribute voxels."""
    total = int(np.sum(field.domain_mask))
    if total == 0:
        return CheckResult("existence_presence", "warn", "Empty domain mask")

    problems: list[str] = []
    for node in graph.nodes():
        if not isinstance(node, StratigraphicUnit):
            continue
        if node.p_exists >= 0.5:
            continue
        try:
            uid_idx = field.unit_catalog.index(node.unit_id)
        except ValueError:
            continue
        voxel_count = int(np.sum(field.most_likely_unit[field.domain_mask] == uid_idx))
        voxel_frac = voxel_count / total
        if voxel_frac > node.p_exists + tolerance:
            problems.append(
                f"unit '{node.unit_id}': p_exists={node.p_exists:.3f} "
                f"but voxel_fraction={voxel_frac:.3f}"
            )

    if problems:
        return CheckResult("existence_presence", "fail", "; ".join(problems))
    return CheckResult(
        "existence_presence", "pass",
        "All low-p_exists units are within tolerance",
    )


def run_all_checks(graph: Graph, field: VoxelField) -> list[CheckResult]:
    return [
        check_domain_closure(field),
        check_existence_presence(field, graph),
    ]


__all__ = [
    "CheckResult",
    "check_domain_closure",
    "check_existence_presence",
    "run_all_checks",
]
from graph_to_voxel.analyses.checks import (
    CheckResult,
    check_bulk_volume,
    check_domain_closure,
    check_existence_presence,
    check_graph_stratigraphic_order,
    check_orphan_blobs,
    check_voxel_stratigraphic_order,
    run_all_checks,
)

__all__ = [
    "CheckResult",
    "check_bulk_volume",
    "check_domain_closure",
    "check_existence_presence",
    "check_graph_stratigraphic_order",
    "check_orphan_blobs",
    "check_voxel_stratigraphic_order",
    "run_all_checks",
]
