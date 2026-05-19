from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pydantic import ValidationError

from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.graph import EntityGraph, GraphValidationError, RealisationInfeasible
from graph_to_voxel.schema.nodes import Contact
from graph_to_voxel.schema.uncertainty import nominal_value


@dataclass(slots=True)
class Ensemble:
    realisations: list[VoxelField]
    unit_ids: list[str]
    n_rejected: int = 0

    def reduce(self) -> VoxelField:
        if not self.realisations:
            raise ValueError("cannot reduce an empty ensemble")

        first = self.realisations[0]
        accum = np.zeros((len(self.unit_ids), *first.shape), dtype=np.float64)
        support_accum = np.zeros_like(accum)
        domain_mask = np.zeros(first.shape, dtype=bool)

        for field in self.realisations:
            if field.shape != first.shape:
                raise ValueError("all ensemble fields must share the same grid")
            domain_mask |= field.domain_mask
            for local_idx, unit_id in enumerate(field.unit_ids):
                if unit_id not in self.unit_ids:
                    continue
                aggregate_idx = self.unit_ids.index(unit_id)
                accum[aggregate_idx] += field.unit_probs[local_idx]
                support_accum[aggregate_idx] += field.support_membership[local_idx]

        probs = (accum / len(self.realisations)).astype(np.float32)
        support_membership = (support_accum / len(self.realisations)).astype(np.float32)
        most_likely = probs.argmax(axis=0).astype(np.int16)
        min_membership = float(first.attrs.get("min_membership", 0.05))
        most_likely[(probs.max(axis=0) < min_membership) | ~domain_mask] = -1
        entropy = entropy_from_probs(probs, domain_mask)

        feature_names = [f"f_{unit_id}" for unit_id in self.unit_ids]
        scalar = np.zeros((len(feature_names), *first.shape), dtype=np.float32)
        return VoxelField(
            most_likely_unit=most_likely,
            unit_probs=probs,
            support_membership=support_membership,
            entropy=entropy,
            domain_mask=domain_mask,
            scalar_field=scalar,
            x=first.x,
            y=first.y,
            z=first.z,
            unit_ids=list(self.unit_ids),
            feature_names=feature_names,
            epsg=first.epsg,
            attrs={
                **first.attrs,
                "unit_probs_kind": "ensemble",
                "ensemble_size": len(self.realisations),
                "n_realisations": len(self.realisations),
                "n_rejected": self.n_rejected,
                "scalar_field_aggregated": False,
            },
        )


def stratigraphic_constrain(graph: EntityGraph) -> EntityGraph:
    """Reject realisations where contact depths violate declared stratigraphic order.

    After graph.realise(), contact positions are PointUncertainty. Checks that for
    each pair of adjacent units in stratigraphic order the upper boundary has a
    higher mean z than the lower boundary. Raises RealisationInfeasible if violated.
    """
    try:
        order = graph.stratigraphic_order()
    except Exception as exc:
        raise RealisationInfeasible(f"cannot determine stratigraphic order: {exc}") from exc

    # Collect mean z per adjacent-pair boundary key
    boundary_z: dict[frozenset, list[float]] = {}
    for node in graph.nodes():
        if not isinstance(node, Contact):
            continue
        z_val = nominal_value(node.position[2])
        if isinstance(z_val, (int, float, np.floating, np.integer)):
            key: frozenset = frozenset(node.between)
            boundary_z.setdefault(key, []).append(float(z_val))

    unit_nodes = graph.unit_node_by_unit_id()
    layered_order = [unit_id for unit_id in order if unit_nodes[unit_id].topology == "layer"]
    if len(layered_order) < 2:
        return graph

    # Walk only layered units youngest→oldest; embedded bodies do not define a
    # vertical monotone boundary, but layered units remain validated across them.
    prev_mean_z: float | None = None
    for upper, lower in itertools.pairwise(layered_order):
        zs = boundary_z.get(frozenset((upper, lower)))
        if not zs:
            continue
        mean_z = float(np.mean(zs))
        if prev_mean_z is not None and mean_z >= prev_mean_z:
            raise RealisationInfeasible(
                f"layer crossing: boundary {upper}/{lower} at mean z={mean_z:.1f} "
                f"is not below upper boundary at z={prev_mean_z:.1f}"
            )
        prev_mean_z = mean_z

    return graph


def run_ensemble(
    graph: EntityGraph,
    build: Callable[[EntityGraph], VoxelField],
    n: int,
    seed: int,
    oversample_budget: int | None = None,
) -> Ensemble:
    budget = oversample_budget if oversample_budget is not None else 2 * n
    root_rng = np.random.default_rng(seed)
    realisations: list[VoxelField] = []
    rejected = 0

    for _ in range(budget):
        child_seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
        rng = np.random.default_rng(child_seed)
        try:
            realised = graph.realise(rng)
            realised = stratigraphic_constrain(realised)
            realised.validate()
            realisations.append(build(realised))
            if len(realisations) == n:
                break
        except (RealisationInfeasible, GraphValidationError, ValidationError, ValueError):
            rejected += 1

    return Ensemble(realisations=realisations, unit_ids=graph.unit_catalog(), n_rejected=rejected)


def entropy_from_probs(probs: np.ndarray, domain_mask: np.ndarray | None = None) -> np.ndarray:
    probs64 = probs.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs64 > 0.0, np.log2(probs64), 0.0)
    entropy = -(probs64 * logp).sum(axis=0)
    k = probs.shape[0]
    if k > 1:
        entropy = entropy / np.log2(k)
    else:
        entropy = np.zeros_like(entropy)
    if domain_mask is not None:
        entropy = np.where(domain_mask, entropy, 0.0)
    return entropy.astype(np.float32)
