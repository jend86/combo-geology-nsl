"""Dry-run / simulation of the §06 self-supervised refinement criterion.

Builds two initial graphs A and B over the same geography, then a family of
candidate graphs C that extend / mediate / drift away from A and B. Each
candidate is realised to a VoxelField and scored with score_refinement.

The point of the script is to surface friction in the end-to-end pipeline,
not to pass tests. Output is a console table of (structural, fit, score,
gates) per candidate plus a short interpretation block at the end.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from graph_to_voxel.engine import GridSpec, build_voxel_field
from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.refinement import (
    RefinementCriterionConfig,
    score_refinement,
    structural_distance,
)
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Contact, Orientation, Series, StratigraphicUnit
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import OrientationUncertainty, PointUncertainty

# Silence the faults-ignored warning the adapter emits for any Fault node:
# we know v1 voxelisation does not render faults but we still want to inspect
# the structural cost they contribute.
warnings.filterwarnings("ignore", message=".*Fault node.*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(source: str = "dry-run") -> Provenance:
    return Provenance(
        source=source,
        confidence=1.0,
        timestamp=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )


def _pt(value: float) -> PointUncertainty:
    return PointUncertainty(value=value)


def _two_units(g: Graph, names: tuple[str, str] = ("above", "below")) -> None:
    upper, lower = names
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id=upper, unit_id=upper, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id=lower, unit_id=lower, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_edge(GraphEdge(id="e_series_upper", kind=EdgeKind.MEMBER_OF_SERIES, source=upper, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_series_lower", kind=EdgeKind.MEMBER_OF_SERIES, source=lower, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_overlies", kind=EdgeKind.OVERLIES, source=upper, target=lower, provenance=_prov()))


def _planar_contacts(g: Graph, names: tuple[str, str], plane_z: float, jitter: float = 0.0, rng: np.random.Generator | None = None) -> None:
    """Sprinkle 5 contacts on a roughly horizontal plane at z = plane_z."""
    rng = rng or np.random.default_rng(0)
    upper, lower = names
    xs = [200.0, 400.0, 600.0, 800.0, 500.0]
    ys = [200.0, 200.0, 500.0, 800.0, 600.0]
    for i, (x, y) in enumerate(zip(xs, ys)):
        z = plane_z + (rng.normal(0.0, jitter) if jitter > 0 else 0.0)
        g.add_node(Contact(
            id=f"c{i}",
            position=(_pt(x), _pt(y), _pt(z)),
            between=(upper, lower),
            p_exists=1.0,
            provenance=_prov(),
        ))
    g.add_node(Orientation(
        id="o0",
        position=(_pt(500.0), _pt(500.0), _pt(plane_z)),
        dip=OrientationUncertainty(dip_mean=0.0, dip_kappa=1e4, azimuth_mean=0.0, azimuth_kappa=1.0),
        for_unit=upper,
        p_exists=1.0,
        provenance=_prov(),
    ))


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def graph_a() -> Graph:
    """A: horizontal contact at z=500. Same units as B."""
    g = Graph()
    _two_units(g)
    _planar_contacts(g, ("above", "below"), plane_z=500.0)
    return g


def graph_b() -> Graph:
    """B: horizontal contact at z=520 (slightly higher). Same units as A."""
    g = Graph()
    _two_units(g)
    _planar_contacts(g, ("above", "below"), plane_z=520.0)
    return g


def candidate_copy_of_a() -> Graph:
    """C1: literal copy of A."""
    return graph_a()


def candidate_mediator() -> Graph:
    """C2: midway interface between A (z=500) and B (z=520)."""
    g = Graph()
    _two_units(g)
    _planar_contacts(g, ("above", "below"), plane_z=510.0)
    return g


def candidate_extension_with_extra_unit() -> Graph:
    """C3: introduce a thin middle unit between 'above' and 'below'.

    Two contact planes: above/middle at z=540, middle/below at z=480.
    Represents a D-driven extension that A and B did not have.
    """
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="above", unit_id="above", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="middle", unit_id="middle", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="below", unit_id="below", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_edge(GraphEdge(id="es_a", kind=EdgeKind.MEMBER_OF_SERIES, source="above", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="es_m", kind=EdgeKind.MEMBER_OF_SERIES, source="middle", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="es_b", kind=EdgeKind.MEMBER_OF_SERIES, source="below", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_am", kind=EdgeKind.OVERLIES, source="above", target="middle", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_mb", kind=EdgeKind.OVERLIES, source="middle", target="below", provenance=_prov()))
    # above/middle contacts at z=540
    for i, (x, y) in enumerate([(200.0, 200.0), (800.0, 200.0), (500.0, 800.0)]):
        g.add_node(Contact(
            id=f"c_am_{i}", position=(_pt(x), _pt(y), _pt(540.0)),
            between=("above", "middle"), p_exists=1.0, provenance=_prov(),
        ))
    # middle/below contacts at z=480
    for i, (x, y) in enumerate([(200.0, 200.0), (800.0, 200.0), (500.0, 800.0)]):
        g.add_node(Contact(
            id=f"c_mb_{i}", position=(_pt(x), _pt(y), _pt(480.0)),
            between=("middle", "below"), p_exists=1.0, provenance=_prov(),
        ))
    g.add_node(Orientation(
        id="o_am",
        position=(_pt(500.0), _pt(500.0), _pt(540.0)),
        dip=OrientationUncertainty(dip_mean=0.0, dip_kappa=1e4, azimuth_mean=0.0, azimuth_kappa=1.0),
        for_unit="above",
        p_exists=1.0,
        provenance=_prov(),
    ))
    return g


def candidate_drift() -> Graph:
    """C4: drift — interface dropped far below at z=200 (contradicts both A and B)."""
    g = Graph()
    _two_units(g)
    _planar_contacts(g, ("above", "below"), plane_z=200.0)
    return g


def candidate_swapped_polarity() -> Graph:
    """C5: same interface as A, but stratigraphy reversed (below over above).

    Tests physics gate: stratigraphic order should fail.
    """
    g = Graph()
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="above", unit_id="above", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="below", unit_id="below", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_edge(GraphEdge(id="e_series_above", kind=EdgeKind.MEMBER_OF_SERIES, source="above", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_series_below", kind=EdgeKind.MEMBER_OF_SERIES, source="below", target="s1", provenance=_prov()))
    # Reversed OVERLIES: 'below' on top of 'above', but place contact at z=500
    # so the realised geometry contradicts the order → stratigraphic_constrain rejects.
    g.add_edge(GraphEdge(id="e_overlies", kind=EdgeKind.OVERLIES, source="below", target="above", provenance=_prov()))
    _planar_contacts(g, ("above", "below"), plane_z=500.0)
    return g


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------


GRID = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1000.0, 1000.0, 1000.0), nx=12, ny=12, nz=12)


def realise_and_build(graph: Graph, seed: int = 0) -> VoxelField | None:
    """Realise + build a voxel field; return None on failure."""
    try:
        realised = graph.realise(np.random.default_rng(seed))
        return build_voxel_field(realised, GRID)
    except Exception as exc:
        print(f"  ! realise/build failed: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class Row:
    name: str
    description: str
    struct_dist_a: float
    struct_dist_b: float
    score: float
    structural: float
    fit: float
    coverage_a: float
    coverage_b: float
    gates: tuple[str, ...]
    flags: tuple[str, ...]
    extra: dict[str, float]


def format_row(row: Row) -> str:
    gates = ",".join(row.gates) if row.gates else "-"
    flags = ",".join(row.flags) if row.flags else "-"
    return (
        f"{row.name:32} | "
        f"d(C,A)={row.struct_dist_a:5.2f} d(C,B)={row.struct_dist_b:5.2f} | "
        f"score={row.score:8.2f}  L_Δ={row.structural:7.2f}  F={row.fit:7.2f} | "
        f"cov_a={row.coverage_a:.2f} cov_b={row.coverage_b:.2f} | "
        f"gates={gates} flags={flags}"
    )


def run() -> None:
    print("=" * 110)
    print("Self-supervised refinement criterion — dry run")
    print("=" * 110)

    cfg = RefinementCriterionConfig(
        epsilon=0.01,
        effective_sample_size=100.0,  # rough ESS; tweak to inspect sensitivity
        coverage_threshold=0.95,
        dedup_epsilon=1e-3,
    )
    print(f"Config: ε={cfg.epsilon}  α={cfg.effective_sample_size}  κ={cfg.effective_kappa_bits:.3f} bits  "
          f"θ_cov={cfg.coverage_threshold}  ε_struct={cfg.dedup_epsilon}")

    print("\n[Building A and B…]")
    A = graph_a()
    B = graph_b()
    A.validate()
    B.validate()
    print(f"  A: {len(list(A.nodes()))} nodes, {len(A.get_edges())} edges. Contact plane at z=500.")
    print(f"  B: {len(list(B.nodes()))} nodes, {len(B.get_edges())} edges. Contact plane at z=520.")
    print(f"  structural_distance(A, B) = {structural_distance(A, B, config=cfg):.3f}")

    field_a = realise_and_build(A, seed=11)
    field_b = realise_and_build(B, seed=22)
    assert field_a is not None and field_b is not None

    print(f"\n  field_a: unit_ids={field_a.unit_ids}  domain={int(field_a.domain_mask.sum())}/{field_a.domain_mask.size}")
    print(f"  field_b: unit_ids={field_b.unit_ids}  domain={int(field_b.domain_mask.sum())}/{field_b.domain_mask.size}")

    candidates: list[tuple[str, str, Graph]] = [
        ("C1_copy_of_A",        "literal copy of A",                         candidate_copy_of_a()),
        ("C2_mediator",         "horizontal interface at z=510 (mid A,B)",   candidate_mediator()),
        ("C3_extension_unit",   "adds middle unit, two contact planes",      candidate_extension_with_extra_unit()),
        ("C4_drift",            "interface dropped to z=200, contradicts",   candidate_drift()),
        ("C5_swapped_polarity", "stratigraphic order reversed",              candidate_swapped_polarity()),
    ]

    rows: list[Row] = []
    print("\n[Scoring candidates…]")
    for name, desc, C in candidates:
        print(f"\n--- {name}: {desc} ---")
        try:
            C.validate()
        except Exception as exc:
            print(f"  schema validate failed: {exc}")
        d_a = structural_distance(C, A, config=cfg)
        d_b = structural_distance(C, B, config=cfg)
        print(f"  structural_distance(C, A) = {d_a:.3f}   structural_distance(C, B) = {d_b:.3f}")
        field_c = realise_and_build(C, seed=33)
        if field_c is None:
            rows.append(Row(name, desc, d_a, d_b, math.inf, math.inf, math.inf, 0.0, 0.0,
                            ("realise_failed",), (), {}))
            continue
        result = score_refinement(
            candidate_graph=C,
            candidate_field=field_c,
            reference_a_graph=A,
            reference_a_field=field_a,
            reference_b_graph=B,
            reference_b_field=field_b,
            config=cfg,
            pool=[A, B],
        )
        gates = tuple(f.name for f in result.gate_failures)
        rows.append(Row(
            name=name,
            description=desc,
            struct_dist_a=d_a,
            struct_dist_b=d_b,
            score=result.score_bits,
            structural=result.structural_bits,
            fit=result.fit_bits,
            coverage_a=result.diagnostics.get("coverage_a", 0.0),
            coverage_b=result.diagnostics.get("coverage_b", 0.0),
            gates=gates,
            flags=tuple(result.flags),
            extra={
                "reverse_kl_a": result.diagnostics.get("reverse_kl_a_bits", 0.0),
                "reverse_kl_b": result.diagnostics.get("reverse_kl_b_bits", 0.0),
                "added_bits": result.diagnostics.get("structural_added_bits", 0.0),
            },
        ))
        if result.gate_failures:
            for f in result.gate_failures:
                print(f"  gate FAIL  {f.name}: {f.details}")
        else:
            sc = result.structural
            print(f"  L_Δ={result.structural_bits:.2f} (n_added={sc.n_added} n_mod={sc.n_modified} "
                  f"n_del∩={sc.n_deleted_consensus} n_sm={sc.n_split_merge})  "
                  f"F_κ={result.fit_bits:.2f}  score={result.score_bits:.2f}")
            print(f"  reverse_KL(C‖A)={result.diagnostics['reverse_kl_a_bits']:.2f}  "
                  f"reverse_KL(C‖B)={result.diagnostics['reverse_kl_b_bits']:.2f}")

    # ------------------------------------------------------------------ table
    print("\n" + "=" * 110)
    print("Summary")
    print("=" * 110)
    for row in rows:
        print(format_row(row))

    print("\nLegend: d(C,X) = greedy structural distance (smaller = more similar)")
    print("        score = L_Δ + ½(L_κ(A|C) + L_κ(B|C))   (lower = better admission)")
    print("        gates listed only on failure — coverage_a/_b, dedup, schema_validity, stratigraphic_consistency, etc.")

    # ------------------------------------------------------------------ ranking commentary
    print("\nRanking (admitted only):")
    admitted = sorted([r for r in rows if not r.gates and math.isfinite(r.score)],
                      key=lambda r: r.score)
    for rank, row in enumerate(admitted, 1):
        print(f"  {rank}. {row.name:32}  score={row.score:8.2f}  L_Δ={row.structural:6.2f}  F={row.fit:6.2f}")

    rejected = [r for r in rows if r.gates]
    if rejected:
        print("\nRejected:")
        for row in rejected:
            print(f"  - {row.name:32}  gates={','.join(row.gates)}")


if __name__ == "__main__":
    run()
