"""Dry-run v2 — does the criterion's selection pressure align with truthiness?

Setup
-----
- A hidden ground-truth graph G* the criterion never sees.
- A and B are *partial* views of G* with different one-source artefacts.
- A candidate factory produces labelled C variants ranging from
  truthy_extension (recovers G*) to drift / lazy copies / hallucinations.

Three experiments
-----------------
1. Discriminative power: median score gap between candidate classes across
   seed-varied draws.
2. Pool simulation: 30 admission rounds, mostly-bad candidate mix, reservoir
   retention + annealed threshold; track admission rate per class and pool
   fraction near G*.
3. Calibration sweep: vary α and edit-cost scales; report the regime where
   truthy beats hallucinated by the largest margin.

The criterion sees only (A, B, C). G* is used here exclusively to *label*
candidates and to measure pool quality after the fact — never as an input to
the criterion.
"""

from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from graph_to_voxel.engine import GridSpec, build_voxel_field
from graph_to_voxel.engine.voxel_field import VoxelField
from graph_to_voxel.graph.core import Graph
from graph_to_voxel.refinement import (
    RefinementCriterionConfig,
    StructuralCostConfig,
    annealed_threshold,
    reservoir_retain,
    score_refinement,
    structural_distance,
)
from graph_to_voxel.schema.edges import EdgeKind, GraphEdge
from graph_to_voxel.schema.nodes import Contact, Orientation, Series, StratigraphicUnit
from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import OrientationUncertainty, PointUncertainty

warnings.filterwarnings("ignore")  # quiet engine warnings; not under test here


GRID = GridSpec(origin=(0.0, 0.0, 0.0), maximum=(1000.0, 1000.0, 1000.0), nx=10, ny=10, nz=10)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="dry-run-v2", confidence=1.0, timestamp=datetime(2026, 5, 11, tzinfo=timezone.utc))


def _pt(value: float) -> PointUncertainty:
    return PointUncertainty(value=value)


def _series(g: Graph) -> None:
    g.add_node(Series(id="s1", p_exists=1.0, provenance=_prov()))


def _two_unit_layered(g: Graph, plane_z: float) -> None:
    """Two-unit layered model with a horizontal interface at plane_z."""
    _series(g)
    g.add_node(StratigraphicUnit(id="above", unit_id="above", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_node(StratigraphicUnit(id="below", unit_id="below", series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
    g.add_edge(GraphEdge(id="es_a", kind=EdgeKind.MEMBER_OF_SERIES, source="above", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="es_b", kind=EdgeKind.MEMBER_OF_SERIES, source="below", target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_ov", kind=EdgeKind.OVERLIES, source="above", target="below", provenance=_prov()))
    for i, (x, y) in enumerate([(200, 200), (800, 200), (500, 800), (200, 800), (800, 800)]):
        g.add_node(Contact(
            id=f"c{i}", position=(_pt(float(x)), _pt(float(y)), _pt(plane_z)),
            between=("above", "below"), p_exists=1.0, provenance=_prov(),
        ))
    g.add_node(Orientation(
        id="o0", position=(_pt(500.0), _pt(500.0), _pt(plane_z)),
        dip=OrientationUncertainty(dip_mean=0.0, dip_kappa=1e4, azimuth_mean=0.0, azimuth_kappa=1.0),
        for_unit="above", p_exists=1.0, provenance=_prov(),
    ))


def _three_unit_layered(g: Graph, top_z: float, bot_z: float) -> None:
    """Three-unit layered model. above/middle at top_z, middle/below at bot_z."""
    _series(g)
    for uid in ("above", "middle", "below"):
        g.add_node(StratigraphicUnit(id=uid, unit_id=uid, series_id="s1", topology="layer", p_exists=1.0, provenance=_prov()))
        g.add_edge(GraphEdge(id=f"es_{uid}", kind=EdgeKind.MEMBER_OF_SERIES, source=uid, target="s1", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_am", kind=EdgeKind.OVERLIES, source="above", target="middle", provenance=_prov()))
    g.add_edge(GraphEdge(id="e_mb", kind=EdgeKind.OVERLIES, source="middle", target="below", provenance=_prov()))
    for i, (x, y) in enumerate([(200, 200), (800, 200), (500, 800)]):
        g.add_node(Contact(
            id=f"c_am_{i}", position=(_pt(float(x)), _pt(float(y)), _pt(top_z)),
            between=("above", "middle"), p_exists=1.0, provenance=_prov(),
        ))
        g.add_node(Contact(
            id=f"c_mb_{i}", position=(_pt(float(x)), _pt(float(y)), _pt(bot_z)),
            between=("middle", "below"), p_exists=1.0, provenance=_prov(),
        ))
    g.add_node(Orientation(
        id="o_am", position=(_pt(500.0), _pt(500.0), _pt(top_z)),
        dip=OrientationUncertainty(dip_mean=0.0, dip_kappa=1e4, azimuth_mean=0.0, azimuth_kappa=1.0),
        for_unit="above", p_exists=1.0, provenance=_prov(),
    ))


# ---------------------------------------------------------------------------
# Ground truth, A, B
# ---------------------------------------------------------------------------


def ground_truth() -> Graph:
    """G*: 3-unit layered truth. above/middle @ z=540, middle/below @ z=480."""
    g = Graph()
    _three_unit_layered(g, top_z=540.0, bot_z=480.0)
    return g


def reference_a() -> Graph:
    """A: 2-unit approximation, interface at z=500, plus a one-source extra contact."""
    g = Graph()
    _two_unit_layered(g, plane_z=500.0)
    # one-source artefact: an extra contact way off centre
    g.add_node(Contact(
        id="c_artefact_a", position=(_pt(900.0), _pt(900.0), _pt(450.0)),
        between=("above", "below"), p_exists=0.3, provenance=_prov(),
    ))
    return g


def reference_b() -> Graph:
    """B: 2-unit approximation, interface at z=520, plus a different one-source artefact."""
    g = Graph()
    _two_unit_layered(g, plane_z=520.0)
    # one-source artefact: a stray orientation in a different location
    g.add_node(Orientation(
        id="o_artefact_b", position=(_pt(100.0), _pt(100.0), _pt(550.0)),
        dip=OrientationUncertainty(dip_mean=5.0, dip_kappa=10.0, azimuth_mean=45.0, azimuth_kappa=1.0),
        for_unit="above", p_exists=0.3, provenance=_prov(),
    ))
    return g


# ---------------------------------------------------------------------------
# Candidate factory — labelled
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    label: str
    graph: Graph
    descriptor: str


def make_truthy_extension(rng: np.random.Generator) -> Candidate:
    """3-unit decomposition that matches G* up to small jitter."""
    g = Graph()
    _three_unit_layered(g, top_z=540.0 + rng.normal(0.0, 3.0), bot_z=480.0 + rng.normal(0.0, 3.0))
    return Candidate("truthy_extension", g, "3-unit decomposition near G*")


def make_truthy_compression_a(_rng: np.random.Generator) -> Candidate:
    """Drops A's one-source artefact; matches B's 2-unit interpretation."""
    g = Graph()
    _two_unit_layered(g, plane_z=510.0)  # mediates between A=500 and B=520
    return Candidate("truthy_compression", g, "drops A's artefact, mediates A/B")


def make_consensus_mediator(_rng: np.random.Generator) -> Candidate:
    """Same as truthy_compression in this setup — interface at z=510, no artefacts."""
    g = Graph()
    _two_unit_layered(g, plane_z=510.0)
    return Candidate("consensus_mediator", g, "2-unit mediator, no artefacts")


def make_hallucinated_extension(rng: np.random.Generator) -> Candidate:
    """Adds a unit that is NOT in G* — wrong middle, far from the truth."""
    g = Graph()
    # middle squeezed between z=300 and z=250 — nowhere near G*'s 540/480
    _three_unit_layered(g, top_z=300.0 + rng.normal(0.0, 5.0), bot_z=250.0 + rng.normal(0.0, 5.0))
    return Candidate("hallucinated_extension", g, "3-unit decomposition far from G*")


def make_lazy_copy_a(_rng: np.random.Generator) -> Candidate:
    """Tiny perturbation of A — keeps A's artefact, ignores B entirely."""
    g = Graph()
    _two_unit_layered(g, plane_z=500.0)  # same plane as A
    g.add_node(Contact(
        id="c_artefact_a", position=(_pt(900.0), _pt(900.0), _pt(450.0)),
        between=("above", "below"), p_exists=0.3, provenance=_prov(),
    ))
    return Candidate("lazy_copy_a", g, "near-copy of A including artefact")


def make_drift(rng: np.random.Generator) -> Candidate:
    """Interface dropped to z≈200 — contradicts both A and B."""
    g = Graph()
    _two_unit_layered(g, plane_z=200.0 + rng.normal(0.0, 20.0))
    return Candidate("drift", g, "interface far from A/B")


CANDIDATE_MAKERS: dict[str, Callable[[np.random.Generator], Candidate]] = {
    "truthy_extension": make_truthy_extension,
    "truthy_compression": make_truthy_compression_a,
    "consensus_mediator": make_consensus_mediator,
    "hallucinated_extension": make_hallucinated_extension,
    "lazy_copy_a": make_lazy_copy_a,
    "drift": make_drift,
}


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def realise(g: Graph, seed: int) -> VoxelField | None:
    try:
        return build_voxel_field(g.realise(np.random.default_rng(seed)), GRID)
    except Exception:
        return None


def score(C: Candidate, A: Graph, fa: VoxelField, B: Graph, fb: VoxelField,
          cfg: RefinementCriterionConfig, pool: list[Graph], seed: int) -> tuple[float, str, dict]:
    fc = realise(C.graph, seed)
    if fc is None:
        return math.inf, "realise_failed", {}
    r = score_refinement(
        candidate_graph=C.graph, candidate_field=fc,
        reference_a_graph=A, reference_a_field=fa,
        reference_b_graph=B, reference_b_field=fb,
        config=cfg, pool=pool,
    )
    gates = ",".join(f.name for f in r.gate_failures) or "ok"
    diag = {
        "structural": r.structural_bits,
        "fit": r.fit_bits,
        "rev_kl_a": r.diagnostics.get("reverse_kl_a_bits", 0.0),
        "rev_kl_b": r.diagnostics.get("reverse_kl_b_bits", 0.0),
    }
    return r.score_bits, gates, diag


def truth_distance(g: Graph, gstar: Graph, cfg: RefinementCriterionConfig) -> float:
    """How far is `g` from ground truth, in our greedy structural-distance units."""
    return structural_distance(g, gstar, config=cfg)


# ---------------------------------------------------------------------------
# Experiment 1 — discriminative power
# ---------------------------------------------------------------------------


def experiment_discrimination(
    A: Graph, fa: VoxelField, B: Graph, fb: VoxelField,
    cfg: RefinementCriterionConfig, *, n_trials: int = 8,
) -> dict[str, list[float]]:
    """Median score per candidate class across n_trials seeds."""
    scores: dict[str, list[float]] = {label: [] for label in CANDIDATE_MAKERS}
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        for label, maker in CANDIDATE_MAKERS.items():
            C = maker(rng)
            s, _gates, _diag = score(C, A, fa, B, fb, cfg, pool=[], seed=2000 + trial)
            scores[label].append(s)
    return scores


def summarise_class(scores: list[float]) -> tuple[float, float, float, int]:
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return math.inf, math.inf, math.inf, 0
    return (statistics.median(finite), min(finite), max(finite), len(finite))


# ---------------------------------------------------------------------------
# Experiment 2 — pool simulation
# ---------------------------------------------------------------------------


@dataclass
class PoolEntry:
    graph: Graph
    label: str
    admission_round: int
    truth_distance: float


@dataclass
class RoundLog:
    round_idx: int
    candidate_label: str
    score: float
    threshold: float
    admitted: bool
    gates: str
    pool_size_after: int
    truthy_fraction_after: float


def experiment_pool_simulation(
    A: Graph, B: Graph, gstar: Graph,
    cfg: RefinementCriterionConfig,
    *,
    n_rounds: int = 30,
    pool_capacity: int = 8,
    initial_threshold: float = math.inf,
    steady_state_threshold: float = 80.0,
    anneal_horizon: int = 12,
    candidate_mix: dict[str, float] | None = None,
    seed: int = 7,
) -> tuple[list[RoundLog], list[PoolEntry]]:
    """Run a small pool refinement loop and log per-round outcomes.

    Returns the per-round log and the final pool.

    candidate_mix: probability of each class being proposed each round; defaults
    to ~80% bad (drift + hallucination + lazy) and ~20% good (truthy_*).
    """
    if candidate_mix is None:
        candidate_mix = {
            "truthy_extension": 0.10,
            "truthy_compression": 0.10,
            "consensus_mediator": 0.05,
            "hallucinated_extension": 0.30,
            "lazy_copy_a": 0.20,
            "drift": 0.25,
        }
    labels = list(candidate_mix)
    probs = np.array([candidate_mix[label] for label in labels])
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)

    # Realise A and B once.
    fa = realise(A, 11)
    fb = realise(B, 22)
    assert fa is not None and fb is not None

    pool: list[PoolEntry] = [
        PoolEntry(A, "ref_A", 0, truth_distance(A, gstar, cfg)),
        PoolEntry(B, "ref_B", 0, truth_distance(B, gstar, cfg)),
    ]
    fields: dict[int, VoxelField] = {0: fa, 1: fb}
    log: list[RoundLog] = []

    admissions = 0
    for r in range(1, n_rounds + 1):
        # Sample pair from current pool
        pair_idx = rng.choice(len(pool), size=2, replace=False)
        a_entry, b_entry = pool[pair_idx[0]], pool[pair_idx[1]]
        # Use cached fields if present, otherwise compute on the fly
        pair_a_field = fields.get(pair_idx[0]) or realise(a_entry.graph, 100 + int(pair_idx[0]))
        pair_b_field = fields.get(pair_idx[1]) or realise(b_entry.graph, 200 + int(pair_idx[1]))
        if pair_a_field is None or pair_b_field is None:
            continue

        # Propose a candidate
        label = labels[rng.choice(len(labels), p=probs)]
        cand_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        candidate = CANDIDATE_MAKERS[label](cand_rng)
        cand_truth_dist = truth_distance(candidate.graph, gstar, cfg)

        # Score
        s, gates, _diag = score(
            candidate, a_entry.graph, pair_a_field,
            b_entry.graph, pair_b_field,
            cfg, pool=[entry.graph for entry in pool],
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        threshold = annealed_threshold(
            admissions,
            initial_threshold=initial_threshold,
            steady_state_threshold=steady_state_threshold,
            anneal_horizon=anneal_horizon,
        )
        admit = gates == "ok" and s <= threshold

        if admit:
            new_entry = PoolEntry(candidate.graph, label, r, cand_truth_dist)
            pool.append(new_entry)
            fields[len(pool) - 1] = realise(candidate.graph, int(rng.integers(0, 2**31 - 1)))
            admissions += 1
            if len(pool) > pool_capacity:
                # Reservoir prune (exempt the initial two)
                locked = {0, 1}
                keep = reservoir_retain(pool, pool_capacity, rng=rng, locked_indices=locked)
                pool = list(keep)
                # Drop cached fields for entries that no longer exist
                fields = {idx: realise(entry.graph, 9999 + idx) for idx, entry in enumerate(pool)}

        truthy_frac = sum(1 for entry in pool if entry.truth_distance < 0.5) / len(pool)
        log.append(RoundLog(
            round_idx=r,
            candidate_label=label,
            score=s,
            threshold=threshold,
            admitted=admit,
            gates=gates,
            pool_size_after=len(pool),
            truthy_fraction_after=truthy_frac,
        ))

    return log, pool


# ---------------------------------------------------------------------------
# Experiment 3 — calibration sweep
# ---------------------------------------------------------------------------


def experiment_calibration_sweep(
    A: Graph, fa: VoxelField, B: Graph, fb: VoxelField,
    *, n_trials: int = 4,
) -> list[dict]:
    """Vary α, parameter_scale, modified_distance_bits; report rank quality."""
    results = []
    grid = []
    for alpha in [10.0, 100.0, 1000.0]:
        for parameter_scale in [1.0, 50.0, 200.0]:
            for mod_bits in [1.0, 0.1]:
                grid.append((alpha, parameter_scale, mod_bits))
    for alpha, ps, mb in grid:
        sc = StructuralCostConfig(parameter_scale=ps, modified_distance_bits=mb)
        cfg = RefinementCriterionConfig(
            epsilon=0.01,
            effective_sample_size=alpha,
            structural=sc,
        )
        class_scores = experiment_discrimination(A, fa, B, fb, cfg, n_trials=n_trials)
        medians = {label: summarise_class(scores)[0] for label, scores in class_scores.items()}
        truthy_med = min(medians["truthy_extension"], medians["truthy_compression"])
        hallucinated_med = medians["hallucinated_extension"]
        drift_med = medians["drift"]
        gap_vs_hallucination = hallucinated_med - truthy_med
        gap_vs_drift = drift_med - truthy_med
        results.append({
            "alpha": alpha,
            "parameter_scale": ps,
            "modified_distance_bits": mb,
            "truthy_median": truthy_med,
            "hallucinated_median": hallucinated_med,
            "drift_median": drift_med,
            "gap_vs_hallucination": gap_vs_hallucination,
            "gap_vs_drift": gap_vs_drift,
            "all_medians": medians,
        })
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 110)
    print("Dry-run v2 — does the criterion's pressure align with truthiness?")
    print("=" * 110)

    gstar = ground_truth()
    A = reference_a()
    B = reference_b()
    cfg_default = RefinementCriterionConfig(
        epsilon=0.01,
        effective_sample_size=100.0,
        structural=StructuralCostConfig(parameter_scale=50.0, modified_distance_bits=1.0),
    )

    print(f"\n  truth_distance(A, G*) = {truth_distance(A, gstar, cfg_default):.3f}")
    print(f"  truth_distance(B, G*) = {truth_distance(B, gstar, cfg_default):.3f}")
    print(f"  structural_distance(A, B) = {structural_distance(A, B, config=cfg_default):.3f}")

    fa = realise(A, 11)
    fb = realise(B, 22)
    assert fa is not None and fb is not None

    # ------------------------------------------------------------------ exp1
    print("\n" + "-" * 110)
    print("Experiment 1 — discriminative power across 8 seed-varied trials")
    print("-" * 110)
    class_scores = experiment_discrimination(A, fa, B, fb, cfg_default, n_trials=8)
    print(f"{'class':30} | {'median':>10} {'min':>10} {'max':>10} {'truth_dist':>12}")
    rng = np.random.default_rng(0)
    for label, scores in class_scores.items():
        med, lo, hi, _ = summarise_class(scores)
        # one representative truth-distance
        cand = CANDIDATE_MAKERS[label](rng)
        td = truth_distance(cand.graph, gstar, cfg_default)
        print(f"{label:30} | {med:>10.2f} {lo:>10.2f} {hi:>10.2f} {td:>12.3f}")

    truthy_med = min(
        statistics.median([s for s in class_scores["truthy_extension"] if math.isfinite(s)]),
        statistics.median([s for s in class_scores["truthy_compression"] if math.isfinite(s)]),
    )
    hallucinated_med = statistics.median([s for s in class_scores["hallucinated_extension"] if math.isfinite(s)])
    drift_med = statistics.median([s for s in class_scores["drift"] if math.isfinite(s)])
    print(f"\n  truthy median               = {truthy_med:8.2f}")
    print(f"  hallucinated median         = {hallucinated_med:8.2f}  (gap vs truthy: {hallucinated_med - truthy_med:+.2f})")
    print(f"  drift median                = {drift_med:8.2f}  (gap vs truthy: {drift_med - truthy_med:+.2f})")
    discriminative = (hallucinated_med - truthy_med) > 0 and (drift_med - truthy_med) > 0
    print(f"  discriminative (gaps > 0)? {'YES' if discriminative else 'NO'}")

    # ------------------------------------------------------------------ exp2
    print("\n" + "-" * 110)
    print("Experiment 2 — pool simulation (30 rounds, capacity 8, ~80% bad mix)")
    print("-" * 110)
    log, final_pool = experiment_pool_simulation(
        A, B, gstar, cfg_default,
        n_rounds=30, pool_capacity=8,
        steady_state_threshold=80.0, anneal_horizon=12,
        seed=42,
    )
    admitted = sum(1 for entry in log if entry.admitted)
    print(f"  total rounds: {len(log)}  admissions: {admitted}  rejection rate: {1 - admitted/len(log):.2%}")
    class_attempts: dict[str, int] = {}
    class_admits: dict[str, int] = {}
    for entry in log:
        class_attempts[entry.candidate_label] = class_attempts.get(entry.candidate_label, 0) + 1
        if entry.admitted:
            class_admits[entry.candidate_label] = class_admits.get(entry.candidate_label, 0) + 1
    print(f"\n  {'class':30} {'proposed':>10} {'admitted':>10} {'admit_rate':>12}")
    for label in CANDIDATE_MAKERS:
        att = class_attempts.get(label, 0)
        adm = class_admits.get(label, 0)
        rate = (adm / att) if att else 0.0
        print(f"  {label:30} {att:>10d} {adm:>10d} {rate:>11.2%}")

    print(f"\n  final pool ({len(final_pool)} graphs):")
    for entry in final_pool:
        marker = "  truthy " if entry.truth_distance < 0.5 else "        "
        print(f"   {marker}{entry.label:24}  truth_dist={entry.truth_distance:.3f}  (admitted round {entry.admission_round})")
    truthy_in_final = sum(1 for entry in final_pool if entry.truth_distance < 0.5) / len(final_pool)
    print(f"\n  truthy fraction in final pool: {truthy_in_final:.2%}")

    # Trajectory of truthy fraction over rounds
    print(f"\n  truthy_fraction trajectory (every 3 rounds):")
    for entry in log[::3]:
        print(f"    round {entry.round_idx:>2}: pool_size={entry.pool_size_after}  truthy_frac={entry.truthy_fraction_after:.2%}  threshold={entry.threshold:8.2f}")

    # ------------------------------------------------------------------ exp3
    print("\n" + "-" * 110)
    print("Experiment 3 — calibration sweep (α × parameter_scale × modified_distance_bits)")
    print("-" * 110)
    sweep = experiment_calibration_sweep(A, fa, B, fb, n_trials=4)
    print(f"{'α':>6} {'param_scale':>12} {'mod_bits':>10} | "
          f"{'truthy':>10} {'halluc':>10} {'drift':>10} | "
          f"{'gap_halluc':>12} {'gap_drift':>12}")
    for r in sweep:
        print(f"{r['alpha']:>6.1f} {r['parameter_scale']:>12.1f} {r['modified_distance_bits']:>10.2f} | "
              f"{r['truthy_median']:>10.2f} {r['hallucinated_median']:>10.2f} {r['drift_median']:>10.2f} | "
              f"{r['gap_vs_hallucination']:>+12.2f} {r['gap_vs_drift']:>+12.2f}")

    # Find best (max sum of gaps where both positive)
    viable = [r for r in sweep if r["gap_vs_hallucination"] > 0 and r["gap_vs_drift"] > 0]
    if viable:
        best = max(viable, key=lambda r: r["gap_vs_hallucination"] + r["gap_vs_drift"])
        print("\n  best regime (both gaps positive, max sum):")
        print(f"    α={best['alpha']}  parameter_scale={best['parameter_scale']}  "
              f"modified_distance_bits={best['modified_distance_bits']}  "
              f"gap_halluc=+{best['gap_vs_hallucination']:.2f}  gap_drift=+{best['gap_vs_drift']:.2f}")
    else:
        print("\n  no regime where both gaps positive — criterion cannot discriminate in this sweep.")


if __name__ == "__main__":
    main()
