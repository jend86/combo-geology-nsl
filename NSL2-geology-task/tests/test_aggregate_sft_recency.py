"""Unit tests for the recency-weighted SFT composer (Approach C).

These exercise the *pure* ``compose()`` function with synthetic candidate
dicts, so they need none of the heavy transform / task-loading machinery — the
script keeps all non-stdlib imports lazy inside its I/O helpers.

Invariants under test:
  - the modern/legacy era budget split is honoured (newest-first tilt);
  - per-run caps bound any single run (so a verbose old mega-run can't dominate);
  - selection is newest-first across runs;
  - exact prompt/response duplicates are dropped, keeping the *newest* run's copy;
  - the within-run cap is stratified across (source, kind) so capping a run does
    not silently delete a whole row family.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_sft_recency.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("aggregate_sft_recency", _SCRIPT)
    assert spec and spec.loader, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # so @dataclass can resolve the module
    spec.loader.exec_module(module)  # must not import torch/numpy/task at module top
    return module


mod = _load_module()


def _cand(run_id, recency, is_modern, source, kind, pair_hash):
    return {
        "run_id": run_id,
        "recency": recency,
        "is_modern": is_modern,
        "source": source,
        "kind": kind,
        "pair_hash": pair_hash,
        "row": {"row_id": pair_hash, "prompt": pair_hash, "raw_response": "r", "success": True},
    }


def _pool(run_id, recency, is_modern, source, kind, n, *, start=0):
    return [
        _cand(run_id, recency, is_modern, source, kind, f"{run_id}-{kind}-{i}")
        for i in range(start, start + n)
    ]


def test_modern_fraction_is_honoured_when_supply_is_ample():
    cands = (
        _pool("new", "2026-06-06T00:00", True, "asis", "code", 1000)
        + _pool("old", "2026-05-01T00:00", False, "synth", "analysis_plan", 1000)
    )
    out = mod.compose(
        cands,
        target_rows=100,
        modern_fraction=0.8,
        modern_per_run_cap=1000,
        legacy_per_run_cap=1000,
    )
    rep = out["report"]
    assert rep["total"] == 100
    assert rep["modern_count"] == 80
    assert rep["legacy_count"] == 20


def test_legacy_per_run_cap_bounds_a_mega_run():
    cands = (
        _pool("new", "2026-06-06T00:00", True, "asis", "code", 200)
        + _pool("mega_old", "2026-05-24T00:00", False, "synth", "parent_relation", 1000)
    )
    out = mod.compose(
        cands,
        target_rows=300,
        modern_fraction=0.5,  # legacy budget = 150
        modern_per_run_cap=1000,
        legacy_per_run_cap=30,
    )
    rep = out["report"]
    # the mega old run must be capped hard regardless of the 150-row legacy budget
    assert rep["per_run"]["mega_old"] == 30
    assert rep["legacy_count"] == 30


def test_selection_is_newest_first_across_runs():
    cands = (
        _pool("r1_old", "2026-06-01T00:00", True, "asis", "code", 100)
        + _pool("r2_mid", "2026-06-03T00:00", True, "asis", "code", 100)
        + _pool("r3_new", "2026-06-05T00:00", True, "asis", "code", 100)
    )
    out = mod.compose(
        cands,
        target_rows=150,
        modern_fraction=1.0,  # all-modern budget = 150
        modern_per_run_cap=100,
        legacy_per_run_cap=0,
    )
    rep = out["report"]
    assert rep["per_run"]["r3_new"] == 100  # newest fills first
    assert rep["per_run"]["r2_mid"] == 50
    assert rep["per_run"].get("r1_old", 0) == 0


def test_exact_duplicate_is_kept_from_newest_run():
    shared = "SHARED-PAIR"
    cands = [
        _cand("old", "2026-05-01T00:00", False, "synth", "analysis_plan", shared),
        _cand("new", "2026-06-06T00:00", True, "synth", "analysis_plan", shared),
    ]
    out = mod.compose(
        cands,
        target_rows=10,
        modern_fraction=0.5,
        modern_per_run_cap=10,
        legacy_per_run_cap=10,
    )
    rep = out["report"]
    assert rep["total"] == 1  # de-duplicated to a single row
    assert rep["per_run"].get("new") == 1
    assert "old" not in rep["per_run"]  # newest copy wins


def test_within_run_cap_is_stratified_across_kinds():
    cands = (
        _pool("run", "2026-06-06T00:00", True, "asis", "code", 10)
        + _pool("run", "2026-06-06T00:00", True, "synth", "analysis_plan", 10)
    )
    out = mod.compose(
        cands,
        target_rows=4,
        modern_fraction=1.0,
        modern_per_run_cap=4,
        legacy_per_run_cap=0,
    )
    rep = out["report"]
    # round-robin: a cap of 4 over two families must take 2 of each, not 4 of one
    assert rep["per_kind"]["code"] == 2
    assert rep["per_kind"]["analysis_plan"] == 2


def test_pinned_modern_synth_is_never_dropped_by_the_cap():
    # a modern run with 600 pinned synth rows + 600 as-is rows, capped at 100.
    # the cap must NOT touch the pinned synth — they are the guaranteed base.
    synth = _pool("new", "2026-06-06T00:00", True, "synth", "analysis_plan", 600)
    for c in synth:
        c["pin"] = True
    asis = _pool("new", "2026-06-06T00:00", True, "asis", "code", 600)
    out = mod.compose(
        synth + asis,
        target_rows=700,
        modern_fraction=1.0,
        modern_per_run_cap=100,
        legacy_per_run_cap=0,
    )
    rep = out["report"]
    assert rep["per_kind"]["analysis_plan"] == 600  # all pinned synth survive
    assert rep["per_kind"]["code"] == 100  # as-is fills the remainder under the cap
    assert rep["total"] == 700


def test_legacy_never_exceeds_its_budget_even_if_modern_is_short():
    # only 10 modern rows exist but modern_budget would be 80 → total shrinks,
    # legacy is still bounded by its 20-row budget (no dilution past the tilt).
    cands = (
        _pool("new", "2026-06-06T00:00", True, "asis", "code", 10)
        + _pool("old", "2026-05-01T00:00", False, "synth", "analysis_plan", 1000)
    )
    out = mod.compose(
        cands,
        target_rows=100,
        modern_fraction=0.8,
        modern_per_run_cap=1000,
        legacy_per_run_cap=1000,
    )
    rep = out["report"]
    assert rep["modern_count"] == 10
    assert rep["legacy_count"] <= 20  # legacy capped at its budget; total may be < target
