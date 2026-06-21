"""TDD for scripts/downsample_au_sft.py — the global family-balancing downsampler.

The downsampler trims an UNCAPPED, code-free pooled SFT set down to a target row
count, preferentially removing rows from the most over-represented hypothesis
*families* (a coarse lexical key over ``record_meta.hypothesis``) so the surviving
set has greater hypothesis variety. ``dataset_hypothesis`` rows are protected from
removal (mirrors the task's canonical curation, which always keeps them).

Loaded via importlib so we don't have to package scripts/.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "downsample_au_sft",
    Path(__file__).resolve().parents[1] / "scripts" / "downsample_au_sft.py",
)
ds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ds)  # type: ignore[union-attr]


def _row(fam_words: str, task_kind: str, i: int) -> dict:
    return {
        "prompt": f"p{i}",
        "raw_response": f"r{i}",
        "success": True,
        "source_run_id": "runA",
        "record_meta": {"hypothesis": fam_words, "task_kind": task_kind, "pair_kind": task_kind},
        "row_id": f"row-{i}",
    }


# ---- family key ----------------------------------------------------------

def test_family_head_first_seven_content_words():
    # stop words (the, of, in, with) dropped; first 7 of the remainder kept
    h = "High grade gold mineralization in the Coe Fairbairn region with deep structural control"
    assert ds.family_head(h) == "high grade gold mineralization coe fairbairn region"


def test_family_head_empty_is_other():
    assert ds.family_head("") == "other"
    assert ds.family_head("the of in with") == "other"


# ---- downsample ----------------------------------------------------------

def test_noop_when_at_or_below_target():
    rows = [_row("alpha beta", "feature_readout", i) for i in range(10)]
    out = ds.downsample(rows, target=10)
    assert len(out) == 10
    out2 = ds.downsample(rows, target=25)
    assert len(out2) == 10


def test_downsample_hits_exact_target():
    rows = (
        [_row("big family one", "feature_readout", i) for i in range(60)]
        + [_row("small family two", "feature_readout", 100 + i) for i in range(5)]
        + [_row("tiny family three", "feature_readout", 200 + i) for i in range(5)]
    )
    out = ds.downsample(rows, target=40)
    assert len(out) == 40


def test_downsample_trims_largest_family_first():
    rows = (
        [_row("dominant gold hypothesis", "feature_readout", i) for i in range(50)]
        + [_row("rare copper hypothesis", "feature_readout", 100 + i) for i in range(8)]
    )
    out = ds.downsample(rows, target=30)
    fams = [ds.family_head(r["record_meta"]["hypothesis"]) for r in out]
    # the rare family (8) must survive intact; trimming comes from the dominant one
    assert fams.count("rare copper hypothesis") == 8
    assert fams.count("dominant gold hypothesis") == 22


def test_downsample_protects_dataset_hypothesis_rows():
    rows = (
        [_row("dominant gold hypothesis", "feature_readout", i) for i in range(40)]
        + [_row("dominant gold hypothesis", "dataset_hypothesis", 500 + i) for i in range(10)]
    )
    out = ds.downsample(rows, target=20)
    kinds = [r["record_meta"]["task_kind"] for r in out]
    # all 10 protected dataset_hypothesis rows survive; trimming hits the others
    assert kinds.count("dataset_hypothesis") == 10
    assert len(out) == 20


def test_drop_blank_removes_empty_prompt_or_completion():
    rows = [
        _row("alpha beta", "feature_readout", 0),
        {**_row("alpha beta", "outcome_narrative", 1), "raw_response": ""},   # blank completion
        {**_row("alpha beta", "outcome_narrative", 2), "raw_response": "   "},  # whitespace only
        {**_row("alpha beta", "analysis_plan", 3), "prompt": ""},              # blank prompt
        _row("gamma delta", "feature_readout", 4),
    ]
    clean = ds.drop_blank(rows)
    assert [r["row_id"] for r in clean] == ["row-0", "row-4"]


def test_downsample_is_deterministic():
    rows = [_row(f"fam {i % 5}", "feature_readout", i) for i in range(100)]
    a = ds.downsample(rows, target=37)
    b = ds.downsample(rows, target=37)
    assert [r["row_id"] for r in a] == [r["row_id"] for r in b]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
