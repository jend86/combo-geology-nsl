"""Tests for the kg-admission gate on FeatureHypothesisKazakhstanTask.

Regression spec: a mid-run audit (gen 0, run 20260524-rg26xw, ~T+1h after
the duplicate-handling cycle) found 30+ successful episodes with
``bic_delta`` between -40 000 and -76 000 yet ``experiments.jsonl`` frozen
at 3 rows. Root cause: ``voxel-features-mcp/voxel_features/scoring.py``
returns ``stage_completed="mae_bic_completed"`` on every successful
stage-2 path, but the kg gate was hardcoded to
``stage_completed == "stage_2_completed"``. The gate must accept both
strings so post-rewrite scoring keeps admitting.
"""

from __future__ import annotations

import pytest

from tasks.feature_hypothesis_kazakhstan import FeatureHypothesisKazakhstanTask


_GATE = FeatureHypothesisKazakhstanTask._should_persist_to_kg


class TestStageCompletedAllowlist:
    def test_legacy_stage_2_completed_admits(self) -> None:
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-1.0,
            stage_completed="stage_2_completed",
        )

    def test_current_mae_bic_completed_admits(self) -> None:
        # The post-rewrite scoring path: this assertion is the regression
        # pin — if it ever fails again, admissions silently stop.
        assert _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-76738.5,
            stage_completed="mae_bic_completed",
        )

    @pytest.mark.parametrize(
        "stage",
        ["", "stage_1_only", "aborted", "stage_2_partial", "unknown"],
    )
    def test_unrecognised_stage_blocks(self, stage: str) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=-100.0,
            stage_completed=stage,
        )


class TestOtherGateConditions:
    def test_masking_test_failure_blocks(self) -> None:
        assert not _GATE(
            masking_test_passed=False,
            admitted=True,
            bic_delta=-100.0,
            stage_completed="mae_bic_completed",
        )

    def test_not_admitted_blocks(self) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=False,
            bic_delta=-100.0,
            stage_completed="mae_bic_completed",
        )

    def test_none_bic_blocks(self) -> None:
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=None,
            stage_completed="mae_bic_completed",
        )

    @pytest.mark.parametrize("bic", [0.0, 0.001, 100.0])
    def test_non_negative_bic_blocks(self, bic: float) -> None:
        # bic_delta must be strictly negative (lower BIC = better).
        assert not _GATE(
            masking_test_passed=True,
            admitted=True,
            bic_delta=bic,
            stage_completed="mae_bic_completed",
        )
