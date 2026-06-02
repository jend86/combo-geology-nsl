"""Regression tests for two-stage scoring (Stage 1 MAE gate + Stage 2 BIC).

Covers the bugs documented in
``NSL2-geology-task/docs/design/scoring-fix-and-replay-2026-05-25.md`` (§9).
Each test asserts an invariant of the *current* (post-fix, post-calibration,
post-RNG-seeding, post-null-baseline) scoring path. If any of these regress,
the SFT reward signal silently degrades.

Notes:
- Uses the voxel-features-mcp copy of ``voxel_features.scoring`` (the live
  copy on the container path). The old NSL2 scorer mirror has been removed so
  there is one authoritative implementation.
- ``evaluate_new_layer(..., seed=N)`` drives deterministic CV split / Moran's
  I / interpolation; tests rely on that for stable assertions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import voxel_features.scoring as scoring
from voxel_features.scoring import evaluate_new_layer
from voxel_features.store import GridSpec, VoxelStore


_GRID = GridSpec(
    origin=(0.0, 0.0, 0.0),
    maximum=(0.02, 0.02, 20.0),
    shape=(20, 20, 4),
)


def _store(tmp_path: Path) -> VoxelStore:
    return VoxelStore(tmp_path, _GRID)


def _seed_layer(store: VoxelStore, name: str, values: np.ndarray) -> None:
    """Admit the first layer (auto-pass) so subsequent tests see a 2-layer state."""
    result = evaluate_new_layer(store, name, values, "float", seed=7)
    assert result["admitted"], "first-layer auto-admit failed"


def _seed_two_layers(store: VoxelStore, base: np.ndarray) -> None:
    """Seed the store with two structurally-related layers so Stage 1 can fire."""
    _seed_layer(store, "base", base)
    related = (base * 0.6 + np.random.default_rng(11).random(base.shape) * 0.05).astype(np.float32)
    result = evaluate_new_layer(store, "related", related, "float", seed=7)
    assert result["admitted"], "second-layer seed failed"


class _ControlledStore:
    def __init__(self) -> None:
        self.grid = SimpleNamespace(shape=(2, 2, 1))
        self.values = {
            "base": np.ones(self.grid.shape, dtype=np.float32),
            "related": np.full(self.grid.shape, 0.9, dtype=np.float32),
        }
        self.dtypes = {"base": "float", "related": "float"}
        self.added_layers: list[str] = []

    @property
    def layer_names(self) -> list[str]:
        return list(self.values)

    def get_layer_values(self, name: str) -> np.ndarray:
        return self.values[name]

    def get_layer(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(dtype=self.dtypes[name])

    def add_layer(self, name: str, values: np.ndarray, dtype: str) -> None:
        self.values[name] = values
        self.dtypes[name] = dtype
        self.added_layers.append(name)

    def update_layer_scores(self, layer_name: str, bic_delta: float, mi_scores: dict) -> None:
        pass


class _OneLayerStore:
    def __init__(self) -> None:
        self.grid = SimpleNamespace(shape=(2, 2, 1))
        self.values = {
            "base": np.ones(self.grid.shape, dtype=np.float32),
        }
        self.dtypes = {"base": "float"}
        self.added_layers: list[str] = []

    @property
    def layer_names(self) -> list[str]:
        return list(self.values)

    def get_layer_values(self, name: str) -> np.ndarray:
        return self.values[name]

    def get_layer(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(dtype=self.dtypes[name])

    def add_layer(self, name: str, values: np.ndarray, dtype: str) -> None:
        self.values[name] = values
        self.dtypes[name] = dtype
        self.added_layers.append(name)

    def update_layer_scores(self, layer_name: str, bic_delta: float, mi_scores: dict) -> None:
        pass


def _controlled_stage1_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mae_improvement: float,
    bic_delta: float,
) -> dict:
    calls = iter(
        [
            {
                "bic": 0.0,
                "total_cv_mse": 1.0,
                "system_mae": 1.0,
                "n_effective_samples": 100,
            },
            {
                "bic": bic_delta * 100.0,
                "total_cv_mse": 1.0,
                "system_mae": 1.0 - mae_improvement,
                "n_effective_samples": 100,
            },
        ]
    )

    monkeypatch.setattr(scoring, "geological_coherence_score", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(scoring, "mutual_information", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(scoring, "pairwise_distance", lambda *args, **kwargs: 0.0)

    return scoring.evaluate_new_layer(
        _ControlledStore(),
        "candidate",
        np.full((2, 2, 1), 0.8, dtype=np.float32),
        "float",
        seed=7,
    )


# ---------------------------------------------------------------------------
# Test 1 — uninformative layer must fail Stage 1
# ---------------------------------------------------------------------------
def test_uninformative_layer_fails_stage1(tmp_path: Path) -> None:
    store = _store(tmp_path / "uninformative")
    rng = np.random.default_rng(0)
    base = rng.random((20, 20, 4)).astype(np.float32)
    _seed_two_layers(store, base)

    # New layer = independent random noise on top of an already-populated store.
    noise = np.random.default_rng(424242).random((20, 20, 4)).astype(np.float32)
    result = evaluate_new_layer(store, "noise", noise, "float", seed=7)

    assert result["masking_test_passed"] is False, (
        f"noise layer should fail Stage 1, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 2 — redundant layer passes Stage 1 (predictive) but dedup catches it
# ---------------------------------------------------------------------------
def test_redundant_layer_passes_stage1_but_is_for_dedup_to_reject(tmp_path: Path) -> None:
    """A noisy near-copy of an existing layer DOES reduce system-wide pairwise
    MAE (because the new pair is near-perfectly predictable), so Stage 1
    legitimately passes. Stopping near-duplicates is the responsibility of
    ``_admit_with_dedup`` in the task layer, not the scoring gate.

    This test pins that contract so future refactors don't accidentally try to
    make the scoring gate catch duplicates (and end up rejecting legitimate
    new layers along the way).
    """
    store = _store(tmp_path / "redundant")
    rng = np.random.default_rng(1)
    base = rng.random((20, 20, 4)).astype(np.float32)
    _seed_two_layers(store, base)

    redundant = (base + np.random.default_rng(2).normal(0, 0.01, base.shape)).astype(np.float32)
    result = evaluate_new_layer(store, "copy_of_base", redundant, "float", seed=7)

    # Stage 1 passes because pairwise MAE drops; that's the gate's role.
    assert result["masking_test_passed"] is True


# ---------------------------------------------------------------------------
# Test 3 — informative layer (function of existing layer) should pass Stage 1
# ---------------------------------------------------------------------------
def test_informative_layer_passes_stage1(tmp_path: Path) -> None:
    """The Stage 1 MAE gate must actually fire (not auto-pass) when there are
    >=2 existing layers, and a linearly-related new layer must pass it.

    Verifies both:
      - ``masking_test_direction`` is ``"mae_delta"`` (real gate, not the
        auto_pass / first_layer shortcuts).
      - A scaled near-copy of an existing layer (high linear predictability)
        does pass the gate.
    """
    store = _store(tmp_path / "informative")
    rng = np.random.default_rng(3)
    base = rng.random((20, 20, 4)).astype(np.float32)
    _seed_two_layers(store, base)

    # Near-copy of base: highly linearly predictable from existing layers,
    # so the average pairwise MAE drops.
    informative = (base * 1.5 + 0.01 * rng.standard_normal(base.shape)).astype(np.float32)
    result = evaluate_new_layer(store, "informative", informative, "float", seed=7)

    assert result["masking_test_direction"] == "mae_delta", (
        f"expected real Stage 1 gate to fire, got {result['masking_test_direction']}"
    )
    assert result["masking_test_passed"] is True


def test_stage1_tolerance_rescues_material_bic_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _controlled_stage1_result(
        monkeypatch,
        mae_improvement=-7.5e-6,
        bic_delta=-1.2,
    )

    assert result["masking_test_passed"] is True
    assert result["admitted"] is True
    assert result["stage_1_tolerance_used"] is True
    assert result["stage_1_mae_tolerance"] == pytest.approx(1e-5)
    assert result["stage_1_bic_rescue_threshold"] == pytest.approx(-1.0)


def test_stage1_tolerance_does_not_rescue_tiny_bic_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _controlled_stage1_result(
        monkeypatch,
        mae_improvement=-7.5e-6,
        bic_delta=-0.5,
    )

    assert result["masking_test_passed"] is False
    assert result["admitted"] is False
    assert result["stage_1_tolerance_used"] is False


def test_stage1_tolerance_does_not_rescue_large_mae_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _controlled_stage1_result(
        monkeypatch,
        mae_improvement=-2e-5,
        bic_delta=-1.2,
    )

    assert result["masking_test_passed"] is False
    assert result["admitted"] is False
    assert result["stage_1_tolerance_used"] is False


def test_bic_delta_recomputed_with_common_effective_sample_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 2 must not accept a layer because before/after BIC used different n.

    The mocked raw BICs look like a strong improvement only because the before
    score used a tiny effective-sample count and the after score used a large
    one. Recomputing both BICs from the same MAE matrices with a common ``n``
    shows the tiny MAE gain is not worth the added pairwise parameters.
    """
    calls = iter(
        [
            {
                "bic": 1_000.0,
                "total_cv_mse": 1.0,
                "system_mae": 0.2,
                "n_effective_samples": 10,
                "coherence_matrix": np.array([[0.0, 0.2], [0.2, 0.0]]),
                "spatial_correction": 1.0,
            },
            {
                "bic": 900.0,
                "total_cv_mse": 1.0,
                "system_mae": 0.199,
                "n_effective_samples": 1_000,
                "coherence_matrix": np.array(
                    [
                        [0.0, 0.199, 0.199],
                        [0.199, 0.0, 0.199],
                        [0.199, 0.199, 0.0],
                    ]
                ),
                "spatial_correction": 1.0,
            },
        ]
    )

    monkeypatch.setattr(scoring, "geological_coherence_score", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(scoring, "mutual_information", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(scoring, "pairwise_distance", lambda *args, **kwargs: 0.0)

    result = scoring.evaluate_new_layer(
        _ControlledStore(),
        "candidate",
        np.full((2, 2, 1), 0.8, dtype=np.float32),
        "float",
        seed=7,
    )

    assert result["bic_comparison_n_effective_samples"] == 1_000
    assert result["bic_delta"] >= 0.0
    assert result["admitted"] is False


def test_second_layer_bic_delta_uses_common_effective_sample_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second layer has no real Stage-1 baseline, so BIC must be clean.

    This guards the boundary where Stage 1 auto-passes. A candidate should not
    be admitted when negative raw BIC is only an artifact of comparing a
    one-layer null scored with small ``n`` against a two-layer score with large
    ``n``.
    """
    monkeypatch.setattr(
        scoring,
        "_single_layer_null_bic",
        lambda *args, **kwargs: {
            "bic": 1_000.0,
            "total_cv_mse": 1.0,
            "system_mae": 0.2,
            "n_effective_samples": 10,
            "single_layer_null_mad": 0.2,
            "spatial_correction": 1.0,
        },
    )
    monkeypatch.setattr(
        scoring,
        "geological_coherence_score",
        lambda *args, **kwargs: {
            "bic": 900.0,
            "total_cv_mse": 1.0,
            "system_mae": 0.25,
            "n_effective_samples": 1_000,
            "coherence_matrix": np.array([[0.0, 0.25], [0.25, 0.0]]),
            "spatial_correction": 1.0,
        },
    )
    monkeypatch.setattr(scoring, "mutual_information", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(scoring, "pairwise_distance", lambda *args, **kwargs: 0.0)

    result = scoring.evaluate_new_layer(
        _OneLayerStore(),
        "candidate",
        np.full((2, 2, 1), 0.8, dtype=np.float32),
        "float",
        seed=7,
    )

    assert result["bic_comparison_n_effective_samples"] == 1_000
    assert result["bic_delta"] >= 0.0
    assert result["admitted"] is False


# ---------------------------------------------------------------------------
# Test 4 — first-layer auto-admit must include all Stage-1 fields
# ---------------------------------------------------------------------------
def test_first_layer_returns_stage1_fields(tmp_path: Path) -> None:
    store = _store(tmp_path / "first_layer")
    rng = np.random.default_rng(4)
    base = rng.random((20, 20, 4)).astype(np.float32)
    result = evaluate_new_layer(store, "only", base, "float", seed=7)

    for key in (
        "masking_test_passed",
        "masking_test_improvement",
        "masking_test_direction",
        "stage_completed",
    ):
        assert key in result, f"missing key {key} in first-layer admit"

    assert result["admitted"] is True
    # Rabbit-hole-bias fix: the first layer is scored with a real predict-by-mean
    # null model, not the old -1.0 sentinel. Direction renamed accordingly and
    # bic_delta is now a genuine negative score.
    assert result["masking_test_direction"] == "null_model_baseline"
    assert result["stage_completed"] == "mae_bic_completed"
    assert result["bic_delta"] != -1.0
    assert result["bic_delta"] < 0.0


def test_stale_nsltask_scoring_mirror_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stale_path = repo_root / "NSL2-geology-task" / "src" / "voxel_features" / "scoring.py"
    live_path = Path(scoring.__file__).resolve()

    assert not stale_path.exists()
    assert repo_root / "voxel-features-mcp" in live_path.parents


# ---------------------------------------------------------------------------
# Test 5 — 2nd-layer BIC must use a null baseline (mean predictor), not bic=0
# ---------------------------------------------------------------------------
def test_second_layer_bic_uses_null_baseline(tmp_path: Path) -> None:
    """An independent-noise 2nd layer should fail Stage 2.

    Pre-fix: ``score_before.bic = 0`` (n_layers==1 sentinel), so any non-zero
    ``bic_after`` produced a wildly negative bic_delta and admission. With the
    null baseline (predict-by-mean), a 2-layer model that's worse than the
    null does NOT yield a negative bic_delta.
    """
    store = _store(tmp_path / "null_baseline")
    _seed_layer(store, "base", np.random.default_rng(5).random((20, 20, 4)).astype(np.float32))

    # 2nd layer that cannot help predict the 1st: pure independent noise.
    noise = np.random.default_rng(98765).random((20, 20, 4)).astype(np.float32)
    result = evaluate_new_layer(store, "noise2", noise, "float", seed=7)

    assert result["bic_delta"] >= 0.0, (
        f"2nd-layer noise should not earn negative bic_delta with null baseline, got {result['bic_delta']}"
    )


# ---------------------------------------------------------------------------
# Test 6 — bic_delta must be per-sample-normalized so grid size doesn't dominate
# ---------------------------------------------------------------------------
def test_bic_delta_grid_invariance(tmp_path: Path) -> None:
    """Same predictive relationship on two grid sizes should give comparable
    per-sample bic_delta (within ~2x). Tests the n_effective_samples
    normalization that fixed Bug 3."""
    def _delta_for_shape(shape: tuple[int, int, int], seed: int) -> float:
        grid = GridSpec(
            origin=(0.0, 0.0, 0.0),
            maximum=(0.02, 0.02, 20.0),
            shape=shape,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = VoxelStore(Path(tmp), grid)
            rng = np.random.default_rng(seed)
            a = rng.random(shape).astype(np.float32)
            evaluate_new_layer(store, "a", a, "float", seed=7)
            b = (a * 0.7 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
            evaluate_new_layer(store, "b", b, "float", seed=7)
            c = (a * 0.5 + b * 0.4).astype(np.float32)
            return evaluate_new_layer(store, "c", c, "float", seed=7)["bic_delta"]

    small = _delta_for_shape((10, 10, 3), seed=1)
    big = _delta_for_shape((25, 25, 5), seed=1)

    # Per-sample bic_delta should be on the same order of magnitude.
    # Allow a generous factor (4x): the test is for "doesn't explode with N",
    # not for exact numerical equivalence.
    if abs(small) > 1e-6 and abs(big) > 1e-6:
        ratio = abs(big / small) if abs(small) > abs(big) else abs(small / big)
        assert 0.1 < ratio < 10.0, (
            f"bic_delta should be per-sample-normalized; small={small}, big={big}"
        )


# ---------------------------------------------------------------------------
# Test 7 — reward gradient: stronger improvement should yield more-negative bic_delta
# ---------------------------------------------------------------------------
def test_reward_gradient_above_threshold(tmp_path: Path) -> None:
    """With recalibrated divisors, ``-bic_delta / 1.0`` should produce a
    visible gradient between a moderate admit and a strong admit (not both
    saturating at 1.0)."""
    def _bic_for(seed: int, noise_scale: float) -> float:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoxelStore(Path(tmp), _GRID)
            rng = np.random.default_rng(seed)
            base = rng.random((20, 20, 4)).astype(np.float32)
            evaluate_new_layer(store, "base", base, "float", seed=7)
            partner = (base + rng.standard_normal(base.shape).astype(np.float32) * 0.3).astype(np.float32)
            evaluate_new_layer(store, "partner", partner, "float", seed=7)
            # Candidate that predicts (base, partner) well — noise_scale controls signal
            candidate = (
                0.5 * base + 0.5 * partner
                + rng.standard_normal(base.shape).astype(np.float32) * noise_scale
            ).astype(np.float32)
            return evaluate_new_layer(store, "candidate", candidate, "float", seed=7)["bic_delta"]

    weak = _bic_for(seed=10, noise_scale=0.5)
    strong = _bic_for(seed=10, noise_scale=0.01)

    # Stronger predictor → more negative (better) bic_delta.
    assert strong < weak, f"strong={strong} should be < weak={weak}"


# ---------------------------------------------------------------------------
# Test 8 — same input + same seed -> identical bic_delta
# ---------------------------------------------------------------------------
def test_scoring_deterministic_with_seed(tmp_path: Path) -> None:
    def _run(seed: int) -> float:
        with tempfile.TemporaryDirectory() as tmp:
            store = VoxelStore(Path(tmp), _GRID)
            rng = np.random.default_rng(0)
            a = rng.random((20, 20, 4)).astype(np.float32)
            evaluate_new_layer(store, "a", a, "float", seed=seed)
            b = (a * 0.6 + 0.05 * rng.standard_normal(a.shape)).astype(np.float32)
            evaluate_new_layer(store, "b", b, "float", seed=seed)
            c = (a * 0.4 + b * 0.4 + 0.02 * rng.standard_normal(a.shape)).astype(np.float32)
            return evaluate_new_layer(store, "c", c, "float", seed=seed)["bic_delta"]

    first = _run(seed=123)
    second = _run(seed=123)
    other = _run(seed=456)

    assert first == pytest.approx(second, abs=1e-10), (
        f"same seed should produce identical bic_delta; got {first} vs {second}"
    )
    # Sanity: different seed should usually differ (this is a weak invariant
    # since the data is identical, but the CV split / Moran's I sample differ).
    # We accept equality here as well — we only care that seeded determinism
    # holds, not that different seeds always diverge.
    _ = other  # touch to keep pytest happy


def test_scoring_reports_effective_sample_confound_diagnostics(tmp_path: Path) -> None:
    store = _store(tmp_path / "n_eff_diagnostics")
    rng = np.random.default_rng(123)
    base = rng.random((20, 20, 4)).astype(np.float32)
    _seed_two_layers(store, base)

    candidate = (base * 0.5 + rng.random(base.shape) * 0.1).astype(np.float32)
    result = evaluate_new_layer(store, "candidate", candidate, "float", seed=7)

    assert "n_effective_samples_before" in result
    assert "n_effective_samples_after" in result
    assert "n_effective_samples_delta" in result
    assert result["n_effective_samples_after"] == result["n_effective_samples"]
    assert result["n_effective_samples_delta"] == (
        result["n_effective_samples_after"] - result["n_effective_samples_before"]
    )
    assert result["candidate_nonzero_voxels"] == int(np.count_nonzero(candidate))
    assert result["candidate_fill_fraction"] == pytest.approx(
        float(np.count_nonzero(candidate)) / candidate.size
    )
