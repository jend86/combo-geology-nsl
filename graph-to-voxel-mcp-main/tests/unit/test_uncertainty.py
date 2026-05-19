"""Tests for Uncertainty tagged-union model — must FAIL before implementation."""
import math
import pytest
import numpy as np
from pydantic import ValidationError

from graph_to_voxel.schema.uncertainty import (
    PointUncertainty,
    GaussianUncertainty,
    IntervalUncertainty,
    CategoricalUncertainty,
    OrientationUncertainty,
    DistributionUncertainty,
    Uncertainty,
)


# ── Round-trips ──────────────────────────────────────────────────────────────

def test_point_round_trip():
    u = PointUncertainty(value=3.14)
    assert Uncertainty.model_validate(u.model_dump()).root == u


def test_gaussian_round_trip():
    u = GaussianUncertainty(mean=10.0, std=2.0)
    data = u.model_dump()
    u2 = Uncertainty.model_validate(data).root
    assert u2 == u


def test_interval_round_trip():
    u = IntervalUncertainty(lo=1.0, hi=5.0)
    data = u.model_dump()
    u2 = Uncertainty.model_validate(data).root
    assert u2 == u


def test_categorical_round_trip():
    u = CategoricalUncertainty(probs={"sandstone": 0.6, "shale": 0.4})
    data = u.model_dump()
    u2 = Uncertainty.model_validate(data).root
    assert u2 == u


def test_orientation_round_trip():
    u = OrientationUncertainty(dip_mean=30.0, dip_kappa=10.0, azimuth_mean=90.0, azimuth_kappa=5.0)
    data = u.model_dump()
    u2 = Uncertainty.model_validate(data).root
    assert u2 == u


def test_distribution_round_trip():
    u = DistributionUncertainty(name="lognormal", params={"loc": 0.0, "s": 1.0})
    data = u.model_dump()
    u2 = Uncertainty.model_validate(data).root
    assert u2 == u


# ── Validation failures ───────────────────────────────────────────────────────

def test_gaussian_std_zero_rejected():
    with pytest.raises(ValidationError):
        GaussianUncertainty(mean=0.0, std=0.0)


def test_gaussian_std_negative_rejected():
    with pytest.raises(ValidationError):
        GaussianUncertainty(mean=0.0, std=-1.0)


def test_interval_inverted_rejected():
    with pytest.raises(ValidationError):
        IntervalUncertainty(lo=5.0, hi=3.0)


def test_interval_equal_rejected():
    with pytest.raises(ValidationError):
        IntervalUncertainty(lo=3.0, hi=3.0)


def test_categorical_does_not_sum_to_one_rejected():
    with pytest.raises(ValidationError):
        CategoricalUncertainty(probs={"a": 0.4, "b": 0.5})


def test_categorical_zero_prob_rejected():
    with pytest.raises(ValidationError):
        CategoricalUncertainty(probs={"a": 0.0, "b": 1.0})


def test_distribution_unknown_name_rejected():
    with pytest.raises(ValidationError):
        DistributionUncertainty(name="__import__", params={})


def test_distribution_arbitrary_name_rejected():
    with pytest.raises(ValidationError):
        DistributionUncertainty(name="expon", params={"loc": 0.0})


def test_orientation_dip_out_of_range_rejected():
    with pytest.raises(ValidationError):
        OrientationUncertainty(dip_mean=95.0, dip_kappa=1.0, azimuth_mean=0.0, azimuth_kappa=1.0)


def test_orientation_azimuth_out_of_range_rejected():
    with pytest.raises(ValidationError):
        OrientationUncertainty(dip_mean=30.0, dip_kappa=1.0, azimuth_mean=360.0, azimuth_kappa=1.0)


def test_orientation_kappa_zero_rejected():
    with pytest.raises(ValidationError):
        OrientationUncertainty(dip_mean=30.0, dip_kappa=0.0, azimuth_mean=90.0, azimuth_kappa=1.0)


# ── sample() statistical correctness ─────────────────────────────────────────

N = 10_000
SIGMA = 3.0


def test_gaussian_sample_mean_std():
    u = GaussianUncertainty(mean=5.0, std=1.5)
    rng = np.random.default_rng(42)
    samples = np.array([u.sample(rng) for _ in range(N)])
    assert abs(samples.mean() - 5.0) < SIGMA * 1.5 / math.sqrt(N)
    assert abs(samples.std() - 1.5) < SIGMA * 1.5 / math.sqrt(2 * N)


def test_point_sample_is_constant():
    u = PointUncertainty(value=7.0)
    rng = np.random.default_rng(0)
    assert all(u.sample(rng) == 7.0 for _ in range(10))


def test_interval_sample_in_bounds():
    u = IntervalUncertainty(lo=2.0, hi=8.0)
    rng = np.random.default_rng(1)
    samples = [u.sample(rng) for _ in range(N)]
    assert all(2.0 <= s <= 8.0 for s in samples)
    assert abs(np.mean(samples) - 5.0) < SIGMA * 6.0 / (math.sqrt(12) * math.sqrt(N))


def test_orientation_sample_returns_tuple():
    u = OrientationUncertainty(dip_mean=30.0, dip_kappa=10.0, azimuth_mean=90.0, azimuth_kappa=5.0)
    rng = np.random.default_rng(2)
    result = u.sample(rng)
    assert isinstance(result, tuple) and len(result) == 2
    dip, az = result
    # Not clamping here — just check types
    assert isinstance(dip, float)
    assert isinstance(az, float)


def test_distribution_lognormal_positive():
    u = DistributionUncertainty(name="lognormal", params={"loc": 0.0, "s": 0.5})
    rng = np.random.default_rng(3)
    samples = [u.sample(rng) for _ in range(N)]
    assert all(s > 0 for s in samples)


# ── variance() common contract ────────────────────────────────────────────────

def test_scalar_uncertainty_variance_contract():
    assert PointUncertainty(value=3.0).variance() == 0.0
    assert GaussianUncertainty(mean=3.0, std=0.25).variance() == 0.25**2
    assert IntervalUncertainty(lo=0.4, hi=1.2).variance() is None
    assert CategoricalUncertainty(probs={"a": 0.25, "b": 0.75}).variance() is None
    assert OrientationUncertainty(
        dip_mean=30.0,
        dip_kappa=10.0,
        azimuth_mean=90.0,
        azimuth_kappa=5.0,
    ).variance() is None


def test_interval_as_distribution_variance_and_round_trip():
    interval = IntervalUncertainty(lo=0.4, hi=1.2)

    uniform = interval.as_distribution("uniform")
    triangular = interval.as_distribution("triangular")
    beta = interval.as_distribution("beta", alpha=2.0, beta=2.0)

    assert uniform.variance() == pytest.approx((1.2 - 0.4) ** 2 / 12.0)
    assert triangular.variance() == pytest.approx((1.2 - 0.4) ** 2 / 24.0)
    assert beta.variance() == pytest.approx((1.2 - 0.4) ** 2 * 4.0 / (4.0**2 * 5.0))
    assert uniform.params == {"lo": 0.4, "hi": 1.2}
    assert Uncertainty.model_validate(beta.model_dump(mode="json")).root == beta


def test_distribution_variance_dispatches_without_exceptions():
    values = [
        DistributionUncertainty(name="uniform", params={"lo": 0.0, "hi": 6.0}),
        DistributionUncertainty(name="triangular", params={"left": 0.0, "mode": 3.0, "right": 6.0}),
        DistributionUncertainty(name="lognormal", params={"mean": 0.0, "sigma": 0.25}),
        DistributionUncertainty(name="truncnorm", params={"mean": 0.0, "std": 1.0, "lo": -2.0, "hi": 2.0}),
        DistributionUncertainty(name="beta", params={"alpha": 2.0, "beta": 3.0}),
    ]

    variances = [value.variance() for value in values]

    assert all(variance is not None and variance >= 0.0 for variance in variances)


def test_mixed_uncertainty_variances_are_uniformly_optional():
    xs = [
        PointUncertainty(value=1.0),
        GaussianUncertainty(mean=1.0, std=0.2),
        IntervalUncertainty(lo=0.0, hi=1.0),
        CategoricalUncertainty(probs={"a": 1.0}),
        OrientationUncertainty(dip_mean=30.0, dip_kappa=2.0, azimuth_mean=10.0, azimuth_kappa=2.0),
    ]

    stds = [variance if variance is not None else 0.0 for variance in [x.variance() for x in xs]]

    assert stds == [0.0, 0.04000000000000001, 0.0, 0.0, 0.0]


# ── Determinism ───────────────────────────────────────────────────────────────

def test_same_seed_same_sample():
    u = GaussianUncertainty(mean=0.0, std=1.0)
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    assert u.sample(rng1) == u.sample(rng2)


def test_different_seeds_different_samples():
    u = GaussianUncertainty(mean=0.0, std=1.0)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    samples1 = [u.sample(rng1) for _ in range(20)]
    samples2 = [u.sample(rng2) for _ in range(20)]
    assert samples1 != samples2
