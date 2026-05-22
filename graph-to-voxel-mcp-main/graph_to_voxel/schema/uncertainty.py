from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


class PointUncertainty(BaseModel):
    kind: Literal["Point"] = "Point"
    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.value

    def nominal(self) -> float:
        return self.value

    def variance(self) -> float | None:
        return 0.0


class GaussianUncertainty(BaseModel):
    kind: Literal["Gaussian"] = "Gaussian"
    mean: float
    std: float = Field(gt=0.0)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.std))

    def nominal(self) -> float:
        return self.mean

    def variance(self) -> float | None:
        return self.std**2


class IntervalUncertainty(BaseModel):
    kind: Literal["Interval"] = "Interval"
    lo: float
    hi: float

    @model_validator(mode="after")
    def _lo_before_hi(self) -> IntervalUncertainty:
        if self.lo >= self.hi:
            raise ValueError("Interval requires lo < hi")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))

    def nominal(self) -> float:
        return (self.lo + self.hi) / 2.0

    def variance(self) -> float | None:
        return None

    def as_distribution(
        self,
        name: Literal["uniform", "triangular", "beta"],
        **params: float,
    ) -> DistributionUncertainty:
        if name == "uniform":
            distribution_params = {"lo": self.lo, "hi": self.hi}
        elif name == "triangular":
            distribution_params = {
                "left": self.lo,
                "mode": params.get("mode", self.nominal()),
                "right": self.hi,
            }
        else:
            distribution_params = {
                "lo": self.lo,
                "hi": self.hi,
                "alpha": params.get("alpha", params.get("a", 1.0)),
                "beta": params.get("beta", params.get("b", 1.0)),
            }
        return DistributionUncertainty(name=name, params=distribution_params)


class CategoricalUncertainty(BaseModel):
    kind: Literal["Categorical"] = "Categorical"
    probs: dict[str, float]

    @model_validator(mode="after")
    def _valid_probabilities(self) -> CategoricalUncertainty:
        if not self.probs:
            raise ValueError("Categorical requires at least one category")
        if any(prob <= 0.0 for prob in self.probs.values()):
            raise ValueError("Categorical probabilities must be positive")
        total = sum(self.probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Categorical probabilities must sum to 1")
        return self

    def sample(self, rng: np.random.Generator) -> str:
        labels = list(self.probs)
        probs = np.array([self.probs[label] for label in labels], dtype=float)
        return str(rng.choice(labels, p=probs))

    def nominal(self) -> str:
        return max(self.probs, key=self.probs.get)

    def variance(self) -> float | None:
        return None


class OrientationUncertainty(BaseModel):
    kind: Literal["Orientation"] = "Orientation"
    dip_mean: float = Field(ge=0.0, le=90.0)
    dip_kappa: float = Field(gt=0.0)
    azimuth_mean: float = Field(ge=0.0, lt=360.0)
    azimuth_kappa: float = Field(gt=0.0)

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        dip = float(rng.vonmises(np.deg2rad(self.dip_mean), self.dip_kappa))
        azimuth = float(rng.vonmises(np.deg2rad(self.azimuth_mean), self.azimuth_kappa))
        return float(np.clip(np.rad2deg(dip), 0.0, 90.0)), float(np.rad2deg(azimuth) % 360.0)

    def nominal(self) -> tuple[float, float]:
        return self.dip_mean, self.azimuth_mean

    def variance(self) -> float | None:
        return None


class DistributionUncertainty(BaseModel):
    kind: Literal["Distribution"] = "Distribution"
    name: Literal["uniform", "triangular", "lognormal", "truncnorm", "beta"]
    params: dict[str, float]

    @field_validator("params")
    @classmethod
    def _numeric_params(cls, value: dict[str, float]) -> dict[str, float]:
        return {key: float(param) for key, param in value.items()}

    def sample(self, rng: np.random.Generator) -> float:
        if self.name == "uniform":
            loc = self.params.get("loc", self.params.get("lo", 0.0))
            if "scale" in self.params:
                scale = self.params["scale"]
            else:
                scale = self.params.get("hi", 1.0) - loc
            return float(rng.uniform(loc, loc + scale))
        if self.name == "triangular":
            left = self.params.get("left", self.params.get("lo", 0.0))
            mode = self.params.get("mode", self.params.get("c", 0.5))
            right = self.params.get("right", self.params.get("hi", 1.0))
            return float(rng.triangular(left, mode, right))
        if self.name == "lognormal":
            sigma = self.params.get("sigma", self.params.get("s", 1.0))
            mean = self.params.get("mean", self.params.get("loc", 0.0))
            scale = self.params.get("scale", float(np.exp(mean)))
            return float(rng.lognormal(np.log(scale), sigma))
        if self.name == "truncnorm":
            from scipy.stats import truncnorm

            mean = self.params.get("mean", self.params.get("loc", 0.0))
            std = self.params.get("std", self.params.get("scale", 1.0))
            lo = self.params.get("lo", self.params.get("a", mean - 2.0 * std))
            hi = self.params.get("hi", self.params.get("b", mean + 2.0 * std))
            return float(truncnorm.rvs((lo - mean) / std, (hi - mean) / std, loc=mean, scale=std, random_state=rng))
        alpha = self.params.get("alpha", self.params.get("a", 1.0))
        beta = self.params.get("beta", self.params.get("b", 1.0))
        lo = self.params.get("lo", 0.0)
        hi = self.params.get("hi", 1.0)
        return float(lo + (hi - lo) * rng.beta(alpha, beta))

    def nominal(self) -> float:
        if self.name == "uniform":
            loc = self.params.get("loc", self.params.get("lo", 0.0))
            if "scale" in self.params:
                return loc + self.params["scale"] / 2.0
            return (loc + self.params.get("hi", 1.0)) / 2.0
        if self.name == "triangular":
            return self.params.get("mode", self.params.get("c", 0.5))
        if self.name == "lognormal":
            sigma = self.params.get("sigma", self.params.get("s", 1.0))
            mean = self.params.get("mean", self.params.get("loc", 0.0))
            scale = self.params.get("scale", float(np.exp(mean)))
            return float(scale * np.exp((sigma**2) / 2.0))
        if self.name == "truncnorm":
            return self.params.get("mean", self.params.get("loc", 0.0))
        alpha = self.params.get("alpha", self.params.get("a", 1.0))
        beta = self.params.get("beta", self.params.get("b", 1.0))
        lo = self.params.get("lo", 0.0)
        hi = self.params.get("hi", 1.0)
        return lo + (hi - lo) * alpha / (alpha + beta)

    def variance(self) -> float | None:
        if self.name == "uniform":
            loc = self.params.get("loc", self.params.get("lo", 0.0))
            scale = self.params.get("scale", self.params.get("hi", 1.0) - loc)
            return scale**2 / 12.0
        if self.name == "triangular":
            left = self.params.get("left", self.params.get("lo", 0.0))
            mode = self.params.get("mode", self.params.get("c", 0.5))
            right = self.params.get("right", self.params.get("hi", 1.0))
            return (
                left**2 + mode**2 + right**2 - left * mode - left * right - mode * right
            ) / 18.0
        if self.name == "lognormal":
            sigma = self.params.get("sigma", self.params.get("s", 1.0))
            mean = self.params.get("mean", self.params.get("loc", 0.0))
            scale = self.params.get("scale", float(np.exp(mean)))
            return (float(np.exp(sigma**2)) - 1.0) * scale**2 * float(np.exp(sigma**2))
        if self.name == "truncnorm":
            from scipy.stats import truncnorm

            mean = self.params.get("mean", self.params.get("loc", 0.0))
            std = self.params.get("std", self.params.get("scale", 1.0))
            lo = self.params.get("lo", self.params.get("a", mean - 2.0 * std))
            hi = self.params.get("hi", self.params.get("b", mean + 2.0 * std))
            return float(
                truncnorm.var((lo - mean) / std, (hi - mean) / std, loc=mean, scale=std)
            )
        alpha = self.params.get("alpha", self.params.get("a", 1.0))
        beta = self.params.get("beta", self.params.get("b", 1.0))
        scale = self.params.get("hi", 1.0) - self.params.get("lo", 0.0)
        return scale**2 * alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))


UncertaintyValue = Annotated[
    PointUncertainty
    | GaussianUncertainty
    | IntervalUncertainty
    | CategoricalUncertainty
    | OrientationUncertainty
    | DistributionUncertainty,
    Field(discriminator="kind"),
]


class Uncertainty(RootModel[UncertaintyValue]):
    @model_validator(mode="before")
    @classmethod
    def _default_kind(cls, value: Any) -> Any:
        if isinstance(value, dict) and "kind" not in value:
            keys = set(value)
            if {"value"} <= keys:
                return {**value, "kind": "Point"}
            if {"mean", "std"} <= keys:
                return {**value, "kind": "Gaussian"}
            if {"lo", "hi"} <= keys:
                return {**value, "kind": "Interval"}
            if {"probs"} <= keys:
                return {**value, "kind": "Categorical"}
            if {"dip_mean", "dip_kappa", "azimuth_mean", "azimuth_kappa"} <= keys:
                return {**value, "kind": "Orientation"}
            if {"name", "params"} <= keys:
                return {**value, "kind": "Distribution"}
        return value


def nominal_value(value: UncertaintyValue) -> float | str | tuple[float, float]:
    return value.nominal()


def sample_to_point(value: UncertaintyValue, rng: np.random.Generator) -> PointUncertainty:
    sampled = value.sample(rng)
    if not isinstance(sampled, (float, int, np.floating, np.integer)):
        raise TypeError("Only scalar uncertainties can be realised as point coordinates")
    return PointUncertainty(value=float(sampled))
