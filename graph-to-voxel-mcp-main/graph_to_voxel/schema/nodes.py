from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, RootModel, model_validator

from graph_to_voxel.schema.provenance import Provenance
from graph_to_voxel.schema.uncertainty import (
    CategoricalUncertainty,
    IntervalUncertainty,
    OrientationUncertainty,
    UncertaintyValue,
    nominal_value,
)


@dataclass(slots=True, frozen=True)
class PositionWithUncertainty:
    """Position with per-axis 1-sigma (AABB) uncertainty.

    Off-diagonal covariance is zero by construction. For borehole-trace correlation,
    use MCUE realisation; do not rely on these stds for accurate kriging variograms
    with anisotropic positional error.
    """

    nominal: np.ndarray
    std: np.ndarray
    covariance: np.ndarray | None = None


class PositionMixin:
    position: tuple[UncertaintyValue, UncertaintyValue, UncertaintyValue]

    def position_array(self) -> np.ndarray:
        values = []
        for component in self.position:
            value = nominal_value(component)
            if isinstance(value, tuple):
                raise TypeError("orientation tuples cannot be used as scalar coordinates")
            values.append(float(value))
        return np.asarray(values, dtype=float)

    def position_with_std(self) -> PositionWithUncertainty:
        std = []
        for component in self.position:
            variance = component.variance()
            std.append(0.0 if variance is None else float(np.sqrt(variance)))
        return PositionWithUncertainty(
            nominal=self.position_array(),
            std=np.asarray(std, dtype=float),
            covariance=None,
        )


class NodeBase(BaseModel):
    id: str
    p_exists: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)


class StratigraphicUnit(NodeBase):
    kind: Literal["stratigraphic_unit", "StratigraphicUnit"] = "stratigraphic_unit"
    unit_id: str
    series_id: str
    topology: Literal["layer", "embedded"]
    anchor_inside: tuple[float, float, float] | None = None
    lithology: CategoricalUncertainty | None = None
    age_ma: UncertaintyValue | None = None
    bulk_volume_bounds: IntervalUncertainty | None = None

    @model_validator(mode="after")
    def _validate_topology_anchor(self) -> StratigraphicUnit:
        if self.topology == "layer" and self.anchor_inside is not None:
            raise ValueError("layered units cannot specify anchor_inside; use topology='embedded'")
        return self


class Contact(PositionMixin, NodeBase):
    kind: Literal["contact", "Contact"] = "contact"
    position: tuple[UncertaintyValue, UncertaintyValue, UncertaintyValue]
    between: tuple[str, str]
    polarity: Literal[-1, 1] | None = None


class Orientation(PositionMixin, NodeBase):
    kind: Literal["orientation", "Orientation"] = "orientation"
    position: tuple[UncertaintyValue, UncertaintyValue, UncertaintyValue]
    dip: OrientationUncertainty
    for_unit: str
    feature: str | None = None


class Fault(NodeBase):
    kind: Literal["fault", "Fault"] = "fault"
    surface_points: list[str]
    kinematic: CategoricalUncertainty | None = None
    chronology_rank: int | None = None


class ObservationPoint(NodeBase):
    kind: Literal["observation_point", "ObservationPoint"] = "observation_point"
    position: tuple[UncertaintyValue, UncertaintyValue, UncertaintyValue]
    notes: str


class Location(PositionMixin, NodeBase):
    """Physical sampling location shared by one or more Sample nodes."""

    kind: Literal["location", "Location"] = "location"
    position: tuple[UncertaintyValue, UncertaintyValue, UncertaintyValue]
    name: str | None = None


class Sample(PositionMixin, NodeBase):
    kind: Literal["sample", "Sample"] = "sample"
    position: tuple[UncertaintyValue, UncertaintyValue, UncertaintyValue] | None = None
    analyte: str
    unit_of_measure: str
    value: UncertaintyValue

    def position_array(self) -> np.ndarray:
        if self.position is None:
            raise TypeError("Sample position is stored on its AT Location; use Graph.position_array(sample)")
        return super().position_array()

    def position_with_std(self) -> PositionWithUncertainty:
        if self.position is None:
            raise TypeError("Sample position is stored on its AT Location; use Graph.position_with_std(sample)")
        return super().position_with_std()


class Series(NodeBase):
    kind: Literal["series", "Series"] = "series"
    series_id: str | None = None
    name: str | None = None


NodeValue = Annotated[
    StratigraphicUnit | Contact | Orientation | Fault | ObservationPoint | Location | Sample | Series,
    Field(discriminator="kind"),
]


class AnyNode(RootModel[NodeValue]):
    @model_validator(mode="before")
    @classmethod
    def _normalise_kind(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        aliases = {
            "StratigraphicUnit": "stratigraphic_unit",
            "Contact": "contact",
            "Orientation": "orientation",
            "Fault": "fault",
            "ObservationPoint": "observation_point",
            "Location": "location",
            "Sample": "sample",
            "Series": "series",
        }
        kind = value.get("kind")
        if kind in aliases:
            return {**value, "kind": aliases[kind]}
        return value
