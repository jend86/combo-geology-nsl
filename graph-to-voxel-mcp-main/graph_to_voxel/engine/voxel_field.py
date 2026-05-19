from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from graph_to_voxel.schema.provenance import DerivationSpec


class DerivedChannelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    derivation: DerivationSpec
    units: str | None = None


class GridSpec:
    """Regular dense grid specification.

    Accepts two equivalent forms:
      GridSpec(origin=(x0,y0,z0), maximum=(x1,y1,z1), nx=N, ny=N, nz=N)
      GridSpec(bounds=((x0,x1),(y0,y1),(z0,z1)), shape=(nx,ny,nz))
    """

    def __init__(
        self,
        *,
        origin: tuple[float, float, float] | None = None,
        maximum: tuple[float, float, float] | None = None,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        shape: tuple[int, int, int] | None = None,
    ) -> None:
        if bounds is not None and shape is not None:
            (x0, x1), (y0, y1), (z0, z1) = bounds
            nx, ny, nz = shape
            origin = (x0, y0, z0)
            maximum = (x1, y1, z1)
        if origin is None or maximum is None or nx is None or ny is None or nz is None:
            raise ValueError(
                "GridSpec requires either origin/maximum/nx/ny/nz or bounds/shape"
            )
        self.origin = origin
        self.maximum = maximum
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def __repr__(self) -> str:
        return (
            f"GridSpec(origin={self.origin}, maximum={self.maximum}, "
            f"nx={self.nx}, ny={self.ny}, nz={self.nz})"
        )

    def cell_centres_xyz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 1-D coordinate arrays for cell centres."""
        xs = np.linspace(self.origin[0], self.maximum[0], self.nx + 1)
        ys = np.linspace(self.origin[1], self.maximum[1], self.ny + 1)
        zs = np.linspace(self.origin[2], self.maximum[2], self.nz + 1)
        return (
            0.5 * (xs[:-1] + xs[1:]),
            0.5 * (ys[:-1] + ys[1:]),
            0.5 * (zs[:-1] + zs[1:]),
        )

    def cell_centres(self) -> np.ndarray:
        """Return (nx*ny*nz, 3) array of cell-centre coordinates (ij indexing)."""
        xc, yc, zc = self.cell_centres_xyz()
        xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    def sample_points(self, subgrid_factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return Gauss-Legendre sample points and per-cell weights.

        The points are flattened as ``(cell, subcell)`` so callers can reshape
        arrays with ``(..., *grid.shape, subgrid_factor**3)`` before reducing.
        """
        if subgrid_factor < 1:
            raise ValueError("subgrid_factor must be >= 1")
        centres = self.cell_centres()
        if subgrid_factor == 1:
            return centres, np.ones(1, dtype=np.float32)

        nodes_1d, weights_1d = np.polynomial.legendre.leggauss(subgrid_factor)
        nodes_1d = 0.5 * nodes_1d
        weights_1d = 0.5 * weights_1d
        offsets = np.array(
            np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij"),
            dtype=float,
        ).reshape(3, -1).T
        spacing = np.asarray(self.spacing, dtype=float)
        sample_points = centres[:, None, :] + offsets[None, :, :] * spacing[None, None, :]
        wx, wy, wz = np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")
        weights = (wx * wy * wz).ravel().astype(np.float32)
        weights = weights / weights.sum(dtype=np.float32)
        return sample_points.reshape(-1, 3), weights

    # Alias used by the adapter layer
    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.cell_centres_xyz()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (
            (self.maximum[0] - self.origin[0]) / self.nx,
            (self.maximum[1] - self.origin[1]) / self.ny,
            (self.maximum[2] - self.origin[2]) / self.nz,
        )

    @property
    def diagonal_length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.maximum) - np.asarray(self.origin)))

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return (
            (self.origin[0], self.maximum[0]),
            (self.origin[1], self.maximum[1]),
            (self.origin[2], self.maximum[2]),
        )


@dataclass(slots=True)
class VoxelField:
    most_likely_unit: np.ndarray
    unit_probs: np.ndarray
    entropy: np.ndarray
    domain_mask: np.ndarray
    scalar_field: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    unit_ids: list[str]
    feature_names: list[str]
    epsg: int | None = None
    derived_scalars: dict[str, np.ndarray] = field(default_factory=dict)
    derived_scalar_provenance: dict[str, DerivedChannelMetadata] = field(
        default_factory=dict
    )
    attrs: dict[str, Any] = field(default_factory=dict)
    support_membership: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.most_likely_unit = np.asarray(self.most_likely_unit, dtype=np.int16)
        self.unit_probs = np.asarray(self.unit_probs, dtype=np.float32)
        if self.support_membership is None:
            self.support_membership = self.unit_probs.copy()
        else:
            self.support_membership = np.asarray(self.support_membership, dtype=np.float32)
        self.entropy = np.asarray(self.entropy, dtype=np.float32)
        self.domain_mask = np.asarray(self.domain_mask, dtype=bool)
        self.scalar_field = np.asarray(self.scalar_field, dtype=np.float32)
        self.derived_scalars = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in self.derived_scalars.items()
        }
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        expected = (len(self.x), len(self.y), len(self.z))
        if self.most_likely_unit.shape != expected:
            raise ValueError(f"most_likely_unit shape {self.most_likely_unit.shape} != {expected}")
        if self.domain_mask.shape != expected:
            raise ValueError(f"domain_mask shape {self.domain_mask.shape} != {expected}")
        if self.entropy.shape != expected:
            raise ValueError(f"entropy shape {self.entropy.shape} != {expected}")
        if self.unit_probs.shape != (len(self.unit_ids), *expected):
            raise ValueError("unit_probs must have shape (unit, x, y, z)")
        if self.support_membership.shape != (len(self.unit_ids), *expected):
            raise ValueError("support_membership must have shape (unit, x, y, z)")
        if self.scalar_field.shape != (len(self.feature_names), *expected):
            raise ValueError("scalar_field must have shape (feature, x, y, z)")
        for name, values in self.derived_scalars.items():
            if values.shape != expected:
                raise ValueError(f"derived scalar {name!r} shape {values.shape} != {expected}")
        for name in self.derived_scalar_provenance:
            if name not in self.derived_scalars:
                raise ValueError(f"derived scalar provenance references unknown channel {name!r}")
        self.attrs.setdefault("unit_probs_kind", "mixing")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.most_likely_unit.shape

    @property
    def unit_catalog(self) -> list[str]:
        return self.unit_ids

    @property
    def cell_volume(self) -> float:
        return _spacing(self.x) * _spacing(self.y) * _spacing(self.z)

    def to_xarray(self) -> xr.Dataset:
        attrs = dict(self.attrs)
        attrs.update(
            {
                "epsg": self.epsg,
                "coordinate_units": "metres",
                "representation": "graph_to_voxel.v1",
            }
        )
        data_vars: dict[str, Any] = {
            "most_likely_unit": (("x", "y", "z"), self.most_likely_unit),
            "unit_probs": (("unit", "x", "y", "z"), self.unit_probs),
            "support_membership": (("unit", "x", "y", "z"), self.support_membership),
            "entropy": (("x", "y", "z"), self.entropy),
            "domain_mask": (("x", "y", "z"), self.domain_mask),
            "scalar_field": (("feature", "x", "y", "z"), self.scalar_field),
        }
        coords: dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "unit": self.unit_ids,
            "feature": self.feature_names,
        }
        if self.derived_scalars:
            derived_names = list(self.derived_scalars)
            data_vars["derived_scalars"] = (
                ("derived_scalar", "x", "y", "z"),
                np.stack([self.derived_scalars[name] for name in derived_names]).astype(np.float32),
            )
            coords["derived_scalar"] = derived_names
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        )
        if self.derived_scalar_provenance and "derived_scalars" in dataset:
            dataset["derived_scalars"].attrs["channel_provenance"] = {
                name: metadata.model_dump(mode="json")
                for name, metadata in self.derived_scalar_provenance.items()
            }
        dataset["unit_probs"].attrs["unit_probs_kind"] = self.attrs.get("unit_probs_kind", "mixing")
        return dataset

    @classmethod
    def from_xarray(cls, dataset: xr.Dataset) -> VoxelField:
        loaded = dataset.load()
        epsg = loaded.attrs.get("epsg")
        derived_scalars = {}
        derived_scalar_provenance = {}
        if "derived_scalars" in loaded:
            derived_names = [str(value) for value in loaded.coords["derived_scalar"].values.tolist()]
            derived_values = loaded["derived_scalars"].values
            derived_scalars = {
                name: derived_values[idx]
                for idx, name in enumerate(derived_names)
            }
            provenance_attr = loaded["derived_scalars"].attrs.get("channel_provenance", {})
            derived_scalar_provenance = {
                str(name): DerivedChannelMetadata.model_validate(metadata)
                for name, metadata in provenance_attr.items()
            }
        attrs = {key: value for key, value in loaded.attrs.items() if key not in {"epsg"}}
        attrs.setdefault(
            "unit_probs_kind",
            loaded["unit_probs"].attrs.get("unit_probs_kind", "mixing"),
        )
        support_membership = (
            loaded["support_membership"].values
            if "support_membership" in loaded
            else loaded["unit_probs"].values
        )
        return cls(
            most_likely_unit=loaded["most_likely_unit"].values,
            unit_probs=loaded["unit_probs"].values,
            support_membership=support_membership,
            entropy=loaded["entropy"].values,
            domain_mask=loaded["domain_mask"].values,
            scalar_field=loaded["scalar_field"].values,
            x=loaded.coords["x"].values,
            y=loaded.coords["y"].values,
            z=loaded.coords["z"].values,
            unit_ids=[str(value) for value in loaded.coords["unit"].values.tolist()],
            feature_names=[str(value) for value in loaded.coords["feature"].values.tolist()],
            epsg=None if epsg in (None, "", -1) else int(epsg),
            derived_scalars=derived_scalars,
            derived_scalar_provenance=derived_scalar_provenance,
            attrs=attrs,
        )

    def with_derived_scalars(
        self,
        channels: dict[str, np.ndarray],
        provenance: dict[str, DerivedChannelMetadata] | None = None,
    ) -> VoxelField:
        collisions = sorted(set(channels) & set(self.feature_names))
        if collisions:
            collision = collisions[0]
            raise ValueError(
                f"derived scalar channel {collision!r} collides with existing feature "
                f"{collision!r}; choose a distinct derived channel name"
            )
        derived_scalars = {name: values.copy() for name, values in self.derived_scalars.items()}
        derived_scalars.update(
            {name: np.asarray(values, dtype=np.float32) for name, values in channels.items()}
        )
        derived_scalar_provenance = dict(self.derived_scalar_provenance)
        if provenance:
            derived_scalar_provenance.update(provenance)
        return VoxelField(
            most_likely_unit=self.most_likely_unit.copy(),
            unit_probs=self.unit_probs.copy(),
            support_membership=self.support_membership.copy(),
            entropy=self.entropy.copy(),
            domain_mask=self.domain_mask.copy(),
            scalar_field=self.scalar_field.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            unit_ids=list(self.unit_ids),
            feature_names=list(self.feature_names),
            epsg=self.epsg,
            derived_scalars=derived_scalars,
            derived_scalar_provenance=derived_scalar_provenance,
            attrs=dict(self.attrs),
        )

    def save_zarr(self, path: str | Path) -> None:
        from graph_to_voxel.voxel.persistence import save_zarr

        save_zarr(self, path)


def _spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    return float(np.mean(np.diff(values)))
