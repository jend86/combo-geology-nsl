"""Voxel store with feature layer CRUD and versioning."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Specification for the voxel grid geometry."""
    
    # Grid bounds (in coordinate system units, e.g., lon/lat/depth)
    origin: tuple[float, float, float]  # (x_min, y_min, z_min)
    maximum: tuple[float, float, float]  # (x_max, y_max, z_max)
    
    # Grid shape (number of voxels in each dimension)
    shape: tuple[int, int, int]  # (nx, ny, nz)
    
    # Coordinate reference system (optional)
    crs: str | None = None
    
    @property
    def cell_size(self) -> tuple[float, float, float]:
        """Size of each voxel in coordinate units."""
        return (
            (self.maximum[0] - self.origin[0]) / self.shape[0],
            (self.maximum[1] - self.origin[1]) / self.shape[1],
            (self.maximum[2] - self.origin[2]) / self.shape[2],
        )
    
    @property
    def n_voxels(self) -> int:
        """Total number of voxels in the grid."""
        return self.shape[0] * self.shape[1] * self.shape[2]
    
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays of cell center coordinates for each axis."""
        dx, dy, dz = self.cell_size
        x = np.linspace(self.origin[0] + dx/2, self.maximum[0] - dx/2, self.shape[0])
        y = np.linspace(self.origin[1] + dy/2, self.maximum[1] - dy/2, self.shape[1])
        z = np.linspace(self.origin[2] + dz/2, self.maximum[2] - dz/2, self.shape[2])
        return x, y, z
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": list(self.origin),
            "maximum": list(self.maximum),
            "shape": list(self.shape),
            "crs": self.crs,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridSpec:
        return cls(
            origin=tuple(d["origin"]),
            maximum=tuple(d["maximum"]),
            shape=tuple(d["shape"]),
            crs=d.get("crs"),
        )


@dataclass
class FeatureLayer:
    """A single feature layer in the voxel store."""
    
    name: str
    values: np.ndarray  # shape must match grid.shape
    dtype: Literal["float", "categorical", "boolean"]
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    hypothesis_uri: str | None = None
    experiment_id: str | None = None
    added_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Scoring results (set after evaluation)
    mdl_delta: float | None = None
    mutual_info: dict[str, float] | None = None  # layer_name -> MI with this layer
    
    @property
    def content_hash(self) -> str:
        """Content-addressable hash of the layer."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(self.values.tobytes())
        h.update(self.dtype.encode())
        return h.hexdigest()[:16]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.values.shape),
            "metadata": self.metadata,
            "hypothesis_uri": self.hypothesis_uri,
            "experiment_id": self.experiment_id,
            "added_timestamp": self.added_timestamp,
            "mdl_delta": self.mdl_delta,
            "mutual_info": self.mutual_info,
            "content_hash": self.content_hash,
        }


class VoxelStore:
    """
    Persistent voxel store with feature layers.
    
    The store is the source of truth for the geological world model.
    Feature layers are added by agents and scored by the framework.
    """
    
    def __init__(self, store_path: Path | str, grid: GridSpec | None = None):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self._index_path = self.store_path / "index.json"
        self._layers_dir = self.store_path / "layers"
        self._layers_dir.mkdir(exist_ok=True)
        
        # Load or initialize
        if self._index_path.exists():
            self._load_index()
        else:
            if grid is None:
                raise ValueError("grid must be provided when creating a new store")
            self._grid = grid
            self._layers: dict[str, FeatureLayer] = {}
            self._save_index()
    
    @property
    def grid(self) -> GridSpec:
        return self._grid
    
    @property
    def layer_names(self) -> list[str]:
        return list(self._layers.keys())
    
    def _load_index(self) -> None:
        """Load store index from disk."""
        with open(self._index_path) as f:
            data = json.load(f)
        self._grid = GridSpec.from_dict(data["grid"])
        self._layers = {}
        for name, layer_data in data.get("layers", {}).items():
            # Layer values are loaded lazily from zarr
            self._layers[name] = FeatureLayer(
                name=name,
                values=np.array([]),  # placeholder
                dtype=layer_data["dtype"],
                metadata=layer_data.get("metadata", {}),
                hypothesis_uri=layer_data.get("hypothesis_uri"),
                experiment_id=layer_data.get("experiment_id"),
                added_timestamp=layer_data.get("added_timestamp", ""),
                mdl_delta=layer_data.get("mdl_delta"),
                mutual_info=layer_data.get("mutual_info"),
            )
    
    def _save_index(self) -> None:
        """Save store index to disk atomically.

        Writes to a sibling ``.tmp`` file then ``os.replace()``s into place so
        concurrent readers (e.g. parallel episodes in the feature-hypothesis
        task) never observe a truncated/empty ``index.json``. The previous
        ``open(path, "w")`` left a window between truncation and content where
        a peer reader would JSONDecodeError on ``line 1 column 1 (char 0)``.
        """
        data = {
            "grid": self._grid.to_dict(),
            "layers": {name: layer.to_dict() for name, layer in self._layers.items()},
        }
        tmp_path = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._index_path)
    
    def add_layer(
        self,
        name: str,
        values: np.ndarray,
        dtype: Literal["float", "categorical", "boolean"],
        metadata: dict[str, Any] | None = None,
        hypothesis_uri: str | None = None,
        experiment_id: str | None = None,
    ) -> FeatureLayer:
        """Add a new feature layer to the store."""
        if name in self._layers:
            raise ValueError(f"Layer '{name}' already exists")
        
        # Validate shape
        if values.shape != self._grid.shape:
            raise ValueError(
                f"Layer shape {values.shape} does not match grid shape {self._grid.shape}"
            )
        
        # Create layer
        layer = FeatureLayer(
            name=name,
            values=values,
            dtype=dtype,
            metadata=metadata or {},
            hypothesis_uri=hypothesis_uri,
            experiment_id=experiment_id,
        )
        
        # Save to numpy file
        np.save(self._layers_dir / f"{name}.npy", values)
        
        # Update index
        self._layers[name] = layer
        self._save_index()
        
        return layer
    
    def get_layer(self, name: str) -> FeatureLayer:
        """Get a feature layer by name, loading values from disk."""
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        
        layer = self._layers[name]
        if layer.values.size == 0:
            # Load from numpy file
            layer.values = np.load(self._layers_dir / f"{name}.npy")
        
        return layer
    
    def get_layer_values(self, name: str) -> np.ndarray:
        """Get just the values array for a layer."""
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        return np.load(self._layers_dir / f"{name}.npy")
    
    def remove_layer(self, name: str) -> None:
        """Remove a feature layer from the store."""
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        
        del self._layers[name]
        npy_path = self._layers_dir / f"{name}.npy"
        if npy_path.exists():
            npy_path.unlink()
        self._save_index()
    
    def update_layer_scores(
        self,
        name: str,
        mdl_delta: float,
        mutual_info: dict[str, float],
    ) -> None:
        """Update the scoring results for a layer."""
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")
        
        self._layers[name].mdl_delta = mdl_delta
        self._layers[name].mutual_info = mutual_info
        self._save_index()
    
    def list_layers(self) -> list[dict[str, Any]]:
        """List all layers with their metadata (without loading values)."""
        return [layer.to_dict() for layer in self._layers.values()]
    
    def get_all_values(self) -> dict[str, np.ndarray]:
        """Get all layer values as a dict. Use sparingly for large stores."""
        return {name: self.get_layer_values(name) for name in self._layers}
    
    def to_dict(self) -> dict[str, Any]:
        """Export store as a dict."""
        x, y, z = self._grid.cell_centers()
        
        return {
            "grid": self._grid.to_dict(),
            "coords": {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()},
            "layers": {name: self.get_layer_values(name).tolist() for name in self._layers},
        }


# Default grid for Coe Fairbairn dataset - High resolution for spatial features
COE_FAIRBAIRN_GRID = GridSpec(
    origin=(117.832397, -27.441096, 0.0),
    maximum=(117.973493, -27.300000, 80.0),
    shape=(200, 200, 8),  # ~70m x 79m x 10m resolution, 320k total voxels
    crs="EPSG:4326",
)
