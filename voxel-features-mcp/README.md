# Voxel Features MCP

Voxel feature layer store with MDL/MI scoring for hypothesis-driven geological discovery.

## Architecture

```
Voxel Store (source of truth)
    ├── Feature Layers (agent-created)
    │   ├── au_kriged_surface
    │   ├── fault_distance
    │   └── ree_anomaly_flag
    └── Grid Spec (25x25x5 for Coe Fairbairn)

Knowledge Graph (experiment records)
    ├── Hypothesis + Result pairs
    ├── Training data export
    └── Crossbreeding lineage
```

## Installation

```bash
cd voxel-features-mcp
uv sync --extra mcp --extra data
```

## Usage

### CLI

```bash
# Initialize store
vfm init /path/to/store --grid coe-fairbairn

# Check store info
vfm info /path/to/store

# Compute MDL
vfm mdl /path/to/store

# Export training data
vfm export-training /path/to/kg /path/to/training.jsonl
```

### MCP Server

```bash
# Run server (stdio)
vfm-mcp

# With custom paths
VFM_STORE_PATH=/path/to/store vfm-mcp
```

### Python API

```python
from voxel_features import VoxelStore, GridSpec, compute_mdl, mutual_information

# Create store with Coe Fairbairn grid
from voxel_features.store import COE_FAIRBAIRN_GRID
store = VoxelStore("/path/to/store", COE_FAIRBAIRN_GRID)

# Add a feature layer
import numpy as np
values = np.random.rand(25, 25, 5)
store.add_layer("test_feature", values, dtype="float")

# Compute MDL
mdl = compute_mdl(store)
print(f"MDL: {mdl} bits")
```

> Experiment persistence is owned by the task framework (writes a JSONL ledger at
> `<kg_dir>/experiments.jsonl`); the previous `KnowledgeGraph` Python class was
> removed in the scoring-fix-and-replay-2026-05-25 cleanup.

## MCP Tools

### Feature Tools

| Tool | Description |
|------|-------------|
| `feature.create` | Add a new feature layer |
| `feature.get` | Get a layer by name |
| `feature.list` | List all layers |
| `feature.delete` | Remove a layer |

### Scoring Tools

| Tool | Description |
|------|-------------|
| `scoring.compute_mdl` | Total MDL of store |
| `scoring.mutual_information` | MI between two layers |
| `scoring.marginal_contribution` | Layer's contribution to compression |
| `scoring.evaluate_layer` | Evaluate + admit/reject a new layer |

## Grid Specification

Default Coe Fairbairn grid:
- Origin: (117.832397, -27.441096, 0.0)
- Maximum: (117.973493, -27.300000, 80.0)
- Shape: 25 × 25 × 5 voxels
- CRS: EPSG:4326

## Scoring

### MDL (Minimum Description Length)

Measures total bits to describe the voxel store:
- **Model complexity**: Cost of encoding structure
- **Data cost**: Cost of encoding values (reduced by correlations)

**Admission rule**: Layer admitted if `mdl_after < mdl_before`

### Mutual Information

Measures shared information between layers:
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

Used for cross-prediction analysis and identifying redundant layers.
