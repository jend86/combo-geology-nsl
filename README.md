# GeoNSL: Voxel Feature Hypothesis System

A geological machine learning system that discovers informative feature layers through hypothesis-driven exploration, scored by BIC on ridge regression.

## Architecture Overview

See [VOXEL_FEATURE_HYPOTHESIS_ARCHITECTURE.md](VOXEL_FEATURE_HYPOTHESIS_ARCHITECTURE.md) for detailed design philosophy and information-theoretic foundations.

## Components

### NSL2-geology-task/
The task harness framework implementing multi-agent workflows:
- **Hypothesis Agent**: Explores data, proposes geological features
- **Coding Agent**: Writes analysis code (isolated from raw data)
- **Framework**: Automated BIC/MI scoring
- **Rewriting Agent**: Creates training data and knowledge graph

### voxel-features-mcp/
MCP server providing voxel store and scoring infrastructure:
- **VoxelStore**: 3D feature layer management with versioning
- **Scoring**: BIC on depth-stratified cross-validated ridge regression
- **KnowledgeGraph**: Experiment tracking and crossbreeding selection

## Key Concepts

1. **Voxels as Truth**: Direct 3D spatial representation, not derived from graphs
2. **Linear Scoring**: Deliberately simple to force agent intelligence
3. **Agent Isolation**: Prevents reward hacking through information boundaries
4. **Depth-Stratified CV**: Tests subsurface prediction from surface data
5. **Crossbreeding**: Combines successful features preferring low mutual information

## Quick Start

```bash
# Set up voxel-features MCP server
cd voxel-features-mcp
uv pip install -e .

# Run feature hypothesis task
cd ../NSL2-geology-task
docker-compose -f docker/feature-hypothesis-compose/docker-compose.yml up
```

## Dataset Requirements

Expects geological data in CSV format with:
- 3D coordinates (longitude, latitude, depth)
- Assay values (Au, Cu, REE elements, etc.)
- Surface and drillhole samples

Place data in a directory structure like:
```
YourDataset/
  amalgamated_csvs/
    geochemDrillhole.csv
    geochemSurface.csv
    tenements.csv
```

## Scoring Philosophy

Uses BIC (Bayesian Information Criterion) on ridge regression:
- **BIC = n·ln(MSE) + k·ln(n)** balances fit vs. complexity
- **Ridge regression** handles correlated geological features
- **Depth folds** test exploration-relevant generalization
- **Joint prediction** ensures features help predict each other

The linear model forces the agent to engineer explicit compositional features rather than relying on the scorer to discover nonlinear patterns.
