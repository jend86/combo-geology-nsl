# Geological AI Training System: Complete Guide

## Overview

This is a sophisticated multi-agent system for automatically discovering informative geological features through hypothesis-driven exploration. The system creates training data for geological AI by having agents propose, test, and validate geological hypotheses against real datasets.

## Core Philosophy

The system is built on **information-theoretic principles**: agents discover features that compress geological understanding (reduce description length). Features are evaluated using **BIC (Bayesian Information Criterion)** on ridge regression, creating a clear learning signal while preventing reward hacking.

### Key Architectural Decisions

1. **Voxels as Ground Truth**: Unlike traditional approaches that derive spatial fields from graphs, this system treats 3D voxel space as the authoritative representation
2. **Linear Scoring**: Uses ridge regression (linear) rather than complex models to force agents to engineer explicit, compositional features
3. **Agent Isolation**: Strict information boundaries prevent gaming of the scoring system
4. **Depth-Stratified Validation**: Cross-validation holds out entire depth slices to test exploration-relevant generalization

## System Architecture

### Multi-Agent Workflow

The system orchestrates 4 specialized agents in a carefully designed pipeline:

```
HYPOTHESIS AGENT    CODING AGENT       FRAMEWORK         REWRITING AGENT
    Survey     ──►                                     
    Hypothesise──►    Code        ──►
    (wait)     ◄──                ◄──   (execute)
    Translate   ──►                     Evaluate    ──►   Rewrite
                                       (auto-score)
```

#### 1. Hypothesis Agent
- **Purpose**: Domain expert that explores geological data and formulates testable hypotheses
- **Phases**: 
  - `Survey`: Explore raw geological data files
  - `Hypothesise`: State falsifiable hypothesis with data specification
  - `Translate`: Convert analysis results to 3D feature layer proposals
- **Information Access**: Sees raw data and file contents, but NOT code results or scores
- **Location**: `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/NSL2-geology-task/src/harness/`

#### 2. Coding Agent
- **Purpose**: Stateless programmer that implements analysis code based on hypotheses
- **Phases**: `Code` - Write Python analysis scripts using polars/scipy/numpy
- **Information Access**: Sees data schemas and hypotheses, but NOT raw data values or scores
- **Isolation**: Cannot cherry-pick data to inflate results

#### 3. Framework (Automated)
- **Purpose**: Executes code and computes objective scores
- **Phases**: `Evaluate` - Run BIC scoring on ridge regression with depth-stratified CV
- **Scoring**: Uses joint prediction model where each layer predicts all other layers
- **Location**: `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp/voxel_features/scoring.py`

#### 4. Rewriting Agent
- **Purpose**: Creates training data and knowledge graph nodes from completed experiments
- **Phases**: `Rewrite` - Transform experiment into prompt/response pairs
- **Information Access**: Sees hypotheses, code, results, and scores, but NOT raw data
- **Output**: Training data + knowledge graph nodes for crossbreeding

### Docker Infrastructure

The system runs in isolated Docker containers to enforce information boundaries:

#### Container Setup (`@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/NSL2-geology-task/docker/feature-hypothesis-compose/docker-compose.yml`)

1. **agent**: Hypothesis and rewriting agents
   - Network: `agent-net` (internal only)
   - No access to raw data or analysis environment

2. **vfm** (voxel-features-mcp): Voxel store and scoring server
   - Network: `task-net` (internal only)  
   - Environment:
     - `VFM_STORE_PATH`: Voxel feature storage
     - `VFM_KG_PATH`: Knowledge graph storage
   - Volumes: Persistent feature and knowledge stores

3. **analysis**: Coding agent execution environment
   - Network: `task-net` (internal only)
   - Volumes: Read-only access to geological datasets
   - Workspace: Temporary execution space

### MCP Server Architecture

The system uses **Model Context Protocol (MCP)** servers to provide standardized tool interfaces:

#### Voxel Features MCP Server (`@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp/`)

**Core Components:**
- `VoxelStore`: 3D feature layer management with versioning
- `SpatialVoxelStore`: Geographic coordinate handling
- `KnowledgeGraph`: Experiment tracking and crossbreeding selection
- `Scoring`: BIC computation on ridge regression

**Tool Categories:**
1. **Feature Tools** (`feature_tools.py`): CRUD operations for feature layers
2. **Scoring Tools** (`scoring_tools.py`): BIC evaluation, mutual information, layer assessment
3. **Experiment Tools** (`experiment_tools.py`): Record experiments, track lineage, export training data
4. **Spatial Tools** (`spatial_tools.py`): Coordinate conversion, spatial queries

## Information-Theoretic Scoring System

### BIC on Ridge Regression

The core insight: use a deliberately simple linear model to force agent intelligence.

```python
BIC = n·ln(MSE) + k·ln(n)
      ^^^^^^^^^^^   ^^^^^^^
      fit quality   complexity penalty
```

**Why Ridge Regression?**
- Linear by design: Forces compositional thinking
- Handles multicollinearity: Geological features often correlate  
- Fast and stable: Scales to many features
- Clear learning signal: Agent must engineer explicit features

### Depth-Stratified Cross-Validation

```python
def _create_depth_folds(shape, n_folds=5):
    """Hold out entire depth slices for CV."""
    # Tests exploration-relevant generalization: surface → subsurface
```

This validation strategy tests what matters for geological exploration: **can surface measurements predict subsurface geology?**

### Joint Prediction Model

Each feature layer must help predict all other layers:

```python
def joint_prediction_score(layer_values, shape, n_folds=5):
    """Each layer ~ all other layers via ridge regression."""
    for i in range(n_layers):
        X = all_layers_except_i  
        y = layer_i
        mse[i] = ridge_cv_mse(X, y, depth_folds)
```

**Admission Rule**: Accept new layer if BIC improves (delta < 0)

## Knowledge Accumulation & Crossbreeding

### Experiment Records

```python
@dataclass
class ExperimentRecord:
    hypothesis: str              # "Au correlates with Y anomalies"
    rationale: str               # Why this might be true  
    code_executed: str           # Analysis script
    result_summary: str          # "p<0.01, r²=0.34"
    feature_layer_name: str      # "au_y_ratio"
    bic_delta: float            # -127.3 (improvement)
    admitted: bool              # True (improved model)
    parent_experiments: list    # Crossbreeding lineage
```

### Crossbreeding Process

After accumulating successful features, agents propose interactions:

```
"Given that:
 1. 'Au correlates with Y' (au_y_ratio layer admitted)
 2. 'Faults correlate with Cu' (fault_cu_proximity layer admitted)
 
What interaction feature might capture their joint behavior?"
```

**Selection Criteria**: Prefer pairs with low mutual information (orthogonal features) to maximize information gain.

## Configuration & Orchestration

### Main Configuration (`@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/NSL2-geology-task/config/config-feature-hypothesis-aiq.toml`)

**Key Parameters:**
- `model_name`: LLM backend (supports OpenRouter, vLLM)
- `container_ids`: Docker container mappings
- `docker_compose_dir`: Container orchestration location
- `dataset_dir`: Geological data source
- `max_episodes`: Training data generation limit
- `target_training_rows`: Desired training examples

**Task Configuration:**
```toml
[task]
class = "tasks.feature_hypothesis.FeatureHypothesisTask"

[task.config]
docker_compose_dir = "./docker/feature-hypothesis-compose"
dataset_dir = "../Coe Fairbairn"  # Your geological dataset
store_dir = "./data/feature-hypothesis/store"
kg_dir = "./data/feature-hypothesis/knowledge"
```

## Dataset Requirements

Expected geological data format:
```
YourDataset/
  amalgamated_csvs/
    geochemDrillhole.csv    # 3D coordinates + assay values
    geochemSurface.csv      # Surface samples
    tenements.csv           # Geographic boundaries
```

**Required Columns:**
- 3D coordinates: longitude, latitude, depth
- Assay values: Au, Cu, REE elements, etc.
- Sample metadata: drill IDs, lithology, etc.

## Running the System

### Setup Process

```bash
# 1. Install uv dependency manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup environment (from repository root)
./setup_uv_env.sh

# 3. Build Docker images
cd NSL2-geology-task
docker compose -f docker/feature-hypothesis-compose/docker-compose.yml build

# 4. Build harness images  
uv run python scripts/build_harness_images.py \
  --config config/config-feature-hypothesis-aiq.toml
```

### Running Episodes

```bash
cd NSL2-geology-task

# Single episode
uv run python scripts/run_episode.py \
  --config config/config-feature-hypothesis-aiq.toml

# Full training data generation
uv run python scripts/generate.py \
  --config config/config-feature-hypothesis-aiq.toml
```

### Crossbreeding Mode

After accumulating successful experiments:
```bash
# Test crossbreeding workflow
./test_crossbreed_workflow.sh
```

## Key Implementation Files

### Core Task Implementation
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/NSL2-geology-task/tasks/feature_hypothesis.py`: Main workflow orchestration
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/NSL2-geology-task/src/harness/`: Agent harness framework

### Voxel & Scoring System  
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp/voxel_features/store.py`: Voxel storage and versioning
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp/voxel_features/scoring.py`: BIC computation on ridge regression
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp/voxel_features/knowledge_graph.py`: Experiment tracking

### MCP Tools
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/voxel-features-mcp/voxel_features/mcp/tools/`: Feature CRUD, scoring, experiments, spatial operations

### Architecture Documentation
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/VOXEL_FEATURE_HYPOTHESIS_ARCHITECTURE.md`: Detailed design philosophy
- `@/home/jen/Desktop/GeoNSL_monorepo-pre-kazakh-complete/README.md`: Quick start guide

## Understanding the Learning Signal

### What Gets Rewarded
- Features that improve joint predictability (reduce BIC)
- Compositional representations (linear combinations work well)
- Exploration-relevant patterns (surface predicts subsurface)
- Novel but useful combinations (crossbreeding successful features)

### What Gets Penalized  
- Redundant features (already captured by existing layers)
- Overfitting (BIC complexity penalty)
- Noise (doesn't improve cross-validated prediction)
- Gaming attempts (information boundaries prevent this)

### Training Data Generation
Every experiment produces:
1. **Prompt**: "The data leads us to hypothesize that..."
2. **Response**: "Upon trying X we discovered that Y."

Both successful AND failed experiments become training data, with successful ones also added to the knowledge graph for crossbreeding.

## Future Extensions

The architecture supports several extension points:

1. **New Agent Types**: Add specialized agents for specific geological domains
2. **Alternative Scoring**: Experiment with different information-theoretic criteria
3. **Advanced Crossbreeding**: Implement more sophisticated feature combination strategies
4. **Multi-Dataset Learning**: Extend to handle multiple geological surveys simultaneously
5. **Real-Time Integration**: Connect to live geological data streams

## Troubleshooting

### Common Issues
- **Container networking**: Ensure Docker networks allow MCP communication
- **Memory limits**: Geological datasets can be large; adjust container memory
- **Model timeouts**: Some hypotheses require longer analysis; tune timeout values
- **Storage space**: Feature stores grow over time; monitor disk usage

### Debug Commands
```bash
# Check container logs
docker logs feature-hypothesis-compose-vfm-1

# Inspect MCP server status
docker exec -it feature-hypothesis-compose-vfm-1 python -c "from voxel_features.mcp.server import _get_store; print(_get_store())"

# View current feature layers
docker exec -it feature-hypothesis-compose-vfm-1 python -m voxel_features.cli list-layers
```

---

This system represents a novel approach to geological AI: rather than training on human-labeled features, it discovers informative representations through systematic hypothesis testing, creating a self-supervised learning loop that mimics how expert geologists develop understanding through exploration and experimentation.
