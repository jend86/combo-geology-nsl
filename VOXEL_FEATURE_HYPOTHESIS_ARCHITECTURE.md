# Voxel Feature Hypothesis Architecture: Logic & Design

## Core Concept

We're building an AI system that discovers informative geological features through hypothesis-driven exploration. The key insight: **voxels are the source of truth, not graphs**.

### The Fundamental Inversion

```
OLD:  Graph (source of truth) → Voxel (derived, ephemeral)
NEW:  Voxel (source of truth) ← Feature Layers (agent-created, scored)
```

Instead of building stratigraphic models and deriving spatial fields, we work directly with 3D voxel space and let agents propose feature layers that compress our understanding of geology.

## Philosophy: Why This Design?

### 1. The Learning Signal Problem

**Challenge**: How do we reward an agent for discovering genuinely useful geological features?

**Solution**: Use a deliberately "dumb" linear scorer (ridge regression) that forces the agent to be smart:
- A random forest scorer would discover nonlinear patterns itself → no learning signal for agent
- A linear scorer can only leverage what the agent explicitly proposes → clear credit assignment

### 2. Information-Theoretic Foundation

We're building a compressed "world model" where feature layers form a dependency graph:
- **Good features**: Create useful shortcuts in the graph (e.g., "gold above clay" predicts multiple phenomena)
- **Bad features**: Add nodes without reducing description length (redundant or noisy)

This aligns with **Minimum Description Length** principles: the best model is the one that describes the data most compactly.

### 3. Preventing Reward Hacking

Agents are isolated to prevent gaming:

| Agent | Sees | Doesn't See | Why |
|-------|------|-------------|-----|
| **Hypothesis Agent** | Raw data, file contents | Code results, scores | Can't write hypotheses that are easy to "prove" |
| **Coding Agent** | Data schemas, hypothesis | Raw data values, scores | Can't cherry-pick data to inflate results |
| **Framework** | Everything | N/A | Automated, no LLM |
| **Rewriting Agent** | Hypothesis, code, results, scores | Raw data | Creates training data from verified experiments |

## The Scoring System: BIC on Ridge CV

### Why Ridge Regression?

1. **Linear by design**: Forces compositional thinking in the agent
2. **Handles multicollinearity**: Geological features often correlate
3. **Fast and stable**: Scales to many features without numerical issues

### Why Depth-Stratified Cross-Validation?

```python
def _create_depth_folds(shape, n_folds=5):
    """Hold out entire depth slices for CV."""
    # Tests what matters: can surface predict subsurface?
```

This tests the generalization that matters for exploration: **subsurface prediction from surface data**.

### Why BIC?

Bayesian Information Criterion balances fit vs. complexity:

```
BIC = n·ln(MSE) + k·ln(n)
      ^^^^^^^^^^   ^^^^^^^
      fit quality  complexity penalty
```

- **Prevents overfitting**: Adding redundant features increases k without reducing MSE enough
- **Model selection**: BIC ≈ MDL for Gaussian linear models
- **Clear admission rule**: Accept if BIC improves (delta < 0)

### The Joint Prediction Model

Each layer should help predict all other layers:

```python
def joint_prediction_score(layer_values, shape, n_folds=5, alpha=1.0):
    """Each layer ~ all other layers via ridge regression."""
    for i in range(n_layers):
        X = all_layers_except_i
        y = layer_i
        mse[i] = ridge_cv_mse(X, y, depth_folds, alpha)
```

This creates a mutual predictability criterion: features that don't participate in the joint model are noise.

## The Workflow: How It All Connects

### Phase Flow

```
HYPOTHESIS AGENT         CODING AGENT          FRAMEWORK           REWRITING AGENT
    Survey      ─────►                                     
    Hypothesise ─────►    Code        ─────►
    (wait)     ◄─────                ◄─────   (execute)
    Translate   ─────►                         Evaluate    ─────►   Rewrite
                                              (auto-score)
```

### Phase Details

**1. Survey**: Agent explores raw geological data, identifies opportunities
- Sees: CSV files, drill logs, assay data
- Output: List of candidate features to investigate

**2. Hypothesise**: Agent states falsifiable hypothesis with data specification
- Example: "High Au in drillholes correlates with elevated Y in surface samples"
- Must specify: files, columns, analysis type, expected output

**3. Code**: Different agent writes analysis code (blind to raw data)
- Sees: Only schemas and hypothesis
- Output: Python script using polars/scipy/numpy

**4. Translate**: Hypothesis agent converts results to 3D feature layer
- Decides: How to handle missing data, subsurface interpolation, dtype
- Output: Feature layer proposal

**5. Evaluate**: Framework computes BIC scores automatically
- Before/after BIC with new layer
- Admission decision: BIC improved?

**6. Rewrite**: Final agent creates knowledge graph node + training data
- Sees: All phases + scores (but not raw data)
- Output: Structured record for future crossbreeding

## Knowledge Accumulation

### Experiment Records

```python
@dataclass
class ExperimentRecord:
    hypothesis: str              # "Au correlates with Y anomalies"
    rationale: str               # Why this might be true
    code_executed: str           # The analysis script
    result_summary: str          # "p<0.01, r²=0.34"
    feature_layer_name: str      # "au_y_ratio"
    bic_delta: float            # -127.3 (improvement)
    admitted: bool              # True (improved model)
    parent_experiments: list    # For crossbreeding lineage
```

### Crossbreeding

After accumulating successful features, the system can propose interactions:

```
"Given that:
 1. 'Au correlates with Y' (au_y_ratio layer admitted)
 2. 'Faults correlate with Cu' (fault_cu_proximity layer admitted)
 
What interaction feature might capture their joint behavior?"
```

This creates an evolutionary process where successful features beget more sophisticated combinations.

## Information-Theoretic Intuition

We're approximating the question: **"How many bits to describe this geology?"**

1. **Each voxel** has values across multiple feature layers
2. **Good features** allow predicting some layers from others (compression)
3. **BIC** measures the total description length (model + residuals)
4. **Admission** means we found a more compact description

This is practical **Solomonoff Induction**: finding the shortest program (feature set) that generates the observations (voxel values).

## Why This Will Work

1. **Clear learning signal**: BIC improvement is unambiguous
2. **Compositional discovery**: Agent must propose explicit features
3. **Prevents overfitting**: Both via ridge regularization and BIC penalty
4. **Domain-relevant validation**: Depth-stratified CV mimics exploration
5. **Knowledge accumulation**: Successful experiments inform future hypotheses

The agent is incentivized to discover features that genuinely compress our understanding of geological relationships—exactly what human geologists do when they identify "indicator minerals" or "pathfinder elements".
