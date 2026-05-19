"""BIC-based scoring for feature layers using ridge regression.

The scoring criterion determines whether a new feature layer improves
the joint predictive model (compression + generalization).

Core idea: Each layer should help predict all other layers.
We use ridge regression with depth-stratified CV to test this.

Scoring:
- Fit multivariate ridge: each layer ~ all other layers
- Evaluate via cross-validated MSE (depth folds for exploration relevance)
- Apply BIC penalty for model complexity
- Admission: CV-MSE improvement > BIC penalty

This is essentially model selection for a "world model" where
features are nodes in a dependency graph. Good features create
useful shortcuts; redundant features add complexity without benefit.

Mutual Information:
- Measures shared information between layers
- Used for crossbreeding pair selection (prefer orthogonal pairs)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from voxel_features.store import VoxelStore


def _entropy_continuous(values: np.ndarray, n_bins: int = 50) -> float:
    """Estimate entropy of continuous values using histogram binning."""
    # Remove NaN values
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return 0.0
    
    # Histogram-based entropy estimation
    hist, _ = np.histogram(valid, bins=n_bins, density=True)
    # Avoid log(0)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    
    # Entropy in bits
    bin_width = (valid.max() - valid.min()) / n_bins
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) * bin_width
    return max(0.0, entropy)


def _entropy_discrete(values: np.ndarray) -> float:
    """Compute entropy of discrete/categorical values."""
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return 0.0
    
    # Count unique values
    _, counts = np.unique(valid, return_counts=True)
    probs = counts / counts.sum()
    
    # Entropy in bits
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return max(0.0, entropy)


def _joint_entropy_continuous(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bins: int = 30,
) -> float:
    """Estimate joint entropy of two continuous variables."""
    # Align and remove NaN
    mask = ~(np.isnan(values_a) | np.isnan(values_b))
    a = values_a[mask]
    b = values_b[mask]
    
    if len(a) < 2:
        return 0.0
    
    # 2D histogram
    hist, _, _ = np.histogram2d(a, b, bins=n_bins, density=True)
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0.0
    
    bin_area = ((a.max() - a.min()) / n_bins) * ((b.max() - b.min()) / n_bins)
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) * bin_area
    return max(0.0, entropy)


def compute_layer_entropy(values: np.ndarray, dtype: str) -> float:
    """Compute entropy of a single layer in bits."""
    flat = values.flatten()
    
    if dtype in ("categorical", "boolean"):
        return _entropy_discrete(flat)
    else:
        return _entropy_continuous(flat)


def mutual_information(
    store: VoxelStore,
    layer_a: str,
    layer_b: str,
) -> float:
    """
    Compute mutual information between two layers.
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Returns bits of shared information.
    """
    values_a = store.get_layer_values(layer_a).flatten()
    values_b = store.get_layer_values(layer_b).flatten()
    
    layer_a_obj = store.get_layer(layer_a)
    layer_b_obj = store.get_layer(layer_b)
    
    h_a = compute_layer_entropy(values_a, layer_a_obj.dtype)
    h_b = compute_layer_entropy(values_b, layer_b_obj.dtype)
    h_ab = _joint_entropy_continuous(values_a, values_b)
    
    # MI = H(A) + H(B) - H(A,B)
    mi = h_a + h_b - h_ab
    return max(0.0, mi)  # MI is non-negative


# =============================================================================
# Joint Prediction Scoring (Ridge Regression + BIC)
# =============================================================================

def _create_depth_folds(
    shape: tuple[int, int, int],
    n_folds: int = 5,
) -> list[np.ndarray]:
    """
    Create depth-stratified CV folds.
    
    Each fold holds out one or more depth slices.
    This tests whether surface data can predict subsurface.
    """
    nx, ny, nz = shape
    
    # Assign each voxel to a fold based on its depth slice
    folds = []
    depth_indices = np.arange(nz)
    fold_assignments = np.array_split(depth_indices, min(n_folds, nz))
    
    for fold_depths in fold_assignments:
        mask = np.zeros(shape, dtype=bool)
        for z in fold_depths:
            mask[:, :, z] = True
        folds.append(mask.flatten())
    
    return folds


def _ridge_cv_mse(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    alpha: float = 1.0,
) -> float:
    """
    Compute cross-validated MSE for ridge regression.
    
    Args:
        X: Feature matrix (n_voxels, n_features)
        y: Target vector (n_voxels,)
        folds: List of boolean masks for held-out voxels
        alpha: Ridge regularization strength
    
    Returns:
        Mean squared error averaged across folds
    """
    from scipy.linalg import solve
    
    fold_mses = []
    
    for fold_mask in folds:
        train_mask = ~fold_mask
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[fold_mask]
        y_test = y[fold_mask]
        
        # Remove NaN rows
        train_valid = ~(np.any(np.isnan(X_train), axis=1) | np.isnan(y_train))
        test_valid = ~(np.any(np.isnan(X_test), axis=1) | np.isnan(y_test))
        
        if train_valid.sum() < X.shape[1] + 1 or test_valid.sum() < 1:
            continue
        
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        
        # Ridge regression: (X'X + αI)β = X'y
        n_features = X_train.shape[1]
        XtX = X_train.T @ X_train + alpha * np.eye(n_features)
        Xty = X_train.T @ y_train
        
        try:
            beta = solve(XtX, Xty, assume_a='pos')
            y_pred = X_test @ beta
            mse = np.mean((y_test - y_pred) ** 2)
            fold_mses.append(mse)
        except Exception:
            continue
    
    return np.mean(fold_mses) if fold_mses else float('inf')


def _compute_bic(
    n_samples: int,
    n_params: int,
    mse: float,
) -> float:
    """
    Compute BIC for Gaussian model.
    
    BIC = n*ln(MSE) + k*ln(n)
    
    Lower is better.
    """
    if mse <= 0 or n_samples <= 0:
        return float('inf')
    
    return n_samples * np.log(mse) + n_params * np.log(n_samples)


def joint_prediction_score(
    layer_values: list[np.ndarray],
    shape: tuple[int, int, int],
    n_folds: int = 5,
    alpha: float = 1.0,
) -> dict:
    """
    Compute joint prediction score for a set of layers.
    
    Each layer is predicted from all other layers using ridge regression.
    Score combines CV-MSE across all layers with BIC penalty.
    
    Args:
        layer_values: List of flattened layer arrays
        shape: Original (nx, ny, nz) shape for depth folds
        n_folds: Number of CV folds
        alpha: Ridge regularization
    
    Returns:
        dict with:
            - total_cv_mse: Sum of CV-MSE across all prediction tasks
            - bic: BIC score for the joint model
            - per_layer_mse: Dict mapping layer index to MSE
    """
    n_layers = len(layer_values)
    n_voxels = layer_values[0].shape[0] if layer_values else 0
    
    if n_layers < 2:
        return {
            "total_cv_mse": 0.0,
            "bic": 0.0,
            "per_layer_mse": {},
        }
    
    # Create depth-stratified folds
    folds = _create_depth_folds(shape, n_folds)
    
    # Stack all layers into matrix
    all_layers = np.column_stack(layer_values)
    
    # Predict each layer from all others
    per_layer_mse = {}
    total_mse = 0.0
    
    for i in range(n_layers):
        # Features: all layers except i
        X = np.delete(all_layers, i, axis=1)
        # Add intercept
        X = np.column_stack([np.ones(n_voxels), X])
        y = layer_values[i]
        
        mse = _ridge_cv_mse(X, y, folds, alpha)
        per_layer_mse[i] = mse
        if mse < float('inf'):
            total_mse += mse
    
    # BIC penalty: k = n_layers * (n_layers - 1) coefficients + n_layers intercepts
    n_params = n_layers * n_layers
    avg_mse = total_mse / n_layers if n_layers > 0 else 0
    bic = _compute_bic(n_voxels, n_params, avg_mse) if avg_mse > 0 else float('inf')
    
    return {
        "total_cv_mse": total_mse,
        "bic": bic,
        "per_layer_mse": per_layer_mse,
    }


# =============================================================================
# Legacy MDL Scoring (kept for comparison)
# =============================================================================

def _model_complexity(n_layers: int, n_voxels: int) -> float:
    """
    Compute model complexity term for MDL.
    
    This is the "cost" of describing the model structure itself.
    More layers = more complexity.
    """
    if n_layers == 0:
        return 0.0
    
    # Cost to encode number of layers + basic structure per layer
    # Using a simple log-based cost model
    structure_cost = math.log2(n_layers + 1) * n_layers
    
    # Each layer has some overhead (name, dtype, metadata)
    overhead_per_layer = 32  # bits for basic metadata
    
    return structure_cost + (overhead_per_layer * n_layers)


def _data_cost(store: VoxelStore) -> float:
    """
    Compute data cost term for MDL.
    
    This is the cost of encoding all layer values given the model.
    Correlated layers can be compressed together.
    """
    layer_names = store.layer_names
    if not layer_names:
        return 0.0
    
    # Start with sum of individual layer entropies
    total_entropy = 0.0
    for name in layer_names:
        values = store.get_layer_values(name)
        layer = store.get_layer(name)
        entropy = compute_layer_entropy(values, layer.dtype)
        total_entropy += entropy
    
    # Subtract mutual information between layers (joint coding benefit)
    # This represents the compression gain from correlated layers
    if len(layer_names) > 1:
        mi_reduction = 0.0
        for i, name_a in enumerate(layer_names):
            for name_b in layer_names[i+1:]:
                mi = mutual_information(store, name_a, name_b)
                mi_reduction += mi
        total_entropy -= mi_reduction
    
    # Scale by number of voxels
    n_voxels = store.grid.n_voxels
    return max(0.0, total_entropy * n_voxels)


def compute_mdl(store: VoxelStore) -> float:
    """
    Compute Minimum Description Length of the voxel store.
    
    MDL = model_complexity + data_cost
    
    Returns total bits needed to describe the store.
    Lower is better (more compressed representation).
    """
    n_layers = len(store.layer_names)
    n_voxels = store.grid.n_voxels
    
    complexity = _model_complexity(n_layers, n_voxels)
    data = _data_cost(store)
    
    return complexity + data


def evaluate_new_layer(
    store: VoxelStore,
    layer_name: str,
    layer_values: np.ndarray,
    layer_dtype: str,
    *,
    ridge_alpha: float = 1.0,
    n_folds: int = 5,
) -> dict:
    """
    Evaluate adding a new layer to the store.
    
    Uses BIC on depth-stratified CV ridge regression:
    1. Compute joint prediction score WITHOUT the new layer
    2. Compute joint prediction score WITH the new layer
    3. Admit if BIC improves (lower is better)
    
    Args:
        store: VoxelStore to evaluate against
        layer_name: Name for the new layer
        layer_values: 3D array of values
        layer_dtype: Data type ("float", "categorical", "boolean")
        ridge_alpha: Ridge regularization strength
        n_folds: Number of CV folds (depth-stratified)
    
    Returns:
        dict with:
            - bic_before/after/delta: BIC scores
            - cv_mse_before/after/delta: Cross-validated MSE
            - mutual_info: MI with existing layers
            - admitted: whether layer improves BIC
            - predicted_value: BIC improvement (higher = better hypothesis)
    """
    existing_layers = list(store.layer_names)
    grid_shape = store.grid.shape
    
    # Flatten new layer values
    new_values_flat = layer_values.flatten()
    
    # Get existing layer values (flattened)
    existing_values = [store.get_layer_values(n).flatten() for n in existing_layers]
    
    # Score WITHOUT new layer
    if len(existing_values) >= 2:
        score_before = joint_prediction_score(
            existing_values, grid_shape, n_folds, ridge_alpha
        )
    else:
        score_before = {"bic": 0.0, "total_cv_mse": 0.0, "per_layer_mse": {}}
    
    # Score WITH new layer
    all_values = existing_values + [new_values_flat]
    score_after = joint_prediction_score(
        all_values, grid_shape, n_folds, ridge_alpha
    )
    
    # BIC delta (negative = improved)
    bic_before = score_before["bic"]
    bic_after = score_after["bic"]
    bic_delta = bic_after - bic_before
    
    cv_mse_before = score_before["total_cv_mse"]
    cv_mse_after = score_after["total_cv_mse"]
    cv_mse_delta = cv_mse_after - cv_mse_before
    
    # Admission: BIC improved (delta < 0)
    admitted = bic_delta < 0
    
    if admitted:
        # Add the layer
        store.add_layer(
            name=layer_name,
            values=layer_values,
            dtype=layer_dtype,
        )
        
        # Compute MI with existing layers
        mi_scores = {}
        for other_name in existing_layers:
            mi_scores[other_name] = mutual_information(store, layer_name, other_name)
        
        store.update_layer_scores(layer_name, bic_delta, mi_scores)
    else:
        mi_scores = {}
    
    return {
        "bic_before": bic_before,
        "bic_after": bic_after,
        "bic_delta": bic_delta,
        "cv_mse_before": cv_mse_before,
        "cv_mse_after": cv_mse_after,
        "cv_mse_delta": cv_mse_delta,
        "mutual_info": mi_scores,
        "admitted": admitted,
        # For hypothesis agent training: BIC improvement (higher = better)
        "predicted_value": -bic_delta,
    }


def marginal_contribution(
    store: VoxelStore,
    layer_name: str,
    ridge_alpha: float = 1.0,
    n_folds: int = 5,
) -> float:
    """
    Compute how much removing this layer would change BIC.
    
    Positive value means the layer is contributing (BIC would increase without it).
    Negative value means the layer is hurting (BIC would decrease without it).
    """
    if layer_name not in store.layer_names:
        raise KeyError(f"Layer '{layer_name}' not found")
    
    all_layers = list(store.layer_names)
    grid_shape = store.grid.shape
    
    # BIC with all layers
    all_values = [store.get_layer_values(n).flatten() for n in all_layers]
    score_with = joint_prediction_score(all_values, grid_shape, n_folds, ridge_alpha)
    bic_with = score_with["bic"]
    
    # BIC without target layer
    layer_idx = all_layers.index(layer_name)
    values_without = [v for i, v in enumerate(all_values) if i != layer_idx]
    
    if len(values_without) >= 2:
        score_without = joint_prediction_score(values_without, grid_shape, n_folds, ridge_alpha)
        bic_without = score_without["bic"]
    else:
        bic_without = 0.0
    
    # Contribution = how much BIC increases if we remove it (positive = layer helps)
    return bic_without - bic_with
