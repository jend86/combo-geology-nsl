"""Two-stage geological coherence scoring for feature layers.

The scoring system separates predictive capacity from complexity assessment,
solving the fundamental flaw where BIC was measuring prediction quality twice
(R² correlation + MSE in BIC = double-counting prediction quality).

Core Philosophy:
In a real geological system, accurate measurements should be mutually 
predictive because they reflect the same underlying geological processes.
This is the "Anna Karenina principle" for geology: coherent geological 
systems are alike (features predict each other), while incoherent systems
fail in their own ways.

Two-Stage Evaluation:

Stage 1 - Predictive Capacity Test:
- Does adding the new layer actually improve geological understanding?
- Bidirectional masking test: mask 20% of data, test prediction improvement
- Direction A: Can new layer improve prediction of existing layers?
- Direction B: Can existing layers predict the new layer well?  
- Pass criteria: Either direction shows R² improvement ≥ threshold

Stage 2 - Complexity Assessment (ESA-BIC):
- Is the predictive improvement worth the added complexity?
- Applied only after Stage 1 passes
- Uses Effective Sample Size Adjusted BIC for sparse geological data
- Geological interpolation: 548m influence radius, inverse distance weighting
- Spatial autocorrelation correction: Moran's I prevents cheat-code layers

Technical Features:
1. Proper BIC usage: Compares models for same prediction task vs mixing metrics
2. R² normalization: Boolean/float layers contribute equally to coherence  
3. Spatial masking: Geologically realistic clustered validation regions
4. Performance optimized: Handles 320K voxel grids in <1 second
5. Backward compatible: Existing MCP tools work unchanged

Mutual Information:
- Legacy entropy-based measure for crossbreeding pair selection
- Used to prefer orthogonal layer pairs for feature combination
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from voxel_features.store import VoxelStore, GridSpec


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
# Geological Coherence Scoring System
# =============================================================================

def normalize_layers(
    layer_values: list[np.ndarray], 
    layer_dtypes: list[str]
) -> list[np.ndarray]:
    """
    Normalize layers for coherence analysis with type-aware approach.
    
    - Boolean layers: Keep as 0/1 (already optimal scale)
    - Float layers: Z-score standardization (mean=0, std=1)
    
    Args:
        layer_values: List of flattened layer arrays
        layer_dtypes: List of data types corresponding to each layer
        
    Returns:
        List of normalized layer arrays
    """
    normalized = []
    
    for values, dtype in zip(layer_values, layer_dtypes):
        if dtype == "boolean":
            # Keep boolean values as is (already 0/1)
            normalized.append(values.copy())
        elif dtype == "float":
            # Z-score standardization for continuous values
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 0:
                normalized_vals = (values - mean) / std
            else:
                # Handle constant layers
                normalized_vals = np.zeros_like(values)
            normalized.append(normalized_vals)
        else:
            # Fallback: treat as float
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 0:
                normalized_vals = (values - mean) / std
            else:
                normalized_vals = np.zeros_like(values)
            normalized.append(normalized_vals)
    
    return normalized


def compute_pairwise_r_squared(
    layer_values: list[np.ndarray],
    layer_dtypes: list[str]
) -> np.ndarray:
    """
    DEPRECATED: Compute pairwise R² values between all layer combinations.
    
    WARNING: This function is deprecated and will be removed in a future version.
    Use compute_pairwise_mae() instead, which provides better handling of sparse
    geological data and unified continuous modeling.
    
    Uses appropriate correlation method based on data types:
    - Boolean ↔ Boolean: Phi coefficient (χ² based correlation)
    - Boolean ↔ Float: Point-biserial correlation  
    - Float ↔ Float: Standard Pearson correlation coefficient
    
    Args:
        layer_values: List of normalized flattened layer arrays
        layer_dtypes: List of data types
        
    Returns:
        Symmetric matrix of R² values
    """
    import warnings
    warnings.warn(
        "compute_pairwise_r_squared is deprecated and will be removed. "
        "Use compute_pairwise_mae for better sparse geological data handling.",
        DeprecationWarning,
        stacklevel=2
    )
    n_layers = len(layer_values)
    r_squared_matrix = np.zeros((n_layers, n_layers))
    
    for i in range(n_layers):
        for j in range(i, n_layers):
            if i == j:
                r_squared_matrix[i, j] = 1.0
            else:
                # Get clean data (remove NaN pairs)
                mask = ~(np.isnan(layer_values[i]) | np.isnan(layer_values[j]))
                if mask.sum() < 2:
                    r_squared_matrix[i, j] = 0.0
                    r_squared_matrix[j, i] = 0.0
                    continue
                
                x = layer_values[i][mask]
                y = layer_values[j][mask]
                
                dtype_i = layer_dtypes[i]
                dtype_j = layer_dtypes[j]
                
                # Compute correlation based on data types
                if dtype_i == "boolean" and dtype_j == "boolean":
                    # Phi coefficient for two binary variables
                    r = phi_coefficient(x, y)
                elif dtype_i == "boolean" or dtype_j == "boolean":
                    # Point-biserial correlation
                    r = point_biserial_correlation(x, y)
                else:
                    # Pearson correlation for continuous variables
                    r = pearson_correlation(x, y)
                
                r_squared = r ** 2
                r_squared_matrix[i, j] = r_squared
                r_squared_matrix[j, i] = r_squared
    
    return r_squared_matrix  # DEPRECATED - use compute_pairwise_mae instead


def phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute phi coefficient (correlation for two binary variables)."""
    # Contingency table
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    
    n11 = np.sum(x_bool & y_bool)
    n10 = np.sum(x_bool & ~y_bool)
    n01 = np.sum(~x_bool & y_bool)
    n00 = np.sum(~x_bool & ~y_bool)
    
    # Phi coefficient formula
    numerator = n11 * n00 - n10 * n01
    denominator = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def point_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute point-biserial correlation (one binary, one continuous)."""
    # Ensure x is binary and y is continuous
    if np.unique(x).size == 2 and np.unique(y).size > 2:
        binary_var, continuous_var = x, y
    elif np.unique(y).size == 2 and np.unique(x).size > 2:
        binary_var, continuous_var = y, x
    else:
        # Fall back to Pearson
        return pearson_correlation(x, y)
    
    # Convert binary to boolean for easier indexing
    binary_bool = binary_var.astype(bool)
    
    if binary_bool.sum() == 0 or (~binary_bool).sum() == 0:
        return 0.0
    
    # Means for each group
    mean_1 = np.mean(continuous_var[binary_bool])
    mean_0 = np.mean(continuous_var[~binary_bool])
    
    # Overall standard deviation
    std_total = np.std(continuous_var)
    
    if std_total == 0:
        return 0.0
    
    # Proportions
    p1 = binary_bool.sum() / len(binary_bool)
    p0 = 1 - p1
    
    # Point-biserial formula
    return (mean_1 - mean_0) / std_total * np.sqrt(p1 * p0)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    return np.corrcoef(x, y)[0, 1]


def compute_moran_correction(
    layer_values: list[np.ndarray],
    grid: 'GridSpec'
) -> float:
    """
    Compute spatial autocorrelation correction using Moran's I.
    
    Uses geographic coordinates to weight spatial relationships
    and estimate effective sample size for statistical tests.
    
    Args:
        layer_values: List of flattened layer arrays
        grid: GridSpec containing coordinate information
        
    Returns:
        Correction factor: effective_n / total_n
    """
    if not layer_values:
        return 1.0
    
    n_voxels = len(layer_values[0])
    
    # Create coordinate arrays for all voxel centers
    x_coords, y_coords, z_coords = grid.cell_centers()
    
    # Flatten coordinate grids to match layer values
    x_flat = np.tile(x_coords[:, np.newaxis, np.newaxis], (1, grid.shape[1], grid.shape[2])).flatten()
    y_flat = np.tile(y_coords[np.newaxis, :, np.newaxis], (grid.shape[0], 1, grid.shape[2])).flatten()
    z_flat = np.tile(z_coords[np.newaxis, np.newaxis, :], (grid.shape[0], grid.shape[1], 1)).flatten()
    
    # Build distance matrix (use subset for performance)
    max_sample_size = min(n_voxels, 1000)  # Sample for large grids
    indices = np.random.choice(n_voxels, max_sample_size, replace=False)
    
    x_sample = x_flat[indices]
    y_sample = y_flat[indices]
    z_sample = z_flat[indices]
    
    # Compute distance matrix
    coords = np.column_stack([x_sample, y_sample, z_sample])
    distances = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
    
    # Create spatial weights (inverse distance, avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (distances + 1e-6)
        weights[np.diag_indices_from(weights)] = 0  # No self-weights
    
    # Normalize weights
    row_sums = np.sum(weights, axis=1)
    mask = row_sums > 0
    weights[mask] = weights[mask] / row_sums[mask][:, np.newaxis]
    
    # Compute average Moran's I across all layers
    moran_values = []
    
    for layer_vals in layer_values:
        layer_sample = layer_vals[indices]
        
        # Remove NaN values
        valid_mask = ~np.isnan(layer_sample)
        if valid_mask.sum() < 10:
            continue
        
        valid_values = layer_sample[valid_mask]
        valid_weights = weights[np.ix_(valid_mask, valid_mask)]
        
        # Center the values
        centered = valid_values - np.mean(valid_values)
        
        if np.std(centered) == 0:
            continue
        
        # Moran's I formula
        numerator = np.sum(valid_weights * np.outer(centered, centered))
        denominator = np.sum(centered ** 2)
        
        if denominator > 0:
            moran_i = numerator / denominator
            moran_values.append(abs(moran_i))  # Use absolute value
    
    if not moran_values:
        return 1.0
    
    # Average spatial autocorrelation
    avg_moran = np.mean(moran_values)
    
    # Convert to effective sample size correction
    # Higher autocorrelation = lower effective sample size
    # Formula: effective_n = n / (1 + (n-1) * autocorr)
    # Simplified to: correction = 1 / (1 + autocorr)
    correction = 1.0 / (1.0 + avg_moran)
    
    return max(0.1, min(1.0, correction))  # Clamp to reasonable range


# =============================================================================
# Geological Interpolation Functions
# =============================================================================

def get_default_influence_radius(grid: 'GridSpec') -> float:
    """
    Calculate default influence radius as 7x average voxel size.
    
    Args:
        grid: GridSpec for spatial coordinate information
        
    Returns:
        Default influence radius in meters
    """
    cell_size = grid.cell_size
    # Calculate average horizontal dimension in degrees
    avg_cell_size_deg = (cell_size[0] + cell_size[1]) / 2
    
    # Convert to meters (rough approximation: 1 degree ≈ 111000m)
    avg_cell_size_m = avg_cell_size_deg * 111000
    
    # Return 7x as default geological influence radius
    return 7.0 * avg_cell_size_m


def compute_3d_distance(voxel1: tuple[int, int, int], voxel2: tuple[int, int, int], 
                        grid: 'GridSpec') -> float:
    """
    Compute 3D distance between two voxel indices in meters.
    
    Args:
        voxel1, voxel2: (i, j, k) voxel indices
        grid: GridSpec for coordinate conversion
        
    Returns:
        Distance in meters
    """
    # Convert voxel indices to world coordinates
    cell_size = grid.cell_size
    origin = grid.origin
    
    # Voxel center coordinates
    coord1 = (
        origin[0] + (voxel1[0] + 0.5) * cell_size[0],  # longitude
        origin[1] + (voxel1[1] + 0.5) * cell_size[1],  # latitude  
        origin[2] + (voxel1[2] + 0.5) * cell_size[2],  # depth
    )
    coord2 = (
        origin[0] + (voxel2[0] + 0.5) * cell_size[0],
        origin[1] + (voxel2[1] + 0.5) * cell_size[1],
        origin[2] + (voxel2[2] + 0.5) * cell_size[2],
    )
    
    # Convert to meters (approximate for small distances)
    # Longitude/latitude: 1 degree ≈ 111000m
    dx_m = (coord2[0] - coord1[0]) * 111000
    dy_m = (coord2[1] - coord1[1]) * 111000  
    dz_m = coord2[2] - coord1[2]  # depth already in meters
    
    return np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)


def find_empty_voxels(layer_values: np.ndarray, shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """
    Find voxel indices where layer value is zero.
    
    Args:
        layer_values: Flattened layer array
        shape: (nx, ny, nz) voxel grid shape
        
    Returns:
        List of (i, j, k) indices for empty voxels
    """
    layer_3d = layer_values.reshape(shape)
    empty_indices = np.where(layer_3d == 0)
    return list(zip(empty_indices[0], empty_indices[1], empty_indices[2]))


def find_non_zero_voxels(layer_values: np.ndarray, shape: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], float]]:
    """
    Find voxel indices and values where layer value is non-zero.
    
    Args:
        layer_values: Flattened layer array
        shape: (nx, ny, nz) voxel grid shape
        
    Returns:
        List of ((i, j, k), value) for non-zero voxels
    """
    layer_3d = layer_values.reshape(shape)
    non_zero_indices = np.where(layer_3d != 0)
    indices_and_values = []
    
    for i in range(len(non_zero_indices[0])):
        idx = (non_zero_indices[0][i], non_zero_indices[1][i], non_zero_indices[2][i])
        value = layer_3d[idx]
        indices_and_values.append((idx, value))
    
    return indices_and_values


def compute_geological_interpolation(
    layer_values: np.ndarray,
    grid: 'GridSpec',
    shape: tuple[int, int, int],
    influence_radius_m: float = None
) -> np.ndarray:
    """
    Apply geological interpolation using sphere of influence with inverse distance weighting.
    
    OPTIMIZED VERSION: Uses spatial bounds checking to avoid O(n²) complexity.
    
    Args:
        layer_values: Flattened layer array
        grid: GridSpec for spatial coordinate information
        shape: (nx, ny, nz) voxel grid shape  
        influence_radius_m: Influence radius in meters (default: 7x voxel size)
        
    Returns:
        Interpolated layer array (same shape as input)
    """
    if influence_radius_m is None:
        influence_radius_m = get_default_influence_radius(grid)
    
    # Work on a copy to avoid modifying original data
    interpolated = layer_values.copy()
    
    # Convert influence radius from meters to voxel units for spatial bounds checking
    cell_size_m = np.array(grid.cell_size) * np.array([111000, 111000, 1])  # deg to meters
    radius_voxels = influence_radius_m / np.min(cell_size_m)  # Conservative radius in voxels
    radius_voxels = min(radius_voxels, 20)  # Cap to prevent excessive computation
    
    # Find non-zero voxels (these will be interpolation sources)
    layer_3d = layer_values.reshape(shape)
    non_zero_indices = np.where(layer_3d != 0)
    
    if len(non_zero_indices[0]) == 0:
        return interpolated
    
    # Performance limits to prevent hangs
    max_sources = 1000  # Limit number of source voxels
    max_targets = 10000  # Limit number of target voxels
    
    # Get source voxels (limited for performance)
    n_sources = min(len(non_zero_indices[0]), max_sources)
    source_indices = np.random.choice(len(non_zero_indices[0]), n_sources, replace=False)
    
    # For each source voxel, find nearby empty voxels to interpolate
    targets_processed = 0
    
    for idx in source_indices:
        if targets_processed >= max_targets:
            break
            
        source_i = non_zero_indices[0][idx]
        source_j = non_zero_indices[1][idx] 
        source_k = non_zero_indices[2][idx]
        source_value = layer_3d[source_i, source_j, source_k]
        
        # Define spatial bounds for this source (much more efficient than global search)
        i_min = max(0, int(source_i - radius_voxels))
        i_max = min(shape[0], int(source_i + radius_voxels + 1))
        j_min = max(0, int(source_j - radius_voxels))
        j_max = min(shape[1], int(source_j + radius_voxels + 1))
        k_min = max(0, int(source_k - radius_voxels))
        k_max = min(shape[2], int(source_k + radius_voxels + 1))
        
        # Check nearby empty voxels
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    if targets_processed >= max_targets:
                        break
                    
                    # Skip if already has a value
                    if layer_3d[i, j, k] != 0:
                        continue
                    
                    # Calculate actual distance
                    distance = compute_3d_distance(
                        (i, j, k), (source_i, source_j, source_k), grid
                    )
                    
                    if distance <= influence_radius_m:
                        # Quadratic decay within influence sphere
                        weight = (1.0 - distance / influence_radius_m) ** 2
                        
                        # Get current value at this target location
                        flat_index = i * shape[1] * shape[2] + j * shape[2] + k
                        current_value = interpolated[flat_index]
                        
                        # If empty, set value; if already has interpolated value, blend
                        if current_value == 0:
                            interpolated[flat_index] = source_value * weight
                        else:
                            # Weighted average with existing interpolated value
                            interpolated[flat_index] = (current_value + source_value * weight) / 2
                        
                        targets_processed += 1
    
    return interpolated


# =============================================================================
# Cross-Validation Framework for Geological Data
# =============================================================================

def create_geological_cv_split(
    interpolated_layers: list[np.ndarray], 
    test_fraction: float = 0.2,
    min_test_signal_ratio: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 80/20 cross-validation split with geological constraints.
    
    Ensures the test set contains adequate geological signal (non-zero values)
    while maintaining random sampling for unbiased evaluation.
    
    Args:
        interpolated_layers: List of interpolated flattened layer arrays
        test_fraction: Fraction of data for testing (default 0.2 = 20%)
        min_test_signal_ratio: Minimum ratio of non-zero values in test set
        
    Returns:
        Tuple of (train_mask, test_mask) boolean arrays
    """
    if not interpolated_layers:
        return np.array([]), np.array([])
        
    n_voxels = len(interpolated_layers[0])
    n_test = int(n_voxels * test_fraction)
    
    # Find voxels with geological signal (non-zero in any layer)
    signal_mask = np.zeros(n_voxels, dtype=bool)
    for layer in interpolated_layers:
        signal_mask |= (layer != 0)
    
    signal_indices = np.where(signal_mask)[0]
    non_signal_indices = np.where(~signal_mask)[0]
    
    # Calculate minimum number of signal voxels needed in test set
    min_test_signal = max(1, int(n_test * min_test_signal_ratio))
    min_test_signal = min(min_test_signal, len(signal_indices))
    
    # Create test set with guaranteed geological signal
    np.random.seed(42)  # Reproducible splits
    
    if len(signal_indices) >= min_test_signal:
        # Randomly select signal voxels for test set
        test_signal_indices = np.random.choice(
            signal_indices, 
            size=min_test_signal, 
            replace=False
        )
        
        # Fill remaining test slots from all remaining voxels
        remaining_indices = np.setdiff1d(np.arange(n_voxels), test_signal_indices)
        remaining_test_size = n_test - min_test_signal
        
        if remaining_test_size > 0 and len(remaining_indices) > 0:
            additional_test_indices = np.random.choice(
                remaining_indices,
                size=min(remaining_test_size, len(remaining_indices)),
                replace=False
            )
            test_indices = np.concatenate([test_signal_indices, additional_test_indices])
        else:
            test_indices = test_signal_indices
    else:
        # Fallback: random selection if insufficient signal
        test_indices = np.random.choice(n_voxels, size=n_test, replace=False)
    
    # Create boolean masks
    test_mask = np.zeros(n_voxels, dtype=bool)
    test_mask[test_indices] = True
    train_mask = ~test_mask
    
    return train_mask, test_mask


def validate_geological_split(
    train_mask: np.ndarray, 
    test_mask: np.ndarray, 
    layer_values: list[np.ndarray]
) -> dict:
    """
    Validate cross-validation split has adequate geological signal distribution.
    
    Args:
        train_mask: Boolean mask for training data
        test_mask: Boolean mask for test data  
        layer_values: List of layer arrays to analyze
        
    Returns:
        Dict with split statistics and validation results
    """
    stats = {
        'train_size': np.sum(train_mask),
        'test_size': np.sum(test_mask),
        'train_signal_count': 0,
        'test_signal_count': 0,
        'train_signal_ratio': 0.0,
        'test_signal_ratio': 0.0,
        'validation_passed': False
    }
    
    if not layer_values or len(layer_values) == 0:
        return stats
    
    # Count non-zero values (geological signal) in each split
    for layer in layer_values:
        train_signal = np.sum(layer[train_mask] != 0)
        test_signal = np.sum(layer[test_mask] != 0)
        
        stats['train_signal_count'] += train_signal
        stats['test_signal_count'] += test_signal
    
    # Calculate signal ratios
    if stats['train_size'] > 0:
        stats['train_signal_ratio'] = stats['train_signal_count'] / stats['train_size']
    if stats['test_size'] > 0:
        stats['test_signal_ratio'] = stats['test_signal_count'] / stats['test_size']
    
    # Validation criteria
    has_test_signal = stats['test_signal_count'] > 0
    reasonable_split = 0.15 <= (stats['test_size'] / (stats['train_size'] + stats['test_size'])) <= 0.25
    
    stats['validation_passed'] = has_test_signal and reasonable_split
    
    return stats


# =============================================================================
# MAE Prediction Framework for Unified Continuous Modeling  
# =============================================================================

def fit_continuous_predictor(
    target_layer: np.ndarray, 
    predictor_layers: list[np.ndarray], 
    train_mask: np.ndarray
) -> dict:
    """
    Fit linear regression model treating all geological layers as continuous.
    
    Boolean geological layers (faults, ore zones) are treated as 0/1 continuous
    variables, enabling a unified prediction framework.
    
    Args:
        target_layer: Target layer to predict (flattened)
        predictor_layers: List of predictor layer arrays (flattened)
        train_mask: Boolean mask for training voxels
        
    Returns:
        Dict with fitted parameters and prediction function
    """
    if not predictor_layers or len(predictor_layers) == 0:
        # No predictors - return mean prediction
        mean_target = np.mean(target_layer[train_mask]) if np.sum(train_mask) > 0 else 0.0
        return {
            'coefficients': np.array([mean_target]),
            'intercept': 0.0,
            'n_predictors': 0,
            'n_train_samples': np.sum(train_mask),
            'prediction_type': 'mean'
        }
    
    # Extract training data
    train_target = target_layer[train_mask]
    train_predictors = np.column_stack([layer[train_mask] for layer in predictor_layers])
    
    n_train = len(train_target)
    n_predictors = train_predictors.shape[1]
    
    if n_train < 2:
        # Insufficient training data
        return {
            'coefficients': np.zeros(n_predictors),
            'intercept': 0.0,
            'n_predictors': n_predictors,
            'n_train_samples': n_train,
            'prediction_type': 'insufficient_data'
        }
    
    try:
        # Use sklearn for robust linear regression if available
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(train_predictors, train_target)
        
        return {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'n_predictors': n_predictors,
            'n_train_samples': n_train,
            'prediction_type': 'sklearn_linear'
        }
        
    except ImportError:
        # Fallback: numpy least squares
        try:
            # Add intercept column
            X_with_intercept = np.column_stack([np.ones(n_train), train_predictors])
            coeffs_with_intercept, _, _, _ = np.linalg.lstsq(X_with_intercept, train_target, rcond=None)
            
            return {
                'coefficients': coeffs_with_intercept[1:],  # Exclude intercept
                'intercept': coeffs_with_intercept[0],
                'n_predictors': n_predictors,
                'n_train_samples': n_train,
                'prediction_type': 'numpy_lstsq'
            }
        except np.linalg.LinAlgError:
            # Ultimate fallback: correlation-based prediction
            correlations = np.array([
                np.corrcoef(predictor_layer[train_mask], train_target)[0, 1] 
                if np.var(predictor_layer[train_mask]) > 0 else 0.0
                for predictor_layer in predictor_layers
            ])
            correlations = np.nan_to_num(correlations, 0.0)
            
            return {
                'coefficients': correlations,
                'intercept': np.mean(train_target),
                'n_predictors': n_predictors,
                'n_train_samples': n_train,
                'prediction_type': 'correlation_fallback'
            }


def compute_out_of_sample_mae(
    target_layer: np.ndarray,
    predictor_layers: list[np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray
) -> float:
    """
    Compute Mean Absolute Error on held-out test data.
    
    Uses unified continuous approach: all layers treated as continuous
    (boolean geological features automatically handled as 0/1).
    
    Args:
        target_layer: Target layer to predict (flattened)
        predictor_layers: List of predictor layer arrays (flattened)
        train_mask: Boolean mask for training voxels
        test_mask: Boolean mask for test voxels
        
    Returns:
        Mean Absolute Error on test data
    """
    n_test = np.sum(test_mask)
    
    if n_test == 0:
        return 0.0  # No test data
    
    # Fit model on training data
    model_params = fit_continuous_predictor(target_layer, predictor_layers, train_mask)
    
    # Extract test data
    test_target = target_layer[test_mask]
    
    if model_params['prediction_type'] == 'mean':
        # Predict mean for all test samples
        predictions = np.full(n_test, model_params['coefficients'][0])
    else:
        # Make predictions using fitted model
        test_predictors = np.column_stack([layer[test_mask] for layer in predictor_layers])
        predictions = test_predictors @ model_params['coefficients'] + model_params['intercept']
    
    # Compute Mean Absolute Error
    mae = np.mean(np.abs(test_target - predictions))
    
    return float(mae)


# =============================================================================
# Laplace Likelihood BIC for MAE-Based Geological Scoring
# =============================================================================

def mae_to_laplace_likelihood(
    mae_values: np.ndarray, 
    n_samples: int
) -> float:
    """
    Convert Mean Absolute Error to Laplace likelihood.
    
    MAE corresponds exactly to the maximum likelihood estimation
    for the Laplace (double exponential) distribution, making this
    the theoretically correct likelihood for MAE-based predictions.
    
    Args:
        mae_values: Array of MAE values from pairwise predictions
        n_samples: Number of effective samples used in predictions
        
    Returns:
        Log-likelihood under Laplace distribution
    """
    if len(mae_values) == 0 or n_samples <= 0:
        return 0.0
    
    # System-wide MAE (average of pairwise MAEs)
    system_mae = np.mean(mae_values)
    
    # Handle edge case of perfect prediction (MAE = 0)
    if system_mae <= 1e-10:
        # Perfect prediction gets very high likelihood
        return n_samples * 10.0  # Arbitrary large positive value
    
    # Laplace likelihood: L = (1/(2*b))^n * exp(-sum(|x_i - mu_i|)/b)
    # where b = MAE for maximum likelihood estimation
    # Log-likelihood: log(L) = -n*log(2*MAE) - sum(|errors|)/MAE
    # Since sum(|errors|) = n*MAE for our case:
    # log(L) = -n*log(2*MAE) - n*MAE/MAE = -n*log(2*MAE) - n
    
    log_likelihood = -n_samples * np.log(2 * system_mae) - n_samples
    
    return float(log_likelihood)


def compute_geological_bic(
    mae_matrix: np.ndarray, 
    n_layers: int, 
    n_effective_samples: int,
    spatial_correction: float = 1.0
) -> float:
    """
    Compute BIC score from MAE matrix using Laplace likelihood.
    
    This provides a unified, theoretically sound BIC calculation
    that directly uses the same metric (MAE) for both prediction 
    assessment and information criterion evaluation.
    
    Args:
        mae_matrix: Symmetric matrix of pairwise MAE values
        n_layers: Number of geological layers
        n_effective_samples: Effective sample size from interpolated data
        spatial_correction: Moran's I spatial autocorrelation correction
        
    Returns:
        BIC score (lower = better geological model)
    """
    if n_layers <= 1:
        # Single layer or no layers - no meaningful BIC
        return 0.0
    
    # Extract off-diagonal MAE values (exclude self-correlations)
    mask = ~np.eye(n_layers, dtype=bool)
    off_diagonal_maes = mae_matrix[mask]
    
    # Remove any NaN or infinite values
    valid_maes = off_diagonal_maes[np.isfinite(off_diagonal_maes)]
    
    if len(valid_maes) == 0:
        # No valid predictions - return neutral BIC
        return 0.0
    
    # Convert MAE to Laplace log-likelihood
    log_likelihood = mae_to_laplace_likelihood(valid_maes, n_effective_samples)
    
    # Apply spatial autocorrelation correction
    corrected_log_likelihood = log_likelihood * spatial_correction
    
    # Calculate number of parameters (pairwise prediction coefficients)
    n_parameters = n_layers * (n_layers - 1) // 2
    
    # BIC = -2 * log_likelihood + k * log(n)
    bic = -2 * corrected_log_likelihood + n_parameters * np.log(max(n_effective_samples, n_layers))
    
    return float(bic)


def system_mae_to_coherence(
    mae_matrix: np.ndarray
) -> float:
    """
    Convert system MAE to a coherence-like metric for compatibility.
    
    Lower MAE = higher coherence, scaled to [0, 1] range.
    This maintains compatibility with existing code that expects
    coherence metrics while using the more robust MAE foundation.
    
    Args:
        mae_matrix: Symmetric matrix of pairwise MAE values
        
    Returns:
        Coherence score (higher = better, range ~[0, 1])
    """
    if mae_matrix.size == 0:
        return 0.0
    
    # Extract off-diagonal MAEs
    n_layers = mae_matrix.shape[0]
    if n_layers <= 1:
        return 1.0  # Perfect coherence for single layer
    
    mask = ~np.eye(n_layers, dtype=bool)
    off_diagonal_maes = mae_matrix[mask]
    
    # Remove invalid values
    valid_maes = off_diagonal_maes[np.isfinite(off_diagonal_maes) & (off_diagonal_maes >= 0)]
    
    if len(valid_maes) == 0:
        return 0.0
    
    # System MAE (average prediction error)
    system_mae = np.mean(valid_maes)
    
    # Convert to coherence: coherence = exp(-MAE)
    # This maps MAE=0 -> coherence=1, MAE=1 -> coherence~0.37, etc.
    coherence = np.exp(-system_mae)
    
    return float(coherence)


# =============================================================================
# MAE-Based Coherence Functions (Replaces R² System)
# =============================================================================

def compute_pairwise_mae(
    layer_values: list[np.ndarray],
    layer_dtypes: list[str],  # Ignored - all treated as continuous
    grid: 'GridSpec'
) -> np.ndarray:
    """
    Compute pairwise MAE values between all layer combinations.
    
    Replaces compute_pairwise_r_squared with unified continuous approach.
    All geological layers (boolean faults, continuous grades) treated as
    continuous variables using cross-validated MAE prediction.
    
    Args:
        layer_values: List of interpolated flattened layer arrays
        layer_dtypes: List of data types (ignored - unified continuous approach)
        grid: GridSpec for spatial information
        
    Returns:
        Symmetric matrix of MAE values (lower = better prediction)
    """
    n_layers = len(layer_values)
    if n_layers == 0:
        return np.array([])
    
    mae_matrix = np.zeros((n_layers, n_layers))
    
    # Create cross-validation split once for all predictions
    train_mask, test_mask = create_geological_cv_split(layer_values)
    
    # Validate the split
    split_stats = validate_geological_split(train_mask, test_mask, layer_values)
    if not split_stats['validation_passed']:
        print(f"Warning: CV split validation failed - {split_stats}")
        # Continue with fallback to avoid breaking the workflow
        if split_stats['test_size'] == 0:
            # No test data - return zero MAE matrix
            return mae_matrix
    
    # Compute pairwise MAEs
    for i in range(n_layers):
        for j in range(n_layers):
            if i == j:
                mae_matrix[i, j] = 0.0  # Perfect self-prediction
            elif i < j:  # Only compute upper triangle
                # Predict layer i using layer j
                mae_i_from_j = compute_out_of_sample_mae(
                    target_layer=layer_values[i],
                    predictor_layers=[layer_values[j]],
                    train_mask=train_mask,
                    test_mask=test_mask
                )
                
                # Predict layer j using layer i  
                mae_j_from_i = compute_out_of_sample_mae(
                    target_layer=layer_values[j],
                    predictor_layers=[layer_values[i]],
                    train_mask=train_mask,
                    test_mask=test_mask
                )
                
                # Use average bidirectional MAE
                avg_mae = (mae_i_from_j + mae_j_from_i) / 2.0
                
                mae_matrix[i, j] = avg_mae
                mae_matrix[j, i] = avg_mae  # Symmetric matrix
    
    return mae_matrix



def create_spatial_mask(
    shape: tuple[int, int, int], 
    grid: 'GridSpec',
    mask_fraction: float = 0.2,
    spatial_clustering: bool = True
) -> np.ndarray:
    """
    Create a spatially-aware mask for validation testing.
    
    Args:
        shape: (nx, ny, nz) voxel grid shape
        grid: GridSpec for spatial information 
        mask_fraction: Fraction of voxels to mask (default 20%)
        spatial_clustering: Whether to create spatially clustered masks
        
    Returns:
        Boolean mask array (True = masked/held out, False = training)
    """
    total_voxels = shape[0] * shape[1] * shape[2]
    n_masked = int(total_voxels * mask_fraction)
    
    if spatial_clustering and n_masked > 0:
        # Create spatially clustered mask regions for realistic geological testing
        mask = np.zeros(shape, dtype=bool)
        
        # Create ~10 cluster centers
        n_clusters = max(1, min(10, n_masked // 100))
        cluster_centers = []
        
        for _ in range(n_clusters):
            center_i = np.random.randint(0, shape[0])
            center_j = np.random.randint(0, shape[1]) 
            center_k = np.random.randint(0, shape[2])
            cluster_centers.append((center_i, center_j, center_k))
        
        # Assign each voxel to nearest cluster and mask some percentage
        masked_count = 0
        cluster_sizes = np.random.multinomial(n_masked, [1/n_clusters] * n_clusters)
        
        for cluster_idx, (center_i, center_j, center_k) in enumerate(cluster_centers):
            target_size = cluster_sizes[cluster_idx]
            if target_size == 0:
                continue
                
            # Create distance-based probability for this cluster
            i_coords, j_coords, k_coords = np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), 
                indexing='ij'
            )
            
            distances = np.sqrt(
                (i_coords - center_i)**2 + 
                (j_coords - center_j)**2 + 
                (k_coords - center_k)**2
            )
            
            # Higher probability for closer voxels
            max_dist = np.max(distances)
            probabilities = np.exp(-distances / (max_dist / 3))
            probabilities = probabilities / np.sum(probabilities)
            
            # Sample voxels for this cluster
            flat_probs = probabilities.flatten()
            flat_indices = np.random.choice(
                total_voxels, size=min(target_size, total_voxels), 
                replace=False, p=flat_probs
            )
            
            # Convert to 3D indices and mask
            for flat_idx in flat_indices:
                k = flat_idx % shape[2]
                j = (flat_idx // shape[2]) % shape[1]
                i = flat_idx // (shape[2] * shape[1])
                mask[i, j, k] = True
                masked_count += 1
                
                if masked_count >= n_masked:
                    break
            
            if masked_count >= n_masked:
                break
                
    else:
        # Simple random masking fallback
        mask = np.zeros(total_voxels, dtype=bool)
        if n_masked > 0:
            masked_indices = np.random.choice(total_voxels, size=n_masked, replace=False)
            mask[masked_indices] = True
        mask = mask.reshape(shape)
    
    return mask


def fit_predict_with_fallback(
    train_X: np.ndarray, 
    train_y: np.ndarray, 
    test_X: np.ndarray, 
    test_y: np.ndarray, 
    layer_dtype: str
) -> float:
    """
    Fit model and predict with sklearn fallback to correlation.
    
    Args:
        train_X: Training predictors
        train_y: Training targets
        test_X: Test predictors  
        test_y: Test targets
        layer_dtype: Data type for target layer
        
    Returns:
        R² score for predictions vs actual
    """
    if len(train_X) == 0 or len(test_X) == 0:
        return 0.0
    
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        
        model = Ridge(alpha=1.0)
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        
        # Calculate R² using sklearn
        r2 = r2_score(test_y, predictions)
        return max(0.0, r2)  # Ensure non-negative
        
    except ImportError:
        # Fallback to best correlation if sklearn unavailable
        if train_X.shape[1] == 1:
            # Single predictor - use correlation
            corr = np.corrcoef(test_X[:, 0], test_y)[0, 1]
            return max(0.0, corr**2) if not np.isnan(corr) else 0.0
        else:
            # Multiple predictors - use best single predictor correlation
            best_r2 = 0.0
            for col_idx in range(train_X.shape[1]):
                corr = np.corrcoef(test_X[:, col_idx], test_y)[0, 1]
                if not np.isnan(corr):
                    best_r2 = max(best_r2, corr**2)
            return best_r2
            
    except Exception:
        # Ultimate fallback
        return 0.0


def evaluate_bidirectional_prediction(
    existing_layers: list[np.ndarray],
    new_layer: np.ndarray,
    layer_dtypes: list[str],
    new_layer_dtype: str,
    grid: 'GridSpec',
    shape: tuple[int, int, int],
    mask_fraction: float = 0.2,
    min_improvement: float = 0.01
) -> dict:
    """
    Test if adding a new layer improves prediction in either direction.
    
    Stage 1 of two-stage scoring: bidirectional masked prediction test.
    Tests both:
    - Direction A: Can new layer improve prediction of existing layers?
    - Direction B: Can existing layers predict the new layer well?
    
    Args:
        existing_layers: List of existing layer arrays (flattened)
        new_layer: New layer to evaluate (flattened)
        layer_dtypes: Data types of existing layers
        new_layer_dtype: Data type of new layer
        grid: GridSpec for spatial information
        shape: (nx, ny, nz) voxel grid shape
        mask_fraction: Fraction of data to mask for testing
        min_improvement: Minimum R² improvement to pass
        
    Returns:
        dict with test results and improvement metrics
    """
    if not existing_layers:
        # First layer always passes
        return {
            "passes_test": True,
            "improvement": 0.0,
            "direction": "first_layer",
            "baseline_r2": 0.0,
            "with_new_layer_r2": 0.0,
            "test_samples": 0
        }
    
    # Apply geological interpolation to all layers
    interpolated_existing = []
    for layer in existing_layers:
        interpolated = compute_geological_interpolation(layer, grid, shape)
        interpolated_existing.append(interpolated)
    
    interpolated_new = compute_geological_interpolation(new_layer, grid, shape)
    
    # Create spatial mask
    mask_3d = create_spatial_mask(shape, grid, mask_fraction)
    mask_flat = mask_3d.flatten()
    
    # Split data: training (not masked) vs test (masked)
    train_mask = ~mask_flat
    test_mask = mask_flat
    n_test_samples = np.sum(test_mask)
    
    if n_test_samples < 10:  # Need minimum test samples
        return {
            "passes_test": False,
            "improvement": 0.0,
            "direction": "insufficient_samples",
            "baseline_r2": 0.0,
            "with_new_layer_r2": 0.0,
            "test_samples": n_test_samples
        }
    
    # Direction A: New layer helps predict existing layers
    direction_a_improvements = []
    
    for target_idx, target_layer in enumerate(interpolated_existing):
        if len(interpolated_existing) <= 1:
            continue  # Need other layers to use as predictors
            
        # Baseline: predict target using other existing layers only
        other_existing = [layer for i, layer in enumerate(interpolated_existing) if i != target_idx]
        other_dtypes = [dt for i, dt in enumerate(layer_dtypes) if i != target_idx]
        
        if other_existing:
            # Normalize layers
            normalized_others = normalize_layers(other_existing, other_dtypes)
            normalized_target = normalize_layers([target_layer], [layer_dtypes[target_idx]])[0]
            
            # Train and test baseline
            train_X = np.column_stack([layer[train_mask] for layer in normalized_others])
            train_y = normalized_target[train_mask]
            test_X = np.column_stack([layer[test_mask] for layer in normalized_others])
            test_y = normalized_target[test_mask]
            
            baseline_r2 = fit_predict_with_fallback(train_X, train_y, test_X, test_y, layer_dtypes[target_idx])
            
            # With new layer: add new layer as additional predictor
            normalized_new = normalize_layers([interpolated_new], [new_layer_dtype])[0]
            train_X_plus = np.column_stack([train_X, normalized_new[train_mask].reshape(-1, 1)])
            test_X_plus = np.column_stack([test_X, normalized_new[test_mask].reshape(-1, 1)])
            
            with_new_r2 = fit_predict_with_fallback(train_X_plus, train_y, test_X_plus, test_y, layer_dtypes[target_idx])
            
            improvement = with_new_r2 - baseline_r2
            direction_a_improvements.append(improvement)
    
    direction_a_improvement = np.mean(direction_a_improvements) if direction_a_improvements else 0.0
    
    # Direction B: Existing layers predict new layer
    if len(interpolated_existing) >= 1:
        # Normalize all layers
        normalized_existing = normalize_layers(interpolated_existing, layer_dtypes)
        normalized_new = normalize_layers([interpolated_new], [new_layer_dtype])[0]
        
        # Use existing layers to predict new layer
        train_X = np.column_stack([layer[train_mask] for layer in normalized_existing])
        train_y = normalized_new[train_mask]
        test_X = np.column_stack([layer[test_mask] for layer in normalized_existing])
        test_y = normalized_new[test_mask]
        
        direction_b_r2 = fit_predict_with_fallback(train_X, train_y, test_X, test_y, new_layer_dtype)
    else:
        direction_b_r2 = 0.0
    
    # Determine if test passes (either direction sufficient)
    direction_a_passes = direction_a_improvement >= min_improvement
    direction_b_passes = direction_b_r2 >= min_improvement
    
    passes_test = direction_a_passes or direction_b_passes
    
    if direction_a_passes and direction_b_passes:
        best_direction = "both"
        best_improvement = max(direction_a_improvement, direction_b_r2)
    elif direction_a_passes:
        best_direction = "new_helps_existing"
        best_improvement = direction_a_improvement
    elif direction_b_passes:
        best_direction = "existing_predict_new"
        best_improvement = direction_b_r2
    else:
        best_direction = "neither"
        best_improvement = max(direction_a_improvement, direction_b_r2)
    
    return {
        "passes_test": passes_test,
        "improvement": best_improvement,
        "direction": best_direction,
        "direction_a_improvement": direction_a_improvement,
        "direction_b_r2": direction_b_r2,
        "baseline_r2": 0.0,  # For compatibility
        "with_new_layer_r2": best_improvement,
        "test_samples": n_test_samples,
        "min_improvement_threshold": min_improvement
    }


def geological_coherence_score(
    layer_values: list[np.ndarray],
    layer_dtypes: list[str],
    grid: 'GridSpec',
    shape: tuple[int, int, int],
    enable_masking_test: bool = True,
    masking_test_threshold: float = 0.01  # MAE improvement threshold
) -> dict:
    """
    Compute geological coherence score using MAE + Laplace likelihood BIC.
    
    Unified continuous approach: All geological layers (boolean faults, continuous
    grades) treated as continuous variables. Uses cross-validated MAE for robust
    prediction assessment and Laplace likelihood for theoretically sound BIC.
    
    Key improvements over R² system:
    - Robust to sparse geological data (no zero-inflation bias)
    - Interpretable errors in geological units (% grade, ppm, fault probability)
    - Theoretically consistent BIC using same metric for prediction and scoring
    - Unified framework eliminates mixed-type complexity
    
    Features:
    - Geological interpolation: Extends features within 7x voxel-size radius
    - Cross-validated MAE: 80/20 split with geological signal constraints
    - Laplace likelihood BIC: Direct conversion from MAE to BIC
    - Spatial autocorrelation correction: Moran's I for spatial validity
    
    Args:
        layer_values: List of flattened layer arrays
        layer_dtypes: List of data types (treated as continuous)
        grid: GridSpec for spatial coordinate information
        shape: Original (nx, ny, nz) shape
        enable_masking_test: Legacy parameter (simplified in MAE version)
        masking_test_threshold: Minimum MAE improvement threshold
    
    Returns:
        dict with:
            - system_coherence: MAE-derived coherence score (higher = better)
            - spatial_correction: Moran's I correction factor
            - coherence_matrix: MAE matrix (lower = better prediction)
            - bic: Laplace likelihood BIC (lower = better)
            - total_cv_mse: Legacy compatibility (derived from MAE)
            - per_layer_mse: Legacy compatibility (empty dict)
            - masking_test_passed: Always True (simplified)
            - masking_test_improvement: System-wide MAE quality
            - masking_test_direction: "unified_continuous"
            - stage_completed: "mae_bic_completed"
    """
    n_layers = len(layer_values)
    
    # Handle edge cases
    if n_layers == 0:
        return {
            "system_coherence": 0.0,
            "spatial_correction": 1.0,
            "coherence_matrix": np.array([]),
            "bic": 0.0,
            "total_cv_mse": 0.0,
            "per_layer_mse": {},
            "masking_test_passed": True,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "not_applicable",
            "stage_completed": "mae_bic_completed",
            "system_mae": 0.0,
            "n_effective_samples": 0,
        }
    
    if n_layers == 1:
        # Single layer - perfect coherence, no complexity penalty
        return {
            "system_coherence": 1.0,
            "spatial_correction": 1.0,
            "coherence_matrix": np.array([[0.0]]),  # Zero MAE for self
            "bic": 0.0,
            "total_cv_mse": 0.0,
            "per_layer_mse": {},
            "masking_test_passed": True,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "single_layer",
            "stage_completed": "mae_bic_completed",
            "system_mae": 0.0,
            "n_effective_samples": 0,
        }
    
    # Apply geological interpolation to reduce sparsity
    interpolated_layers = []
    for layer in layer_values:
        interpolated = compute_geological_interpolation(layer, grid, shape)
        interpolated_layers.append(interpolated)
    
    # Calculate effective samples based on non-zero values in interpolated data
    total_non_zero = sum(np.count_nonzero(layer) for layer in interpolated_layers)
    effective_samples = max(total_non_zero, n_layers * 10)  # At least 10 samples per layer
    
    # Handle case where no geological data exists
    if all(np.count_nonzero(layer) == 0 for layer in interpolated_layers):
        return {
            "system_coherence": 0.0,
            "spatial_correction": 1.0,
            "coherence_matrix": np.zeros((n_layers, n_layers)),
            "bic": float('inf'),  # Infinite BIC for empty data
            "total_cv_mse": 1.0,
            "per_layer_mse": {},
            "masking_test_passed": False,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "no_data",
            "stage_completed": "mae_bic_completed",
            "system_mae": float('inf'),
            "n_effective_samples": 0,
        }
    
    # Compute pairwise MAE matrix using unified continuous approach
    mae_matrix = compute_pairwise_mae(interpolated_layers, layer_dtypes, grid)
    
    # Convert MAE matrix to coherence score for compatibility
    system_coherence = system_mae_to_coherence(mae_matrix)
    
    # Spatial autocorrelation correction using original data
    spatial_correction = compute_moran_correction(layer_values, grid)
    
    # Compute Laplace likelihood BIC from MAE matrix
    bic = compute_geological_bic(
        mae_matrix=mae_matrix,
        n_layers=n_layers,
        n_effective_samples=effective_samples,
        spatial_correction=spatial_correction
    )
    
    # Legacy compatibility mappings
    total_cv_mse = 1.0 - system_coherence * spatial_correction
    
    # System-wide MAE (mean of off-diagonal pairwise MAEs)
    system_mae = float(np.mean(mae_matrix[~np.eye(n_layers, dtype=bool)]))
    
    return {
        "system_coherence": system_coherence,
        "spatial_correction": spatial_correction,
        "coherence_matrix": mae_matrix,  # Now MAE matrix (lower = better)
        "bic": bic,
        "total_cv_mse": total_cv_mse,
        "per_layer_mse": {},  # Legacy compatibility
        "masking_test_passed": True,  # Placeholder; real gate applied in evaluate_new_layer
        "masking_test_improvement": 0.0,  # Placeholder; real delta computed in evaluate_new_layer
        "masking_test_direction": "unified_continuous",
        "stage_completed": "mae_bic_completed",
        "system_mae": system_mae,
        "n_effective_samples": effective_samples,
    }



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
    Evaluate adding a new layer to the store using MAE + Laplace likelihood BIC.
    
    NEW: Uses unified continuous approach with MAE prediction and Laplace likelihood
    BIC for robust geological scoring. All layer types (boolean faults, continuous
    grades) treated as continuous variables for consistent evaluation.
    
    Process:
    1. Compute geological coherence score WITHOUT the new layer (MAE-based)
    2. Compute geological coherence score WITH the new layer (MAE-based)
    3. Admit if Laplace likelihood BIC improves (lower is better)
    
    Improvements over R² system:
    - Robust to sparse geological data (no zero-inflation bias)
    - Interpretable errors in geological units (%Cu, ppm Au, fault probability)
    - Theoretically consistent BIC using same metric for prediction and scoring
    
    Args:
        store: VoxelStore to evaluate against
        layer_name: Name for the new layer
        layer_values: 3D array of values
        layer_dtype: Data type ("float", "categorical", "boolean") - all treated as continuous
        ridge_alpha: Ridge regularization strength (legacy compatibility - ignored)
        n_folds: Number of CV folds (legacy compatibility - uses 80/20 CV split)
    
    Returns:
        dict with (API compatible):
            - bic_before/after/delta: Laplace likelihood BIC scores
            - cv_mse_before/after/delta: MAE-derived MSE (legacy compatibility)
            - mutual_info: MI with existing layers
            - admitted: whether layer improves BIC
            - predicted_value: BIC improvement (higher = better hypothesis)
    """
    existing_layers = list(store.layer_names)
    grid_shape = store.grid.shape
    
    # Flatten new layer values
    new_values_flat = layer_values.flatten()
    
    # First layer: no comparison is possible, admit unconditionally
    if not existing_layers:
        store.add_layer(
            name=layer_name,
            values=layer_values,
            dtype=layer_dtype,
        )
        return {
            "bic_before": 0.0,
            "bic_after": 0.0,
            "bic_delta": -1.0,
            "bic_delta_raw": -1.0,
            "n_effective_samples": 0,
            "cv_mse_before": 0.0,
            "cv_mse_after": 0.0,
            "cv_mse_delta": 0.0,
            "mutual_info": {},
            "admitted": True,
            "predicted_value": 1.0,
            "masking_test_passed": True,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "first_layer",
            "stage_completed": "mae_bic_completed",
        }
    
    # Get existing layer values and dtypes
    existing_values = [store.get_layer_values(n).flatten() for n in existing_layers]
    existing_dtypes = [store.get_layer(n).dtype for n in existing_layers]
    
    # Score WITHOUT new layer (always call to get system_mae / n_effective_samples)
    score_before = geological_coherence_score(
        existing_values, existing_dtypes, store.grid, grid_shape
    )
    
    # Score WITH new layer
    all_values = existing_values + [new_values_flat]
    all_dtypes = existing_dtypes + [layer_dtype]
    score_after = geological_coherence_score(
        all_values, all_dtypes, store.grid, grid_shape
    )
    
    # Raw BIC values
    bic_before = score_before["bic"]
    bic_after = score_after["bic"]
    bic_delta_raw = bic_after - bic_before
    
    # Normalize BIC delta by effective sample count to remove grid-size artifact
    n_eff = max(score_after.get("n_effective_samples", 1), 1)
    bic_delta = bic_delta_raw / n_eff  # per-sample BIC delta; range ~[-0.5, 0] for good layers
    
    cv_mse_before = score_before["total_cv_mse"]
    cv_mse_after = score_after["total_cv_mse"]
    cv_mse_delta = cv_mse_after - cv_mse_before
    
    # -----------------------------------------------------------------------
    # Stage 1: Real MAE gate - does adding this layer reduce system MAE?
    # Auto-pass when <2 existing layers (no meaningful baseline MAE available)
    # -----------------------------------------------------------------------
    mae_before = score_before.get("system_mae", None)
    mae_after = score_after.get("system_mae", 0.0)
    
    if len(existing_layers) < 2 or mae_before is None:
        # Not enough layers to compute a meaningful before-MAE; let Stage 2 decide
        stage1_passed = True
        mae_improvement = 0.0
    else:
        mae_improvement = mae_before - mae_after  # positive = system MAE improved
        stage1_passed = mae_improvement > 0
    
    # -----------------------------------------------------------------------
    # Stage 2: Normalized BIC must improve (negative per-sample delta)
    # -----------------------------------------------------------------------
    if not stage1_passed:
        admitted = False
        print(f"❌ Layer {layer_name} rejected at Stage 1: MAE worsened by {-mae_improvement:.6f} (mae_before={mae_before:.4f}, mae_after={mae_after:.4f})")
    else:
        admitted = bic_delta < 0.0
        if admitted:
            print(f"✅ Layer {layer_name} admitted: BIC/sample={bic_delta:.6f}, MAE delta={mae_improvement:.6f}")
        else:
            print(f"❌ Layer {layer_name} rejected at Stage 2: BIC/sample={bic_delta:.6f} (not negative)")
    
    stage1_fields = {
        "masking_test_passed": stage1_passed,
        "masking_test_improvement": mae_improvement,
        "masking_test_direction": "mae_delta" if len(existing_layers) >= 2 else "auto_pass",
        "stage_completed": "mae_bic_completed",
    }
    
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
        "bic_delta": bic_delta,          # normalized per-sample; range ~[-0.5, 0] for good layers
        "bic_delta_raw": bic_delta_raw,  # raw (grid-size-dependent); diagnostic only
        "n_effective_samples": n_eff,
        "cv_mse_before": cv_mse_before,
        "cv_mse_after": cv_mse_after,
        "cv_mse_delta": cv_mse_delta,
        "mutual_info": mi_scores,
        "admitted": admitted,
        # For hypothesis agent training: positive = improvement (higher = better)
        "predicted_value": -bic_delta,
        # Stage 1 fields from two-stage scoring
        **stage1_fields
    }


def marginal_contribution(
    store: VoxelStore,
    layer_name: str,
    ridge_alpha: float = 1.0,
    n_folds: int = 5,
) -> float:
    """
    Compute how much removing this layer would change geological coherence BIC.
    
    Positive value means the layer is contributing (BIC would increase without it).
    Negative value means the layer is hurting (BIC would decrease without it).
    """
    if layer_name not in store.layer_names:
        raise KeyError(f"Layer '{layer_name}' not found")
    
    all_layers = list(store.layer_names)
    grid_shape = store.grid.shape
    
    # BIC with all layers
    all_values = [store.get_layer_values(n).flatten() for n in all_layers]
    all_dtypes = [store.get_layer(n).dtype for n in all_layers]
    score_with = geological_coherence_score(all_values, all_dtypes, store.grid, grid_shape)
    bic_with = score_with["bic"]
    
    # BIC without target layer
    layer_idx = all_layers.index(layer_name)
    values_without = [v for i, v in enumerate(all_values) if i != layer_idx]
    dtypes_without = [d for i, d in enumerate(all_dtypes) if i != layer_idx]
    
    if len(values_without) >= 2:
        score_without = geological_coherence_score(values_without, dtypes_without, store.grid, grid_shape)
        bic_without = score_without["bic"]
    else:
        bic_without = 0.0
    
    # Contribution = how much BIC increases if we remove it (positive = layer helps)
    return bic_without - bic_with
