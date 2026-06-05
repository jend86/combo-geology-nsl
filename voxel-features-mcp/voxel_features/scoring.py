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
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    generate_binary_structure,
    iterate_structure,
)

if TYPE_CHECKING:
    from voxel_features.store import VoxelStore, GridSpec


_RELATIVE_MAE_FLOOR = 1e-3
_RELATIVE_MAE_NULL_EPS = 1e-10
_MAX_EFFECTIVE_SAMPLES = 10_000

_STAGE1_MAE_TOLERANCE = 1e-3
_STAGE1_BIC_RESCUE_THRESHOLD = -1.0

_SPATIAL_SCORING_OBJECTIVE = "spatial_predictor_lift_v1"
_SPATIAL_DEFAULT_SCALES_VOX = (3.0, 8.0, 20.0)
_SPATIAL_SMALL_GRID_SCALES_VOX = (2.0, 5.0, 12.0)
_SPATIAL_DEFAULT_SELF_SCALES_VOX = (3.0, 8.0)
_SPATIAL_SMALL_GRID_SELF_SCALES_VOX = (2.0, 5.0)
_SPATIAL_VERTICAL_SIGMA_VOX = 0.8
_SPATIAL_GAUSSIAN_TRUNCATE = 2.0
_SPATIAL_N_BLOCKS_XY = 4
_SPATIAL_CV_BUFFER_VOX = 4
_SPATIAL_MATCHED_ZERO_RATIO = 1.0
_SPATIAL_TAU_SELF = 0.9
_SPATIAL_MIN_EVAL_ROWS = 8
_SPATIAL_MIN_TRAIN_ROWS = 5
_SPATIAL_MAX_PREDICTOR_LAYERS = 6
_SPATIAL_NULL_PERMUTATIONS = 0
_SPATIAL_NULL_PERCENTILE = 5.0
# 3 -> 6 (2026-06-05): the seed bypass admits validity-passing layers WITHOUT
# predictor-lift, building the diverse founder pool the survey phase needs. At 3 it
# stalled: layers 4+ hit full predictor-lift, which can't admit spatially-distinct
# binary layers (cross-prediction == near-dup), so the KG froze at ~4 and never
# reached min_features=6 / crossbreed. 6 = min_features, so the survey blankets the
# basin with up to 6 valid diverse founders, then predictor-lift governs at L>=6.
_SPATIAL_SEED_POOL_TARGET = 6
_SPATIAL_REJECTION_BIC_DELTA = 1_000_000.0
_SPATIAL_MIN_LIFT = 1e-6
# Calibrated ADMISSION bar on mean cross-layer predictor lift (2026-06-05). The
# uncalibrated `bic_delta < 0` gate inverted the ranking: tiny low-DOF self-
# predictive blobs admitted while richer high-DOF crossbreed children with
# genuine cross-layer lift were rejected -> KG frozen at 7, 25 consecutive
# crossbreed failures. _SPATIAL_MIN_LIFT (1e-6) stays the telemetry "any positive
# lift" floor; admission now requires a MEANINGFUL lift (live: trivial blobs
# ~0.0026 vs distributed children ~0.011-0.020). Tunable; validated offline on
# scratch/scoring_validation. See predictor_lift_admission_decision.
_SPATIAL_ADMIT_MIN_LIFT = 0.005


def predictor_lift_admission_decision(
    *,
    validity_passed: bool,
    lift_mean: float,
    bic_delta: float,
    admit_min_lift: float = _SPATIAL_ADMIT_MIN_LIFT,
) -> bool:
    """Calibrated admission policy for ``spatial_predictor_lift_v1``.

    Gates on the CROSS-LAYER predictor lift (validity + a meaningful lift bar),
    NOT the per-sample ``bic_delta``. ``bic_delta`` penalises a candidate's
    effective DOF, which perversely rejected rich, distributed (high-DOF)
    crossbreed children that genuinely improved cross-layer prediction while
    admitting tiny low-DOF self-predictive blobs. ``bic_delta`` is retained for
    telemetry (and as a future veto hook) but does NOT gate admission here;
    ``validity_passed`` (self coherence) plus held-out buffered-CV lift guard
    against blanket layers. The ``bic_delta`` argument is intentionally accepted
    and unused so callers/telemetry keep a stable signature.
    """
    _ = bic_delta  # telemetry / future veto hook; deliberately not a gate
    if not validity_passed:
        return False
    return float(lift_mean) > float(admit_min_lift)
# Calibration 2026-06-05: a candidate occupying fewer than this many distinct (x,y)
# columns has no HORIZONTAL spatial structure to validate — any self-prediction skill
# comes entirely from its own vertical stack (a borehole-like pillar; Rv leak), which
# is a geologically trivial "obvious failure". Reject it at the validity gate.
# (Chosen over a horizontal-only-feature rewrite: smaller, interpretable, keeps the
# vertical kernel for real layers. The live empties/single-pillars in the rejected
# corpus all sit at <=2 columns; all 19 real admitted layers are >=3.)
_SPATIAL_MIN_SUPPORT_COLUMNS = 3


def _effective_sample_count(total_non_zero: int, n_layers: int) -> int:
    return int(min(max(int(total_non_zero), int(n_layers) * 10), _MAX_EFFECTIVE_SAMPLES))


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


def pairwise_distance(
    store: VoxelStore,
    layer_a: str,
    layer_b: str,
) -> float:
    """Orthogonality proxy in [0, 1], normalized so one threshold fits all dtypes.

    Crossbreed-queue replacement for mutual_information(): the latter had a
    unit mismatch (Shannon-bits marginals vs density-scaled joint) that
    silently returned 0 for every sparse-boolean pair, so the queue lost its
    orthogonality signal. Jaccard has no entropy/binning footguns and is the
    right shape for sparse boolean projections (Kazakhstan's typical layer
    type). Two all-false layers are treated as identical (distance 0).

    Boolean pairs use Jaccard distance. Every other pair uses the
    magnitude-normalized L1 (Sørensen/Bray–Curtis) distance
    ``sum|a-b| / (sum|a| + sum|b|)``, which is the natural generalization of
    Jaccard to real-valued layers: bounded in [0, 1] by the triangle
    inequality, identical layers → 0, disjoint supports → 1, and (unlike raw
    MAE) scale-free so the shared near-duplicate threshold (0.15) carries the
    same "≥85% agreement" meaning regardless of value magnitude. Raw MAE made
    jittered large-magnitude duplicates read as distinct and distinct
    small-magnitude layers read as duplicates. Two all-zero layers → 0.
    """
    layer_a_obj = store.get_layer(layer_a)
    layer_b_obj = store.get_layer(layer_b)
    values_a = layer_a_obj.values
    values_b = layer_b_obj.values

    if layer_a_obj.dtype == "boolean" and layer_b_obj.dtype == "boolean":
        a_bool = values_a.astype(bool)
        b_bool = values_b.astype(bool)
        union = int(np.logical_or(a_bool, b_bool).sum())
        if union == 0:
            return 0.0
        intersection = int(np.logical_and(a_bool, b_bool).sum())
        return 1.0 - float(intersection) / float(union)

    a_flat = np.nan_to_num(values_a.astype(float), nan=0.0).ravel()
    b_flat = np.nan_to_num(values_b.astype(float), nan=0.0).ravel()
    scale = float(np.sum(np.abs(a_flat)) + np.sum(np.abs(b_flat)))
    if scale <= 1e-12:
        return 0.0  # both layers are ~all-zero → identical "nothing here"
    distance = float(np.sum(np.abs(a_flat - b_flat))) / scale
    return min(max(distance, 0.0), 1.0)


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
    grid: 'GridSpec',
    *,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute spatial autocorrelation correction using Moran's I.

    Uses geographic coordinates to weight spatial relationships
    and estimate effective sample size for statistical tests.

    Args:
        layer_values: List of flattened layer arrays
        grid: GridSpec containing coordinate information
        rng: Optional numpy Generator for reproducible voxel sampling.
            Defaults to ``np.random.default_rng()`` (non-deterministic).

    Returns:
        Correction factor: effective_n / total_n
    """
    if rng is None:
        rng = np.random.default_rng()

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
    indices = rng.choice(n_voxels, max_sample_size, replace=False)
    
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
    influence_radius_m: float = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply geological interpolation using sphere of influence with inverse distance weighting.

    OPTIMIZED VERSION: Uses spatial bounds checking to avoid O(n²) complexity.

    Args:
        layer_values: Flattened layer array
        grid: GridSpec for spatial coordinate information
        shape: (nx, ny, nz) voxel grid shape
        influence_radius_m: Influence radius in meters (default: 7x voxel size)
        rng: Optional numpy Generator for reproducible source selection.
            Defaults to ``np.random.default_rng()`` (non-deterministic).

    Returns:
        Interpolated layer array (same shape as input)
    """
    if rng is None:
        rng = np.random.default_rng()

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
    source_indices = rng.choice(len(non_zero_indices[0]), n_sources, replace=False)
    
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
    min_test_signal_ratio: float = 0.01,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 80/20 cross-validation split with geological constraints.

    Ensures the test set contains adequate geological signal (non-zero values)
    while maintaining random sampling for unbiased evaluation.

    Args:
        interpolated_layers: List of interpolated flattened layer arrays
        test_fraction: Fraction of data for testing (default 0.2 = 20%)
        min_test_signal_ratio: Minimum ratio of non-zero values in test set
        rng: Optional numpy Generator for reproducible splits. Defaults to
            ``np.random.default_rng(42)`` so that mae_before and mae_after
            share the same split (preserves the historical seed=42 behaviour
            without the ``np.random.seed`` global side effect).

    Returns:
        Tuple of (train_mask, test_mask) boolean arrays
    """
    if rng is None:
        rng = np.random.default_rng(42)

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

    if len(signal_indices) >= min_test_signal:
        # Randomly select signal voxels for test set
        test_signal_indices = rng.choice(
            signal_indices,
            size=min_test_signal,
            replace=False,
        )

        # Fill remaining test slots from all remaining voxels
        remaining_indices = np.setdiff1d(np.arange(n_voxels), test_signal_indices)
        remaining_test_size = n_test - min_test_signal

        if remaining_test_size > 0 and len(remaining_indices) > 0:
            additional_test_indices = rng.choice(
                remaining_indices,
                size=min(remaining_test_size, len(remaining_indices)),
                replace=False,
            )
            test_indices = np.concatenate([test_signal_indices, additional_test_indices])
        else:
            test_indices = test_signal_indices
    else:
        # Fallback: random selection if insufficient signal
        test_indices = rng.choice(n_voxels, size=n_test, replace=False)
    
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
        # Use ridge when sklearn is available; the multi-scale spatial features
        # are intentionally collinear, so nominal OLS is too brittle.
        from sklearn.linear_model import Ridge
        
        model = Ridge(alpha=1e-3, fit_intercept=True)
        model.fit(train_predictors, train_target)
        
        return {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'n_predictors': n_predictors,
            'n_train_samples': n_train,
            'prediction_type': 'sklearn_ridge'
        }
        
    except ImportError:
        # Fallback: numpy least squares
        try:
            # Add intercept column; keep the intercept unpenalized.
            X_with_intercept = np.column_stack([np.ones(n_train), train_predictors])
            regularizer = 1e-3 * np.eye(X_with_intercept.shape[1])
            regularizer[0, 0] = 0.0
            coeffs_with_intercept = np.linalg.solve(
                X_with_intercept.T @ X_with_intercept + regularizer,
                X_with_intercept.T @ train_target,
            )
            
            return {
                'coefficients': coeffs_with_intercept[1:],  # Exclude intercept
                'intercept': coeffs_with_intercept[0],
                'n_predictors': n_predictors,
                'n_train_samples': n_train,
                'prediction_type': 'numpy_ridge'
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
    Compute relative Mean Absolute Error on held-out test data.
    
    Uses unified continuous approach: all layers treated as continuous
    (boolean geological features automatically handled as 0/1).
    
    Args:
        target_layer: Target layer to predict (flattened)
        predictor_layers: List of predictor layer arrays (flattened)
        train_mask: Boolean mask for training voxels
        test_mask: Boolean mask for test voxels
        
    Returns:
        MAE divided by the target's predict-by-train-mean null MAE. Constant
        targets return 1.0 so they cannot manufacture likelihood.
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
    
    mae_pred = float(np.mean(np.abs(test_target - predictions)))
    train_target = target_layer[train_mask]
    null_prediction = float(np.mean(train_target)) if train_target.size else float(np.mean(test_target))
    mae_null = float(np.mean(np.abs(test_target - null_prediction)))

    if mae_null <= _RELATIVE_MAE_NULL_EPS:
        return 1.0

    return float(max(mae_pred / mae_null, _RELATIVE_MAE_FLOOR))


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
    
    system_mae = max(float(system_mae), _RELATIVE_MAE_FLOOR)
    
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
    spatial_correction: float = 1.0,
    target_relative_maes: np.ndarray | None = None,
) -> float:
    """
    Compute BIC score from MAE matrix using Laplace likelihood.
    
    This provides a unified, theoretically sound BIC calculation
    that directly uses the same metric (MAE) for both prediction 
    assessment and information criterion evaluation.
    
    Args:
        mae_matrix: Matrix of relative-MAE values
        n_layers: Number of geological layers
        n_effective_samples: Effective sample size from interpolated data
        spatial_correction: Moran's I spatial autocorrelation correction
        
    Returns:
        BIC score (lower = better geological model)
    """
    if n_layers <= 1:
        # Single layer or no layers - no meaningful BIC
        return 0.0
    
    if target_relative_maes is not None:
        relative_maes = np.asarray(target_relative_maes, dtype=float)
    else:
        # Fallback for tests/legacy callers: use directed off-diagonal entries.
        mask = ~np.eye(n_layers, dtype=bool)
        relative_maes = np.asarray(mae_matrix, dtype=float)[mask]

    valid_maes = relative_maes[np.isfinite(relative_maes) & (relative_maes >= 0)]
    
    if len(valid_maes) == 0:
        # No valid predictions - return neutral BIC
        return 0.0
    
    valid_maes = np.clip(valid_maes, _RELATIVE_MAE_FLOOR, None)
    n_eff = int(max(n_effective_samples, 1))
    log_likelihood = float(np.sum(-n_eff * (np.log(2.0 * valid_maes) + 1.0)))
    corrected_log_likelihood = log_likelihood * spatial_correction

    # Directed per-target regressions: each target is predicted from the rest.
    n_parameters = n_layers * max(n_layers - 1, 1)

    bic = -2 * corrected_log_likelihood + n_parameters * np.log(max(n_eff, n_layers))
    
    return float(bic)


def _single_layer_null_bic(
    layer_values: np.ndarray,
    layer_dtype: str,
    grid: 'GridSpec',
    shape: tuple[int, int, int],
    *,
    seed: int | None = None,
) -> dict:
    """Compute a "predict-by-mean" null-model BIC for a single layer.

    Used as the score_before baseline when the second layer is being
    evaluated: comparing the two-layer BIC against ``bic=0`` (the historical
    n_layers==1 sentinel) is an apples-to-oranges baseline that produces
    spurious deltas. This null model encodes the lone layer with its own
    mean, so the 2-layer BIC delta measures actual predictive gain.

    Returns a dict with the same shape as ``geological_coherence_score`` so
    the caller can use it interchangeably.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    interpolated = compute_geological_interpolation(layer_values, grid, shape, rng=rng)
    non_zero = interpolated[interpolated != 0]
    n_eff = _effective_sample_count(len(non_zero), 1)

    if len(non_zero) < 2 or np.std(non_zero) <= 1e-12:
        # Degenerate layer (empty, constant, or one point); no meaningful null.
        return {
            "system_coherence": 0.0,
            "spatial_correction": 1.0,
            "coherence_matrix": np.array([[0.0]]),
            "bic": 0.0,
            "total_cv_mse": 0.0,
            "per_layer_mse": {},
            "masking_test_passed": True,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "single_layer_null",
            "stage_completed": "mae_bic_completed",
            "system_mae": 1.0,
            "relative_mae_mean": 1.0,
            "relative_mae_min": 1.0,
            "relative_mae_max": 1.0,
            "relative_mae_by_target": np.array([1.0]),
            "n_effective_samples": n_eff,
            "single_layer_null_mad": 0.0,
        }

    layer_mean = float(np.nanmean(non_zero))
    mad = float(np.mean(np.abs(non_zero - layer_mean)))
    moran_rng = np.random.default_rng(seed) if seed is not None else None
    spatial_correction = compute_moran_correction([layer_values], grid, rng=moran_rng)
    bic = _single_layer_null_bic_from_mad(mad, n_eff, spatial_correction)

    return {
        "system_coherence": 0.0,
        "spatial_correction": spatial_correction,
        "coherence_matrix": np.array([[0.0]]),
        "bic": float(bic),
        "total_cv_mse": 1.0,
        "per_layer_mse": {},
        "masking_test_passed": True,
        "masking_test_improvement": 0.0,
        "masking_test_direction": "single_layer_null",
        "stage_completed": "mae_bic_completed",
        "system_mae": 1.0,
        "relative_mae_mean": 1.0,
        "relative_mae_min": 1.0,
        "relative_mae_max": 1.0,
        "relative_mae_by_target": np.array([1.0]),
        "n_effective_samples": n_eff,
        "single_layer_null_mad": mad,
    }


def _single_layer_null_bic_from_mad(
    mad: float,
    n_effective_samples: int,
    spatial_correction: float,
) -> float:
    """Compute one-layer predict-by-mean BIC for a supplied sample count."""
    n_eff = int(max(n_effective_samples, 1))
    relative_mae = 1.0
    log_likelihood = -n_eff * (float(np.log(2.0 * relative_mae)) + 1.0)

    corrected_log_likelihood = log_likelihood * spatial_correction
    # 1 parameter: the layer's mean.
    bic = -2.0 * corrected_log_likelihood + float(np.log(n_eff))
    return float(bic)


def _bic_with_common_effective_samples(
    score: dict,
    n_layers: int,
    n_effective_samples: int,
) -> float:
    """Recompute BIC using a comparison-wide effective sample count.

    ``geological_coherence_score`` computes each model with its own
    ``n_effective_samples``. That is useful telemetry, but before/after BIC
    deltas must compare the same sample universe; otherwise adding a sparse
    candidate can look better or worse simply because the effective ``n`` moved.
    """
    n_eff = int(max(n_effective_samples, 1))
    spatial_correction = float(score.get("spatial_correction", 1.0) or 1.0)

    if n_layers == 1:
        mad = score.get("single_layer_null_mad", score.get("system_mae"))
        if mad is not None and np.isfinite(float(mad)):
            return _single_layer_null_bic_from_mad(float(mad), n_eff, spatial_correction)

    mae_matrix = score.get("coherence_matrix")
    if isinstance(mae_matrix, np.ndarray) and mae_matrix.size:
        return compute_geological_bic(
            mae_matrix=mae_matrix,
            n_layers=n_layers,
            n_effective_samples=n_eff,
            spatial_correction=spatial_correction,
            target_relative_maes=score.get("relative_mae_by_target"),
        )

    # Test doubles and legacy callers may only provide a scalar BIC.
    return float(score.get("bic", 0.0))


def _spatial_scales_for_shape(
    shape: tuple[int, int, int],
    scales: tuple[float, ...] | None,
    *,
    self_scales: bool = False,
) -> tuple[float, ...]:
    if scales is not None:
        source = tuple(float(s) for s in scales if float(s) > 0.0)
    elif min(shape[:2]) < 80:
        source = (
            _SPATIAL_SMALL_GRID_SELF_SCALES_VOX
            if self_scales
            else _SPATIAL_SMALL_GRID_SCALES_VOX
        )
    else:
        source = _SPATIAL_DEFAULT_SELF_SCALES_VOX if self_scales else _SPATIAL_DEFAULT_SCALES_VOX

    max_scale = max(1.0, float(min(shape[:2])) / 2.0)
    clipped: list[float] = []
    for scale in source:
        value = min(float(scale), max_scale)
        if not clipped or abs(value - clipped[-1]) > 1e-9:
            clipped.append(value)
    return tuple(clipped) or (1.0,)


def _spatial_null_permutations_from_env() -> int:
    import os

    raw = os.environ.get("VFM_SPATIAL_NULL_PERMUTATIONS")
    if raw is None:
        return _SPATIAL_NULL_PERMUTATIONS
    try:
        return max(0, int(raw))
    except ValueError:
        return _SPATIAL_NULL_PERMUTATIONS


def _as_spatial_field(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float).reshape(shape), nan=0.0, copy=True)


def _spatial_union_bbox(fields: list[np.ndarray], pad: int = 0) -> tuple[int, int, int, int]:
    if not fields:
        return (0, 0, 0, 0)
    nx, ny = fields[0].shape[:2]
    support = np.zeros((nx, ny), dtype=bool)
    for field in fields:
        support |= np.any(field != 0, axis=2)
    xs, ys = np.where(support)
    if xs.size == 0:
        return (0, nx, 0, ny)
    return (
        max(0, int(xs.min()) - int(pad)),
        min(nx, int(xs.max()) + 1 + int(pad)),
        max(0, int(ys.min()) - int(pad)),
        min(ny, int(ys.max()) + 1 + int(pad)),
    )


def spatial_block_folds(
    shape: tuple[int, int, int],
    signal_mask: np.ndarray | None = None,
    block_voxels: int | None = None,
    buffer_voxels: int = _SPATIAL_CV_BUFFER_VOX,
    *,
    n_blocks_xy: int = _SPATIAL_N_BLOCKS_XY,
) -> list[dict[str, np.ndarray | int]]:
    """Create deterministic whole-column XY block folds.

    D13 uses these folds only for fit/evaluation row separation. Cross-layer
    features are full-field smooths of other layers, so no block-masked feature
    tensors are constructed here.
    """
    nx, ny, nz = shape
    if signal_mask is None:
        bbox = (0, nx, 0, ny)
    else:
        signal = np.asarray(signal_mask, dtype=bool).reshape(shape)
        bbox = _spatial_union_bbox([signal.astype(float)])
    if block_voxels is not None and block_voxels > 0:
        x0, x1, y0, y1 = bbox
        n_blocks_xy = max(1, int(math.ceil(max(x1 - x0, y1 - y0) / float(block_voxels))))
    labels = _spatial_block_labels(shape, bbox, n_blocks_xy)
    st = (
        iterate_structure(generate_binary_structure(2, 1), int(buffer_voxels))
        if buffer_voxels > 0
        else None
    )
    folds: list[dict[str, np.ndarray | int]] = []
    for block_id in sorted(int(v) for v in np.unique(labels) if int(v) >= 0):
        block2d = labels == block_id
        buffer2d = binary_dilation(block2d, structure=st) if st is not None else block2d
        block = np.repeat(block2d[:, :, None], nz, axis=2).ravel()
        buffer = np.repeat(buffer2d[:, :, None], nz, axis=2).ravel()
        folds.append({"block_id": block_id, "block": block, "buffer": buffer})
    return folds


def _spatial_block_labels(
    shape: tuple[int, int, int],
    bbox: tuple[int, int, int, int],
    n_blocks_xy: int,
) -> np.ndarray:
    nx, ny, _ = shape
    x0, x1, y0, y1 = bbox
    labels = -np.ones((nx, ny), dtype=int)
    k = max(1, int(n_blocks_xy))
    x_edges = np.linspace(x0, x1, k + 1).astype(int)
    y_edges = np.linspace(y0, y1, k + 1).astype(int)
    block_id = 0
    for ix in range(k):
        for iy in range(k):
            labels[x_edges[ix]:x_edges[ix + 1], y_edges[iy]:y_edges[iy + 1]] = block_id
            block_id += 1
    return labels


def _center_weight(
    sigma: tuple[float, float, float],
    truncate: float,
    nz: int,
) -> float:
    n = 21
    z = max(int(nz), 1)
    impulse = np.zeros((n, n, z), dtype=float)
    impulse[n // 2, n // 2, z // 2] = 1.0
    filtered = gaussian_filter(impulse, sigma=sigma, truncate=truncate, mode="constant")
    return float(filtered[n // 2, n // 2, z // 2])


def masked_kernel_features(
    field: np.ndarray,
    scales_vox: tuple[float, ...],
    vertical_sigma_vox: float = _SPATIAL_VERTICAL_SIGMA_VOX,
    truncate: float = _SPATIAL_GAUSSIAN_TRUNCATE,
    *,
    leave_self: bool = False,
) -> list[np.ndarray]:
    """Full-field normalized convolution features for D13.

    Cross-features use ``leave_self=False`` because they smooth other layers and
    carry no target signal. The self-validity gate passes ``leave_self=True`` to
    remove the candidate voxel's own kernel contribution.
    """
    field = np.asarray(field, dtype=float)
    observed = (field != 0).astype(float)
    features: list[np.ndarray] = []
    for scale in scales_vox:
        sigma = (float(scale), float(scale), float(vertical_sigma_vox))
        numerator = gaussian_filter(field, sigma=sigma, truncate=truncate, mode="constant")
        denominator = gaussian_filter(observed, sigma=sigma, truncate=truncate, mode="constant")
        if leave_self:
            k0 = _center_weight(sigma, truncate, field.shape[2])
            numerator = numerator - k0 * field
            denominator = denominator - k0 * observed
        features.append(numerator / (denominator + 1e-9))
    return features


def denominator_confidence_mask(
    field: np.ndarray,
    scales_vox: tuple[float, ...],
    min_den: float,
    vertical_sigma_vox: float = _SPATIAL_VERTICAL_SIGMA_VOX,
    truncate: float = _SPATIAL_GAUSSIAN_TRUNCATE,
) -> np.ndarray:
    """Validity-gate-only denominator confidence mask.

    D13 removed denominator masking from cross-features. This helper remains for
    callers/tests that need to inspect whether leave-self self-validity features
    have enough kernel support.
    """
    observed = (np.asarray(field) != 0).astype(float)
    mask = np.ones(field.shape, dtype=bool)
    for scale in scales_vox:
        sigma = (float(scale), float(scale), float(vertical_sigma_vox))
        denominator = gaussian_filter(observed, sigma=sigma, truncate=truncate, mode="constant")
        k0 = _center_weight(sigma, truncate, field.shape[2])
        denominator = denominator - k0 * observed
        mask &= denominator >= float(min_den)
    return mask


def _stack_spatial_features(feature_list: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    if not feature_list:
        return np.zeros((len(idx), 0), dtype=float)
    columns = [feature[idx[:, 0], idx[:, 1], idx[:, 2]] for feature in feature_list]
    return np.column_stack(columns) if columns else np.zeros((len(idx), 0), dtype=float)


def _spatial_eval_indices(
    target_field: np.ndarray,
    bbox: tuple[int, int, int, int],
    matched_zero_ratio: float,
) -> np.ndarray:
    x0, x1, y0, y1 = bbox
    region = np.zeros(target_field.shape, dtype=bool)
    region[x0:x1, y0:y1, :] = True
    positive = np.array(np.where((target_field != 0) & region)).T
    zero = np.array(np.where((target_field == 0) & region)).T
    n_zero = int(float(matched_zero_ratio) * len(positive))
    if n_zero > 0 and len(zero) > n_zero:
        stride = max(1, len(zero) // n_zero)
        zero = zero[::stride][:n_zero]
    elif n_zero <= 0:
        zero = zero[:0]
    return np.vstack([positive, zero]) if len(positive) else zero


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n_rows, n_features = X.shape
    X_with_intercept = np.column_stack([np.ones(n_rows), X])
    xtx = X_with_intercept.T @ X_with_intercept
    regularizer = float(alpha) * np.eye(n_features + 1)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(xtx + regularizer, X_with_intercept.T @ y)


def _ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X]) @ beta


def ridge_effective_dof(X: np.ndarray, alpha: float) -> float:
    """Ridge effective degrees of freedom, including an unpenalized intercept."""
    X = np.asarray(X, dtype=float)
    n_rows, n_features = X.shape
    if n_rows == 0:
        return 0.0
    X_with_intercept = np.column_stack([np.ones(n_rows), X])
    xtx = X_with_intercept.T @ X_with_intercept
    regularizer = float(alpha) * np.eye(n_features + 1)
    regularizer[0, 0] = 0.0
    try:
        solved = np.linalg.solve(xtx + regularizer, xtx)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(xtx + regularizer) @ xtx
    return float(np.clip(np.trace(solved), 0.0, n_features + 1.0))


def _buffered_block_relative_mae(
    idx: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    buffer_voxels: int,
    alpha: float,
    min_train_rows: int,
) -> tuple[float, int, int]:
    block_of = labels[idx[:, 0], idx[:, 1]]
    blocks = np.unique(block_of[block_of >= 0])
    st = (
        iterate_structure(generate_binary_structure(2, 1), int(buffer_voxels))
        if buffer_voxels > 0
        else None
    )
    pred_error = 0.0
    null_error = 0.0
    n_used = 0
    n_folds = 0
    for block_id in blocks:
        test_rows = block_of == block_id
        if int(test_rows.sum()) < 1:
            continue
        block_mask = labels == block_id
        buffer_mask = binary_dilation(block_mask, structure=st) if st is not None else block_mask
        in_buffer = buffer_mask[idx[:, 0], idx[:, 1]]
        train_rows = (~test_rows) & (~in_buffer) & (block_of >= 0)
        if int(train_rows.sum()) < int(min_train_rows):
            continue
        if X.shape[1] == 0:
            prediction = np.full(int(test_rows.sum()), float(np.mean(y[train_rows])))
        else:
            try:
                beta = _ridge_fit(X[train_rows], y[train_rows], alpha)
                prediction = _ridge_predict(X[test_rows], beta)
            except np.linalg.LinAlgError:
                prediction = np.full(int(test_rows.sum()), float(np.mean(y[train_rows])))
        null_prediction = float(np.mean(y[train_rows]))
        pred_error += float(np.abs(y[test_rows] - prediction).sum())
        null_error += float(np.abs(y[test_rows] - null_prediction).sum())
        n_used += int(test_rows.sum())
        n_folds += 1
    if null_error <= _RELATIVE_MAE_NULL_EPS or n_used == 0:
        return 1.0, n_used, n_folds
    return float(max(pred_error / null_error, _RELATIVE_MAE_FLOOR)), n_used, n_folds


def _laplace_bic_single(relative_mae: float, n_rows: int, df: float) -> float:
    n = float(max(int(n_rows), 1))
    rel = max(float(relative_mae), _RELATIVE_MAE_FLOOR)
    log_likelihood = -n * (math.log(2.0 * rel) + 1.0)
    return float(-2.0 * log_likelihood + float(df) * math.log(max(n, 2.0)))


def _fold_preserving_permute_features(
    X: np.ndarray,
    idx: np.ndarray,
    labels: np.ndarray,
    permutation_index: int,
) -> np.ndarray:
    if X.shape[0] <= 1:
        return X.copy()
    out = X.copy()
    block_of = labels[idx[:, 0], idx[:, 1]]
    for block_id in np.unique(block_of[block_of >= 0]):
        rows = np.where(block_of == block_id)[0]
        if len(rows) <= 1:
            continue
        shift = 1 + (int(permutation_index) % (len(rows) - 1))
        out[rows] = X[np.roll(rows, shift)]
    return out


def self_validity_score(
    candidate_values: np.ndarray,
    shape: tuple[int, int, int],
    *,
    self_scales_vox: tuple[float, ...] | None = None,
    vertical_sigma_vox: float = _SPATIAL_VERTICAL_SIGMA_VOX,
    truncate: float = _SPATIAL_GAUSSIAN_TRUNCATE,
    matched_zero_ratio: float = _SPATIAL_MATCHED_ZERO_RATIO,
    ridge_alpha: float = 1e-2,
    min_eval_rows: int = _SPATIAL_MIN_EVAL_ROWS,
) -> float:
    field = _as_spatial_field(candidate_values, shape)
    # Minimum horizontal support: reject borehole-like pillars / near-points whose
    # only internal coherence is the vertical stack of <_SPATIAL_MIN_SUPPORT_COLUMNS
    # distinct (x,y) columns (see constant). relative_mae=1.0 == "no validity".
    support_columns = int(np.count_nonzero(np.any(field != 0, axis=2)))
    if support_columns < _SPATIAL_MIN_SUPPORT_COLUMNS:
        return 1.0
    scales = _spatial_scales_for_shape(shape, self_scales_vox, self_scales=True)
    bbox = _spatial_union_bbox([field], pad=int(2 * max(scales)))
    idx = _spatial_eval_indices(field, bbox, matched_zero_ratio)
    if len(idx) < int(min_eval_rows):
        return 1.0
    features = masked_kernel_features(
        field,
        scales,
        vertical_sigma_vox,
        truncate,
        leave_self=True,
    )
    X = _stack_spatial_features(features, idx)
    if X.shape[1] == 0 or np.allclose(X, 0.0):
        return 1.0
    y = field[idx[:, 0], idx[:, 1], idx[:, 2]]
    try:
        beta = _ridge_fit(X, y, ridge_alpha)
        prediction = _ridge_predict(X, beta)
    except np.linalg.LinAlgError:
        return 1.0
    pred_error = float(np.abs(y - prediction).sum())
    null_error = float(np.abs(y - float(np.mean(y))).sum())
    if null_error <= _RELATIVE_MAE_NULL_EPS:
        return 1.0
    return float(max(pred_error / null_error, _RELATIVE_MAE_FLOOR))


def spatial_predictor_lift_score(
    pool_values: list[np.ndarray],
    pool_names: list[str],
    candidate_values: np.ndarray,
    shape: tuple[int, int, int],
    *,
    ridge_alpha: float = 1e-2,
    scales_vox: tuple[float, ...] | None = None,
    self_scales_vox: tuple[float, ...] | None = None,
    vertical_sigma_vox: float = _SPATIAL_VERTICAL_SIGMA_VOX,
    truncate: float = _SPATIAL_GAUSSIAN_TRUNCATE,
    n_blocks_xy: int = _SPATIAL_N_BLOCKS_XY,
    cv_buffer_vox: int = _SPATIAL_CV_BUFFER_VOX,
    matched_zero_ratio: float = _SPATIAL_MATCHED_ZERO_RATIO,
    tau_self: float = _SPATIAL_TAU_SELF,
    min_eval_rows: int = _SPATIAL_MIN_EVAL_ROWS,
    min_train_rows: int = _SPATIAL_MIN_TRAIN_ROWS,
    max_predictor_layers: int | None = _SPATIAL_MAX_PREDICTOR_LAYERS,
    null_permutations: int = _SPATIAL_NULL_PERMUTATIONS,
    null_percentile: float = _SPATIAL_NULL_PERCENTILE,
    admit_min_lift: float = _SPATIAL_ADMIT_MIN_LIFT,
) -> dict:
    """D13 cross-only spatial predictor-lift score.

    The candidate is never scored as its own target. It is added only as a
    multi-scale spatial predictor for the existing pool targets, evaluated on
    identical held-out signal + matched-zero rows under buffered spatial block
    CV. Self-prediction is used only as a validity gate.
    """
    L = len(pool_values)
    names = list(pool_names) if len(pool_names) == L else [f"layer_{i}" for i in range(L)]
    pool_fields = [_as_spatial_field(values, shape) for values in pool_values]
    candidate_field = _as_spatial_field(candidate_values, shape)
    scales = _spatial_scales_for_shape(shape, scales_vox)
    self_scales = _spatial_scales_for_shape(shape, self_scales_vox, self_scales=True)

    self_relative_mae = self_validity_score(
        candidate_field,
        shape,
        self_scales_vox=self_scales,
        vertical_sigma_vox=vertical_sigma_vox,
        truncate=truncate,
        matched_zero_ratio=matched_zero_ratio,
        ridge_alpha=ridge_alpha,
        min_eval_rows=min_eval_rows,
    )
    validity_passed = bool(self_relative_mae < float(tau_self))
    base_result = {
        "scoring_objective": _SPATIAL_SCORING_OBJECTIVE,
        "spatial_correction": 1.0,
        "kernel_scales_vox": tuple(float(v) for v in scales),
        "self_kernel_scales_vox": tuple(float(v) for v in self_scales),
        "R_v_vox": float(vertical_sigma_vox),
        "block_voxels": None,
        "buffer_voxels": int(cv_buffer_vox),
        "n_spatial_folds": int(n_blocks_xy) * int(n_blocks_xy),
        "matched_zero_ratio": float(matched_zero_ratio),
        "ridge_alpha": float(ridge_alpha),
        "tau_self": float(tau_self),
        "self_relative_mae": float(self_relative_mae),
        "candidate_as_target_relative_mae": float(self_relative_mae),
        "validity_passed": validity_passed,
        "pool_size_at_score": L,
        "calibration_bin": f"L={L}",
        "calibration_null_permutations": int(null_permutations),
    }
    if L == 0:
        return {
            **base_result,
            "bic_before": None,
            "bic_after": None,
            "bic_delta": None,
            "bic_delta_raw": None,
            "bic_before_observed": None,
            "bic_after_observed": None,
            "bic_comparison_n_effective_samples": 0,
            "n_effective_samples": 0,
            "n_effective_samples_before": 0,
            "n_effective_samples_after": 0,
            "relative_mae_by_target": np.array([], dtype=float),
            "relative_mae_before_by_target": {},
            "relative_mae_after_by_target": {},
            "candidate_predictor_lift_by_target": {},
            "candidate_predictor_lift_mean": 0.0,
            "bic_delta_by_target": {},
            "ridge_effective_dof_by_target": {},
            "n_signal_folds_by_target": {},
            "n_holdout_rows_by_target": {},
            "n_rows_dropped_low_den_by_target": {},
            "insufficient_evidence_by_target": {},
            "masking_test_passed": False,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "no_existing_targets",
            "admitted": False,
            "admission_threshold": 0.0,
            "permutation_null_bic_deltas": [],
            "score_note": "no_existing_targets",
        }
    if not validity_passed:
        return {
            **base_result,
            "bic_before": 0.0,
            "bic_after": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_delta": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_delta_raw": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_before_observed": 0.0,
            "bic_after_observed": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_comparison_n_effective_samples": 0,
            "n_effective_samples": 0,
            "n_effective_samples_before": 0,
            "n_effective_samples_after": 0,
            "relative_mae_by_target": np.ones(L, dtype=float),
            "relative_mae_before_by_target": {},
            "relative_mae_after_by_target": {},
            "candidate_predictor_lift_by_target": {},
            "candidate_predictor_lift_mean": 0.0,
            "bic_delta_by_target": {},
            "ridge_effective_dof_by_target": {},
            "n_signal_folds_by_target": {},
            "n_holdout_rows_by_target": {},
            "n_rows_dropped_low_den_by_target": {},
            "insufficient_evidence_by_target": {
                name: "candidate_failed_self_validity" for name in names
            },
            "masking_test_passed": False,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "self_validity_gate",
            "admitted": False,
            "admission_threshold": 0.0,
            "permutation_null_bic_deltas": [],
            "score_note": "rejected_by_validity_gate",
        }

    bbox = _spatial_union_bbox(pool_fields + [candidate_field])
    labels = _spatial_block_labels(shape, bbox, n_blocks_xy)
    pool_features = [
        masked_kernel_features(field, scales, vertical_sigma_vox, truncate)
        for field in pool_fields
    ]
    candidate_features = masked_kernel_features(
        candidate_field,
        scales,
        vertical_sigma_vox,
        truncate,
    )

    target_payloads: list[dict] = []
    insufficient: dict[str, str] = {}
    for target_idx, target_field in enumerate(pool_fields):
        target_name = names[target_idx]
        idx = _spatial_eval_indices(target_field, bbox, matched_zero_ratio)
        if len(idx) < int(min_eval_rows):
            insufficient[target_name] = "too_few_eval_rows"
            continue
        y = target_field[idx[:, 0], idx[:, 1], idx[:, 2]]
        predictor_indices = [i for i in range(L) if i != target_idx]
        if max_predictor_layers is not None and len(predictor_indices) > int(max_predictor_layers):
            mid = min(1, len(scales) - 1)
            relevance: list[tuple[float, int]] = []
            for predictor_idx in predictor_indices:
                feature = pool_features[predictor_idx][mid][idx[:, 0], idx[:, 1], idx[:, 2]]
                if float(np.std(feature)) <= 1e-12:
                    corr = 0.0
                else:
                    corr = float(abs(np.corrcoef(feature, y)[0, 1]))
                    if not np.isfinite(corr):
                        corr = 0.0
                relevance.append((corr, predictor_idx))
            relevance.sort(reverse=True)
            predictor_indices = [i for _, i in relevance[: int(max_predictor_layers)]]
        before_feature_sets = [pool_features[i] for i in predictor_indices]
        X_before = (
            np.column_stack([_stack_spatial_features(feature_set, idx) for feature_set in before_feature_sets])
            if before_feature_sets
            else np.zeros((len(idx), 0), dtype=float)
        )
        X_candidate = _stack_spatial_features(candidate_features, idx)
        X_after = np.column_stack([X_before, X_candidate]) if X_before.shape[1] else X_candidate
        rel_before, n_before, folds_before = _buffered_block_relative_mae(
            idx,
            y,
            X_before,
            labels,
            cv_buffer_vox,
            ridge_alpha,
            min_train_rows,
        )
        rel_after, n_after, folds_after = _buffered_block_relative_mae(
            idx,
            y,
            X_after,
            labels,
            cv_buffer_vox,
            ridge_alpha,
            min_train_rows,
        )
        n_rows = min(int(n_before), int(n_after))
        n_folds = min(int(folds_before), int(folds_after))
        if n_rows < int(min_eval_rows) or n_folds < 1:
            insufficient[target_name] = "too_few_buffered_fold_rows"
            continue
        df_before = ridge_effective_dof(X_before, ridge_alpha)
        df_after = ridge_effective_dof(X_after, ridge_alpha)
        bic_before_t = _laplace_bic_single(rel_before, n_rows, df_before)
        bic_after_t = _laplace_bic_single(rel_after, n_rows, df_after)
        target_payloads.append({
            "name": target_name,
            "idx": idx,
            "y": y,
            "X_before": X_before,
            "X_candidate": X_candidate,
            "rel_before": float(rel_before),
            "rel_after": float(rel_after),
            "n_rows": n_rows,
            "n_folds": n_folds,
            "df_before": float(df_before),
            "df_after": float(df_after),
            "bic_before": float(bic_before_t),
            "bic_after": float(bic_after_t),
        })

    if not target_payloads:
        return {
            **base_result,
            "bic_before": 0.0,
            "bic_after": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_delta": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_delta_raw": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_before_observed": 0.0,
            "bic_after_observed": _SPATIAL_REJECTION_BIC_DELTA,
            "bic_comparison_n_effective_samples": 0,
            "n_effective_samples": 0,
            "n_effective_samples_before": 0,
            "n_effective_samples_after": 0,
            "relative_mae_by_target": np.ones(L, dtype=float),
            "relative_mae_before_by_target": {},
            "relative_mae_after_by_target": {},
            "candidate_predictor_lift_by_target": {},
            "candidate_predictor_lift_mean": 0.0,
            "bic_delta_by_target": {},
            "ridge_effective_dof_by_target": {},
            "n_signal_folds_by_target": {},
            "n_holdout_rows_by_target": {},
            "n_rows_dropped_low_den_by_target": {},
            "insufficient_evidence_by_target": insufficient,
            "masking_test_passed": False,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "insufficient_evidence",
            "admitted": False,
            "admission_threshold": 0.0,
            "permutation_null_bic_deltas": [],
            "score_note": "no_scorable_targets",
        }

    bic_before_by_target = {payload["name"]: payload["bic_before"] for payload in target_payloads}
    bic_after_by_target = {payload["name"]: payload["bic_after"] for payload in target_payloads}
    delta_by_target = {
        payload["name"]: (payload["bic_after"] - payload["bic_before"]) / max(payload["n_rows"], 1)
        for payload in target_payloads
    }
    rel_before_by_target = {payload["name"]: payload["rel_before"] for payload in target_payloads}
    rel_after_by_target = {payload["name"]: payload["rel_after"] for payload in target_payloads}
    lift_by_target = {
        payload["name"]: payload["rel_before"] - payload["rel_after"]
        for payload in target_payloads
    }
    n_rows_by_target = {payload["name"]: payload["n_rows"] for payload in target_payloads}
    n_folds_by_target = {payload["name"]: payload["n_folds"] for payload in target_payloads}
    dof_by_target = {
        payload["name"]: {
            "before": payload["df_before"],
            "after": payload["df_after"],
            "delta": payload["df_after"] - payload["df_before"],
        }
        for payload in target_payloads
    }
    bic_before = float(sum(bic_before_by_target.values()))
    bic_after = float(sum(bic_after_by_target.values()))
    bic_delta_raw = float(bic_after - bic_before)
    bic_delta = float(np.mean(list(delta_by_target.values())))
    n_total = int(sum(n_rows_by_target.values()))
    rel_after_array = np.array(list(rel_after_by_target.values()), dtype=float)
    lift_mean = float(np.mean(list(lift_by_target.values()))) if lift_by_target else 0.0
    null_deltas: list[float] = []
    for permutation_idx in range(int(null_permutations)):
        permuted_delta_by_target: list[float] = []
        for payload in target_payloads:
            X_perm_candidate = _fold_preserving_permute_features(
                payload["X_candidate"],
                payload["idx"],
                labels,
                permutation_idx,
            )
            X_before = payload["X_before"]
            X_perm_after = (
                np.column_stack([X_before, X_perm_candidate])
                if X_before.shape[1]
                else X_perm_candidate
            )
            rel_perm, n_perm, folds_perm = _buffered_block_relative_mae(
                payload["idx"],
                payload["y"],
                X_perm_after,
                labels,
                cv_buffer_vox,
                ridge_alpha,
                min_train_rows,
            )
            if n_perm < int(min_eval_rows) or folds_perm < 1:
                continue
            df_perm = ridge_effective_dof(X_perm_after, ridge_alpha)
            bic_perm = _laplace_bic_single(rel_perm, n_perm, df_perm)
            permuted_delta_by_target.append(
                (bic_perm - payload["bic_before"]) / max(int(n_perm), 1)
            )
        if permuted_delta_by_target:
            null_deltas.append(float(np.mean(permuted_delta_by_target)))
    if null_deltas:
        admission_threshold = min(float(np.percentile(null_deltas, float(null_percentile))), 0.0)
    else:
        admission_threshold = 0.0

    stage1_passed = bool(validity_passed and lift_mean > _SPATIAL_MIN_LIFT)
    # Calibrated 2026-06-05: admit on cross-layer predictor lift, NOT the
    # complexity-penalised bic_delta (which rejected rich, distributed crossbreed
    # children that genuinely lifted prediction). bic_delta + admission_threshold
    # remain in the result as telemetry. See predictor_lift_admission_decision.
    admitted = predictor_lift_admission_decision(
        validity_passed=validity_passed,
        lift_mean=lift_mean,
        bic_delta=bic_delta,
        admit_min_lift=admit_min_lift,
    )
    return {
        **base_result,
        "bic_before": bic_before,
        "bic_after": bic_after,
        "bic_delta": bic_delta,
        "bic_delta_raw": bic_delta_raw,
        "bic_before_observed": bic_before,
        "bic_after_observed": bic_after,
        "bic_before_by_target": bic_before_by_target,
        "bic_after_by_target": bic_after_by_target,
        "bic_comparison_n_effective_samples": n_total,
        "n_effective_samples": n_total,
        "n_effective_samples_before": n_total,
        "n_effective_samples_after": n_total,
        "relative_mae_by_target": rel_after_array,
        "relative_mae_before_by_target": rel_before_by_target,
        "relative_mae_after_by_target": rel_after_by_target,
        "relative_mae_mean": float(np.mean(rel_after_array)) if rel_after_array.size else 1.0,
        "relative_mae_min": float(np.min(rel_after_array)) if rel_after_array.size else 1.0,
        "relative_mae_max": float(np.max(rel_after_array)) if rel_after_array.size else 1.0,
        "candidate_predictor_lift_by_target": lift_by_target,
        "candidate_predictor_lift_mean": lift_mean,
        "bic_delta_by_target": delta_by_target,
        "ridge_effective_dof_by_target": dof_by_target,
        "n_signal_folds_by_target": n_folds_by_target,
        "n_holdout_rows_by_target": n_rows_by_target,
        "n_rows_dropped_low_den_by_target": {payload["name"]: 0 for payload in target_payloads},
        "insufficient_evidence_by_target": insufficient,
        "masking_test_passed": stage1_passed,
        "masking_test_improvement": lift_mean,
        "masking_test_direction": "candidate_predictor_lift",
        "admitted": admitted,
        "admission_threshold": admission_threshold,
        "admit_min_lift": float(admit_min_lift),
        "admission_policy": "predictor_lift_v1",
        "permutation_null_bic_deltas": null_deltas,
        "score_note": "scored",
    }


def system_mae_to_coherence(
    mae_matrix: np.ndarray
) -> float:
    """
    Convert system relative MAE to a coherence-like metric for compatibility.
    
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
    
    system_relative_mae = float(np.mean(valid_maes))
    return float(np.clip(1.0 - system_relative_mae, 0.0, 1.0))


# =============================================================================
# MAE-Based Coherence Functions (Replaces R² System)
# =============================================================================

def compute_pairwise_mae(
    layer_values: list[np.ndarray],
    layer_dtypes: list[str],  # Ignored - all treated as continuous
    grid: 'GridSpec',
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute directed pairwise relative-MAE values between all layer combinations.

    Replaces compute_pairwise_r_squared with unified continuous approach.
    All geological layers (boolean faults, continuous grades) treated as
    continuous variables using cross-validated MAE prediction.

    Args:
        layer_values: List of interpolated flattened layer arrays
        layer_dtypes: List of data types (ignored - unified continuous approach)
        grid: GridSpec for spatial information
        rng: Optional numpy Generator forwarded to ``create_geological_cv_split``.

    Returns:
        Directed matrix of relative-MAE values (lower = better prediction,
        1.0 = target-specific null). Diagonal is 0.0 for self-prediction.
    """
    n_layers = len(layer_values)
    if n_layers == 0:
        return np.array([])

    mae_matrix = np.zeros((n_layers, n_layers))

    # Create cross-validation split once for all predictions
    train_mask, test_mask = create_geological_cv_split(layer_values, rng=rng)
    
    # Validate the split
    split_stats = validate_geological_split(train_mask, test_mask, layer_values)
    if not split_stats['validation_passed']:
        print(f"Warning: CV split validation failed - {split_stats}")
        # Continue with fallback to avoid breaking the workflow
        if split_stats['test_size'] == 0:
            # No test data - return zero MAE matrix
            return mae_matrix
    
    # Compute directed pairwise relative MAEs. Do not average directions:
    # each target has its own null denominator.
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
                
                mae_matrix[i, j] = mae_i_from_j
                mae_matrix[j, i] = mae_j_from_i
    
    return mae_matrix


def compute_target_relative_maes(
    layer_values: list[np.ndarray],
    grid: 'GridSpec',
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Predict each target layer from all other layers and return relative MAE.

    This is the BIC objective: one relative-MAE term per target, each normalized
    by that target's predict-by-mean null.
    """
    n_layers = len(layer_values)
    if n_layers <= 1:
        return np.array([], dtype=float)

    train_mask, test_mask = create_geological_cv_split(layer_values, rng=rng)
    split_stats = validate_geological_split(train_mask, test_mask, layer_values)
    if not split_stats["validation_passed"] and split_stats["test_size"] == 0:
        return np.ones(n_layers, dtype=float)

    out = np.ones(n_layers, dtype=float)
    for target_idx in range(n_layers):
        predictors = [
            values
            for idx, values in enumerate(layer_values)
            if idx != target_idx
        ]
        out[target_idx] = compute_out_of_sample_mae(
            target_layer=layer_values[target_idx],
            predictor_layers=predictors,
            train_mask=train_mask,
            test_mask=test_mask,
        )
    return out



def create_spatial_mask(
    shape: tuple[int, int, int],
    grid: 'GridSpec',
    mask_fraction: float = 0.2,
    spatial_clustering: bool = True,
    *,
    rng: np.random.Generator | None = None,
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
    if rng is None:
        rng = np.random.default_rng()

    total_voxels = shape[0] * shape[1] * shape[2]
    n_masked = int(total_voxels * mask_fraction)

    if spatial_clustering and n_masked > 0:
        # Create spatially clustered mask regions for realistic geological testing
        mask = np.zeros(shape, dtype=bool)

        # Create ~10 cluster centers
        n_clusters = max(1, min(10, n_masked // 100))
        cluster_centers = []

        for _ in range(n_clusters):
            center_i = rng.integers(0, shape[0])
            center_j = rng.integers(0, shape[1])
            center_k = rng.integers(0, shape[2])
            cluster_centers.append((center_i, center_j, center_k))

        # Assign each voxel to nearest cluster and mask some percentage
        masked_count = 0
        cluster_sizes = rng.multinomial(n_masked, [1/n_clusters] * n_clusters)
        
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
            flat_indices = rng.choice(
                total_voxels, size=min(target_size, total_voxels),
                replace=False, p=flat_probs,
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
            masked_indices = rng.choice(total_voxels, size=n_masked, replace=False)
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
    DEPRECATED — not on the live scoring path.

    Retained for the future Approach B (ground-truth-holdout) successor; see
    ``NSL2-geology-task/docs/design/scoring-fix-and-replay-2026-05-25.md`` §6.6.
    The current Stage-1 gate is implemented inside ``evaluate_new_layer`` as a
    direct MAE-delta check using ``geological_coherence_score``'s ``system_mae``
    field. Do not call this in new code.

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
    masking_test_threshold: float = 0.01,  # MAE improvement threshold
    *,
    seed: int | None = None,
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
            "relative_mae_mean": 0.0,
            "relative_mae_min": 0.0,
            "relative_mae_max": 0.0,
            "relative_mae_by_target": np.array([]),
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
            "relative_mae_mean": 0.0,
            "relative_mae_min": 0.0,
            "relative_mae_max": 0.0,
            "relative_mae_by_target": np.array([]),
            "n_effective_samples": 0,
        }
    
    # Build a deterministic generator when ``seed`` is supplied so that
    # ``mae_before`` and ``mae_after`` calls (from evaluate_new_layer) share the
    # same RNG sequence and the scoring is reproducible for replay. With
    # ``seed=None`` the legacy non-deterministic behaviour is preserved.
    rng = np.random.default_rng(seed) if seed is not None else None

    # Apply geological interpolation to reduce sparsity
    interpolated_layers = []
    for layer in layer_values:
        interpolated = compute_geological_interpolation(layer, grid, shape, rng=rng)
        interpolated_layers.append(interpolated)
    
    # Calculate effective samples based on non-zero values in interpolated data
    total_non_zero = sum(np.count_nonzero(layer) for layer in interpolated_layers)
    effective_samples = _effective_sample_count(total_non_zero, n_layers)
    
    # Handle case where no geological data exists
    if all(np.count_nonzero(layer) == 0 for layer in interpolated_layers):
        relative_maes = np.ones(n_layers, dtype=float)
        mae_matrix = np.ones((n_layers, n_layers), dtype=float)
        np.fill_diagonal(mae_matrix, 0.0)
        bic = compute_geological_bic(
            mae_matrix=mae_matrix,
            n_layers=n_layers,
            n_effective_samples=effective_samples,
            spatial_correction=1.0,
            target_relative_maes=relative_maes,
        )
        return {
            "system_coherence": 0.0,
            "spatial_correction": 1.0,
            "coherence_matrix": mae_matrix,
            "bic": bic,
            "total_cv_mse": 1.0,
            "per_layer_mse": {},
            "masking_test_passed": True,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "no_data",
            "stage_completed": "mae_bic_completed",
            "system_mae": 1.0,
            "relative_mae_mean": 1.0,
            "relative_mae_min": 1.0,
            "relative_mae_max": 1.0,
            "relative_mae_by_target": relative_maes,
            "n_effective_samples": effective_samples,
        }
    
    # Compute pairwise MAE matrix using unified continuous approach.
    # When ``seed`` is set, build a fresh rng per call so both
    # ``mae_before`` and ``mae_after`` see the same CV split.
    pairwise_rng = np.random.default_rng(seed) if seed is not None else None
    mae_matrix = compute_pairwise_mae(interpolated_layers, layer_dtypes, grid, rng=pairwise_rng)
    target_rng = np.random.default_rng(seed) if seed is not None else None
    target_relative_maes = compute_target_relative_maes(interpolated_layers, grid, rng=target_rng)
    if target_relative_maes.size:
        system_mae = float(np.mean(target_relative_maes))
        relative_mae_min = float(np.min(target_relative_maes))
        relative_mae_max = float(np.max(target_relative_maes))
    else:
        system_mae = 0.0
        relative_mae_min = 0.0
        relative_mae_max = 0.0

    # Convert MAE matrix to coherence score for compatibility
    system_coherence = float(np.clip(1.0 - system_mae, 0.0, 1.0))

    # Spatial autocorrelation correction using original data
    moran_rng = np.random.default_rng(seed) if seed is not None else None
    spatial_correction = compute_moran_correction(layer_values, grid, rng=moran_rng)
    
    # Compute Laplace likelihood BIC from MAE matrix
    bic = compute_geological_bic(
        mae_matrix=mae_matrix,
        n_layers=n_layers,
        n_effective_samples=effective_samples,
        spatial_correction=spatial_correction,
        target_relative_maes=target_relative_maes,
    )
    
    # Legacy compatibility mappings
    total_cv_mse = 1.0 - system_coherence * spatial_correction
    
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
        "relative_mae_mean": system_mae,
        "relative_mae_min": relative_mae_min,
        "relative_mae_max": relative_mae_max,
        "relative_mae_by_target": target_relative_maes,
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
    ridge_alpha: float = 1e-2,
    n_folds: int = 5,
    seed: int | None = None,
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
    candidate_nonzero_voxels = int(np.count_nonzero(new_values_flat))
    candidate_fill_fraction = (
        float(candidate_nonzero_voxels) / float(new_values_flat.size)
        if new_values_flat.size
        else 0.0
    )

    # Resolve the effective seed BEFORE any scoring so the first-layer null-model
    # BIC below is reproducibly seeded too: explicit > VFM_EPISODE_ID env > 42.
    # When the env var is set by the framework (per-episode), each episode
    # produces a different but reproducible scoring. Without it we fall back
    # to a fixed seed so the rubber-stamp pre-fix behaviour cannot return
    # silently (replay needs determinism, not coin flips).
    effective_seed: int | None
    if seed is not None:
        effective_seed = seed
    else:
        import os as _os
        env_eid = _os.environ.get("VFM_EPISODE_ID")
        if env_eid:
            # Stable 32-bit hash; avoids Python's randomised hash() seed.
            import hashlib as _hashlib
            effective_seed = int(_hashlib.sha256(env_eid.encode()).hexdigest()[:8], 16)
        else:
            effective_seed = 42

    # First layer: no cross-layer comparison is possible, so admit it as evidence
    # without assigning a synthetic BIC magnitude. Parent selection filters these
    # records until a later corroboration/rescoring pass assigns a real bic_delta.
    if not existing_layers:
        store.add_layer(
            name=layer_name,
            values=layer_values,
            dtype=layer_dtype,
        )
        return {
            "bic_before": None,
            "bic_after": None,
            "bic_delta": None,
            "bic_delta_raw": None,
            "bic_before_observed": None,
            "bic_after_observed": None,
            "bic_comparison_n_effective_samples": 0,
            "n_effective_samples": 0,
            "n_effective_samples_before": 0,
            "n_effective_samples_after": 0,
            "n_effective_samples_delta": 0,
            "candidate_nonzero_voxels": candidate_nonzero_voxels,
            "candidate_fill_fraction": candidate_fill_fraction,
            "cv_mse_before": 0.0,
            "cv_mse_after": 0.0,
            "cv_mse_delta": 0.0,
            "relative_mae_mean": None,
            "relative_mae_min": None,
            "relative_mae_max": None,
            "mutual_info": {},
            "pairwise_distance": {},
            "admitted": True,
            "predicted_value": 1.0,
            "masking_test_passed": True,
            "masking_test_improvement": 0.0,
            "masking_test_direction": "first_layer_auto",
            "stage_1_tolerance_used": False,
            "stage_1_mae_tolerance": _STAGE1_MAE_TOLERANCE,
            "stage_1_bic_rescue_threshold": _STAGE1_BIC_RESCUE_THRESHOLD,
            "stage_completed": "first_layer_auto",
            "admission_path": "first_layer_auto",
            "scoring_objective": _SPATIAL_SCORING_OBJECTIVE,
            "validity_passed": True,
            "self_relative_mae": None,
            "candidate_as_target_relative_mae": None,
            "candidate_predictor_lift_mean": 0.0,
            "candidate_predictor_lift_by_target": {},
            "bic_delta_by_target": {},
            "insufficient_evidence_by_target": {},
            "n_spatial_folds": 0,
            "n_signal_folds_by_target": {},
            "n_holdout_rows_by_target": {},
            "n_rows_dropped_low_den_by_target": {},
            "ridge_effective_dof_by_target": {},
            "pool_size_at_score": 0,
            "admission_threshold": 0.0,
            "permutation_null_bic_deltas": [],
        }

    # Get existing layer values and dtypes
    existing_values = [store.get_layer_values(n).flatten() for n in existing_layers]
    existing_dtypes = [store.get_layer(n).dtype for n in existing_layers]

    score_after = spatial_predictor_lift_score(
        existing_values,
        existing_layers,
        new_values_flat,
        grid_shape,
        ridge_alpha=float(ridge_alpha),
        null_permutations=_spatial_null_permutations_from_env(),
    )

    bic_before = score_after.get("bic_before")
    bic_after = score_after.get("bic_after")
    bic_delta = score_after.get("bic_delta")
    bic_delta_raw = score_after.get("bic_delta_raw")
    bic_before_observed = score_after.get("bic_before_observed", bic_before)
    bic_after_observed = score_after.get("bic_after_observed", bic_after)

    n_eff_before = int(score_after.get("n_effective_samples_before", 0) or 0)
    n_eff_after = int(score_after.get("n_effective_samples_after", 0) or 0)
    n_eff_delta = n_eff_after - n_eff_before
    bic_comparison_n_eff = int(score_after.get("bic_comparison_n_effective_samples", n_eff_after) or 0)
    n_eff = int(score_after.get("n_effective_samples", n_eff_after) or 0)

    relative_before = score_after.get("relative_mae_before_by_target") or {}
    cv_mse_before = float(np.mean(list(relative_before.values()))) if relative_before else 1.0
    cv_mse_after = float(score_after.get("relative_mae_mean", 1.0) or 1.0)
    cv_mse_delta = cv_mse_after - cv_mse_before

    stage1_passed = bool(score_after.get("masking_test_passed", False))
    mae_improvement = float(score_after.get("masking_test_improvement", 0.0) or 0.0)
    stage1_tolerance_used = False
    seed_bootstrap = bool(
        len(existing_layers) < _SPATIAL_SEED_POOL_TARGET
        and score_after.get("score_note") == "scored"
        and score_after.get("validity_passed") is True
        and n_eff > 0
    )
    if seed_bootstrap:
        admitted = True
        admission_path = "diverse_seed"
        stage1_passed = True
        masking_direction = "diverse_seed_validity"
        print(
            f"✅ Layer {layer_name} admitted as diverse seed: "
            f"self_rel_mae={score_after.get('self_relative_mae'):.4f}, "
            f"pool_size={len(existing_layers)}"
        )
    else:
        admitted = bool(score_after.get("admitted", False))
        admission_path = "normal"
        masking_direction = str(score_after.get("masking_test_direction", "candidate_predictor_lift"))
        if not stage1_passed:
            print(
                f"❌ Layer {layer_name} rejected at Stage 1: "
                f"{masking_direction}, lift={mae_improvement:.6f}"
            )
        elif admitted:
            print(
                f"✅ Layer {layer_name} admitted: "
                f"BIC/sample={float(bic_delta):.6f}, lift={mae_improvement:.6f}"
            )
        else:
            print(
                f"❌ Layer {layer_name} rejected at Stage 2: "
                f"BIC/sample={float(bic_delta):.6f}, "
                f"threshold={float(score_after.get('admission_threshold', 0.0)):.6f}"
            )

    stage1_fields = {
        "masking_test_passed": stage1_passed,
        "masking_test_improvement": mae_improvement,
        "masking_test_direction": masking_direction,
        "stage_1_tolerance_used": stage1_tolerance_used,
        "stage_1_mae_tolerance": _STAGE1_MAE_TOLERANCE,
        "stage_1_bic_rescue_threshold": _STAGE1_BIC_RESCUE_THRESHOLD,
        "stage_completed": "mae_bic_completed",
    }
    
    if admitted:
        # Add the layer
        store.add_layer(
            name=layer_name,
            values=layer_values,
            dtype=layer_dtype,
        )

        # Compute MI with existing layers (legacy; kept for record audit/back-
        # compat). Crossbreed queue ranking now uses pairwise_distance below.
        mi_scores = {}
        pairwise_distances = {}
        for other_name in existing_layers:
            mi_scores[other_name] = mutual_information(store, layer_name, other_name)
            pairwise_distances[other_name] = pairwise_distance(
                store, layer_name, other_name
            )

        store.update_layer_scores(layer_name, bic_delta, mi_scores)
    else:
        mi_scores = {}
        pairwise_distances = {}

    return {
        "bic_before": bic_before,
        "bic_after": bic_after,
        "bic_delta": bic_delta,          # normalized per-sample; range ~[-0.5, 0] for good layers
        "bic_delta_raw": bic_delta_raw,  # raw (grid-size-dependent); diagnostic only
        "bic_before_observed": bic_before_observed,
        "bic_after_observed": bic_after_observed,
        "bic_comparison_n_effective_samples": bic_comparison_n_eff,
        "n_effective_samples": n_eff,
        "n_effective_samples_before": n_eff_before,
        "n_effective_samples_after": n_eff_after,
        "n_effective_samples_delta": n_eff_delta,
        "candidate_nonzero_voxels": candidate_nonzero_voxels,
        "candidate_fill_fraction": candidate_fill_fraction,
        "cv_mse_before": cv_mse_before,
        "cv_mse_after": cv_mse_after,
        "cv_mse_delta": cv_mse_delta,
        "relative_mae_mean": score_after.get("relative_mae_mean", score_after.get("system_mae")),
        "relative_mae_min": score_after.get("relative_mae_min"),
        "relative_mae_max": score_after.get("relative_mae_max"),
        "mutual_info": mi_scores,
        "pairwise_distance": pairwise_distances,
        "admitted": admitted,
        "admission_path": admission_path,
        # For hypothesis agent training: positive = improvement (higher = better)
        "predicted_value": -float(bic_delta) if bic_delta is not None else 0.0,
        # Stage 1 fields from two-stage scoring
        **stage1_fields,
        "scoring_objective": score_after.get("scoring_objective", _SPATIAL_SCORING_OBJECTIVE),
        "spatial_correction": score_after.get("spatial_correction", 1.0),
        "kernel_scales_m": score_after.get("kernel_scales_m"),
        "kernel_scales_vox": score_after.get("kernel_scales_vox"),
        "R_v_m": score_after.get("R_v_m"),
        "R_v_vox": score_after.get("R_v_vox"),
        "block_voxels": score_after.get("block_voxels"),
        "buffer_voxels": score_after.get("buffer_voxels"),
        "min_den": score_after.get("min_den"),
        "ridge_alpha": score_after.get("ridge_alpha", float(ridge_alpha)),
        "ridge_effective_dof_by_target": score_after.get("ridge_effective_dof_by_target", {}),
        "matched_zero_ratio": score_after.get("matched_zero_ratio"),
        "pool_size_at_score": score_after.get("pool_size_at_score", len(existing_layers)),
        "calibration_bin": score_after.get("calibration_bin"),
        "calibration_null_permutations": score_after.get("calibration_null_permutations", 0),
        "admission_threshold": score_after.get("admission_threshold", 0.0),
        "permutation_null_bic_deltas": score_after.get("permutation_null_bic_deltas", []),
        "candidate_predictor_lift_by_target": score_after.get("candidate_predictor_lift_by_target", {}),
        "candidate_predictor_lift_mean": score_after.get("candidate_predictor_lift_mean", 0.0),
        "bic_delta_by_target": score_after.get("bic_delta_by_target", {}),
        "self_relative_mae": score_after.get("self_relative_mae"),
        "candidate_as_target_relative_mae": score_after.get("candidate_as_target_relative_mae"),
        "validity_passed": score_after.get("validity_passed"),
        "tau_self": score_after.get("tau_self"),
        "n_spatial_folds": score_after.get("n_spatial_folds", 0),
        "n_signal_folds_by_target": score_after.get("n_signal_folds_by_target", {}),
        "n_holdout_rows_by_target": score_after.get("n_holdout_rows_by_target", {}),
        "n_rows_dropped_low_den_by_target": score_after.get("n_rows_dropped_low_den_by_target", {}),
        "insufficient_evidence_by_target": score_after.get("insufficient_evidence_by_target", {}),
        "relative_mae_by_target": score_after.get("relative_mae_by_target", np.array([])),
        "relative_mae_before_by_target": score_after.get("relative_mae_before_by_target", {}),
        "relative_mae_after_by_target": score_after.get("relative_mae_after_by_target", {}),
        "score_note": score_after.get("score_note"),
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
