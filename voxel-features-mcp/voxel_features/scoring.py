"""Geological coherence scoring for feature layers using information-theoretic principles.

The scoring criterion determines whether a new feature layer improves
the geological system's mutual predictability (geological harmony).

Core Philosophy: 
In a real geological system, accurate measurements should be mutually 
predictive because they reflect the same underlying geological processes.
This is the "Anna Karenina principle" for geology: coherent geological 
systems are alike (features predict each other), while incoherent systems
fail in their own ways.

Geological Coherence Scoring:
- Compute pairwise R² between all layer combinations (type-aware correlation)
- System coherence = average R² across mutual predictability matrix  
- Apply Moran's I spatial autocorrelation correction to prevent cheat-code layers
- BIC = -system_coherence × spatial_correction + complexity_penalty
- Admission: BIC improvement (lower is better)

Key Improvements over Legacy System:
1. Proper BIC: Uses layers as features, not voxels as samples
2. R² normalization: Boolean/float layers contribute equally to coherence
3. True joint model: Measures system-wide geological harmony
4. Spatial correction: Prevents near-identical layers from gaming the system
5. Type-aware correlation: Phi coefficient, point-biserial, and Pearson

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
    Compute pairwise R² values between all layer combinations.
    
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
    
    return r_squared_matrix


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
# Adaptive Multi-Resolution for Sparse Geological Grids
# =============================================================================

def compute_local_data_density(
    layer_values: np.ndarray,
    shape: tuple[int, int, int],
    window_size: tuple[int, int, int] = (20, 20, 2)
) -> np.ndarray:
    """
    Analyze local data density across the grid to inform resolution decisions.
    
    Args:
        layer_values: Flattened layer array
        shape: Original (nx, ny, nz) grid shape
        window_size: Size of analysis windows for density calculation
        
    Returns:
        3D density map showing data coverage in each region
    """
    # Reshape to 3D for analysis
    layer_3d = layer_values.reshape(shape)
    
    # Initialize density map
    density_map = np.zeros(shape)
    
    # Calculate density in overlapping windows
    wx, wy, wz = window_size
    for x in range(0, shape[0], wx//2):
        for y in range(0, shape[1], wy//2):
            for z in range(0, shape[2], wz//2):
                # Define window bounds
                x_end = min(x + wx, shape[0])
                y_end = min(y + wy, shape[1])
                z_end = min(z + wz, shape[2])
                
                # Extract window
                window = layer_3d[x:x_end, y:y_end, z:z_end]
                
                # Calculate density (proportion of non-zero values)
                total_voxels = window.size
                data_voxels = np.count_nonzero(window)
                density = data_voxels / total_voxels if total_voxels > 0 else 0.0
                
                # Assign density to all voxels in window
                density_map[x:x_end, y:y_end, z:z_end] = np.maximum(
                    density_map[x:x_end, y:y_end, z:z_end], density
                )
    
    return density_map


def create_adaptive_resolution_map(
    density_map: np.ndarray,
    shape: tuple[int, int, int]
) -> dict:
    """
    Assign resolution levels based on local data density.
    
    Args:
        density_map: 3D density map from compute_local_data_density
        shape: Original grid shape
        
    Returns:
        Dict with resolution assignments and aggregation info
    """
    # Resolution thresholds and corresponding scales
    thresholds = {
        'full': 0.01,      # > 1% coverage → full resolution
        'medium': 0.001,   # 0.1-1% → medium resolution (4x4x2 blocks)
        'coarse': 0.0001,  # 0.01-0.1% → coarse resolution (8x8x4 blocks) 
        'regional': 0.0    # < 0.01% → regional resolution (20x20x8 blocks)
    }
    
    # Scale factors for each resolution level
    scale_factors = {
        'full': (1, 1, 1),
        'medium': (4, 4, 2),
        'coarse': (8, 8, 4),
        'regional': (20, 20, 8)
    }
    
    # Create resolution assignment map
    resolution_map = np.full(shape, 'regional', dtype='U8')
    
    for level, threshold in thresholds.items():
        if level == 'regional':
            continue  # Already initialized
        mask = density_map >= threshold
        resolution_map[mask] = level
    
    return {
        'resolution_map': resolution_map,
        'scale_factors': scale_factors,
        'density_map': density_map
    }


def aggregate_sparse_regions(
    layer_values: np.ndarray,
    resolution_info: dict,
    shape: tuple[int, int, int]
) -> tuple[np.ndarray, int]:
    """
    Aggregate sparse regions into larger blocks while preserving dense areas.
    
    Uses maximum pooling to preserve geological signal in sparse data.
    
    Args:
        layer_values: Flattened layer array  
        resolution_info: Resolution map and scale factors
        shape: Original grid shape
        
    Returns:
        Tuple of (aggregated_values, effective_sample_count)
    """
    layer_3d = layer_values.reshape(shape)
    resolution_map = resolution_info['resolution_map']
    scale_factors = resolution_info['scale_factors']
    
    # Track effective samples
    effective_samples = 0
    aggregated_data = []
    
    # Process each resolution level
    for resolution_level in ['full', 'medium', 'coarse', 'regional']:
        # Find regions at this resolution level
        mask = resolution_map == resolution_level
        if not np.any(mask):
            continue
            
        scale_x, scale_y, scale_z = scale_factors[resolution_level]
        
        # Aggregate data at this resolution level
        if scale_x == 1 and scale_y == 1 and scale_z == 1:
            # Full resolution - use original data where mask is True
            masked_data = layer_3d * mask
            non_zero_data = masked_data[masked_data != 0]
            if len(non_zero_data) > 0:
                aggregated_data.extend(non_zero_data)
                effective_samples += len(non_zero_data)
        else:
            # Downsample using maximum pooling
            nx, ny, nz = shape
            new_nx = max(1, nx // scale_x)
            new_ny = max(1, ny // scale_y) 
            new_nz = max(1, nz // scale_z)
            
            for i in range(new_nx):
                for j in range(new_ny):
                    for k in range(new_nz):
                        # Define block boundaries
                        x_start, x_end = i * scale_x, min((i + 1) * scale_x, nx)
                        y_start, y_end = j * scale_y, min((j + 1) * scale_y, ny)
                        z_start, z_end = k * scale_z, min((k + 1) * scale_z, nz)
                        
                        # Extract block
                        data_block = layer_3d[x_start:x_end, y_start:y_end, z_start:z_end]
                        mask_block = mask[x_start:x_end, y_start:y_end, z_start:z_end]
                        
                        # Only process if this block is assigned to current resolution
                        if np.any(mask_block):
                            # Use maximum pooling to preserve geological signal
                            block_max = np.max(data_block * mask_block)
                            if block_max > 0:
                                aggregated_data.append(block_max)
                                effective_samples += 1
    
    # Convert to array and add zeros to maintain some zero representation
    aggregated_array = np.array(aggregated_data) if aggregated_data else np.array([])
    
    return aggregated_array, effective_samples


def compute_effective_sample_size(
    layer_values_list: list[np.ndarray],
    grid: 'GridSpec',
    shape: tuple[int, int, int]
) -> tuple[list[np.ndarray], int]:
    """
    Compute effective sample sizes for all layers using adaptive resolution.
    
    Args:
        layer_values_list: List of flattened layer arrays
        grid: Grid specification  
        shape: Original grid shape
        
    Returns:
        Tuple of (adaptively_aggregated_layers, total_effective_samples)
    """
    if not layer_values_list:
        return [], 0
    
    # Analyze data density across all layers
    combined_density = np.zeros(shape)
    for layer_values in layer_values_list:
        layer_density = compute_local_data_density(layer_values, shape)
        combined_density = np.maximum(combined_density, layer_density)
    
    # Create unified resolution map based on combined density
    resolution_info = create_adaptive_resolution_map(combined_density, shape)
    
    # Aggregate all layers using the same resolution map
    aggregated_layers = []
    layer_sample_counts = []
    
    for layer_values in layer_values_list:
        aggregated_layer, layer_samples = aggregate_sparse_regions(
            layer_values, resolution_info, shape
        )
        aggregated_layers.append(aggregated_layer)
        layer_sample_counts.append(layer_samples)
    
    # Find the maximum effective samples across all layers
    max_effective_samples = max(layer_sample_counts) if layer_sample_counts else 0
    
    # Pad all layers to have the same effective sample size
    # This ensures pairwise R² calculations work correctly
    padded_layers = []
    for aggregated_layer in aggregated_layers:
        if len(aggregated_layer) < max_effective_samples:
            # Pad with zeros to reach consistent size
            padding_needed = max_effective_samples - len(aggregated_layer)
            padded_layer = np.concatenate([aggregated_layer, np.zeros(padding_needed)])
        elif len(aggregated_layer) > max_effective_samples:
            # Truncate to consistent size (shouldn't happen with current logic)
            padded_layer = aggregated_layer[:max_effective_samples]
        else:
            padded_layer = aggregated_layer
        
        padded_layers.append(padded_layer)
    
    return padded_layers, max_effective_samples


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


def compute_esa_bic(
    system_coherence: float,
    spatial_correction: float,
    n_layers: int,
    effective_samples: int,
    total_voxels: int
) -> float:
    """
    Compute Effective Sample Size Adjusted BIC.
    
    Applies density-weighted penalty to standard BIC calculation
    to account for sparsity in geological data.
    
    Args:
        system_coherence: Average R² across layer pairs
        spatial_correction: Moran's I correction factor
        n_layers: Number of layers
        effective_samples: Effective sample size from adaptive resolution
        total_voxels: Total number of voxels in grid
        
    Returns:
        ESA-BIC score (lower = better)
    """
    # Standard BIC components
    n_params = n_layers * (n_layers - 1) // 2
    n_samples = max(effective_samples, n_layers)
    
    # Base coherence term
    coherence_term = -system_coherence * spatial_correction
    
    # Standard complexity penalty
    if n_samples > 0 and n_params > 0:
        complexity_penalty = n_params * np.log(n_samples) / n_samples
    else:
        complexity_penalty = 0.0
    
    standard_bic = coherence_term + complexity_penalty
    
    # Effective Sample Size Adjustment
    density_weight = effective_samples / max(total_voxels, 1)
    density_weight = max(density_weight, 0.001)  # Minimum threshold for numerical stability
    
    # Apply sparsity penalty: penalize low-density datasets more heavily
    sparsity_penalty = 1.0 + np.log(1.0 / density_weight)
    
    esa_bic = standard_bic * sparsity_penalty
    
    return esa_bic


def geological_coherence_score(
    layer_values: list[np.ndarray],
    layer_dtypes: list[str],
    grid: 'GridSpec',
    shape: tuple[int, int, int],
) -> dict:
    """
    Compute geological coherence score for a set of layers.
    
    Measures how well layers mutually predict each other (geological harmony).
    System coherence = average R² across the mutual predictability matrix.
    
    Features:
    - Geological interpolation: Extends features within 7x voxel-size radius
    - Spatial autocorrelation correction: Moran's I to prevent cheat-code layers
    - ESA-BIC: Effective Sample Size adjustment for sparse data
    
    Args:
        layer_values: List of flattened layer arrays
        layer_dtypes: List of data types for each layer
        grid: GridSpec for spatial coordinate information
        shape: Original (nx, ny, nz) shape (for compatibility)
    
    Returns:
        dict with:
            - system_coherence: Average R² across layer pairs
            - spatial_correction: Moran's I correction factor
            - coherence_matrix: Full R² matrix between layers
            - bic: ESA-BIC score (lower = better)
            - total_cv_mse: Legacy compatibility (set to 1 - coherence)
            - per_layer_mse: Legacy compatibility (empty dict)
    """
    n_layers = len(layer_values)
    
    if n_layers < 2:
        return {
            "system_coherence": 0.0,
            "spatial_correction": 1.0,
            "coherence_matrix": np.array([]),
            "bic": 0.0,
            "total_cv_mse": 0.0,
            "per_layer_mse": {},
        }
    
    # Apply geological interpolation to reduce sparsity
    # This extends known geological features within influence spheres
    interpolated_layers = []
    for layer in layer_values:
        interpolated = compute_geological_interpolation(layer, grid, shape)
        interpolated_layers.append(interpolated)
    
    # Use standard resolution on interpolated data
    # No adaptive aggregation - work directly with interpolated layers
    
    # Calculate effective samples based on non-zero values in interpolated data
    total_non_zero = sum(np.count_nonzero(layer) for layer in interpolated_layers)
    effective_samples = max(total_non_zero, n_layers)  # At least n_layers samples
    
    # Handle case where no data exists
    if all(np.count_nonzero(layer) == 0 for layer in interpolated_layers):
        return {
            "system_coherence": 0.0,
            "spatial_correction": 1.0,
            "coherence_matrix": np.array([]),
            "bic": float('inf'),  # Infinite BIC for empty data
            "total_cv_mse": 1.0,
            "per_layer_mse": {},
        }
    
    # Normalize interpolated layers for fair comparison
    normalized_values = normalize_layers(interpolated_layers, layer_dtypes)
    
    # Compute pairwise R² matrix on interpolated data
    r_squared_matrix = compute_pairwise_r_squared(normalized_values, layer_dtypes)
    
    # System coherence = average off-diagonal R²
    mask = ~np.eye(n_layers, dtype=bool)  # Exclude diagonal (self-correlations)
    system_coherence = np.mean(r_squared_matrix[mask])
    
    # Spatial autocorrelation correction (still use original data for spatial analysis)
    spatial_correction = compute_moran_correction(layer_values, grid)
    
    # Effective Sample Size Adjusted BIC score with geological interpolation
    # ESA-BIC accounts for sparsity without adaptive resolution
    # Uses interpolated data at standard grid resolution
    total_voxels = grid.n_voxels
    bic = compute_esa_bic(
        system_coherence=system_coherence,
        spatial_correction=spatial_correction,
        n_layers=n_layers,
        effective_samples=effective_samples,
        total_voxels=total_voxels
    )
    
    # For compatibility with existing code, map coherence to MSE-like metric
    # High coherence -> low "MSE", low coherence -> high "MSE"
    total_cv_mse = 1.0 - system_coherence * spatial_correction
    
    return {
        "system_coherence": system_coherence,
        "spatial_correction": spatial_correction,
        "coherence_matrix": r_squared_matrix,
        "bic": bic,
        "total_cv_mse": total_cv_mse,
        "per_layer_mse": {},  # Legacy compatibility
    }


# =============================================================================
# Joint Prediction Scoring (Ridge Regression + BIC) - LEGACY
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
    
    Uses geological coherence scoring:
    1. Compute geological coherence score WITHOUT the new layer
    2. Compute geological coherence score WITH the new layer
    3. Admit if BIC improves (lower is better)
    
    Args:
        store: VoxelStore to evaluate against
        layer_name: Name for the new layer
        layer_values: 3D array of values
        layer_dtype: Data type ("float", "categorical", "boolean")
        ridge_alpha: Ridge regularization strength (legacy compatibility)
        n_folds: Number of CV folds (legacy compatibility)
    
    Returns:
        dict with:
            - bic_before/after/delta: BIC scores
            - cv_mse_before/after/delta: Cross-validated MSE (coherence-based)
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
            "cv_mse_before": 0.0,
            "cv_mse_after": 0.0,
            "cv_mse_delta": 0.0,
            "mutual_info": {},
            "admitted": True,
            "predicted_value": 1.0,
        }
    
    # Get existing layer values and dtypes
    existing_values = [store.get_layer_values(n).flatten() for n in existing_layers]
    existing_dtypes = [store.get_layer(n).dtype for n in existing_layers]
    
    # Score WITHOUT new layer
    if len(existing_values) >= 2:
        score_before = geological_coherence_score(
            existing_values, existing_dtypes, store.grid, grid_shape
        )
    else:
        score_before = {"bic": 0.0, "total_cv_mse": 0.0, "per_layer_mse": {}}
    
    # Score WITH new layer
    all_values = existing_values + [new_values_flat]
    all_dtypes = existing_dtypes + [layer_dtype]
    score_after = geological_coherence_score(
        all_values, all_dtypes, store.grid, grid_shape
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
