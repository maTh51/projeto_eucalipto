"""Heuristic trunk extraction based on geometric features.

Refactored from remove_old.py. Intended primarily for use on per-tree
point clouds produced by treeiso (artemis_treeiso), where no semantic
trunk labels are available.
"""

from typing import Tuple, Dict, Optional
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

# Configure module logger
logger = logging.getLogger(__name__)


def remove_outliers_iqr(points: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """Remove outliers using IQR method on horizontal distances from median.
    
    Helps axis estimation by removing branches far from the trunk center.
    Returns boolean mask of inliers.
    """
    xy = points[:, :2]
    median = np.median(xy, axis=0)
    distances = np.linalg.norm(xy - median, axis=1)
    
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_factor * iqr
    
    mask = distances <= upper_bound
    n_before = len(points)
    n_after = np.sum(mask)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Outlier removal: {n_before} → {n_after} points "
                    f"(removed {n_before - n_after})")
    
    return mask


def estimate_trunk_axis(points: np.ndarray, 
                       base_percentile: float = 40.0,
                       remove_xy_outliers: bool = True,
                       iqr_factor: float = 1.5) -> Tuple[np.ndarray, Dict]:
    """Estimate trunk axis using PCA on lower half, with outlier removal.
    
    Returns
    -------
    axis : np.ndarray - unit vector pointing upward
    diagnostics : Dict - information about axis estimation
    """
    z = points[:, 2]
    
    # Take lower half of tree
    mask = z < np.percentile(z, base_percentile)
    lower_trunk_points = points[mask]
    
    if len(lower_trunk_points) < 10:
        logger.warning(f"Few points in lower half ({len(lower_trunk_points)}), using full cloud")
        lower_trunk_points = points
    
    n_before_outlier = len(lower_trunk_points)
    
    # Remove XY outliers if requested
    if remove_xy_outliers:
        outlier_mask = remove_outliers_iqr(lower_trunk_points, iqr_factor=iqr_factor)
        lower_trunk_points = lower_trunk_points[outlier_mask]
    
    n_after_outlier = len(lower_trunk_points)
    
    cov = np.cov(lower_trunk_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    
    # Force axis to point upward (Z positive)
    if axis[2] < 0:
        axis = -axis
    
    axis = axis / np.linalg.norm(axis)
    
    diagnostics = {
        "base_percentile": base_percentile,
        "n_lower_points": n_after_outlier,
        "n_outliers_removed": n_before_outlier - n_after_outlier,
        "axis_z_component": float(axis[2]),
        "eigvals": eigvals.tolist(),
    }
    
    return axis, diagnostics



def estimate_trunk_center(points: np.ndarray, base_percentile: float) -> np.ndarray:
    z = points[:, 2]
    mask = z < np.percentile(z, base_percentile)
    base_pts = points[mask]
    center_xy = base_pts[:, :2].mean(axis=0)
    return center_xy


def distance_to_axis(points: np.ndarray,
                     axis: np.ndarray,
                     center_xy: np.ndarray) -> np.ndarray:
    p0 = np.array([center_xy[0], center_xy[1], points[:, 2].min()])
    vecs = points - p0
    proj = np.dot(vecs, axis)
    proj_points = np.outer(proj, axis)
    dist = np.linalg.norm(vecs - proj_points, axis=1)
    return dist


def compute_geometric_features(points: np.ndarray,
                               k_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Compute local geometric features: linearity, scattering, and verticality.
    
    Returns
    -------
    linearity : np.ndarray - (l1 - l2) / l1
    scattering : np.ndarray - l3 / l1
    verticality : np.ndarray - |z-component of first eigenvector|
    diagnostics : Dict - statistics about features
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    _, idx = nbrs.kneighbors(points)

    linearity = np.zeros(len(points))
    scattering = np.zeros(len(points))
    verticality = np.zeros(len(points))

    for i in range(len(points)):
        neighbors = points[idx[i]]
        cov = np.cov(neighbors.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort from largest to smallest
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices] + 1e-8
        eigvecs = eigvecs[:, sort_indices]

        l1, l2, l3 = eigvals

        linearity[i] = (l1 - l2) / l1
        scattering[i] = l3 / l1
        verticality[i] = abs(eigvecs[2, 0])

    diagnostics = {
        "k_neighbors": k_neighbors,
        "linearity_mean": float(np.mean(linearity)),
        "linearity_std": float(np.std(linearity)),
        "linearity_range": [float(np.min(linearity)), float(np.max(linearity))],
        "scattering_mean": float(np.mean(scattering)),
        "scattering_std": float(np.std(scattering)),
        "scattering_range": [float(np.min(scattering)), float(np.max(scattering))],
        "verticality_mean": float(np.mean(verticality)),
        "verticality_std": float(np.std(verticality)),
        "verticality_range": [float(np.min(verticality)), float(np.max(verticality))],
    }

    return linearity, scattering, verticality, diagnostics



def classify_wood(points: np.ndarray,
                  base_percentile: float = 20.0,
                  k_neighbors: int = 25,
                  linearity_threshold: float = 0.3,
                  scattering_threshold: float = 0.4,
                  verticality_threshold: float = 0.85,
                  max_trunk_radius: float = 0.4,
                  remove_xy_outliers: bool = True,
                  use_verticality_as_hard_criterion: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Classify wood points using multiple geometric criteria.
    
    Parameters
    ----------
    use_verticality_as_hard_criterion : bool
        If False, verticality is only used in wood_score (more permissive).
        If True, verticality is a hard threshold (original behavior).
    
    Returns
    -------
    wood_mask : np.ndarray (bool) - points classified as trunk
    wood_score : np.ndarray (float) - confidence score [0, 1]
    dist_axis : np.ndarray (float) - distance to trunk axis
    diagnostics : Dict - detailed extraction diagnostics
    """
    axis, axis_diag = estimate_trunk_axis(points, base_percentile=base_percentile, 
                                         remove_xy_outliers=remove_xy_outliers)
    center_xy = estimate_trunk_center(points, base_percentile)
    dist_axis = distance_to_axis(points, axis, center_xy)
    
    lin, sca, vert, features_diag = compute_geometric_features(points, k_neighbors)

    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

    # Apply classification criteria
    crit_dist = dist_axis < max_trunk_radius
    crit_lin = lin > linearity_threshold
    crit_sca = sca < scattering_threshold
    crit_height = z_norm < 1.0  # Allow up to top of tree (was 0.90)

    if use_verticality_as_hard_criterion:
        crit_vert = vert > verticality_threshold
        wood_mask = crit_dist & crit_lin & crit_sca & crit_vert & crit_height
    else:
        # Verticality only used for scoring, not hard cutoff
        # This is more permissive and avoids losing trunk points
        wood_mask = crit_dist & crit_lin & crit_sca & crit_height
    
    # Compute confidence score - incorporate verticality as weight
    dist_score = np.clip(1 - (dist_axis / max_trunk_radius), 0, 1)
    vert_weight = vert  # Higher verticality = higher confidence
    
    if use_verticality_as_hard_criterion:
        wood_score = 0.5 * dist_score + 0.3 * lin + 0.15 * (1 - sca) + 0.05 * vert_weight
    else:
        # Verticality has less weight when it's not a hard criterion
        wood_score = 0.4 * dist_score + 0.4 * lin + 0.15 * (1 - sca) + 0.05 * vert_weight

    # Count points passing each criterion
    n_points = len(points)
    n_pass_dist = np.sum(crit_dist)
    n_pass_lin = np.sum(crit_lin)
    n_pass_sca = np.sum(crit_sca)
    n_pass_vert = np.sum(vert > verticality_threshold) if use_verticality_as_hard_criterion else np.sum(vert > 0.6)
    n_pass_height = np.sum(crit_height)
    n_wood = np.sum(wood_mask)
    
    diagnostics = {
        "n_total_points": n_points,
        "distance_criterion": {
            "threshold_m": max_trunk_radius,
            "n_pass": int(n_pass_dist),
            "pct_pass": float(100 * n_pass_dist / n_points),
        },
        "linearity_criterion": {
            "threshold": linearity_threshold,
            "n_pass": int(n_pass_lin),
            "pct_pass": float(100 * n_pass_lin / n_points),
        },
        "scattering_criterion": {
            "threshold": scattering_threshold,
            "n_pass": int(n_pass_sca),
            "pct_pass": float(100 * n_pass_sca / n_points),
        },
        "verticality_criterion": {
            "threshold": verticality_threshold,
            "n_pass": int(n_pass_vert),
            "pct_pass": float(100 * n_pass_vert / n_points),
            "used_as_hard_criterion": use_verticality_as_hard_criterion,
        },
        "height_criterion": {
            "n_pass": int(n_pass_height),
            "pct_pass": float(100 * n_pass_height / n_points),
        },
        "n_wood_points": int(n_wood),
        "pct_wood": float(100 * n_wood / n_points),
        "axis_diagnostics": axis_diag,
        "features_diagnostics": features_diag,
    }
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Wood classification: {n_wood}/{n_points} points "
                    f"({100*n_wood/n_points:.1f}%)")
        logger.debug(f"  Distance: {n_pass_dist}/{n_points} ({100*n_pass_dist/n_points:.1f}%)")
        logger.debug(f"  Linearity: {n_pass_lin}/{n_points} ({100*n_pass_lin/n_points:.1f}%)")
        logger.debug(f"  Scattering: {n_pass_sca}/{n_points} ({100*n_pass_sca/n_points:.1f}%)")
        logger.debug(f"  Verticality: {n_pass_vert}/{n_points} ({100*n_pass_vert/n_points:.1f}%)")

    return wood_mask, wood_score, dist_axis, diagnostics




def expand_wood(points: np.ndarray,
                wood_mask: np.ndarray,
                expansion_radius: float = 0.2) -> np.ndarray:
    nbrs = NearestNeighbors(radius=expansion_radius).fit(points)
    wood_points = points[wood_mask]
    expanded_mask = wood_mask.copy()

    for p in wood_points:
        neighbors = nbrs.radius_neighbors([p], return_distance=False)[0]
        expanded_mask[neighbors] = True

    return expanded_mask


def largest_component(points: np.ndarray,
                      mask: np.ndarray,
                      component_radius: float = 0.2) -> np.ndarray:
    filtered_points = points[mask]
    tree = cKDTree(filtered_points)

    n = len(filtered_points)
    visited = np.zeros(n, dtype=bool)
    components = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        comp = []

        while stack:
            idx = stack.pop()

            if visited[idx]:
                continue

            visited[idx] = True
            comp.append(idx)

            neighbors = tree.query_ball_point(filtered_points[idx], component_radius)
            stack.extend(neighbors)

        components.append(comp)

    largest = max(components, key=len)

    new_mask = np.zeros(len(points), dtype=bool)
    idxs = np.where(mask)[0]
    new_mask[idxs[largest]] = True

    return new_mask


def extract_trunk(points: np.ndarray,
                  base_percentile: float = 20.0,
                  k_neighbors: int = 25,
                  linearity_threshold: float = 0.4,
                  scattering_threshold: float = 0.4,
                  verticality_threshold: float = 0.85,
                  max_trunk_radius: float = 0.4,
                  expansion_radius: float = 0.2,
                  component_radius: float = 0.2,
                  min_trunk_points: int = 10,
                  use_verticality_as_hard_criterion: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Full heuristic pipeline: return trunk-only mask and scores with diagnostics.

    Parameters
    ----------
    use_verticality_as_hard_criterion : bool
        If False (default), verticality is soft (used in scoring only).
        If True, verticality is a hard threshold (stricter, original behavior).

    Returns
    -------
    trunk_mask : np.ndarray (bool)
    wood_score : np.ndarray (float)
    dist_axis : np.ndarray (float)
    diagnostics : Dict - comprehensive extraction diagnostics
    """
    logger.debug(f"Extracting trunk from {len(points)} points")
    
    wood_mask, wood_score, dist_axis, class_diag = classify_wood(
        points,
        base_percentile=base_percentile,
        k_neighbors=k_neighbors,
        linearity_threshold=linearity_threshold,
        scattering_threshold=scattering_threshold,
        verticality_threshold=verticality_threshold,
        max_trunk_radius=max_trunk_radius,
        use_verticality_as_hard_criterion=use_verticality_as_hard_criterion,
    )

    n_before_expand = np.sum(wood_mask)
    wood_mask = expand_wood(points, wood_mask, expansion_radius=expansion_radius)
    n_after_expand = np.sum(wood_mask)
    
    logger.debug(f"After expansion: {n_before_expand} → {n_after_expand} points")
    
    n_before_cc = np.sum(wood_mask)
    wood_mask = largest_component(points, wood_mask, component_radius=component_radius)
    n_after_cc = np.sum(wood_mask)
    
    logger.debug(f"After connected component: {n_before_cc} → {n_after_cc} points")
    
    # Check if trunk extraction was successful
    has_warning = False
    if n_after_cc < min_trunk_points:
        logger.warning(f"Trunk has only {n_after_cc} points (threshold: {min_trunk_points})")
        has_warning = True
    
    trunk_ratio = n_after_cc / len(points)
    if trunk_ratio < 0.005:
        logger.warning(f"Trunk is only {100*trunk_ratio:.2f}% of the cloud (likely foreground/noise)")
        has_warning = True
    elif trunk_ratio > 0.9:
        logger.warning(f"Trunk is {100*trunk_ratio:.2f}% of cloud (likely all foreground)")
        has_warning = True
    
    diagnostics = {
        "n_total_points": len(points),
        "classification": class_diag,
        "expansion": {
            "n_before": int(n_before_expand),
            "n_after": int(n_after_expand),
            "radius_m": expansion_radius,
        },
        "connected_component": {
            "n_before": int(n_before_cc),
            "n_after": int(n_after_cc),
            "radius_m": component_radius,
        },
        "n_final_trunk": int(n_after_cc),
        "trunk_percentage": float(100 * trunk_ratio),
        "has_warnings": has_warning,
    }
    
    return wood_mask, wood_score, dist_axis, diagnostics


