"""Filter and clean extracted trunk points before DBH calculation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _estimate_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a robust trunk axis from lower points using PCA."""
    z = points[:, 2]
    z_cut = np.percentile(z, 35.0)
    base_pts = points[z <= z_cut]
    if base_pts.shape[0] < 10:
        base_pts = points

    centroid = np.mean(base_pts, axis=0)
    centered = base_pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:
        axis = -axis
    axis /= np.linalg.norm(axis) + 1e-12
    return axis, centroid


def _distance_to_axis(points: np.ndarray, axis: np.ndarray, origin: np.ndarray) -> np.ndarray:
    vecs = points - origin
    proj_len = np.dot(vecs, axis)
    proj = np.outer(proj_len, axis)
    return np.linalg.norm(vecs - proj, axis=1)


def clean_trunk_points(trunk_points: np.ndarray,
                      outlier_method: str = "isolation_forest",
                      percentile_method: bool = True,
                      axis_distance_filter: bool = True,
                      n_height_bins: int = 10,
                      min_points_per_bin: int = 25,
                      mad_scale: float = 3.5,
                      global_distance_percentile: float = 97.0,
                      return_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Clean trunk points by removing extreme outliers.
    
    Uses percentile and local-density filtering, plus an axis-distance
    robust filter to suppress off-trunk outliers while preserving taper.
    """
    n_total = len(trunk_points)
    if n_total < 50:
        if return_mask:
            return trunk_points, np.ones(n_total, dtype=bool)
        return trunk_points  # Too few to filter

    keep_mask = np.ones(n_total, dtype=bool)

    def _apply_submask(submask: np.ndarray) -> None:
        idx = np.where(keep_mask)[0]
        keep_mask[idx[~submask]] = False
    
    # Get XY coordinates
    xy = trunk_points[:, :2]
    
    # Method 1: Remove points far from median center
    if percentile_method:
        median_xy = np.median(xy, axis=0)
        distances = np.linalg.norm(xy - median_xy, axis=1)
        
        # Keep points within top 95% percentile (remove very far outliers)
        dist_threshold = np.percentile(distances, 95)
        mask_dist = distances <= dist_threshold
        
        _apply_submask(mask_dist)
        xy = trunk_points[keep_mask, :2]

    # Method 1.5: Robust filter by distance to estimated trunk axis per height band.
    # This is useful when neural predictions have sparse artifacts around the trunk.
    if axis_distance_filter and np.count_nonzero(keep_mask) >= 80:
        current = trunk_points[keep_mask]
        axis, origin = _estimate_axis(current)
        dist_axis = _distance_to_axis(current, axis, origin)
        z = current[:, 2]

        z_min, z_max = float(z.min()), float(z.max())
        if z_max > z_min:
            edges = np.linspace(z_min, z_max, n_height_bins + 1)
            keep = np.zeros(len(current), dtype=bool)

            global_thr = np.percentile(dist_axis, global_distance_percentile)

            for i in range(n_height_bins):
                mask_bin = (z >= edges[i]) & (z <= edges[i + 1] if i == n_height_bins - 1 else z < edges[i + 1])
                idx = np.where(mask_bin)[0]
                if idx.size == 0:
                    continue

                d = dist_axis[idx]
                if idx.size < min_points_per_bin:
                    keep[idx] = d <= global_thr
                    continue

                med = np.median(d)
                mad = np.median(np.abs(d - med)) + 1e-9
                robust_std = 1.4826 * mad
                local_thr = min(global_thr, med + mad_scale * robust_std)
                keep[idx] = d <= local_thr

            if np.count_nonzero(keep) >= max(30, int(0.25 * len(current))):
                _apply_submask(keep)
    
    # Method 2: Remove points with very low local density
    if np.count_nonzero(keep_mask) > 50:
        current = trunk_points[keep_mask]
        tree = cKDTree(current)
        distances, _ = tree.query(current, k=10)
        
        # Mean distance to 10 nearest neighbors
        mean_dists = np.mean(distances, axis=1)
        
        # Keep points with density below 2x median (not isolated)
        density_threshold = 2.0 * np.median(mean_dists)
        mask_density = mean_dists <= density_threshold
        
        _apply_submask(mask_density)

    filtered = trunk_points[keep_mask]
    if return_mask:
        return filtered, keep_mask
    
    return filtered
