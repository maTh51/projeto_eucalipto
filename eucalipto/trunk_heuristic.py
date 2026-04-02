"""Heuristic trunk extraction based on geometric features.

Refactored from remove_old.py. Intended primarily for use on per-tree
point clouds produced by treeiso (artemis_treeiso), where no semantic
trunk labels are available.
"""

from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def estimate_trunk_axis(points: np.ndarray) -> np.ndarray:
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return axis / np.linalg.norm(axis)


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
                               k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    _, idx = nbrs.kneighbors(points)

    linearity = np.zeros(len(points))
    scattering = np.zeros(len(points))

    for i in range(len(points)):
        neighbors = points[idx[i]]
        cov = np.cov(neighbors.T)

        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1] + 1e-8

        l1, l2, l3 = eigvals

        linearity[i] = (l1 - l2) / l1
        scattering[i] = l3 / l1

    return linearity, scattering


def classify_wood(points: np.ndarray,
                  base_percentile: float = 20.0,
                  k_neighbors: int = 25,
                  linearity_threshold: float = 0.4,
                  scattering_threshold: float = 0.4,
                  dist_percentile: float = 30.0):
    """Return (wood_mask, wood_score, dist_axis) for a per-tree cloud."""
    axis = estimate_trunk_axis(points)
    center_xy = estimate_trunk_center(points, base_percentile)
    dist_axis = distance_to_axis(points, axis, center_xy)
    lin, sca = compute_geometric_features(points, k_neighbors)

    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min())

    dist_score = 1 - (dist_axis / (np.percentile(dist_axis, 90) + 1e-6))
    dist_score = np.clip(dist_score, 0, 1)

    wood_score = 0.5 * dist_score + 0.3 * lin + 0.2 * (1 - sca)

    wood_mask = (
        (dist_axis < np.percentile(dist_axis, dist_percentile))
        & (lin > linearity_threshold)
        & (sca < scattering_threshold)
        & (z_norm < 0.95)
    )

    return wood_mask, wood_score, dist_axis


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
                  dist_percentile: float = 30.0,
                  expansion_radius: float = 0.2,
                  component_radius: float = 0.2):
    """Full heuristic pipeline: return trunk-only mask and scores.

    Returns
    -------
    trunk_mask : np.ndarray (bool)
    wood_score : np.ndarray (float)
    dist_axis  : np.ndarray (float)
    """
    wood_mask, wood_score, dist_axis = classify_wood(
        points,
        base_percentile=base_percentile,
        k_neighbors=k_neighbors,
        linearity_threshold=linearity_threshold,
        scattering_threshold=scattering_threshold,
        dist_percentile=dist_percentile,
    )

    wood_mask = expand_wood(points, wood_mask, expansion_radius=expansion_radius)
    wood_mask = largest_component(points, wood_mask, component_radius=component_radius)

    return wood_mask, wood_score, dist_axis
