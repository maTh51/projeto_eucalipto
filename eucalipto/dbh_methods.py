"""DBH estimation methods extracted and simplified from dbh.ipynb.

Main entry point: estimate_dbh(points, method="ensemble", **kwargs)
"""

from typing import Literal, Tuple, Optional, List

import numpy as np
import pyransac3d as pyrsc
from scipy import optimize


def _slice_points(points: np.ndarray,
                  z_center: float,
                  thickness: float) -> np.ndarray:
    z = points[:, 2]
    half = thickness / 2.0
    mask = (z >= z_center - half) & (z <= z_center + half)
    return points[mask]


def _dbh_ransac_on_slice(points_slice: np.ndarray,
                          thresh: float = 0.02,
                          radius_min: float = 0.01,
                          radius_max: float = 1.0) -> Tuple[Optional[float], int, Optional[np.ndarray], Optional[float]]:
    if points_slice.shape[0] < 3:
        return None, 0, None, None

    circle = pyrsc.Circle()
    center, normal, radius, inliers = circle.fit(points_slice, thresh=thresh)

    if radius <= 0 or not (radius_min <= radius <= radius_max):
        return None, 0, None, None

    dbh_cm = radius * 2.0 * 100.0
    return dbh_cm, len(inliers), center, radius


def _dbh_ls_on_slice(points_slice: np.ndarray,
                     radius_min: float = 0.01,
                     radius_max: float = 1.0) -> Optional[float]:
    if points_slice.shape[0] < 3:
        return None

    x = points_slice[:, 0]
    y = points_slice[:, 1]

    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m

    def calc_R(xc, yc):
        return np.sqrt((u - xc) ** 2 + (v - yc) ** 2)

    def f_2(c):
        ri = calc_R(*c)
        return ri - ri.mean()

    center_estimate = np.array([0.0, 0.0])

    try:
        center_2, ier = optimize.leastsq(f_2, center_estimate)
        ri_2 = calc_R(*center_2)
        radius = ri_2.mean()
        if radius_min <= radius <= radius_max:
            return radius * 2.0 * 100.0
    except Exception:
        return None

    return None


def estimate_dbh_ensemble(points: np.ndarray,
                          breast_height_offset: float = 1.3,
                          offset_range: float = 0.30,
                          n_slices: int = 7,
                          slice_thickness: float = 0.16,
                          ransac_thresh: float = 0.02) -> Tuple[Optional[float], List[float]]:
    """Estimate DBH using a multi-slice ensemble around breast height.

    Returns (dbh_cm, per_slice_dbh_values).
    """
    if points.shape[0] < 3:
        return None, []

    z = points[:, 2]
    z_min = float(z.min())
    breast_height = z_min + breast_height_offset

    offsets = np.linspace(-offset_range, offset_range, n_slices)
    dbh_values: List[float] = []

    for off in offsets:
        z_center = breast_height + off
        slice_pts = _slice_points(points, z_center=z_center, thickness=slice_thickness)
        if slice_pts.shape[0] < 3:
            continue

        dbh_cm, n_inliers, center, radius = _dbh_ransac_on_slice(slice_pts, thresh=ransac_thresh)
        if dbh_cm is not None and n_inliers >= 3:
            dbh_values.append(dbh_cm)

    if len(dbh_values) == 0:
        return None, []

    dbh_cm_final = float(np.median(dbh_values))
    return dbh_cm_final, dbh_values


def estimate_dbh(points: np.ndarray,
                 method: Literal["ensemble", "single_ransac", "ls"] = "ensemble",
                 **kwargs) -> Tuple[Optional[float], dict]:
    """High-level DBH estimator.

    Parameters
    ----------
    points : np.ndarray
        Trunk-only points (N, 3).
    method : {"ensemble", "single_ransac", "ls"}
        Ensemble is the recommended default.

    Returns
    -------
    dbh_cm : float or None
    info : dict
        Additional diagnostics (per-slice DBHs, method, etc.).
    """
    info: dict = {"method": method}

    if method == "ensemble":
        dbh_cm, values = estimate_dbh_ensemble(points, **kwargs)
        info["per_slice_dbh_cm"] = values
        return dbh_cm, info

    z = points[:, 2]
    z_min = float(z.min())
    breast_height = z_min + kwargs.get("breast_height_offset", 1.3)
    slice_thickness = kwargs.get("slice_thickness", 0.16)

    slice_pts = _slice_points(points, z_center=breast_height, thickness=slice_thickness)

    if method == "single_ransac":
        dbh_cm, n_inliers, center, radius = _dbh_ransac_on_slice(slice_pts,
                                                                 thresh=kwargs.get("ransac_thresh", 0.02))
        info.update({
            "n_inliers": n_inliers,
            "radius_m": radius,
        })
        return dbh_cm, info

    if method == "ls":
        dbh_cm = _dbh_ls_on_slice(slice_pts)
        return dbh_cm, info

    raise ValueError(f"Unsupported DBH method: {method}")
