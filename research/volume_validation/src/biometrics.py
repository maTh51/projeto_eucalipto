"""
Biometric estimation engines: DBH, Height, and Volumetric methods.
"""

import os
import sys
import numpy as np

# Import existing core modules relative to repository root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from eucalipto.dbh_methods import estimate_dbh
from eucalipto.volume_methods import estimate_volume


def compute_tree_biometrics_grid(tree_dict: dict, config: dict):
    """
    Compute DBH and evaluate volume across all viable parametric methods for a tree.
    """
    pts = tree_dict["pts"]
    tmask = tree_dict["trunk_mask"]
    trunk = pts[tmask] if tmask.sum() > 10 else pts
    h = tree_dict["height"]

    # 1. Estimate DBH
    dbh_cfg = config.get("dbh", {})
    try:
        dbh_val, _ = estimate_dbh(
            trunk,
            method=dbh_cfg.get("method", "ensemble"),
            offset_range=dbh_cfg.get("offset_range", 0.3),
            n_slices=dbh_cfg.get("n_slices", 7),
            slice_thickness=dbh_cfg.get("slice_thickness", 0.16),
            ransac_thresh=dbh_cfg.get("ransac_thresh", 0.02)
        )
        dbh = float(dbh_val)
    except Exception:
        dbh = np.nan

    configs_out = {}

    # 2. Voxel Grid
    voxel_cfg = config.get("voxel", {})
    for vs in voxel_cfg.get("voxel_sizes", [0.06, 0.07, 0.08, 0.09, 0.10]):
        cfg_name = f"voxel_vs{vs:.2f}"
        try:
            r = estimate_volume(trunk, method="voxel", voxel_size=vs)
            configs_out[cfg_name] = float(r["volume_m3"])
        except Exception:
            configs_out[cfg_name] = np.nan

    # 3. Axis Profile Grid
    axis_cfg = config.get("axis_profile", {})
    for rp in axis_cfg.get("radius_percentiles", [15.0, 20.0, 25.0]):
        for ns in axis_cfg.get("n_slices_list", [15, 20]):
            cfg_name = f"axis_rp{rp:.0f}_ns{ns}"
            try:
                r = estimate_volume(
                    trunk, method="axis_profile",
                    n_slices=ns,
                    slice_thickness=axis_cfg.get("slice_thickness", 0.20),
                    radius_percentile=rp,
                    min_points_per_slice=axis_cfg.get("min_points_per_slice", 5)
                )
                configs_out[cfg_name] = float(r["volume_m3"])
            except Exception:
                configs_out[cfg_name] = np.nan

    # 4. Form Factor Calibrated Cylinder Grid
    cyl_cfg = config.get("cylinder_ff", {})
    if not np.isnan(dbh) and dbh > 0:
        v_cyl_raw = (np.pi / 4.0) * ((dbh / 100.0) ** 2) * h
        for f in cyl_cfg.get("form_factors", [0.35, 0.38, 0.40, 0.42, 0.45]):
            cfg_name = f"cyl_ff{f:.2f}"
            configs_out[cfg_name] = float(v_cyl_raw * f)
    else:
        for f in cyl_cfg.get("form_factors", [0.35, 0.38, 0.40, 0.42, 0.45]):
            configs_out[f"cyl_ff{f:.2f}"] = np.nan

    return {
        "height_m": float(h),
        "dbh_cm": dbh,
        "configs": configs_out
    }
