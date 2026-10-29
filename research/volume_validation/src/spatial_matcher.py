"""
Adaptive spatial matching between segmented LiDAR trees and field planting grid.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def align_and_match_plot(qc_trees: list, plot_df: pd.DataFrame,
                         max_dist: float = 1.30,
                         angle_step_deg: int = 3,
                         translation_range: tuple = (-5.0, 5.0),
                         translation_steps: int = 21,
                         height_penalty_weight: float = 0.20):
    """
    Find optimal rigid 2D transformation (rotation + translation) aligning
    LiDAR QC trees with relative field grid positions (Fila/Cova), then match pairs.
    """
    if len(qc_trees) < 3 or len(plot_df) == 0:
        return []

    F = plot_df[["x_grid", "y_grid"]].values
    T_pts = np.array([[t["x"], t["y"]] for t in qc_trees])
    T_h = np.array([t["height"] for t in qc_trees])
    mA = T_pts.mean(axis=0)

    best_score = -float("inf")
    best_Fr = None

    for swap in [False, True]:
        F_curr = F[:, [1, 0]] if swap else F
        mB = F_curr.mean(axis=0)
        Bc = F_curr - mB

        for deg in range(0, 360, angle_step_deg):
            t = np.radians(deg)
            c, s = np.cos(t), np.sin(t)
            R = np.array([[c, -s], [s, c]])
            Br = Bc @ R.T

            for dx in np.linspace(translation_range[0], translation_range[1], translation_steps):
                for dy in np.linspace(translation_range[0], translation_range[1], translation_steps):
                    Fr_cand = Br + mA + np.array([dx, dy])
                    D = cdist(T_pts, Fr_cand)
                    min_d = np.min(D, axis=1)
                    close_mask = min_d < max_dist
                    n_close = close_mask.sum()
                    if n_close < min(len(qc_trees), 4):
                        continue

                    score = float(np.sum(1.5 - min_d[close_mask]))
                    closest = np.argmin(D, axis=1)
                    for ti, si in enumerate(closest):
                        if min_d[ti] < max_dist and plot_df.iloc[si]["has_gt_volume"]:
                            score += 2.0
                            gt_h = plot_df.iloc[si]["alt_m"]
                            if not np.isnan(gt_h):
                                score -= min(2.0, height_penalty_weight * abs(T_h[ti] - gt_h))

                    if score > best_score:
                        best_score = score
                        best_Fr = Fr_cand

    matches = []
    if best_Fr is not None:
        D = cdist(best_Fr, T_pts)
        ri, ci = linear_sum_assignment(D)
        for f_idx, t_idx in zip(ri, ci):
            if D[f_idx, t_idx] < max_dist:
                row = plot_df.iloc[f_idx]
                t = qc_trees[t_idx]
                matches.append({
                    "plot_id": row["PAR_CODIGO"],
                    "field_id": row["field_id"],
                    "tid": t["tid"],
                    "global_tid": t["global_tid"],
                    "has_gt_volume": row["has_gt_volume"],
                    "gt_volume": row["gt_volume"] if row["has_gt_volume"] else np.nan,
                    "gt_dbh": row["dap_cm"] if row["has_gt_volume"] else np.nan,
                    "gt_height": row["alt_m"] if row["has_gt_volume"] else np.nan,
                    "dist": float(D[f_idx, t_idx]),
                    "tree_data": t
                })
    return matches
