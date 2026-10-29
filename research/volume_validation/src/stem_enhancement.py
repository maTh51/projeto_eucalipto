"""
Cylinder-following stem enhancement and trunk-centric quality control (QC).
"""

import numpy as np
from scipy.spatial import cKDTree


def enhance_stem_points(pts: np.ndarray, inst: np.ndarray, sem: np.ndarray,
                       kdtree_distance_thresh: float = 0.35,
                       cylinder_dz_thresh: float = 0.80,
                       cylinder_dxy_thresh: float = 0.30):
    """
    Reassign unassigned trunk points (tree_id <= 0 & trunk_leaf_label == 1)
    to existing isolated tree stems within vertical cylinder tolerance.
    """
    enhanced_inst = inst.copy()

    known_trunk_mask = (enhanced_inst > 0) & (sem == 1)
    known_trunk_pts = pts[known_trunk_mask]
    known_trunk_ids = enhanced_inst[known_trunk_mask]

    bg_trunk_mask = (enhanced_inst <= 0) & (sem == 1)
    bg_trunk_pts = pts[bg_trunk_mask]
    bg_trunk_indices = np.where(bg_trunk_mask)[0]

    reassigned_count = 0
    if len(known_trunk_pts) > 0 and len(bg_trunk_pts) > 0:
        kdt = cKDTree(known_trunk_pts)
        dists, idxs = kdt.query(bg_trunk_pts, distance_upper_bound=kdtree_distance_thresh)
        valid = dists < kdtree_distance_thresh
        
        for i, is_v in enumerate(valid):
            if is_v:
                match_pt = known_trunk_pts[idxs[i]]
                if abs(bg_trunk_pts[i, 2] - match_pt[2]) <= cylinder_dz_thresh:
                    if np.linalg.norm(bg_trunk_pts[i, :2] - match_pt[:2]) <= cylinder_dxy_thresh:
                        enhanced_inst[bg_trunk_indices[i]] = known_trunk_ids[idxs[i]]
                        reassigned_count += 1

    return enhanced_inst, reassigned_count


def extract_qc_trees(pts: np.ndarray, inst: np.ndarray, sem: np.ndarray, plot_id: int,
                     min_tree_points: int = 40,
                     min_stem_points: int = 15,
                     min_height_ratio: float = 0.40,
                     max_ground_offset: float = 2.50,
                     max_stem_gap: float = 4.0,
                     max_r90: float = 1.45):
    """
    Extract tree instances and filter out invalid/broken stems using strict QC criteria.
    """
    z_min_plot = pts[:, 2].min()
    plot_p90 = np.percentile(pts[:, 2] - z_min_plot, 90)

    qc_trees = []
    for tid in np.unique(inst):
        if tid <= 0:
            continue
        tmask = (inst == tid)
        t_pts = pts[tmask]
        t_sem = sem[tmask]

        if len(t_pts) < min_tree_points:
            continue

        z0, z1 = t_pts[:, 2].min(), t_pts[:, 2].max()
        h = z1 - z0
        if h < (plot_p90 * min_height_ratio):
            continue
        if (z0 - z_min_plot) > max_ground_offset:
            continue

        p_trunk = t_pts[t_sem == 1]
        if len(p_trunk) < min_stem_points:
            continue

        tz0, tz1 = p_trunk[:, 2].min(), p_trunk[:, 2].max()
        if (tz1 - tz0) < 6.0:
            continue

        # Vertical continuity check
        tbins = np.arange(tz0, tz1 + 1.0, 1.0)
        tcounts, _ = np.histogram(p_trunk[:, 2], bins=tbins)
        tgap_mask = tcounts < 2
        max_tgap, cgap = 0, 0
        for is_g in tgap_mask:
            if is_g:
                cgap += 1
            else:
                max_tgap, cgap = max(max_tgap, cgap), 0
        max_tgap = max(max_tgap, cgap)
        if max_tgap > max_stem_gap:
            continue

        # Stem compactness check
        xy_trunk = p_trunk[:, :2]
        center = np.median(xy_trunk, axis=0)
        r90 = np.percentile(np.linalg.norm(xy_trunk - center, axis=1), 90)
        if r90 > max_r90:
            continue

        cx, cy = np.median(t_pts[:, 0]), np.median(t_pts[:, 1])
        t_centered = t_pts.copy()
        t_centered[:, 0] -= cx
        t_centered[:, 1] -= cy
        t_centered[:, 2] -= z0

        qc_trees.append({
            "plot_id": plot_id,
            "tid": int(tid),
            "global_tid": int(plot_id * 10000 + tid),
            "x": cx,
            "y": cy,
            "height": float(h),
            "pts": t_centered,
            "trunk_mask": (t_sem == 1)
        })

    return qc_trees
