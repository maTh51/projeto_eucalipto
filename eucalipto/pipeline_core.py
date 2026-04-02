"""Core orchestration helpers for the Eucalyptus pipeline.

These functions are intended to be used by small top-level scripts such
as run_ff3d_pipeline.py.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np

from . import io
from . import dbh_methods
from . import volume_methods
from . import trunk_heuristic
from . import isolation_ff3d
from . import isolation_treeiso


def run_isolation_ff3d(
    ff3d_repo_dir: str,
    bucket_in_dir: str,
    bucket_out_dir: str,
    input_laz: str,
    instance_dim: str,
    semantic_dim: str,
    trunk_label: int,
) -> Dict[int, np.ndarray]:
    """Run FF3D Docker inference and return per-tree trunk point clouds."""
    isolation_ff3d.run_ff3d_docker(
        ff3d_repo_dir=ff3d_repo_dir,
        bucket_in_dir=bucket_in_dir,
        bucket_out_dir=bucket_out_dir,
        input_laz=input_laz,
    )

    per_tree_trunk = isolation_ff3d.collect_trunk_points_from_ff3d_output(
        bucket_out_dir=bucket_out_dir,
        instance_dim=instance_dim,
        semantic_dim=semantic_dim,
        trunk_label=trunk_label,
    )
    return per_tree_trunk


def run_isolation_treeiso(
    input_laz: str,
    tree_id_dim: str = "treeID",
) -> Dict[int, np.ndarray]:
    """Split the input cloud into per-tree clouds using a tree ID field.

    This is a placeholder for a more complete integration with the
    treeiso (artemis_treeiso) framework.
    """
    return isolation_treeiso.split_by_tree_id(input_laz, tree_id_dim=tree_id_dim)


def run_trunk_extraction_for_treeiso(
    per_tree_points: Dict[int, np.ndarray],
    **trunk_params,
) -> Dict[int, np.ndarray]:
    """Apply heuristic trunk extraction to per-tree clouds (treeiso path)."""
    trunks: Dict[int, np.ndarray] = {}
    for tid, pts in per_tree_points.items():
        wood_mask, wood_score, dist_axis = trunk_heuristic.extract_trunk(pts, **trunk_params)
        trunks[tid] = pts[wood_mask]
    return trunks


def run_metrics_on_trunks(
    per_tree_trunks: Dict[int, np.ndarray],
    dbh_method: Literal["ensemble", "single_ransac", "ls"] = "ensemble",
    volume_method: str = "cylinder",
    wood_density_kg_m3: float | None = None,
) -> Dict[int, dict]:
    """Compute DBH and volume for each tree trunk cloud."""
    results: Dict[int, dict] = {}

    for tid, trunk_pts in per_tree_trunks.items():
        if trunk_pts.shape[0] < 3:
            results[tid] = {"dbh_cm": None, "volume_m3": None}
            continue

        dbh_cm, dbh_info = dbh_methods.estimate_dbh(trunk_pts, method=dbh_method)

        if dbh_cm is None:
            results[tid] = {"dbh_cm": None, "volume_m3": None, "dbh_info": dbh_info}
            continue

        height_m = io.tree_height(trunk_pts)
        vol_info = volume_methods.estimate_volume(
            trunk_pts,
            dbh_cm=dbh_cm,
            height_m=height_m,
            method=volume_method,
            wood_density_kg_m3=wood_density_kg_m3,
        )

        res = {"dbh_cm": dbh_cm, "dbh_info": dbh_info}
        res.update(vol_info)
        results[tid] = res

    return results
