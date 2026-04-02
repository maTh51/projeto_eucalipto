"""FF3D-based tree isolation and trunk extraction.

This module assumes the FF3D_inference repository is available on disk
and that its run_docker_locally.sh script is used to perform inference.

The exact output format of FF3D can vary; this wrapper therefore
exposes flexible parameters for the names of the instance and semantic
label dimensions and for the numeric label corresponding to trunk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob
from typing import Dict, Tuple

import laspy
import numpy as np


def run_ff3d_docker(
    ff3d_repo_dir: str,
    bucket_in_dir: str,
    bucket_out_dir: str,
    input_laz: str,
    rebuild_image: str = "auto",
    recreate_container: str = "auto",
) -> str:
    """Run FF3D Docker inference for a single input cloud.

    Parameters
    ----------
    ff3d_repo_dir : str
        Path to FF3D_inference/ff3d_forestsens directory.
    bucket_in_dir : str
        Host bucket_in_folder path used by FF3D_inference.
    bucket_out_dir : str
        Host bucket_out_folder path used by FF3D_inference.
    input_laz : str
        Path to the LAS/LAZ file to process.

    Returns
    -------
    output_dir : str
        The bucket_out_dir, where FF3D outputs will be found.
    """
    os.makedirs(bucket_in_dir, exist_ok=True)
    os.makedirs(bucket_out_dir, exist_ok=True)

    # Copy input into bucket_in_dir; FF3D scripts will treat everything
    # inside as test data.
    shutil.copy2(input_laz, bucket_in_dir)

    env = os.environ.copy()
    env.setdefault("HOST_PROJECT_DIR", ff3d_repo_dir)
    env.setdefault("HOST_BUCKET_IN", bucket_in_dir)
    env.setdefault("HOST_BUCKET_OUT", bucket_out_dir)
    env.setdefault("REBUILD_IMAGE", rebuild_image)
    env.setdefault("RECREATE_CONTAINER", recreate_container)

    script_path = os.path.join(ff3d_repo_dir, "run_docker_locally.sh")

    subprocess.run(
        ["bash", script_path],
        cwd=ff3d_repo_dir,
        env=env,
        check=True,
    )

    return bucket_out_dir


def _load_labelled_laz(path: str,
                       instance_dim: str,
                       semantic_dim: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    instance_ids = np.asarray(getattr(las, instance_dim))
    semantic_ids = np.asarray(getattr(las, semantic_dim))
    return points, instance_ids, semantic_ids


def collect_trunk_points_from_ff3d_output(
    bucket_out_dir: str,
    instance_dim: str,
    semantic_dim: str,
    trunk_label: int,
) -> Dict[int, np.ndarray]:
    """Collect per-tree trunk points from FF3D outputs.

    This function searches for LAS/LAZ files in bucket_out_dir, assumes
    they contain per-point instance and semantic labels, and groups
    trunk points (semantic==trunk_label) by instance id.

    Because FF3D output conventions can change, the caller must specify
    the names of the instance and semantic dimensions and the label
    value corresponding to trunk.
    """
    pattern = os.path.join(bucket_out_dir, "**", "*.la*")
    files = sorted(glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under {bucket_out_dir}")

    # For now, assume a single combined prediction file is produced.
    labelled_path = files[0]

    points, instance_ids, semantic_ids = _load_labelled_laz(
        labelled_path, instance_dim=instance_dim, semantic_dim=semantic_dim
    )

    trunk_mask = semantic_ids == trunk_label
    trunk_points = points[trunk_mask]
    trunk_instances = instance_ids[trunk_mask]

    per_tree: Dict[int, list] = {}
    for p, inst in zip(trunk_points, trunk_instances):
        per_tree.setdefault(int(inst), []).append(p)

    per_tree_np: Dict[int, np.ndarray] = {
        tid: np.vstack(pts) for tid, pts in per_tree.items() if len(pts) > 0
    }

    return per_tree_np
