import os
from typing import Dict, Tuple, Optional

import laspy
import numpy as np
from plyfile import PlyData, PlyElement


def load_laz(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load a LAS/LAZ file and return (points, extra_fields).

    points: (N, 3) float64 array with XYZ.
    extra_fields: mapping from field name to numpy array (length N).
    """
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    extras: Dict[str, np.ndarray] = {}
    for dim in las.point_format.extra_dimension_names:
        arr = getattr(las, dim)
        extras[dim] = np.asarray(arr)

    # Standard LAS dimensions that are often useful
    for dim in ("classification", "intensity"):
        if hasattr(las, dim):
            extras[dim] = np.asarray(getattr(las, dim))

    return points, extras


def save_laz(path: str,
             points: np.ndarray,
             extras: Optional[Dict[str, np.ndarray]] = None,
             like_path: Optional[str] = None) -> None:
    """Save points (and optional extra fields) to LAS/LAZ.

    If like_path is given, its header/point_format is reused when possible.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if like_path is not None:
        like_las = laspy.read(like_path)
        header = like_las.header
        point_format = header.point_format
    else:
        header = laspy.LasHeader(point_format=3, version="1.2")
        point_format = header.point_format

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if extras:
        for name, values in extras.items():
            if name in point_format.dimension_names:
                setattr(las, name, values)
            else:
                # register as extra dimension
                if name not in las.point_format.extra_dimension_names:
                    las.add_extra_dim(laspy.ExtraBytesParams(name=name,
                                                             type=values.dtype))
                setattr(las, name, values)

    las.write(path)


def save_trunk_ply(points: np.ndarray,
                   wood_mask: np.ndarray,
                   wood_score: np.ndarray,
                   dist_axis: np.ndarray,
                   path: str) -> None:
    """Save trunk classification outputs to a PLY with scalar fields.

    This mirrors the structure produced in remove_old.py so results can be
    inspected in CloudCompare.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    wood_flag = wood_mask.astype(np.float32)

    vertex = np.array(
        [
            (
                float(points[i, 0]),
                float(points[i, 1]),
                float(points[i, 2]),
                float(wood_flag[i]),
                float(wood_score[i]),
                float(dist_axis[i]),
            )
            for i in range(points.shape[0])
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("wood", "f4"),
            ("wood_score", "f4"),
            ("dist_axis", "f4"),
        ],
    )

    ply = PlyData([PlyElement.describe(vertex, "vertex")])
    ply.write(path)


def normalize_xy(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Center X and Y coordinates for numerical stability.

    Returns (points_normalized, x_center, y_center).
    """
    pts = points.copy()
    x_center = pts[:, 0].mean()
    y_center = pts[:, 1].mean()
    pts[:, 0] -= x_center
    pts[:, 1] -= y_center
    return pts, float(x_center), float(y_center)


def tree_height(points: np.ndarray) -> float:
    """Return approximate tree height as max(z) - min(z)."""
    z = points[:, 2]
    return float(z.max() - z.min())
