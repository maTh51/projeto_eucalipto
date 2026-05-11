"""Export helpers for treeiso pipeline outputs.

Includes:
- per-point classified cloud with isolation/trunk/volume attributes;
- cylinder-based geometric representation for presentation/validation.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from . import io as eio


def save_classified_cloud_laz(output_path: str,
                              points: np.ndarray,
                              tree_segment_id: np.ndarray,
                              trunk_mask: np.ndarray,
                              wood_score: np.ndarray,
                              dist_axis: np.ndarray,
                              dbh_cm_tree: np.ndarray,
                              volume_m3_tree: np.ndarray,
                              extra_fields: Dict[str, np.ndarray] | None = None) -> None:
    """Save per-point classification and metrics in a LAZ file."""
    extras = {
        "tree_segment_id": np.asarray(tree_segment_id, dtype=np.int32),
        "trunk_mask": np.asarray(trunk_mask, dtype=np.uint8),
        "wood_score": np.asarray(wood_score, dtype=np.float32),
        "dist_axis": np.asarray(dist_axis, dtype=np.float32),
        "dbh_cm_tree": np.asarray(dbh_cm_tree, dtype=np.float32),
        "volume_m3_tree": np.asarray(volume_m3_tree, dtype=np.float32),
    }
    if extra_fields:
        for key, values in extra_fields.items():
            extras[key] = np.asarray(values)
    eio.save_laz(output_path, points, extras=extras)


def save_cylinder_primitives_csv(output_path: str,
                                 cylinders: List[Dict[str, float]]) -> None:
    """Save cylinder primitive parameters as CSV."""
    import csv

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "tree_id",
        "center_x",
        "center_y",
        "z_min",
        "height_m",
        "radius_m",
        "volume_m3",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cylinders:
            writer.writerow({k: row.get(k) for k in fieldnames})


def save_cylinder_representation_ply(output_path: str,
                                     cylinders: List[Dict[str, float]],
                                     n_theta: int = 48,
                                     n_z: int = 24) -> None:
    """Save a point-based PLY representation of cylinder volumes."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise ImportError(
            "Dependência 'plyfile' não encontrada. Instale com: pip install plyfile"
        ) from exc

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_rows = []
    for cyl in cylinders:
        tree_id = int(cyl["tree_id"])
        cx = float(cyl["center_x"])
        cy = float(cyl["center_y"])
        z_min = float(cyl["z_min"])
        height = float(cyl["height_m"])
        radius = float(cyl["radius_m"])
        volume_m3 = float(cyl["volume_m3"])

        if height <= 0 or radius <= 0:
            continue

        thetas = np.linspace(0.0, 2.0 * np.pi, num=n_theta, endpoint=False)
        zs = np.linspace(z_min, z_min + height, num=n_z)

        for z in zs:
            for theta in thetas:
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                all_rows.append((
                    float(x),
                    float(y),
                    float(z),
                    int(tree_id),
                    float(volume_m3),
                ))

    if len(all_rows) == 0:
        vertex = np.array([], dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("tree_id", "i4"),
            ("volume_m3", "f4"),
        ])
    else:
        vertex = np.array(
            all_rows,
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("tree_id", "i4"),
                ("volume_m3", "f4"),
            ],
        )

    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(output_path)
