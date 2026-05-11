"""Native-to-canonical adapters for supported providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .contracts import MISSING_FLOAT, TreeInstancesCanonical, TreeSegmentationCanonical


@dataclass
class AdapterResult:
    field_mapping: Dict[str, str]


def adapter_ff3d_to_instances(
    points: np.ndarray,
    instance_ids: np.ndarray,
) -> tuple[TreeInstancesCanonical, AdapterResult]:
    mapping = {"treeID": "tree_id"}
    canonical = TreeInstancesCanonical(
        points_xyz=points,
        tree_id=instance_ids.astype(np.int32),
        provider="ff3d",
        field_mapping=mapping,
    )
    return canonical, AdapterResult(field_mapping=mapping)


def adapter_ff3d_to_segmentation(
    points: np.ndarray,
    tree_id: np.ndarray,
    semantic_ids: np.ndarray,
    trunk_label: int,
) -> tuple[TreeSegmentationCanonical, AdapterResult]:
    # canonical: 0=leaf/non-trunk, 1=trunk
    trunk_leaf = (semantic_ids == trunk_label).astype(np.int32)
    mapping = {"semantic_seg": "trunk_leaf_label"}
    canonical = TreeSegmentationCanonical(
        points_xyz=points,
        tree_id=tree_id.astype(np.int32),
        trunk_leaf_label=trunk_leaf,
        provider="ff3d",
        confidence=np.full(points.shape[0], MISSING_FLOAT, dtype=np.float32),
        field_mapping=mapping,
    )
    return canonical, AdapterResult(field_mapping=mapping)


def adapter_treeiso_to_instances(
    points: np.ndarray,
    segment_ids: np.ndarray,
) -> tuple[TreeInstancesCanonical, AdapterResult]:
    mapping = {"final_segs": "tree_id"}
    canonical = TreeInstancesCanonical(
        points_xyz=points,
        tree_id=segment_ids.astype(np.int32),
        provider="treeiso",
        field_mapping=mapping,
    )
    return canonical, AdapterResult(field_mapping=mapping)


def adapter_leafwood_to_segmentation(
    points: np.ndarray,
    tree_id: np.ndarray,
    leafwood_label: np.ndarray,
) -> tuple[TreeSegmentationCanonical, AdapterResult]:
    # leafwood returns 0=leaf, 1=wood(trunk)
    mapping = {"leafwood_pred": "trunk_leaf_label"}
    canonical = TreeSegmentationCanonical(
        points_xyz=points,
        tree_id=tree_id.astype(np.int32),
        trunk_leaf_label=leafwood_label.astype(np.int32),
        provider="leafwood",
        confidence=np.full(points.shape[0], MISSING_FLOAT, dtype=np.float32),
        field_mapping=mapping,
    )
    return canonical, AdapterResult(field_mapping=mapping)

