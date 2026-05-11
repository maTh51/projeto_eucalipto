"""Canonical contracts shared across all pipeline providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


SCHEMA_VERSION = "1.0.0"

MISSING_INT = -1
MISSING_FLOAT = float("nan")


@dataclass
class PlotCloudCanonical:
    """Normalized point cloud representation for pipeline input."""

    points_xyz: np.ndarray
    source_path: str
    source_format: str
    point_count: int
    extras: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class TreeInstancesCanonical:
    """Tree instance IDs aligned to input points."""

    points_xyz: np.ndarray
    tree_id: np.ndarray
    provider: str
    field_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class TreeSegmentationCanonical:
    """Canonical trunk/leaf segmentation aligned to input points."""

    points_xyz: np.ndarray
    tree_id: np.ndarray
    trunk_leaf_label: np.ndarray
    provider: str
    confidence: Optional[np.ndarray] = None
    field_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class TreeMetricRow:
    """Per-tree metrics in canonical schema."""

    tree_id: int
    dbh_cm: Optional[float]
    height_m: Optional[float]
    volume_m3: Optional[float]
    mass_kg: Optional[float]
    metric_provider: str
    warnings: List[str] = field(default_factory=list)

