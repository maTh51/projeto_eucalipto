"""Declarative capabilities per provider for preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ProviderCapabilities:
    stages: List[str]
    input_formats: List[str]
    requires_gpu: bool
    notes: str


CAPABILITIES: Dict[str, ProviderCapabilities] = {
    "ff3d": ProviderCapabilities(
        stages=["isolation", "segmentation"],
        input_formats=["laz", "las", "ply"],
        requires_gpu=True,
        notes="FF3D inference via Docker; high-memory GPU recommended.",
    ),
    "treeiso": ProviderCapabilities(
        stages=["isolation"],
        input_formats=["laz", "las", "ply"],
        requires_gpu=False,
        notes="Treeiso isolates instances; segmentation must come from leafwood/other.",
    ),
    "leafwood": ProviderCapabilities(
        stages=["segmentation"],
        input_formats=["ply", "txt", "npy"],
        requires_gpu=True,
        notes="Leaf-wood only; consumes per-tree/per-segment clouds.",
    ),
    "rayextract": ProviderCapabilities(
        stages=["isolation", "segmentation", "metrics"],
        input_formats=["ply", "laz", "las"],
        requires_gpu=False,
        notes="Designed for plot-level clouds; may over-segment single-tree inputs.",
    ),
}

