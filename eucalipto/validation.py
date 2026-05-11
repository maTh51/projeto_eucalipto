"""Preflight validation and schema checks for pipeline config."""

from __future__ import annotations

from pathlib import Path

from .capabilities import CAPABILITIES
from .config_schema import PipelineConfig


MODE_TO_PROVIDERS = {
    "ff3d_full": ["ff3d"],
    "treeiso_leafwood": ["treeiso", "leafwood"],
    "treeiso_leafwood_rctqsm": ["treeiso", "leafwood", "rayextract"],
    "rayextract_full": ["rayextract"],
    "rct_qsm_metrics": ["rayextract"],
}


def validate_pipeline_config(cfg: PipelineConfig) -> None:
    input_path = Path(cfg.input.path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {cfg.input.path}")

    expected_providers = MODE_TO_PROVIDERS[cfg.pipeline_mode]
    for provider in expected_providers:
        if provider not in CAPABILITIES:
            raise ValueError(f"Unknown provider in capabilities table: {provider}")

    ext = input_path.suffix.lower().lstrip(".")
    if ext == "":
        raise ValueError("Input path must have an extension (.laz/.las/.ply).")

