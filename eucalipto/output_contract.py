"""Canonical output writers (cloud, metrics CSV and run manifest)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .contracts import MISSING_FLOAT, MISSING_INT, TreeMetricRow


def ensure_output_dir(path: str) -> Path:
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_canonical_cloud(
    output_dir: Path,
    points_xyz: np.ndarray,
    tree_id: np.ndarray,
    trunk_leaf_label: np.ndarray,
    source_format: str,
    cloud_format: str,
) -> Path:
    """Write standardized cloud with required canonical fields."""
    from . import io as eio  # lazy import: only needed when writing clouds

    extras = {
        "tree_id": tree_id.astype(np.int32),
        "trunk_leaf_label": trunk_leaf_label.astype(np.int32),
        "semantic_class": np.full(points_xyz.shape[0], MISSING_INT, dtype=np.int16),
        "confidence": np.full(points_xyz.shape[0], MISSING_FLOAT, dtype=np.float32),
    }

    if cloud_format == "laz":
        out_path = output_dir / "classified_cloud.laz"
        eio.save_laz(str(out_path), points_xyz, extras=extras)
        return out_path

    out_path = output_dir / "classified_cloud.ply"
    eio.save_ply_with_fields(str(out_path), points_xyz, extras)
    return out_path


def write_metrics_csv(output_dir: Path, rows: Iterable[TreeMetricRow]) -> Path:
    out_path = output_dir / "metrics.csv"
    fieldnames = [
        "tree_id",
        "dbh_cm",
        "height_m",
        "volume_m3",
        "mass_kg",
        "metric_provider",
        "warnings",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tree_id": row.tree_id,
                    "dbh_cm": row.dbh_cm,
                    "height_m": row.height_m,
                    "volume_m3": row.volume_m3,
                    "mass_kg": row.mass_kg,
                    "metric_provider": row.metric_provider,
                    "warnings": ";".join(row.warnings),
                }
            )
    return out_path


def write_run_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    out_path = output_dir / "run_manifest.json"
    full_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    out_path.write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")
    return out_path

