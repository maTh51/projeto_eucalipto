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
    import os
    import time
    from . import io as eio  # lazy import: only needed when writing clouds

    extras = {
        "tree_id": tree_id.astype(np.int32),
        "trunk_leaf_label": trunk_leaf_label.astype(np.int32),
        "semantic_class": np.full(points_xyz.shape[0], MISSING_INT, dtype=np.int16),
        "confidence": np.full(points_xyz.shape[0], MISSING_FLOAT, dtype=np.float32),
    }

    if cloud_format == "laz":
        out_path = output_dir / "classified_cloud.laz"
        out_path_str = str(out_path)
        
        # Write with retries and verification
        max_retries = 3
        for attempt in range(max_retries):
            try:
                eio.save_laz(out_path_str, points_xyz, extras=extras)
                
                # Verify file was written
                if not out_path.exists():
                    raise FileNotFoundError(f"File not written: {out_path}")
                
                file_size = out_path.stat().st_size
                print(f"✓ Wrote LAZ cloud: {out_path.name} ({file_size} bytes)")
                return out_path
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠ Retry writing LAZ ({attempt + 1}/{max_retries}): {e}")
                    os.sync()
                    time.sleep(1)
                else:
                    raise

    out_path = output_dir / "classified_cloud.ply"
    out_path_str = str(out_path)
    
    # Write with retries and verification
    max_retries = 3
    for attempt in range(max_retries):
        try:
            eio.save_ply_with_fields(out_path_str, points_xyz, extras)
            
            # Verify file was written
            if not out_path.exists():
                raise FileNotFoundError(f"File not written: {out_path}")
            
            file_size = out_path.stat().st_size
            print(f"✓ Wrote PLY cloud: {out_path.name} ({file_size} bytes)")
            return out_path
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Retry writing PLY ({attempt + 1}/{max_retries}): {e}")
                os.sync()
                time.sleep(1)
            else:
                raise


def write_metrics_csv(output_dir: Path, rows: Iterable[TreeMetricRow]) -> Path:
    import os
    import time
    
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
    
    # Convert iterable to list for potential retries
    rows_list = list(rows)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows_list:
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
            
            # Verify file was written
            if not out_path.exists():
                raise FileNotFoundError(f"Metrics CSV not written: {out_path}")
            file_size = out_path.stat().st_size
            print(f"✓ Wrote metrics CSV: {out_path.name} ({file_size} bytes)")
            return out_path
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Retry writing metrics CSV ({attempt + 1}/{max_retries}): {e}")
                os.sync()
                time.sleep(1)
            else:
                raise


def write_run_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    import os
    import time
    
    out_path = output_dir / "run_manifest.json"
    full_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    manifest_text = json.dumps(full_manifest, indent=2)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            out_path.write_text(manifest_text, encoding="utf-8")
            
            # Verify file was written
            if not out_path.exists():
                raise FileNotFoundError(f"Manifest JSON not written: {out_path}")
            file_size = out_path.stat().st_size
            print(f"✓ Wrote run manifest: {out_path.name} ({file_size} bytes)")
            return out_path
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Retry writing manifest JSON ({attempt + 1}/{max_retries}): {e}")
                os.sync()
                time.sleep(1)
            else:
                raise

