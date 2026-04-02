"""Default FF3D-based pipeline script.

This script demonstrates an end-to-end workflow:

1. Run FF3D_inference via Docker on a full-plot LAS/LAZ file.
2. Extract per-tree trunk point clouds using FF3D semantic labels.
3. Compute DBH (ensemble method) and cylinder volume for each tree.
4. Save a CSV summary in the results directory.

Adjust the CONFIG section below to your environment before running:

    python run_ff3d_pipeline.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from eucalipto import pipeline_core


# ==========================
# CONFIGURATION (EDIT HERE)
# ==========================

# Path to FF3D_inference/ff3d_forestsens directory
FF3D_REPO_DIR = "/home/matheuspimenta/Jobs/Eucalipto/FF3D_inference/ff3d_forestsens"

# Bucket in/out directories used by FF3D_inference
BUCKET_IN_DIR = "/home/matheuspimenta/Jobs/Eucalipto/FF3D_inference/FF3D_oracle/bucket_in_folder"
BUCKET_OUT_DIR = "/home/matheuspimenta/Jobs/Eucalipto/FF3D_inference/FF3D_oracle/bucket_out_folder"

# Input LAS/LAZ file for a plot
INPUT_LAZ = "/path/to/your/input.laz"  # TODO: set this

# Names of FF3D dimensions and trunk label value
INSTANCE_DIM = "instance_id"   # adjust to actual FF3D output
SEMANTIC_DIM = "sem_label"     # adjust to actual FF3D output
TRUNK_LABEL = 1                 # integer label corresponding to trunk class

# Output directory inside projeto_eucalipto
RESULTS_DIR = "results_ff3d"

# Wood density (kg/m^3) for mass estimate; set None to skip
WOOD_DENSITY = 900.0


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_tree_trunks = pipeline_core.run_isolation_ff3d(
        ff3d_repo_dir=FF3D_REPO_DIR,
        bucket_in_dir=BUCKET_IN_DIR,
        bucket_out_dir=BUCKET_OUT_DIR,
        input_laz=INPUT_LAZ,
        instance_dim=INSTANCE_DIM,
        semantic_dim=SEMANTIC_DIM,
        trunk_label=TRUNK_LABEL,
    )

    metrics = pipeline_core.run_metrics_on_trunks(
        per_tree_trunks,
        dbh_method="ensemble",
        volume_method="cylinder",
        wood_density_kg_m3=WOOD_DENSITY,
    )

    csv_path = Path(RESULTS_DIR) / "ff3d_metrics_summary.csv"
    fieldnames = [
        "tree_id",
        "dbh_cm",
        "height_m",
        "volume_m3",
        "volume_liters",
        "mass_kg",
        "dbh_method",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tid, res in sorted(metrics.items()):
            row = {
                "tree_id": tid,
                "dbh_cm": res.get("dbh_cm"),
                "height_m": res.get("height_m"),
                "volume_m3": res.get("volume_m3"),
                "volume_liters": res.get("volume_liters"),
                "mass_kg": res.get("mass_kg"),
                "dbh_method": res.get("dbh_info", {}).get("method"),
            }
            writer.writerow(row)

    print(f"Wrote metrics summary to {csv_path}")


if __name__ == "__main__":
    main()
