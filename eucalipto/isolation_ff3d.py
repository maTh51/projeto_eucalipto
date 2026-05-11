"""FF3D-based tree isolation and trunk extraction."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from glob import glob
from pathlib import Path
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
    """Run FF3D Docker inference for a single input cloud."""
    os.makedirs(bucket_in_dir, exist_ok=True)
    os.makedirs(bucket_out_dir, exist_ok=True)

    for item in Path(bucket_in_dir).iterdir():
        if item.is_file():
            item.unlink()

    input_path = Path(input_laz)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_laz}")

    dest_path = Path(bucket_in_dir) / input_path.name
    shutil.copy2(str(input_path), str(dest_path))
    print(f"Copied {input_laz} to {dest_path}")

    if not dest_path.exists():
        raise FileNotFoundError(f"Failed to copy file to {dest_path}")
    time.sleep(1)

    container_name = os.environ.get("FF3D_CONTAINER_NAME", "forestformer-forestsens-container")
    subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            f"rm -f /workspace/data/ForAINetV2/test_data/{input_path.name} && mkdir -p /workspace/data/ForAINetV2/test_data",
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "cp",
            str(dest_path),
            f"{container_name}:/workspace/data/ForAINetV2/test_data/{input_path.name}",
        ],
        check=True,
    )

    files_in_bucket = list(Path(bucket_in_dir).glob("*"))
    print(f"Files in bucket_in_dir ({bucket_in_dir}): {files_in_bucket}")

    env = os.environ.copy()
    env["HOST_PROJECT_DIR"] = ff3d_repo_dir
    env["HOST_BUCKET_IN"] = bucket_in_dir
    env["HOST_BUCKET_OUT"] = bucket_out_dir
    env["REBUILD_IMAGE"] = rebuild_image
    env["RECREATE_CONTAINER"] = recreate_container

    script_path = os.path.join(ff3d_repo_dir, "run_docker_locally.sh")
    subprocess.run(["bash", script_path], cwd=ff3d_repo_dir, env=env, check=True)

    _extract_ff3d_outputs_from_container(
        container_name=container_name,
        bucket_out_dir=bucket_out_dir,
    )

    return bucket_out_dir


def _extract_ff3d_outputs_from_container(container_name: str, bucket_out_dir: str) -> None:
    """Extract the newest FF3D ZIP from the container and publish outputs on the host."""
    print(f"Extracting FF3D outputs from container {container_name}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            listing = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "bash",
                    "-lc",
                    "ls -1t /workspace/work_dirs/output/results_*.zip 2>/dev/null | head -1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            zip_name = listing.stdout.strip()
            if not zip_name:
                raise FileNotFoundError("No results_*.zip file found in container output directory")

            subprocess.run(
                ["docker", "cp", f"{container_name}:{zip_name}", str(tmpdir_path / "results.zip")],
                check=True,
                capture_output=True,
            )
        except Exception as exc:
            print(f"Warning: Could not extract results ZIP from container: {exc}")
            result = subprocess.run(
                ["docker", "exec", container_name, "ls", "-lah", "/workspace/work_dirs/output/"],
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print(result.stderr)
            return

        zip_path = tmpdir_path / "results.zip"
        if not zip_path.exists():
            print(f"Warning: ZIP not found at {zip_path}")
            return

        extracted_root = tmpdir_path / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)

        print(f"Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(extracted_root))

        for extracted_file in extracted_root.rglob("*"):
            if not extracted_file.is_file():
                continue
            relative_path = extracted_file.relative_to(extracted_root)
            target_path = Path(bucket_out_dir) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(extracted_file), str(target_path))

        ply_files = list(extracted_root.rglob("*.ply"))
        print(f"Found {len(ply_files)} PLY file(s)")

        for ply_file in ply_files:
            print(f"Converting {ply_file.name} to LAZ...")
            laz_file = Path(bucket_out_dir) / ply_file.name.replace(".ply", ".laz")
            try:
                ply_las = laspy.read(str(ply_file))
                ply_las.write(str(laz_file))
                print(f"✓ Converted to {laz_file}")
            except Exception as exc:
                print(f"✗ Failed to convert {ply_file.name}: {exc}")
                shutil.copy2(str(ply_file), str(Path(bucket_out_dir) / ply_file.name))
                print(f"✓ Copied PLY as fallback to {Path(bucket_out_dir) / ply_file.name}")


def _load_labelled_laz(
    path: str,
    instance_dim: str,
    semantic_dim: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    """Collect per-tree trunk points from FF3D outputs."""
    pattern = os.path.join(bucket_out_dir, "**", "*.la*")
    files = sorted(glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under {bucket_out_dir}")

    labelled_path = files[0]
    points, instance_ids, semantic_ids = _load_labelled_laz(
        labelled_path,
        instance_dim=instance_dim,
        semantic_dim=semantic_dim,
    )

    trunk_mask = semantic_ids == trunk_label
    trunk_points = points[trunk_mask]
    trunk_instances = instance_ids[trunk_mask]

    per_tree: Dict[int, list] = {}
    for point, inst in zip(trunk_points, trunk_instances):
        per_tree.setdefault(int(inst), []).append(point)

    return {
        tree_id: np.vstack(points_list)
        for tree_id, points_list in per_tree.items()
        if len(points_list) > 0
    }
