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

    # Use a temporary directory outside of any Docker volumes for reliable file transfer
    with tempfile.TemporaryDirectory(prefix="ff3d_", dir="/tmp") as tmpdir:
        tmpdir_path = Path(tmpdir)
        temp_copy = tmpdir_path / input_path.name
        
        # Stage the file in /tmp first
        shutil.copy2(str(input_path), str(temp_copy))
        print(f"Staged file to temp location: {temp_copy}")
        
        if not temp_copy.exists():
            raise FileNotFoundError(f"Failed to stage file to {temp_copy}")
        
        os.chmod(str(temp_copy), 0o644)
        os.sync()
        time.sleep(0.5)
        
        if not temp_copy.exists():
            raise FileNotFoundError(f"Staged file disappeared: {temp_copy}")
        if not os.access(str(temp_copy), os.R_OK):
            raise PermissionError(f"Cannot read staged file: {temp_copy}")
        
        file_size = temp_copy.stat().st_size
        print(f"Staged file verified: {temp_copy.name} ({file_size} bytes)")
        
        container_name = os.environ.get("FF3D_CONTAINER_NAME", "forestformer-forestsens-container")
        
        # Prepare container
        print(f"Preparing container {container_name}...")
        
        # Ensure the image exists or rebuild if requested
        check_image = subprocess.run(
            ["docker", "images", "-q", "forestformer-forestsens-image"],
            capture_output=True, text=True, check=True
        )
        image_exists = bool(check_image.stdout.strip())
        
        project_root = Path(__file__).resolve().parents[1]
        if not image_exists or rebuild_image == "always":
            print("Image forestformer-forestsens-image not found or rebuild requested. Building via docker compose...")
            subprocess.run(
                ["docker", "compose", "--profile", "workers", "build", "forestformer-forestsens"],
                cwd=str(project_root),
                check=True
            )

        # Ensure the container is created and running
        check_exist = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        exists = container_name in check_exist.stdout.splitlines()
        
        if not exists:
            print(f"Container {container_name} does not exist. Creating it...")
            os.makedirs(bucket_in_dir, exist_ok=True)
            os.makedirs(bucket_out_dir, exist_ok=True)
            subprocess.run([
                "docker", "run", "-d",
                "--gpus", "all",
                "--shm-size", "128g",
                "-p", "127.0.0.1:49218:22",
                "--name", container_name,
                "-v", f"{ff3d_repo_dir}:/workspace",
                "--mount", f"type=bind,source={bucket_in_dir},target=/workspace/data/ForAINetV2/test_data",
                "--mount", f"type=bind,source={bucket_out_dir},target=/workspace/work_dirs/output",
                "--entrypoint", "bash",
                "forestformer-forestsens-image", "-lc", "sleep infinity"
            ], check=True)
        else:
            check_running = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            running = container_name in check_running.stdout.splitlines()
            if not running:
                print(f"Container {container_name} is stopped. Starting it...")
                subprocess.run(["docker", "start", container_name], check=True)

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
        
        # Extra sync and wait to ensure container is ready
        os.sync()
        time.sleep(1)
        
        # Copy from temp location (not from bucket_in_dir) to avoid volume mount issues
        print(f"Copying file from temp location to container...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        str(temp_copy),
                        f"{container_name}:/workspace/data/ForAINetV2/test_data/{input_path.name}",
                    ],
                    check=True,
                )
                print(f"Successfully copied to container (attempt {attempt + 1})")
                break
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    print(f"docker cp failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Temp file exists: {temp_copy.exists()}")
                    if temp_copy.exists():
                        print(f"Temp file size: {temp_copy.stat().st_size} bytes, readable: {os.access(str(temp_copy), os.R_OK)}")
                    time.sleep(2)
                else:
                    raise
        
        # Also copy to bucket_in_dir for reference/debugging
        dest_path = Path(bucket_in_dir) / input_path.name
        shutil.copy2(str(temp_copy), str(dest_path))
        print(f"Copied file to bucket_in_folder for reference: {dest_path}")

    files_in_bucket = list(Path(bucket_in_dir).glob("*"))
    print(f"Files in bucket_in_dir ({bucket_in_dir}): {files_in_bucket}")

    env = os.environ.copy()
    env["HOST_PROJECT_DIR"] = ff3d_repo_dir
    env["HOST_BUCKET_IN"] = bucket_in_dir
    env["HOST_BUCKET_OUT"] = bucket_out_dir
    env["REBUILD_IMAGE"] = "never"
    env["RECREATE_CONTAINER"] = recreate_container
    env["SKIP_PROVISION"] = "true"
    env["SKIP_PREPROCESS"] = "true"

    script_path = os.path.join(ff3d_repo_dir, "run_docker_locally.sh")
    subprocess.run(["bash", script_path], cwd=ff3d_repo_dir, env=env, check=True)

    # Restore ownership of output files to the host user
    uid = os.getuid()
    gid = os.getgid()
    subprocess.run(
        ["docker", "exec", container_name, "chown", "-R", f"{uid}:{gid}", "/workspace/work_dirs/output"],
        check=True
    )

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

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    subprocess.run(
                        ["docker", "cp", f"{container_name}:{zip_name}", str(tmpdir_path / "results.zip")],
                        check=True,
                        capture_output=True,
                    )
                    break
                except subprocess.CalledProcessError as e:
                    if attempt < max_retries - 1:
                        print(f"docker cp from container failed (attempt {attempt + 1}/{max_retries}), retrying...")
                        time.sleep(2)
                    else:
                        raise
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

        # Stage extraction in /tmp to avoid Docker volume issues
        extracted_root = tmpdir_path / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)

        print(f"Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(extracted_root))

        # Use staging directory for output writes to avoid Docker volume issues
        stage_root = tmpdir_path / "stage"
        stage_root.mkdir(parents=True, exist_ok=True)

        # Copy extracted files to staging directory first
        print(f"Staging extracted files...")
        for extracted_file in extracted_root.rglob("*"):
            if not extracted_file.is_file():
                continue
            relative_path = extracted_file.relative_to(extracted_root)
            stage_path = stage_root / relative_path
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(extracted_file), str(stage_path))

        # Convert PLY files to LAZ in staging directory
        ply_files = list(stage_root.rglob("*.ply"))
        print(f"Found {len(ply_files)} PLY file(s), converting to LAZ...")
        for ply_file in ply_files:
            print(f"Converting {ply_file.name} to LAZ...")
            laz_file = stage_root / ply_file.name.replace(".ply", ".laz")
            try:
                ply_las = laspy.read(str(ply_file))
                ply_las.write(str(laz_file))
                print(f"✓ Converted to {laz_file.name}")
            except Exception as exc:
                print(f"✗ Failed to convert {ply_file.name}: {exc}")
                print(f"✓ Will copy PLY as fallback")

        # Ensure output directory exists
        Path(bucket_out_dir).mkdir(parents=True, exist_ok=True)
        
        # Force sync before copying to bucket_out_dir
        os.sync()
        time.sleep(0.5)
        
        # Finally, copy all staged files to the actual output directory
        print(f"Copying staged results to output directory...")
        for staged_file in stage_root.rglob("*"):
            if not staged_file.is_file():
                continue
            relative_path = staged_file.relative_to(stage_root)
            target_path = Path(bucket_out_dir) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            max_copy_retries = 3
            for copy_attempt in range(max_copy_retries):
                try:
                    shutil.copy2(str(staged_file), str(target_path))
                    print(f"  ✓ {relative_path.name}")
                    break
                except Exception as e:
                    if copy_attempt < max_copy_retries - 1:
                        print(f"  ⚠ Retry copying {relative_path.name} ({copy_attempt + 1}/{max_copy_retries})")
                        time.sleep(1)
                    else:
                        print(f"  ✗ Failed to copy {relative_path.name}: {e}")
                        raise
        
        print(f"Successfully extracted and staged all outputs to {bucket_out_dir}")


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
