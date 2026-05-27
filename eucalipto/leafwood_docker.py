"""Leaf-wood Docker integration for treeiso outputs.

This module prepares per-segment point clouds from a treeiso output,
runs leaf-wood inference inside the existing Docker environment, and
returns per-segment predicted labels.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def _sanitize_job_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _write_leafwood_cfg(
    cfg_path: Path,
    dataset_path_in_container: str,
    test_dir: str,
    test_result_folder: str,
    model_ckpt: str,
    device: str,
    batch_size: int,
    grid_size: float = 0.02,
    num_points: int = 65536,
    model_name: str = "RandLANet",
) -> None:
    """Write leaf-wood inference config with adjustable parameters.
    
    Available models: 'RandLANet', 'KPConv', 'PointTransformer'
    grid_size: Voxel downsampling resolution. Smaller = more detail (slower).
               Larger = faster but loses fine details. Default 0.02 for dense clouds.
               For sparse/occluded clouds: 0.03-0.05
    num_points: Points sampled from each segment. Default 65536.
    """
    dataset_section = f"""dataset:
  dataset_path: {dataset_path_in_container}
  test_dir: {test_dir}
  test_result_folder: {test_result_folder}"""
    
    # Generate model-specific configuration
    if model_name == "RandLANet":
        model_section = f"""model:
  name: {model_name}
  ckpt_path: {model_ckpt}
  num_layers: 5
  sub_sampling_ratio: [4, 4, 4, 4, 2]
  dim_output: [16, 64, 128, 256, 512]
  num_points: {num_points}
  in_channels: 3
  dim_features: 3
  num_classes: 2
  class_weights: [34512040, 9127323]
  grid_size: {grid_size}
  augment:
    recenter:
      dim: [0, 1, 2]"""
    
    elif model_name == "KPConv":
        model_section = f"""model:
  name: {model_name}
  ckpt_path: {model_ckpt}
  first_subsampling_dl: {grid_size}
  min_in_points: {num_points}
  in_points_dim: 3
  in_features_dim: 3
  num_classes: 2
  class_weights: [34512040, 9127323]
  lbl_values: [0, 1]
  ignored_label_inds: []
  augment:
    recenter:
      dim: [0, 1, 2]"""
    
    elif model_name == "PointTransformer":
        model_section = f"""model:
  name: {model_name}
  ckpt_path: {model_ckpt}
  blocks: [2, 3, 4, 6, 3]
  in_channels: 3
  num_classes: 2
  ignored_label_inds: []
  voxel_size: {grid_size}
  max_voxels: {num_points}
  augment:
    recenter:
      dim: [0, 1, 2]"""
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: RandLANet, KPConv, PointTransformer")
    
    pipeline_section = f"""pipeline:
  device: '{device}'
  batch_size: {batch_size}"""
    
    cfg_text = f"""{dataset_section}
{model_section}
{pipeline_section}
"""
    cfg_path.write_text(cfg_text, encoding="utf-8")


def _extract_labels(pred_file: Path) -> np.ndarray:
    """Extract predicted labels from a prediction file.
    
    Handles various formats and returns appropriate error diagnostics.
    """
    if not pred_file.stat().st_size > 0:
        raise ValueError(f"Prediction file is empty: {pred_file}")
    
    try:
        arr = np.loadtxt(pred_file)
    except Exception as e:
        raise ValueError(f"Failed to load predictions from {pred_file}: {e}")
    
    arr = np.asarray(arr)

    # Handle different output formats
    if arr.ndim == 0:
        # Single scalar value
        labels = np.array([arr.item()], dtype=np.int32)
    elif arr.ndim == 1:
        labels = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        labels = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] >= 4:
        # Multi-column format, last column is usually the prediction
        labels = arr[:, -1]
    else:
        raise ValueError(f"Unsupported prediction format from {pred_file}: shape={arr.shape}")

    if labels.size == 0:
        raise ValueError(f"No labels extracted from {pred_file}")

    return np.rint(labels).astype(np.int32)



def run_leafwood_for_treeiso_segments(
    points: np.ndarray,
    seg_ids: np.ndarray,
    segment_ids: Iterable[int],
    leafwood_repo_dir: str,
    docker_subdir: str = "docker",
    docker_service: str = "open3dml",
    model_ckpt: str = "/project/model_weights/weights_randlanet.pth",
    device: str = "cuda",
    batch_size: int = 1,
    job_name: str = "treeiso_job",
    grid_size: float = 0.02,
    model_name: str = "RandLANet",
) -> Dict[int, np.ndarray]:
    """Run leaf-wood inference for all tree segments in one treeiso output.

    Returns a mapping: segment_id -> predicted labels (0=leaf, 1=wood).
    
    Parameters:
    -----------
    grid_size : float
        Voxel downsampling before network. Default 0.02 (2cm).
        For sparse/occluded data: try 0.03-0.05
        For very dense data: try 0.01
    model_name : str
        Name of model architecture in checkpoint file.
        Available: 'RandLANet', 'KPConv', 'PointTransformer'
    """
    repo_dir = Path(leafwood_repo_dir).expanduser().resolve()
    docker_dir = repo_dir / docker_subdir
    cfg_dir = repo_dir / "cfg"

    if not repo_dir.exists():
        raise FileNotFoundError(f"Leaf-wood repo not found: {repo_dir}")
    if not docker_dir.exists():
        raise FileNotFoundError(f"Leaf-wood docker dir not found: {docker_dir}")

    safe_job_name = _sanitize_job_name(job_name)
    host_dataset_dir = repo_dir / "data" / "treeiso_jobs" / safe_job_name
    host_inputs_dir = host_dataset_dir / "inputs"
    host_preds_dir = host_dataset_dir / "predictions"

    host_inputs_dir.mkdir(parents=True, exist_ok=True)
    host_preds_dir.mkdir(parents=True, exist_ok=True)

    # Keep each run isolated to avoid stale files in inference folder.
    for file_path in host_inputs_dir.glob("*"):
        if file_path.is_file():
            file_path.unlink()
    for file_path in host_preds_dir.glob("*"):
        if file_path.is_file():
            file_path.unlink()

    seg_ids_arr = np.asarray(seg_ids).astype(np.int32)
    unique_segment_ids = [int(sid) for sid in segment_ids]

    print(f"Preparing {len(unique_segment_ids)} segments for leaf-wood inference...")
    print(f"  Model: {model_name} | Grid: {grid_size}m | Device: {device}")
    saved_segments = 0
    skipped_segments = []
    
    for sid in unique_segment_ids:
        mask = seg_ids_arr == sid
        xyz = points[mask, :3].astype(np.float32)
        
        if xyz.shape[0] == 0:
            skipped_segments.append(sid)
            continue
        
        out_txt = host_inputs_dir / f"tree_segment_id_{sid}.txt"
        np.savetxt(out_txt, xyz, fmt="%.3f")
        saved_segments += 1

    print(f"✓ Saved {saved_segments}/{len(unique_segment_ids)} segments")
    if skipped_segments:
        print(f"⚠ Skipped {len(skipped_segments)} empty segments: {skipped_segments[:5]}{'...' if len(skipped_segments) > 5 else ''}")

    cfg_name = f"rln_infer_treeiso_{safe_job_name}.yml"
    cpu_cfg_name = f"rln_infer_treeiso_{safe_job_name}_cpu.yml"
    cfg_path = cfg_dir / cfg_name
    cpu_cfg_path = cfg_dir / cpu_cfg_name
    dataset_path_in_container = f"/project/data/treeiso_jobs/{safe_job_name}"
    _write_leafwood_cfg(
        cfg_path=cfg_path,
        dataset_path_in_container=dataset_path_in_container,
        test_dir="inputs",
        test_result_folder="predictions",
        model_ckpt=model_ckpt,
        device=device,
        batch_size=batch_size,
        grid_size=grid_size,
        model_name=model_name,
    )

    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate"],
        cwd=str(docker_dir),
        check=True,
    )
    infer_cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        docker_service,
        "python",
        "scripts/infer.py",
        f"cfg/{cfg_name}",
    ]
    cpu_infer_cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        docker_service,
        "python",
        "scripts/infer.py",
        f"cfg/{cpu_cfg_name}",
    ]

    try:
        subprocess.run(infer_cmd, cwd=str(docker_dir), check=True)
    except subprocess.CalledProcessError:
        if device.lower() == "cpu":
            raise
        _write_leafwood_cfg(
            cfg_path=cpu_cfg_path,
            dataset_path_in_container=dataset_path_in_container,
            test_dir="inputs",
            test_result_folder="predictions",
            model_ckpt=model_ckpt,
            device="cpu",
            batch_size=batch_size,
            grid_size=grid_size,
            model_name=model_name,
        )
        subprocess.run(cpu_infer_cmd, cwd=str(docker_dir), check=True)

    labels_by_segment: Dict[int, np.ndarray] = {}
    missing_segments = []
    mismatched_segments = []
    
    for sid in unique_segment_ids:
        mask = seg_ids_arr == sid
        n_expected = int(np.count_nonzero(mask))
        
        # Check if segment was skipped due to 0 points
        if n_expected == 0:
            print(f"  ⚠ Segment {sid} has 0 points (was skipped during save), using default label")
            labels_by_segment[sid] = np.array([], dtype=np.int32)
            continue
        
        pred_file = host_preds_dir / f"tree_segment_id_{sid}.txt"
        
        if not pred_file.exists():
            print(f"  ✗ Prediction file missing for segment {sid}: {pred_file}")
            print(f"    (inference may have skipped this segment, using default label 0=leaf)")
            missing_segments.append(sid)
            # Use default label (0=leaf) for missing predictions
            labels_by_segment[sid] = np.zeros(n_expected, dtype=np.int32)
            continue

        try:
            labels = _extract_labels(pred_file)
        except Exception as e:
            print(f"  ✗ Failed to extract labels from {pred_file}: {e}")
            missing_segments.append(sid)
            labels_by_segment[sid] = np.zeros(n_expected, dtype=np.int32)
            continue
        
        if labels.shape[0] != n_expected:
            print(f"  ⚠ Prediction size mismatch for segment {sid}: "
                  f"pred={labels.shape[0]} vs expected={n_expected}")
            mismatched_segments.append((sid, labels.shape[0], n_expected))
            # Use the predictions we have, padding with 0s if needed
            if labels.shape[0] < n_expected:
                labels = np.pad(labels, (0, n_expected - labels.shape[0]), constant_values=0)
            else:
                labels = labels[:n_expected]

        labels_by_segment[sid] = labels
        wood_count = int(np.sum(labels))
        leaf_count = n_expected - wood_count
        print(f"  ✓ Segment {sid}: {wood_count} wood, {leaf_count} leaf (total {n_expected})")
    
    if missing_segments:
        print(f"\n⚠️  WARNING: {len(missing_segments)} segments had missing predictions")
        print(f"   These will default to leaf class (0)")
    
    if mismatched_segments:
        print(f"⚠️  WARNING: {len(mismatched_segments)} segments had size mismatches (will be padded/truncated)")
    
    if not any(v.size > 0 for v in labels_by_segment.values()):
        print(f"\n⚠️  WARNING: No valid predictions were generated!")
        print(f"   All segments are either empty or missing predictions.")
        print(f"   This may indicate:")
        print(f"   - The input point clouds were too small for the model")
        print(f"   - The inference failed silently in the container")
        print(f"   - The model produced no output")

    return labels_by_segment
