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
) -> None:
    cfg_text = f"""dataset:
  dataset_path: {dataset_path_in_container}
  test_dir: {test_dir}
  test_result_folder: {test_result_folder}
model:
  name: RandLANet
  ckpt_path: {model_ckpt}
  num_layers: 5
  sub_sampling_ratio: [4, 4, 4, 4, 2]
  dim_output: [16, 64, 128, 256, 512]
  num_points: 65536
  in_channels: 3
  dim_features: 3
  num_classes: 2
  class_weights: [34512040, 9127323]
  grid_size: 0.02
  augment:
    recenter:
      dim: [0, 1, 2]
pipeline:
  device: '{device}'
  batch_size: {batch_size}
"""
    cfg_path.write_text(cfg_text, encoding="utf-8")


def _extract_labels(pred_file: Path) -> np.ndarray:
    arr = np.loadtxt(pred_file)
    arr = np.asarray(arr)

    if arr.ndim == 1:
        labels = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        labels = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] >= 4:
        labels = arr[:, -1]
    else:
        raise ValueError(f"Unsupported prediction format: {pred_file}")

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
) -> Dict[int, np.ndarray]:
    """Run leaf-wood inference for all tree segments in one treeiso output.

    Returns a mapping: segment_id -> predicted labels (0=leaf, 1=wood).
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

    for sid in unique_segment_ids:
        mask = seg_ids_arr == sid
        xyz = points[mask, :3].astype(np.float32)
        if xyz.shape[0] == 0:
            continue
        out_txt = host_inputs_dir / f"tree_segment_id_{sid}.txt"
        np.savetxt(out_txt, xyz, fmt="%.3f")

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
        )
        subprocess.run(cpu_infer_cmd, cwd=str(docker_dir), check=True)

    labels_by_segment: Dict[int, np.ndarray] = {}
    for sid in unique_segment_ids:
        pred_file = host_preds_dir / f"tree_segment_id_{sid}.txt"
        if not pred_file.exists():
            raise FileNotFoundError(
                f"Prediction file missing for segment {sid}: {pred_file}"
            )

        labels = _extract_labels(pred_file)
        n_expected = int(np.count_nonzero(seg_ids_arr == sid))
        if labels.shape[0] != n_expected:
            raise ValueError(
                f"Prediction size mismatch for segment {sid}: "
                f"pred={labels.shape[0]} vs expected={n_expected}"
            )

        labels_by_segment[sid] = labels

    return labels_by_segment
