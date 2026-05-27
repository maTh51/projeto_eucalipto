"""Unified pipeline runner driven by canonical YAML/JSON config."""

from __future__ import annotations

from dataclasses import asdict
from glob import glob
from pathlib import Path
import subprocess
import os
import shutil
from typing import Dict, List

import numpy as np

from . import dbh_methods, io, leafwood_docker, volume_methods
from .adapters import (
    adapter_ff3d_to_instances,
    adapter_ff3d_to_segmentation,
    adapter_leafwood_to_segmentation,
    adapter_treeiso_to_instances,
)
from .config_schema import PipelineConfig
from .contracts import MISSING_INT, SCHEMA_VERSION, TreeMetricRow
from .dependencies import resolve_dependencies, validate_dependency_paths
from .isolation_ff3d import run_ff3d_docker
from .isolation_treeiso import run_treeiso_on_dir
from .output_contract import (
    ensure_output_dir,
    write_canonical_cloud,
    write_metrics_csv,
    write_run_manifest,
)
from .preprocess_treeiso import prepare_treeiso_input
from .validation import validate_pipeline_config


MODE_REQUIRED_DEPENDENCIES: Dict[str, List[str]] = {
    "ff3d_full": ["ff3d_inference", "forestformer3d"],
    "treeiso_leafwood": ["treeiso", "leafwood"],
    "treeiso_leafwood_rctqsm": ["treeiso", "leafwood"],
    "rayextract_full": ["rayextract_manual"],
    "rct_qsm_metrics": ["rayextract_manual"],
}


def _collect_metrics(
    points_xyz: np.ndarray,
    tree_id: np.ndarray,
    trunk_leaf_label: np.ndarray,
    metric_provider: str,
    wood_density: float | None = None,
) -> List[TreeMetricRow]:
    rows: List[TreeMetricRow] = []
    for tid in sorted(np.unique(tree_id)):
        if int(tid) == MISSING_INT:
            continue
        mask = tree_id == tid
        trunk_pts = points_xyz[mask & (trunk_leaf_label == 1)]
        warnings: List[str] = []
        if trunk_pts.shape[0] < 10:
            rows.append(
                TreeMetricRow(
                    tree_id=int(tid),
                    dbh_cm=None,
                    height_m=None,
                    volume_m3=None,
                    mass_kg=None,
                    metric_provider=metric_provider,
                    warnings=["insufficient_trunk_points"],
                )
            )
            continue

        dbh_cm, _ = dbh_methods.estimate_dbh(trunk_pts, method="ensemble")
        if dbh_cm is None:
            rows.append(
                TreeMetricRow(
                    tree_id=int(tid),
                    dbh_cm=None,
                    height_m=float(io.tree_height(trunk_pts)),
                    volume_m3=None,
                    mass_kg=None,
                    metric_provider=metric_provider,
                    warnings=["dbh_estimation_failed"],
                )
            )
            continue

        height_m = float(io.tree_height(trunk_pts))
        vol_info = volume_methods.estimate_volume(
            trunk_pts,
            dbh_cm=dbh_cm,
            height_m=height_m,
            method="cylinder",
            wood_density_kg_m3=wood_density,
        )
        rows.append(
            TreeMetricRow(
                tree_id=int(tid),
                dbh_cm=float(dbh_cm),
                height_m=float(vol_info.get("height_m", height_m)),
                volume_m3=float(vol_info["volume_m3"]),
                mass_kg=(
                    float(vol_info["mass_kg"])
                    if vol_info.get("mass_kg") is not None
                    else None
                ),
                metric_provider=metric_provider,
                warnings=warnings,
            )
        )
    return rows


def _run_ff3d_mode(cfg: PipelineConfig, dep_paths: Dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    provider_cfg = cfg.providers.get("ff3d", {})
    input_path = cfg.input.path
    if Path(input_path).suffix.lower() == ".ply":
        input_path = prepare_treeiso_input(input_path, pre_filter_enabled=False, output_root_dir=cfg.output.output_dir)

    default_bucket_in = str(dep_paths["ff3d_inference"] / "FF3D_oracle/bucket_in_folder")
    default_bucket_out = str(dep_paths["ff3d_inference"] / "FF3D_oracle/bucket_out_folder")

    bucket_in = provider_cfg.get("bucket_in_dir")
    bucket_out = provider_cfg.get("bucket_out_dir")

    if not bucket_in or str(bucket_in).startswith("/path/to/"):
        bucket_in = default_bucket_in
    if not bucket_out or str(bucket_out).startswith("/path/to/"):
        bucket_out = default_bucket_out

    instance_dim = provider_cfg.get("instance_dim", "instance_pred")
    semantic_dim = provider_cfg.get("semantic_dim", "semantic_pred")
    trunk_label = int(provider_cfg.get("trunk_label", 1))

    # Clean output directory
    Path(bucket_out).mkdir(parents=True, exist_ok=True)
    for item in Path(bucket_out).iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    run_ff3d_docker(
        ff3d_repo_dir=str(dep_paths["ff3d_inference"]),
        bucket_in_dir=bucket_in,
        bucket_out_dir=bucket_out,
        input_laz=input_path,
    )
    files = sorted(
        glob(str(Path(bucket_out) / "**" / "*.la*"), recursive=True)
        + glob(str(Path(bucket_out) / "**" / "*.ply"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"FF3D output not found in {bucket_out}")
    points, extras, _format = io.load_cloud_auto(files[0])
    inst, inst_meta = adapter_ff3d_to_instances(points, extras[instance_dim])
    seg, seg_meta = adapter_ff3d_to_segmentation(points, inst.tree_id, extras[semantic_dim], trunk_label=trunk_label)
    mapping = {**inst_meta.field_mapping, **seg_meta.field_mapping}
    return points, inst.tree_id, seg.trunk_leaf_label, mapping


def _run_treeiso_leafwood_mode(cfg: PipelineConfig, dep_paths: Dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    provider_cfg = cfg.providers.get("treeiso", {})
    input_path = prepare_treeiso_input(
        cfg.input.path,
        pre_filter_enabled=bool(provider_cfg.get("pre_filter_enabled", False)),
        pre_filter_dim=provider_cfg.get("pre_filter_dim", "treeID"),
        pre_filter_ground_value=int(provider_cfg.get("pre_filter_ground_value", 0)),
        output_root_dir=cfg.output.output_dir,
    )
    outputs = run_treeiso_on_dir(
        treeiso_repo_dir=str(dep_paths["treeiso"]),
        input_dir=input_path,
        force_python_cut_pursuit=bool(provider_cfg.get("force_python_cut_pursuit", True)),
    )
    if not outputs:
        raise RuntimeError("treeiso produced no outputs")

    points_all: list[np.ndarray] = []
    tree_id_all: list[np.ndarray] = []
    trunk_leaf_all: list[np.ndarray] = []

    tree_offset = 0
    tree_field = provider_cfg.get("tree_id_dim", "final_segs")

    inst_meta = None
    seg_meta = None

    lw_cfg = cfg.providers.get("leafwood", {})
    for out_path in outputs:
        points, extras = io.load_laz(out_path)
        seg_ids = np.asarray(extras[tree_field]).astype(np.int32)
        inst, inst_meta_local = adapter_treeiso_to_instances(points, seg_ids)

        print(f"Processing TreeISO output with {len(np.unique(inst.tree_id))} segments...")
        
        # Prepare optional parameters for leafwood inference
        model_ckpt = lw_cfg.get("model_ckpt", "/project/model_weights/weights_randlanet.pth")
        grid_size = float(lw_cfg.get("grid_size", 0.02))
        model_name = lw_cfg.get("model_name", "RandLANet")
        print(f"  ↳ Model: {model_name} | Grid: {grid_size}m")
        
        labels_by_segment = leafwood_docker.run_leafwood_for_treeiso_segments(
            points=points,
            seg_ids=inst.tree_id,
            segment_ids=np.unique(inst.tree_id),
            leafwood_repo_dir=str(dep_paths["leafwood"]),
            docker_subdir=lw_cfg.get("docker_subdir", "docker"),
            docker_service=lw_cfg.get("docker_service", "open3dml"),
            model_ckpt=model_ckpt,
            device=lw_cfg.get("device", "cuda"),
            batch_size=int(lw_cfg.get("batch_size", 1)),
            job_name=lw_cfg.get("job_name", "canonical_treeiso_leafwood"),
            grid_size=grid_size,
            model_name=model_name,
        )

        if not labels_by_segment:
            print(f"⚠️  WARNING: No leaf-wood predictions generated for {out_path.name}!")
            print(f"   Defaulting all points to leaf class (0)")
            trunk_leaf = np.zeros(points.shape[0], dtype=np.int32)
        else:
            trunk_leaf = np.zeros(points.shape[0], dtype=np.int32)
            for sid in np.unique(inst.tree_id):
                mask = inst.tree_id == sid
                if int(sid) not in labels_by_segment:
                    print(f"  ⚠ Segment {sid} missing from predictions, using default (leaf)")
                    trunk_leaf[mask] = 0
                else:
                    try:
                        trunk_leaf[mask] = labels_by_segment[int(sid)]
                    except Exception as e:
                        print(f"  ✗ Error assigning labels for segment {sid}: {e}")
                        trunk_leaf[mask] = 0

        seg, seg_meta_local = adapter_leafwood_to_segmentation(points, inst.tree_id, trunk_leaf)

        # Ensure canonical tree_ids are unique across multiple treeiso outputs.
        new_tree_id = inst.tree_id.astype(np.int32) + tree_offset
        tree_offset += int(np.max(inst.tree_id)) + 1 if inst.tree_id.size > 0 else 0

        points_all.append(points)
        tree_id_all.append(new_tree_id)
        trunk_leaf_all.append(seg.trunk_leaf_label)

        inst_meta = inst_meta_local
        seg_meta = seg_meta_local

    mapping: Dict[str, str] = {}
    if inst_meta:
        mapping.update(inst_meta.field_mapping)
    if seg_meta:
        mapping.update(seg_meta.field_mapping)

    return (
        np.vstack(points_all),
        np.concatenate(tree_id_all),
        np.concatenate(trunk_leaf_all),
        mapping,
    )


def _run_rayextract_full(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    rct_cfg = cfg.providers.get("rayextract", {})
    work_dir = Path(cfg.output.output_dir).resolve() / "rayextract_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(cfg.input.path).resolve()

    pcd_path = input_path
    if input_path.suffix.lower() in {".las", ".laz"}:
        pts, _extras = io.load_laz(str(input_path))
        pcd_path = work_dir / f"{input_path.stem}.ply"
        io.save_ply_with_fields(str(pcd_path), pts, {})

    # Ensure the pointcloud file is available inside the work_dir so the
    # container can access it via the mounted /workspace path. If the input
    # is a .ply located outside work_dir, copy it into work_dir.
    try:
        pcd_resolved = pcd_path.resolve()
    except Exception:
        pcd_resolved = pcd_path

    if not str(pcd_resolved).startswith(str(work_dir.resolve())):
        dest = work_dir / pcd_path.name
        if pcd_resolved.exists():
            shutil.copy2(pcd_resolved, dest)
            pcd_path = dest

    def _run_cmd_try_local_then_docker(cmd: list[str]):
        try:
            subprocess.run(cmd, cwd=work_dir, check=True)
            return
        except FileNotFoundError:
            docker_image = "ghcr.io/csiro-robotics/raycloudtools:latest"
            # Run container as current user and mount workspace read-write.
            user_flag = ["--user", f"{os.getuid()}:{os.getgid()}"]
            mount = f"{work_dir}:/workspace:rw"
            def _translate_arg_to_container(arg: str) -> str:
                try:
                    p = Path(arg)
                except Exception:
                    return arg

                if not p.is_absolute():
                    return arg

                try:
                    rel = p.resolve().relative_to(work_dir.resolve())
                except Exception:
                    return arg

                return str(Path("/workspace") / rel)

            container_args = [_translate_arg_to_container(a) for a in cmd]

            docker_cmd = [
                "docker",
                "run",
                "--rm",
            ] + user_flag + [
                "-v",
                mount,
                "-w",
                "/workspace",
                docker_image,
            ] + container_args

            subprocess.run(docker_cmd, check=True)

    _run_cmd_try_local_then_docker(["rayimport", str(pcd_path), "ray", "0,0,-1", "--max_intensity", "0"])
    raycloud = work_dir / f"{pcd_path.stem}_raycloud.ply"
    
    # Build rayextract terrain command with optional parameters
    terrain_cmd = ["rayextract", "terrain", str(raycloud)]
    if "gradient" in rct_cfg:
        terrain_cmd.extend(["--gradient", str(rct_cfg["gradient"])])
    _run_cmd_try_local_then_docker(terrain_cmd)
    
    terrain = work_dir / f"{pcd_path.stem}_raycloud_mesh.ply"
    
    # Build rayextract trees command with optional parameters
    trees_cmd = ["rayextract", "trees", str(raycloud), str(terrain)]
    
    # Map config keys to rayextract command-line flags (value parameters)
    value_params = {
        "max_diameter": "--max_diameter",
        "crop_length": "--crop_length",
        "height_min": "--height_min",
        "girth_height_ratio": "--girth_height_ratio",
        "gravity_factor": "--gravity_factor",
        "global_taper": "--global_taper",
        "global_taper_factor": "--global_taper_factor",
        "distance_limit": "--distance_limit",
        "grid_width": "--grid_width",
    }
    
    # Flag parameters (boolean, no value needed)
    flag_params = {
        "branch_segmentation": "--branch_segmentation",
        "use_rays": "--use_rays",
    }
    
    for config_key, cli_flag in value_params.items():
        if config_key in rct_cfg:
            value = rct_cfg[config_key]
            # Skip if value is a boolean (invalid for value parameters)
            if isinstance(value, bool):
                continue
            trees_cmd.append(cli_flag)
            trees_cmd.append(str(value))
    
    for config_key, cli_flag in flag_params.items():
        if config_key in rct_cfg and rct_cfg[config_key]:
            trees_cmd.append(cli_flag)
    
    _run_cmd_try_local_then_docker(trees_cmd)

    segmented = work_dir / f"{pcd_path.stem}_raycloud_segmented.ply"
    points, extras = io.load_ply(str(segmented))
    # For RCT segmented clouds, there is no strict canonical field by default.
    if "tree_id" in extras:
        tree_id = np.asarray(extras["tree_id"]).astype(np.int32)
    else:
        tree_id = np.full(points.shape[0], -1, dtype=np.int32)
    # RCT leaf/wood split is separate; fill sentinel unless configured.
    trunk_leaf = np.full(points.shape[0], -1, dtype=np.int32)
    return points, tree_id, trunk_leaf, {"rct_native": "tree_id/trunk_leaf_label"}


def run_pipeline(cfg: PipelineConfig) -> Dict[str, str]:
    validate_pipeline_config(cfg)
    resolved = resolve_dependencies(Path(__file__).resolve().parents[1], cfg.external_paths)
    required_deps = MODE_REQUIRED_DEPENDENCIES.get(cfg.pipeline_mode)
    validate_dependency_paths(resolved, required_dependencies=required_deps)

    output_dir = ensure_output_dir(cfg.output.output_dir)
    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_mode": cfg.pipeline_mode,
        "modules_executed": [],
        "field_mapping": {},
        "conversions": [],
        "warnings": [],
        "errors": [],
        "dependency_paths": {k: str(v) for k, v in resolved.paths.items()},
    }

    if cfg.pipeline_mode == "ff3d_full":
        points, tree_id, trunk_leaf, mapping = _run_ff3d_mode(cfg, resolved.paths)
        manifest["modules_executed"] = ["ff3d", "metrics"]
    elif cfg.pipeline_mode in {"treeiso_leafwood", "treeiso_leafwood_rctqsm"}:
        points, tree_id, trunk_leaf, mapping = _run_treeiso_leafwood_mode(cfg, resolved.paths)
        manifest["modules_executed"] = ["treeiso", "leafwood", "metrics"]
    elif cfg.pipeline_mode in {"rayextract_full", "rct_qsm_metrics"}:
        points, tree_id, trunk_leaf, mapping = _run_rayextract_full(cfg)
        manifest["modules_executed"] = ["rayextract", "metrics"]
    else:
        raise ValueError(f"Unsupported pipeline mode: {cfg.pipeline_mode}")

    manifest["field_mapping"] = mapping

    rows = _collect_metrics(
        points_xyz=points,
        tree_id=tree_id,
        trunk_leaf_label=trunk_leaf,
        metric_provider="canonical_metrics",
        wood_density=cfg.providers.get("metrics", {}).get("wood_density_kg_m3"),
    )
    cloud_path = write_canonical_cloud(
        output_dir=output_dir,
        points_xyz=points,
        tree_id=tree_id,
        trunk_leaf_label=trunk_leaf,
        source_format=cfg.input.format,
        cloud_format=cfg.output.cloud_format,
    )
    metrics_path = write_metrics_csv(output_dir, rows)
    manifest_path = write_run_manifest(output_dir, manifest)

    return {
        "classified_cloud": str(cloud_path),
        "metrics_csv": str(metrics_path),
        "manifest": str(manifest_path),
    }

