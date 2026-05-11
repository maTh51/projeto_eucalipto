#!/usr/bin/env python3
"""Commercial volume and CBC processing helpers.

This script extracts the notebook logic into two CLI modes:

1. compare-methods
   Evaluate all DBH/volume combinations for point-cloud inputs, excluding qsm.
   This is intended for FF3D / TreeISO style clouds with tree ids and trunk labels.

2. fix-rayextract-mesh
   Rebuild the RayExtract *_with_commercial_mesh.csv files using a tree-relative
   CBC reference (local z-min) and a more robust height estimate.

The RayExtract mesh step is intentionally kept separate because the mesh CSVs are
already per-tree summaries and do not expose a trunk/leaf semantic split.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from eucalipto import io
from eucalipto.rayextract_processor import _get_tree_segments_from_root, read_rayextract_tree_file


DEFAULT_DBH_METHODS = ["ensemble", "single_ransac", "ls"]
DEFAULT_VOLUME_METHODS = ["cylinder", "voxel", "taper", "frustum", "axis_profile"]

DEFAULT_RAYEXTRACT_CSVS = [
    Path("/home/matheuspimenta/Jobs/Eucalipto/rel01/rayextract_plot1_with_commercial_mesh.csv"),
    Path("/home/matheuspimenta/Jobs/Eucalipto/rel01/rayextract_plot2_with_commercial_mesh.csv"),
    Path("/home/matheuspimenta/Jobs/Eucalipto/rel01/rayextract_plot3_with_commercial_mesh.csv"),
]

DEFAULT_RAYEXTRACT_TREEFILES = {
    1: Path("/home/matheuspimenta/Jobs/Eucalipto/drive/outputs/ray/CULS_CULS_plot_1_annotated_train_raycloud_trees.txt"),
    2: Path("/home/matheuspimenta/Jobs/Eucalipto/drive/outputs/ray/CULS_CULS_plot_2_annotated_test_raycloud_trees.txt"),
    3: Path("/home/matheuspimenta/Jobs/Eucalipto/drive/outputs/ray/CULS_CULS_plot_3_annotated_val_raycloud_trees.txt"),
}


def estimate_cbc_relative(
    pts_tree: np.ndarray,
    sem_tree: np.ndarray,
    wood_label: int,
    n_slices: int = 50,
    smooth_window: int = 5,
    wood_frac_threshold: float = 0.15,
) -> float:
    """Estimate CBC using tree-relative Z coordinates."""
    if pts_tree is None or len(pts_tree) == 0:
        return float("nan")

    pts = np.asarray(pts_tree, dtype=np.float64).copy()
    z_min = float(pts[:, 2].min())
    pts[:, 2] -= z_min

    z = pts[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    height = z_max - z_min
    if height <= 0:
        return float("nan")

    edges = np.linspace(z_min, z_max, n_slices + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    frac = np.zeros(n_slices)
    for i in range(n_slices):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        if mask.sum() > 0:
            frac[i] = np.sum(sem_tree[mask] == wood_label) / mask.sum()

    kernel = np.ones(smooth_window) / smooth_window
    frac_smooth = np.convolve(frac, kernel, mode="same")

    cbc_height = z_max
    for i in range(n_slices):
        if frac_smooth[i] < wood_frac_threshold:
            cbc_height = float(centers[i])
            break

    cbc_height = max(cbc_height, z_min + 0.2 * height)
    return float(cbc_height)


def _load_cloud(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ext = Path(path).suffix.lower()
    if ext in {".las", ".laz"}:
        return io.load_laz(path)
    if ext == ".ply":
        return io.load_ply(path)
    raise ValueError(f"Unsupported cloud format: {path}")


def _resolve_column(extras: dict[str, np.ndarray], candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in extras:
            return candidate
    available = ", ".join(sorted(extras.keys()))
    raise KeyError(f"No {label} column found. Tried {list(candidates)}. Available: {available}")


def compare_method_combinations(
    cloud_path: str,
    tree_id_candidates: Sequence[str],
    trunk_leaf_candidates: Sequence[str],
    trunk_label: int = 1,
    wood_density_kg_m3: float = 900.0,
    dbh_methods: Sequence[str] = DEFAULT_DBH_METHODS,
    volume_methods_list: Sequence[str] = DEFAULT_VOLUME_METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all DBH/volume combinations for a single cloud."""
    from eucalipto import dbh_methods as dbh_methods_module
    from eucalipto import volume_methods as volume_methods_module

    pts, extras = _load_cloud(cloud_path)
    tree_id_col = _resolve_column(extras, tree_id_candidates, "tree id")
    trunk_leaf_col = _resolve_column(extras, trunk_leaf_candidates, "trunk/leaf")

    tree_ids = np.asarray(extras[tree_id_col]).astype(np.int32)
    trunk_leaf = np.asarray(extras[trunk_leaf_col]).astype(np.int32)

    dbh_rows: list[dict[str, object]] = []
    combo_rows: list[dict[str, object]] = []

    for tree_id in sorted(np.unique(tree_ids)):
        mask = (tree_ids == tree_id) & (trunk_leaf == trunk_label)
        trunk_pts = pts[mask]
        if trunk_pts.shape[0] < 3:
            continue

        trunk_pts_rel = trunk_pts.copy()
        trunk_pts_rel[:, 2] -= trunk_pts_rel[:, 2].min()
        height_m = float(trunk_pts_rel[:, 2].max())

        dbh_results: dict[str, tuple[float | None, dict[str, object] | None]] = {}
        for dbh_method in dbh_methods:
            try:
                dbh_cm, dbh_info = dbh_methods_module.estimate_dbh(trunk_pts, method=dbh_method)
                status = "ok" if dbh_cm is not None else "none"
            except Exception as exc:  # pragma: no cover - defensive
                dbh_cm = None
                dbh_info = {"error": str(exc)}
                status = f"error: {exc}"

            dbh_results[dbh_method] = (dbh_cm, dbh_info)
            dbh_rows.append(
                {
                    "cloud": Path(cloud_path).stem,
                    "tree_id": int(tree_id),
                    "method": dbh_method,
                    "dbh_cm": dbh_cm,
                    "status": status,
                    "n_values": len(dbh_info.get("per_slice_dbh_cm", [])) if isinstance(dbh_info, dict) else None,
                }
            )

        try:
            cbc_rel = estimate_cbc_relative(trunk_pts_rel, trunk_leaf[mask], trunk_label)
        except Exception:
            cbc_rel = float("nan")

        if np.isnan(cbc_rel):
            cbc_abs = float("nan")
        else:
            cbc_abs = float(cbc_rel + float(trunk_pts[:, 2].min()))

        commercial_mask = trunk_pts_rel[:, 2] <= cbc_rel if not np.isnan(cbc_rel) else np.zeros(len(trunk_pts_rel), dtype=bool)
        commercial_pts = trunk_pts_rel[commercial_mask]

        for dbh_method in dbh_methods:
            dbh_cm, _ = dbh_results[dbh_method]
            for volume_method in volume_methods_list:
                try:
                    if volume_method == "cylinder":
                        if dbh_cm is None or np.isnan(cbc_rel):
                            raise ValueError("dbh_cm or CBC is missing for cylinder volume")
                        total_info = volume_methods_module.estimate_volume(
                            trunk_pts_rel,
                            dbh_cm=dbh_cm,
                            height_m=height_m,
                            method=volume_method,
                            wood_density_kg_m3=wood_density_kg_m3,
                        )
                        commercial_info = volume_methods_module.estimate_volume(
                            commercial_pts,
                            dbh_cm=dbh_cm,
                            height_m=cbc_rel,
                            method=volume_method,
                            wood_density_kg_m3=wood_density_kg_m3,
                        )
                    else:
                        total_info = volume_methods_module.estimate_volume(
                            trunk_pts_rel,
                            method=volume_method,
                            wood_density_kg_m3=wood_density_kg_m3,
                        )
                        commercial_info = volume_methods_module.estimate_volume(
                            commercial_pts,
                            method=volume_method,
                            wood_density_kg_m3=wood_density_kg_m3,
                        )

                    volume_status = "ok"
                except Exception as exc:
                    total_info = {"volume_m3": None, "mass_kg": None}
                    commercial_info = {"volume_m3": None, "mass_kg": None}
                    volume_status = f"error: {exc}"

                combo_rows.append(
                    {
                        "cloud": Path(cloud_path).stem,
                        "tree_id": int(tree_id),
                        "dbh_method": dbh_method,
                        "volume_method": volume_method,
                        "dbh_cm": dbh_cm,
                        "height_m": height_m,
                        "cbc_rel_m": cbc_rel,
                        "cbc_abs_m": cbc_abs,
                        "volume_m3": total_info.get("volume_m3"),
                        "mass_kg": total_info.get("mass_kg"),
                        "commercial_volume_m3": commercial_info.get("volume_m3"),
                        "commercial_mass_kg": commercial_info.get("mass_kg"),
                        "status": volume_status,
                        "n_trunk_pts": int(trunk_pts.shape[0]),
                        "n_commercial_pts": int(commercial_pts.shape[0]),
                    }
                )

    return pd.DataFrame(dbh_rows), pd.DataFrame(combo_rows)


def _segment_volume_and_height(tree_segments: pd.DataFrame, cut_height_rel: float | None = None) -> tuple[float, float, float]:
    """Compute volume and height from a tree structure using tree-relative coordinates."""
    if tree_segments.empty:
        return 0.0, 0.0, 0.0

    seg = tree_segments.copy()
    base_z = float(seg["z"].min())
    seg["z_rel"] = seg["z"] - base_z

    total_volume = 0.0
    total_length = 0.0
    for idx, row in seg.iterrows():
        parent_idx = int(row["parent_id"])
        if parent_idx < 0 or parent_idx not in seg.index:
            continue

        parent = seg.loc[parent_idx]
        p0 = parent[["x", "y", "z_rel"]].to_numpy(dtype=np.float64)
        p1 = row[["x", "y", "z_rel"]].to_numpy(dtype=np.float64)
        length = float(np.linalg.norm(p1 - p0))
        if length <= 0:
            continue

        radius = float(row["radius"])
        total_volume += float(np.pi * length * radius * radius)
        total_length += length

    height_from_z = float(seg["z_rel"].max())

    if cut_height_rel is None or np.isnan(cut_height_rel):
        commercial_volume = total_volume
    else:
        commercial_volume = 0.0
        for idx, row in seg.iterrows():
            parent_idx = int(row["parent_id"])
            if parent_idx < 0 or parent_idx not in seg.index:
                continue

            parent = seg.loc[parent_idx]
            z0 = float(parent["z_rel"])
            z1 = float(row["z_rel"])
            p0 = parent[["x", "y", "z_rel"]].to_numpy(dtype=np.float64)
            p1 = row[["x", "y", "z_rel"]].to_numpy(dtype=np.float64)
            length = float(np.linalg.norm(p1 - p0))
            if length <= 0:
                continue

            radius = float(row["radius"])
            z_low = min(z0, z1)
            z_high = max(z0, z1)

            if cut_height_rel <= z_low:
                continue
            if cut_height_rel >= z_high or z_high == z_low:
                commercial_volume += float(np.pi * length * radius * radius)
                continue

            frac = (cut_height_rel - z_low) / (z_high - z_low)
            frac = float(np.clip(frac, 0.0, 1.0))
            commercial_volume += float(np.pi * (length * frac) * radius * radius)

    height_m = max(height_from_z, total_length)
    if cut_height_rel is not None and not np.isnan(cut_height_rel):
        height_m = max(height_m, float(cut_height_rel))

    return float(total_volume), float(commercial_volume), float(height_m)


def _treefile_path_from_csv(csv_path: Path, treefile_dir: Path) -> Path:
    match = re.search(r"plot(\d+)", csv_path.name)
    if not match:
        raise ValueError(f"Cannot infer plot number from {csv_path.name}")

    plot_num = int(match.group(1))
    try:
        default_treefile = DEFAULT_RAYEXTRACT_TREEFILES[plot_num]
    except KeyError as exc:
        raise ValueError(f"Unsupported RayExtract plot number: {plot_num}") from exc

    if treefile_dir == default_treefile.parent:
        return default_treefile
    return treefile_dir / default_treefile.name


def fix_rayextract_mesh_csv(
    csv_path: Path,
    treefile_path: Path,
    wood_density_kg_m3: float = 600.0,
    output_suffix: str = "_fixed",
) -> Path:
    """Fix RayExtract mesh CSV using tree-relative CBC and tree structure height."""
    df = pd.read_csv(csv_path)
    tree_df = read_rayextract_tree_file(str(treefile_path))
    root_nodes = tree_df[tree_df["parent_id"] == -1].index.tolist()

    if len(root_nodes) != len(df):
        print(
            f"Warning: {csv_path.name} has {len(df)} rows but treefile has {len(root_nodes)} trees. "
            "Will process rows by index order."
        )

    new_rows: list[dict[str, object]] = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        if row_idx >= len(root_nodes):
            break

        root_idx = root_nodes[row_idx]
        tree_segments = _get_tree_segments_from_root(tree_df, root_idx)
        base_z = float(tree_segments["z"].min()) if not tree_segments.empty else float("nan")

        cut_height_abs = float(row.get("cut_height_m", np.nan))
        cut_height_rel = cut_height_abs - base_z if not np.isnan(base_z) and not np.isnan(cut_height_abs) else float("nan")

        total_volume, commercial_volume, structure_height = _segment_volume_and_height(tree_segments, cut_height_rel)

        # Preserve the mesh-derived volume from the CSV, but keep the corrected
        # commercial value bounded by it.
        mesh_volume = float(row.get("volume_m3", np.nan))
        if not np.isnan(mesh_volume) and mesh_volume > 0:
            total_volume = mesh_volume

        if not np.isnan(mesh_volume) and commercial_volume > mesh_volume:
            commercial_volume = mesh_volume

        height_corrected = float(row.get("height_m", np.nan))
        if np.isnan(height_corrected) or height_corrected <= 0:
            height_corrected = structure_height

        if not np.isnan(cut_height_rel):
            height_corrected = max(height_corrected, cut_height_rel)

        mass_kg = float(total_volume * wood_density_kg_m3) if not np.isnan(total_volume) else np.nan
        commercial_mass_kg = float(commercial_volume * wood_density_kg_m3) if not np.isnan(commercial_volume) else np.nan

        new_row = row.to_dict()
        new_row.update(
            {
                "height_m_raw": row.get("height_m", np.nan),
                "cut_height_m_raw": row.get("cut_height_m", np.nan),
                "volume_m3_raw": row.get("volume_m3", np.nan),
                "commercial_volume_m3_raw": row.get("commercial_volume_m3", np.nan),
                "mass_kg_raw": row.get("mass_kg", np.nan),
                "commercial_mass_kg_raw": row.get("commercial_mass_kg", np.nan),
                "base_z_m": base_z,
                "cut_height_rel_m": cut_height_rel,
                "height_m": height_corrected,
                "volume_m3": total_volume,
                "mass_kg": mass_kg,
                "commercial_volume_m3": commercial_volume,
                "commercial_mass_kg": commercial_mass_kg,
                "status": "mesh_volume_fixed" if float(row.get("height_m", 0.0) or 0.0) <= 0 else row.get("status", "success"),
                "n_segments": int(len(tree_segments)),
            }
        )
        new_rows.append(new_row)

    fixed_df = pd.DataFrame(new_rows)
    output_path = csv_path.with_name(csv_path.stem + output_suffix + csv_path.suffix)
    fixed_df.to_csv(output_path, index=False)
    return output_path


def run_compare_mode(args: argparse.Namespace) -> int:
    cloud_paths = {
        "FF3D": args.ff3d,
        "TreeISO": args.treeiso,
    }

    column_candidates = {
        "FF3D": (["instance_pred", "tree_id"], ["semantic_pred", "trunk_leaf_label"]),
        "TreeISO": (["tree_id", "final_segs"], ["trunk_leaf_label", "semantic_pred"]),
    }

    dbh_frames: list[pd.DataFrame] = []
    combo_frames: list[pd.DataFrame] = []

    for name, path in cloud_paths.items():
        if not path:
            continue
        if not Path(path).exists():
            print(f"Skipping {name}: missing file {path}")
            continue

        tree_candidates, trunk_candidates = column_candidates[name]
        dbh_df, combo_df = compare_method_combinations(
            path,
            tree_id_candidates=tree_candidates,
            trunk_leaf_candidates=trunk_candidates,
            trunk_label=args.trunk_label,
            wood_density_kg_m3=args.wood_density,
            dbh_methods=args.dbh_methods,
            volume_methods_list=args.volume_methods,
        )
        dbh_frames.append(dbh_df)
        combo_frames.append(combo_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dbh_frames:
        pd.concat(dbh_frames, ignore_index=True).to_csv(output_dir / "dbh_method_grid.csv", index=False)
    if combo_frames:
        pd.concat(combo_frames, ignore_index=True).to_csv(output_dir / "volume_method_grid.csv", index=False)

    print(f"Saved outputs to {output_dir}")
    return 0


def run_fix_mesh_mode(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    treefile_dir = Path(args.treefile_dir)
    csv_files = sorted(input_dir.glob("rayextract_plot*_with_commercial_mesh.csv"))
    if not csv_files:
        csv_files = sorted(DEFAULT_RAYEXTRACT_CSVS)

    outputs = []
    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"Skipping missing CSV: {csv_path}")
            continue

        treefile_path = _treefile_path_from_csv(csv_path, treefile_dir)
        if not treefile_path.exists():
            print(f"Skipping {csv_path.name}: missing treefile {treefile_path}")
            continue

        out_path = fix_rayextract_mesh_csv(
            csv_path,
            treefile_path,
            wood_density_kg_m3=args.wood_density,
            output_suffix=args.output_suffix,
        )
        outputs.append(out_path)
        print(f"Saved: {out_path.name}")

    if not outputs:
        print("No RayExtract mesh CSVs were processed.")
        return 1

    print(f"Processed {len(outputs)} RayExtract mesh CSV files.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commercial volume and CBC utilities")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    compare = subparsers.add_parser("compare-methods", help="Evaluate DBH/volume method combinations")
    compare.add_argument("--ff3d", help="FF3D cloud path (.ply)")
    compare.add_argument("--treeiso", help="TreeISO cloud path (.laz)")
    compare.add_argument("--dbh-methods", nargs="+", default=DEFAULT_DBH_METHODS)
    compare.add_argument("--volume-methods", nargs="+", default=DEFAULT_VOLUME_METHODS)
    compare.add_argument("--trunk-label", type=int, default=1)
    compare.add_argument("--wood-density", type=float, default=900.0)
    compare.add_argument("--output-dir", default="/home/matheuspimenta/Jobs/Eucalipto/rel01/testes")
    compare.set_defaults(func=run_compare_mode)

    fix_mesh = subparsers.add_parser("fix-rayextract-mesh", help="Fix RayExtract mesh CSVs using tree-relative CBC")
    fix_mesh.add_argument("--input-dir", default="/home/matheuspimenta/Jobs/Eucalipto/rel01")
    fix_mesh.add_argument("--treefile-dir", default="/home/matheuspimenta/Jobs/Eucalipto/drive/outputs/ray")
    fix_mesh.add_argument("--wood-density", type=float, default=600.0)
    fix_mesh.add_argument("--output-suffix", default="_fixed")
    fix_mesh.set_defaults(func=run_fix_mesh_mode)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())