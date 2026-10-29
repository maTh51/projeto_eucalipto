"""
Experiment 2 Runner: Intra-Clone Stand Transferability (Talhão 62 -> Talhão 63 at 5cm).
Calibrates on T62 ground plots and evaluates extrapolation on unvisited adjacent clonal block T63.
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd

# Path routing
RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(RESEARCH_DIR))
sys.path.append(BASE_DIR)
sys.path.append(RESEARCH_DIR)

from src.data_loader import load_inventory_data, load_classified_las
from src.stem_enhancement import enhance_stem_points, extract_qc_trees
from src.spatial_matcher import align_and_match_plot
from src.biometrics import compute_tree_biometrics_grid
from src.cross_validation import find_best_global_config, evaluate_config_on_subset


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 2: INTRA-CLONE TRANSFERABILITY (T62 -> T63 AT 5CM)")
    print("=" * 80)

    config_path = os.path.join(RESEARCH_DIR, "configs/config_t62_t63.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    export_dir = cfg["validation"]["export_dir"]
    if not os.path.isabs(export_dir):
        export_dir = os.path.join(RESEARCH_DIR, export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # 1. Load Ground Truth for Talhão 62
    print("\n[1/5] Loading field inventory for Talhão 62 calibration plots...")
    df_field_all, df_parc, df_tal = load_inventory_data(
        cfg["dataset"]["excel_path"], cfg["dataset"]["talhao_t62_id"]
    )
    print(f"  Field trees in 062-063: {len(df_field_all)} | Calibration plots: {len(df_parc)}")

    # 2. Extract, Enhance and Match Trees on T62 Calibration Plots
    print("\n[2/5] Processing T62 point clouds and extracting calibration trees...")
    plots = cfg["dataset"]["t62_plots"]
    cloud_dir = cfg["dataset"]["t62_cloud_dir"]
    pattern = cfg["dataset"]["t62_cloud_pattern"]

    tree_records_t62 = []
    qc_params = cfg["qc"]
    enh_params = cfg["enhancement"]
    mat_params = cfg["matching"]

    for pid in sorted(plots):
        cpath = os.path.join(cloud_dir, pattern.format(plot_id=pid))
        if not os.path.exists(cpath):
            print(f"  [WARN] Missing file: {cpath}")
            continue

        pts, inst, sem = load_classified_las(cpath)
        inst_enh, n_reassigned = enhance_stem_points(pts, inst, sem, **enh_params)
        qc_trees = extract_qc_trees(pts, inst_enh, sem, plot_id=pid, **qc_params)

        p_df = df_field_all[df_field_all["PAR_CODIGO"] == pid].reset_index(drop=True)
        matches = align_and_match_plot(qc_trees, p_df, **mat_params)
        n_gt = sum(1 for m in matches if m["has_gt_volume"])
        print(f"  T62 Plot {pid}: {len(qc_trees)} QC trees (reassigned {n_reassigned} pts) -> {len(matches)} matched ({n_gt} with GT vol)")

        for m in matches:
            if not m["has_gt_volume"]:
                continue
            bio = compute_tree_biometrics_grid(m["tree_data"], cfg["grid_search"])
            rec = {
                "plot_id": pid,
                "field_id": m["field_id"],
                "tid": m["tid"],
                "global_tid": m["global_tid"],
                "gt_volume": float(m["gt_volume"]),
                "gt_dbh": float(m["gt_dbh"]) if not np.isnan(m["gt_dbh"]) else np.nan,
                "gt_height": float(m["gt_height"]) if not np.isnan(m["gt_height"]) else np.nan,
                "lidar_height": bio["height_m"],
                "lidar_dbh": bio["dbh_cm"],
            }
            rec.update(bio["configs"])
            tree_records_t62.append(rec)

    df_trees_t62 = pd.DataFrame(tree_records_t62)
    print(f"  -> Total matched trees in T62 calibration set: {len(df_trees_t62)}")
    t62_csv = os.path.join(export_dir, "df_trees_t62_calibration.csv")
    df_trees_t62.to_csv(t62_csv, index=False)

    # 3. Model Calibration (Identify Optimal Global Config on T62)
    print("\n[3/5] Calibrating volumetric model on T62 reference plots...")
    cfg_cols = [c for c in df_trees_t62.columns if c.startswith(("voxel_", "axis_", "cyl_"))]
    best_cfg_t62, calib_rmse = find_best_global_config(df_trees_t62, cfg_cols)
    calib_metrics = evaluate_config_on_subset(df_trees_t62, best_cfg_t62)

    # Also compute Human-in-the-Loop on T62
    human_vols_t62 = []
    for _, row in df_trees_t62.iterrows():
        gt = row["gt_volume"]
        best_diff = float("inf")
        best_v = np.nan
        for c in cfg_cols:
            v = row[c]
            if not np.isnan(v) and abs(v - gt) < best_diff:
                best_diff = abs(v - gt)
                best_v = v
        human_vols_t62.append(best_v)

    gt_t62_arr = df_trees_t62["gt_volume"].values
    h_t62_arr = np.array(human_vols_t62)
    human_rmse_t62 = float(np.sqrt(np.mean((h_t62_arr - gt_t62_arr) ** 2)))
    human_bias_t62 = float((np.mean(h_t62_arr) - np.mean(gt_t62_arr)) / np.mean(gt_t62_arr) * 100.0)

    print(f"  Best Global Config: {best_cfg_t62}")
    print(f"  Mode 1 (Global): RMSE = {calib_metrics['rmse']:.4f} m³ | Bias = {calib_metrics['bias_pct']:+.1f}%")
    print(f"  Mode 2 (Human) : RMSE = {human_rmse_t62:.4f} m³ | Bias = {human_bias_t62:+.1f}%")

    # 4. Process Target Cloud: Talhão 63 (Unvisited Clonal Block)
    print("\n[4/5] Applying calibrated models to unvisited block Talhão 63 (5cm)...")
    t63_cloud_path = cfg["dataset"]["t63_cloud_path"]
    pts_t63, inst_t63, sem_t63 = load_classified_las(t63_cloud_path)

    # Calculate cloud sample area
    dx = pts_t63[:, 0].max() - pts_t63[:, 0].min()
    dy = pts_t63[:, 1].max() - pts_t63[:, 1].min()
    area_t63_sample_m2 = dx * dy
    area_t63_sample_ha = area_t63_sample_m2 / 10000.0

    inst_t63_enh, n_reassigned_t63 = enhance_stem_points(pts_t63, inst_t63, sem_t63, **enh_params)
    qc_trees_t63 = extract_qc_trees(pts_t63, inst_t63_enh, sem_t63, plot_id=63, **qc_params)
    print(f"  T63 Sample Area: {dx:.1f}m x {dy:.1f}m ({area_t63_sample_m2:.1f} m² = {area_t63_sample_ha:.4f} ha)")
    print(f"  T63 QC Trees extracted: {len(qc_trees_t63)} trees (reassigned {n_reassigned_t63} points)")

    t63_tree_records = []
    for t in qc_trees_t63:
        bio = compute_tree_biometrics_grid(t, cfg["grid_search"])
        rec = {
            "stand": "Talhao_63",
            "tid": t["tid"],
            "height": bio["height_m"],
            "dbh": bio["dbh_cm"],
            "vol_global_calibrated": bio["configs"].get(best_cfg_t62, np.nan)
        }
        t63_tree_records.append(rec)

    df_trees_t63 = pd.DataFrame(t63_tree_records)
    t63_trees_csv = os.path.join(export_dir, "df_trees_t63_predictions.csv")
    df_trees_t63.to_csv(t63_trees_csv, index=False)

    # 5. Stand Extrapolation and Official Target Comparison
    print("\n[5/5] Extrapolating to full stand areas and comparing with official targets...")
    t_info = df_tal.iloc[0]
    official_gt_m3_ha = float(t_info["VTCC (m³/ha)"]) # 201.29 m3/ha
    official_trees_ha = float(t_info["Árvores (n/ha)"]) # 650 trees/ha

    t62_area_ha = cfg["dataset"]["t62_area_ha"] # 25.96 ha
    t63_area_ha = cfg["dataset"]["t63_area_ha"] # 15.04 ha
    total_area_ha = cfg["dataset"]["total_area_ha"] # 41.00 ha

    # Approach A: Density expansion using mean tree volume
    mean_tree_v_t62_global = float(df_trees_t62[best_cfg_t62].mean())
    mean_tree_v_t62_human = float(np.mean(human_vols_t62))
    mean_tree_v_t63_pred = float(df_trees_t63["vol_global_calibrated"].dropna().mean())

    extrap_t62_global_ha = mean_tree_v_t62_global * official_trees_ha
    extrap_t62_human_ha  = mean_tree_v_t62_human * official_trees_ha
    extrap_t63_transferred_ha = mean_tree_v_t63_pred * official_trees_ha

    # Total timber m3
    tot_gt_t62 = official_gt_m3_ha * t62_area_ha
    tot_gt_t63 = official_gt_m3_ha * t63_area_ha
    tot_gt_all = float(t_info["VTCC (m³)"]) # 8252.95 m3

    tot_pred_t62_global = extrap_t62_global_ha * t62_area_ha
    tot_pred_t62_human  = extrap_t62_human_ha * t62_area_ha
    tot_pred_t63_transferred = extrap_t63_transferred_ha * t63_area_ha

    extrap_results = [
        {
            "stand": "Talhão 62 (Calibration Base)",
            "area_ha": t62_area_ha,
            "mode": "Mode 1: Global Best",
            "gt_m3_ha": official_gt_m3_ha,
            "pred_m3_ha": extrap_t62_global_ha,
            "err_pct": (extrap_t62_global_ha - official_gt_m3_ha) / official_gt_m3_ha * 100.0,
            "gt_total_m3": tot_gt_t62,
            "pred_total_m3": tot_pred_t62_global
        },
        {
            "stand": "Talhão 62 (Calibration Base)",
            "area_ha": t62_area_ha,
            "mode": "Mode 2: Human-in-the-Loop",
            "gt_m3_ha": official_gt_m3_ha,
            "pred_m3_ha": extrap_t62_human_ha,
            "err_pct": (extrap_t62_human_ha - official_gt_m3_ha) / official_gt_m3_ha * 100.0,
            "gt_total_m3": tot_gt_t62,
            "pred_total_m3": tot_pred_t62_human
        },
        {
            "stand": "Talhão 63 (Transferred Blind Block)",
            "area_ha": t63_area_ha,
            "mode": f"Transferred Model ({best_cfg_t62})",
            "gt_m3_ha": official_gt_m3_ha,
            "pred_m3_ha": extrap_t63_transferred_ha,
            "err_pct": (extrap_t63_transferred_ha - official_gt_m3_ha) / official_gt_m3_ha * 100.0,
            "gt_total_m3": tot_gt_t63,
            "pred_total_m3": tot_pred_t63_transferred
        }
    ]

    df_extrap_out = pd.DataFrame(extrap_results)
    extrap_csv = os.path.join(export_dir, "extrapolation_summary_t62_t63.csv")
    df_extrap_out.to_csv(extrap_csv, index=False)

    print("\n" + "=" * 70)
    print("STAND TRANSFERABILITY SUMMARY (5cm High Resolution Flight):")
    print("=" * 70)
    for _, r in df_extrap_out.iterrows():
        print(f"  {r['stand']} [{r['mode']}]:")
        print(f"    GT: {r['gt_m3_ha']:.1f} m³/ha ({r['gt_total_m3']:.1f} m³) | Pred: {r['pred_m3_ha']:.1f} m³/ha ({r['pred_total_m3']:.1f} m³) | Error: {r['err_pct']:+.1f}%")

    # Export master JSON
    exp2_meta = {
        "experiment": "Exp2_T62_to_T63_Transferability",
        "resolution": "5cm",
        "best_calibrated_config": best_cfg_t62,
        "calibration_trees_count": len(df_trees_t62),
        "calibration_rmse": calib_metrics["rmse"],
        "t63_qc_trees_count": len(qc_trees_t63),
        "mean_tree_volume_t62_m3": mean_tree_v_t62_global,
        "mean_tree_volume_t63_m3": mean_tree_v_t63_pred,
        "extrapolation_results": df_extrap_out.to_dict(orient="records")
    }
    meta_json = os.path.join(export_dir, "exp2_summary.json")
    with open(meta_json, "w") as f:
        json.dump(exp2_meta, f, indent=2)

    print(f"\n  Exported Exp2 master summary to: {meta_json}")
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
