"""
Experiment 4: Cross-Stand Transferability Benchmark at Identical 10cm Resolution.
Compares cross-resolution transfer (10cm <-> 5cm) against matched-resolution transfer
(T37 10cm <-> T62 10cm) under both Tree-RMSE calibration and Volume-Bias calibration.
"""

import json
import os
import sys
import yaml
import numpy as np
import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(RESEARCH_DIR))
RESULTS_DIR = os.path.join(RESEARCH_DIR, "results")
EXP4_DIR = os.path.join(RESULTS_DIR, "exp4_cross_stand_10cm")
os.makedirs(EXP4_DIR, exist_ok=True)

sys.path.append(BASE_DIR)
sys.path.append(RESEARCH_DIR)

from src.cross_validation import (
    find_best_global_config,
    find_best_unbiased_config,
    evaluate_config_on_subset,
)
from src.data_loader import load_inventory_data


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 4: CROSS-STAND TRANSFER AT IDENTICAL 10CM RESOLUTION")
    print("=" * 80)

    # 1. Load Official Inventory Metadata (No Hardcoded Stand Constants)
    cfg_t37_path = os.path.join(RESEARCH_DIR, "configs/config_t37_cv.yaml")
    with open(cfg_t37_path, "r") as f:
        cfg_t37 = yaml.safe_load(f)
    excel_path = cfg_t37["dataset"]["excel_path"]

    _, _, df_tal_37 = load_inventory_data(excel_path, 37)
    _, _, df_tal_62 = load_inventory_data(excel_path, "062-063")

    t37_info = df_tal_37.iloc[0]
    t62_info = df_tal_62.iloc[0]

    t37_gt_ha = float(t37_info["VTCC (m³/ha)"])
    trees_ha_t37 = float(t37_info["Árvores (n/ha)"])

    t62_gt_ha = float(t62_info["VTCC (m³/ha)"])
    trees_ha_t62 = float(t62_info["Árvores (n/ha)"])

    print(f"Official Inventory T37: GT = {t37_gt_ha:.2f} m³/ha | Density = {trees_ha_t37:.2f} trees/ha")
    print(f"Official Inventory T62: GT = {t62_gt_ha:.2f} m³/ha | Density = {trees_ha_t62:.2f} trees/ha")

    # 2. Load Matched Tree Biometrics Datasets
    t37_csv = os.path.join(RESULTS_DIR, "exp1_t37/df_trees_t37_matched.csv")
    t62_csv = os.path.join(RESULTS_DIR, "exp3_resolution_5vs10cm/trees_resolution_comparison.csv")

    df_t37 = pd.read_csv(t37_csv)
    df_res = pd.read_csv(t62_csv)
    df_t62_10 = df_res[df_res["resolution"].str.contains("10cm")].reset_index(drop=True)
    df_t62_5 = df_res[df_res["resolution"].str.contains("5cm")].reset_index(drop=True)

    print(f"\nLoaded Talhão 37 (10cm): {len(df_t37)} trees across 11 plots.")
    print(f"Loaded Talhão 62 (10cm): {len(df_t62_10)} trees across 4 plots.")
    print(f"Loaded Talhão 62 (5cm):  {len(df_t62_5)} trees across 4 plots.")

    cfg_cols = [c for c in df_t37.columns if c.startswith(("voxel_", "axis_", "cyl_"))]

    # -------------------------------------------------------------------------
    # Side-by-Side Calibration on T37 (10cm), T62 (10cm), and T62 (5cm)
    # -------------------------------------------------------------------------
    rmse_cfg_t37, rmse_val_t37 = find_best_global_config(df_t37, cfg_cols)
    bias_cfg_t37, bias_val_t37 = find_best_unbiased_config(df_t37, cfg_cols)

    rmse_cfg_t62_10, rmse_val_t62_10 = find_best_global_config(df_t62_10, cfg_cols)
    bias_cfg_t62_10, bias_val_t62_10 = find_best_unbiased_config(df_t62_10, cfg_cols)

    rmse_cfg_t62_5, rmse_val_t62_5 = find_best_global_config(df_t62_5, cfg_cols)
    bias_cfg_t62_5, bias_val_t62_5 = find_best_unbiased_config(df_t62_5, cfg_cols)

    print("\n--- CALIBRATION SELECTION BY STAND & CRITERION ---")
    print(f"  T37 (10cm): Tree-RMSE Cal = {rmse_cfg_t37} (RMSE={rmse_val_t37:.4f} m³) | Plot-Bias Cal = {bias_cfg_t37} (|Bias|={bias_val_t37:.1f}%)")
    print(f"  T62 (10cm): Tree-RMSE Cal = {rmse_cfg_t62_10} (RMSE={rmse_val_t62_10:.4f} m³) | Plot-Bias Cal = {bias_cfg_t62_10} (|Bias|={bias_val_t62_10:.1f}%)")
    print(f"  T62 (5cm) : Tree-RMSE Cal = {rmse_cfg_t62_5} (RMSE={rmse_val_t62_5:.4f} m³) | Plot-Bias Cal = {bias_cfg_t62_5} (|Bias|={bias_val_t62_5:.1f}%)")

    def eval_transfer(df_target, cfg_rmse, cfg_bias, trees_ha_target, gt_ha_target, exp_type, direction, target_res):
        # Tree-RMSE calibrated model
        m_vol_rmse = float(df_target[cfg_rmse].mean())
        pred_ha_rmse = m_vol_rmse * trees_ha_target
        err_rmse_pct = (pred_ha_rmse - gt_ha_target) / gt_ha_target * 100.0
        ev_rmse = evaluate_config_on_subset(df_target, cfg_rmse)

        # Plot-Bias calibrated model
        m_vol_bias = float(df_target[cfg_bias].mean())
        pred_ha_bias = m_vol_bias * trees_ha_target
        err_bias_pct = (pred_ha_bias - gt_ha_target) / gt_ha_target * 100.0
        ev_bias = evaluate_config_on_subset(df_target, cfg_bias)

        return {
            "Experiment": exp_type,
            "Transfer Direction": direction,
            "Target Resolution": target_res,
            "Calibrated Model (Tree-RMSE)": cfg_rmse,
            "Stand Pred Tree-RMSE (m³/ha)": round(pred_ha_rmse, 1),
            "Stand Error Tree-RMSE (%)": round(err_rmse_pct, 1),
            "Tree Volume RMSE (m³)": round(ev_rmse["rmse"], 4),
            "Calibrated Model (Plot-Bias)": cfg_bias,
            "Stand Pred Plot-Bias (m³/ha)": round(pred_ha_bias, 1),
            "Stand Error Plot-Bias (%)": round(err_bias_pct, 1),
            "Tree Volume RMSE Plot-Bias (m³)": round(ev_bias["rmse"], 4),
            "Ground Truth (m³/ha)": gt_ha_target,
            # Keep backward-compatible keys for downstream plots
            "Calibrated Model": cfg_rmse,
            "Stand Pred (m³/ha)": round(pred_ha_rmse, 1),
            "Stand Relative Error (%)": round(err_rmse_pct, 1),
        }

    summary_rows = [
        eval_transfer(
            df_t62_5, rmse_cfg_t37, bias_cfg_t37, trees_ha_t62, t62_gt_ha,
            "Mismatched Resolution (10cm -> 5cm)", "T37 (10cm) -> T62 (5cm)", "5cm"
        ),
        eval_transfer(
            df_t62_10, rmse_cfg_t37, bias_cfg_t37, trees_ha_t62, t62_gt_ha,
            "Matched 10cm Resolution (10cm -> 10cm)", "T37 (10cm) -> T62 (10cm)", "10cm"
        ),
        eval_transfer(
            df_t37, rmse_cfg_t62_5, bias_cfg_t62_5, trees_ha_t37, t37_gt_ha,
            "Mismatched Resolution (5cm -> 10cm)", "T62 (5cm) -> T37 (10cm)", "10cm"
        ),
        eval_transfer(
            df_t37, rmse_cfg_t62_10, bias_cfg_t62_10, trees_ha_t37, t37_gt_ha,
            "Matched 10cm Resolution (10cm -> 10cm)", "T62 (10cm) -> T37 (10cm)", "10cm"
        ),
    ]

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(EXP4_DIR, "cross_stand_resolution_matching_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    print("\n" + "=" * 95)
    print("CROSS-STAND TRANSFER SUMMARY (SIDE-BY-SIDE: TREE-RMSE vs PLOT-BIAS CALIBRATION):")
    print("=" * 95)
    cols_print = [
        "Transfer Direction",
        "Ground Truth (m³/ha)",
        "Calibrated Model (Tree-RMSE)",
        "Stand Pred Tree-RMSE (m³/ha)",
        "Stand Error Tree-RMSE (%)",
        "Calibrated Model (Plot-Bias)",
        "Stand Pred Plot-Bias (m³/ha)",
        "Stand Error Plot-Bias (%)",
    ]
    print(df_summary[cols_print].to_string(index=False))
    print(f"\nSaved summary table to: {csv_path}")

    meta = {
        "experiment": "Exp4_Cross_Stand_Matched_Resolution",
        "description": "Validation of stand-to-stand transfer when matching 10cm LiDAR resolution",
        "official_trees_ha_t37": trees_ha_t37,
        "official_trees_ha_t62": trees_ha_t62,
        "best_model_t37_10cm_rmse": rmse_cfg_t37,
        "best_model_t37_10cm_bias": bias_cfg_t37,
        "best_model_t62_10cm_rmse": rmse_cfg_t62_10,
        "best_model_t62_10cm_bias": bias_cfg_t62_10,
        "summary": summary_rows,
    }
    json_path = os.path.join(EXP4_DIR, "exp4_summary.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved summary JSON to: {json_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 4 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
