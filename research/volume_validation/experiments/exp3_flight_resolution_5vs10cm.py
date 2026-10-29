"""
Experiment 3 Runner: Point Cloud Resolution Comparison (5cm vs 10cm on Talhão 62).
Evaluates biometric estimation accuracy across point cloud densities on the same physical plots.
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


def process_resolution_dataset(res_name: str, cloud_dir: str, cfg: dict, df_field_all: pd.DataFrame, df_parc: pd.DataFrame):
    plots = cfg["dataset"]["plots"]
    pattern = cfg["dataset"]["cloud_pattern"]
    qc_params = cfg["qc"]
    enh_params = cfg["enhancement"]
    mat_params = cfg["matching"]

    tree_records = []
    plot_results = []

    for pid in sorted(plots):
        cpath = os.path.join(cloud_dir, pattern.format(plot_id=pid))
        if not os.path.exists(cpath):
            continue

        pts, inst, sem = load_classified_las(cpath)
        inst_enh, n_reassigned = enhance_stem_points(pts, inst, sem, **enh_params)
        qc_trees = extract_qc_trees(pts, inst_enh, sem, plot_id=pid, **qc_params)

        p_df = df_field_all[df_field_all["PAR_CODIGO"] == pid].reset_index(drop=True)
        matches = align_and_match_plot(qc_trees, p_df, **mat_params)

        for m in matches:
            if not m["has_gt_volume"]:
                continue
            bio = compute_tree_biometrics_grid(m["tree_data"], cfg["grid_search"])
            rec = {
                "resolution": res_name,
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
            tree_records.append(rec)

    df_trees = pd.DataFrame(tree_records)
    cfg_cols = [c for c in df_trees.columns if c.startswith(("voxel_", "axis_", "cyl_"))]

    best_cfg, rmse = find_best_global_config(df_trees, cfg_cols)
    metrics_global = evaluate_config_on_subset(df_trees, best_cfg)

    # Human in the loop
    human_vols = []
    for _, row in df_trees.iterrows():
        gt = row["gt_volume"]
        best_diff = float("inf")
        best_v = np.nan
        for c in cfg_cols:
            v = row[c]
            if not np.isnan(v) and abs(v - gt) < best_diff:
                best_diff = abs(v - gt)
                best_v = v
        human_vols.append(best_v)

    gt_arr = df_trees["gt_volume"].values
    h_arr = np.array(human_vols)
    rmse_h = float(np.sqrt(np.mean((h_arr - gt_arr) ** 2)))
    bias_h = float((np.mean(h_arr) - np.mean(gt_arr)) / np.mean(gt_arr) * 100.0)

    # Plot level m3/ha
    for pid in sorted(plots):
        sub_t = df_trees[df_trees["plot_id"] == pid]
        if len(sub_t) == 0:
            continue
        p_row = df_parc[df_parc["Parcela"] == pid]
        gt_ha = float(p_row["VTCC (m³/ha)"].iloc[0]) if len(p_row) > 0 else np.nan
        area_m2 = float(p_row["Área (m²)"].iloc[0]) if len(p_row) > 0 else 400.0
        exp_factor = 10000.0 / area_m2
        n_living = len(df_field_all[(df_field_all["PAR_CODIGO"] == pid) & df_field_all["has_gt_volume"]])

        m_v_g = float(sub_t[best_cfg].mean())
        h_sub = [row[cfg_cols].iloc[(row[cfg_cols] - row["gt_volume"]).abs().argmin()] for _, row in sub_t.iterrows()]
        m_v_h = float(np.mean(h_sub))

        pred_g_ha = m_v_g * n_living * exp_factor
        pred_h_ha = m_v_h * n_living * exp_factor

        plot_results.append({
            "resolution": res_name,
            "plot_id": pid,
            "n_matched": len(sub_t),
            "gt_m3_ha": gt_ha,
            "pred_global_m3_ha": pred_g_ha,
            "err_global_ha_pct": (pred_g_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan,
            "pred_human_m3_ha": pred_h_ha,
            "err_human_ha_pct": (pred_h_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan,
        })

    return {
        "res_name": res_name,
        "n_trees": len(df_trees),
        "best_cfg": best_cfg,
        "tree_rmse_global": metrics_global["rmse"],
        "tree_bias_global_pct": metrics_global["bias_pct"],
        "tree_rmse_human": rmse_h,
        "tree_bias_human_pct": bias_h,
        "df_trees": df_trees,
        "df_plots": pd.DataFrame(plot_results)
    }


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 3: POINT CLOUD RESOLUTION COMPARISON (5CM vs 10CM ON T62)")
    print("=" * 80)

    config_path = os.path.join(RESEARCH_DIR, "configs/config_flight_res.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    export_dir = cfg["validation"]["export_dir"]
    if not os.path.isabs(export_dir):
        export_dir = os.path.join(RESEARCH_DIR, export_dir)
    os.makedirs(export_dir, exist_ok=True)

    df_field_all, df_parc, df_tal = load_inventory_data(
        cfg["dataset"]["excel_path"], cfg["dataset"]["talhao_id"]
    )

    print("\n[1/3] Processing 5cm Point Clouds on Talhão 62...")
    res_5cm = process_resolution_dataset(
        "5cm", cfg["dataset"]["clouds_5cm_dir"], cfg, df_field_all, df_parc
    )
    print(f"  5cm: {res_5cm['n_trees']} trees | Best: {res_5cm['best_cfg']} | RMSE: {res_5cm['tree_rmse_global']:.4f} m³")

    print("\n[2/3] Processing 10cm Point Clouds on Talhão 62...")
    res_10cm = process_resolution_dataset(
        "10cm", cfg["dataset"]["clouds_10cm_dir"], cfg, df_field_all, df_parc
    )
    print(f"  10cm: {res_10cm['n_trees']} trees | Best: {res_10cm['best_cfg']} | RMSE: {res_10cm['tree_rmse_global']:.4f} m³")

    # Combine datasets
    df_trees_all = pd.concat([res_5cm["df_trees"], res_10cm["df_trees"]], ignore_index=True)
    df_plots_all = pd.concat([res_5cm["df_plots"], res_10cm["df_plots"]], ignore_index=True)

    trees_csv = os.path.join(export_dir, "trees_resolution_comparison.csv")
    plots_csv = os.path.join(export_dir, "plots_resolution_comparison.csv")
    df_trees_all.to_csv(trees_csv, index=False)
    df_plots_all.to_csv(plots_csv, index=False)

    # 3. Build Resolution Comparison Summary Table
    print("\n[3/3] Generating Resolution Comparison Summary Table...")
    summary_table = pd.DataFrame([
        {
            "Resolution": "5 cm",
            "Approx_Point_Density": "~400 pts/m²",
            "Matched_Trees": res_5cm["n_trees"],
            "Optimal_Model": res_5cm["best_cfg"],
            "Tree_Volume_RMSE_m3": round(res_5cm["tree_rmse_global"], 4),
            "Tree_Volume_Bias_pct": round(res_5cm["tree_bias_global_pct"], 1),
            "Mean_Plot_Error_Global_pct": round(float(res_5cm["df_plots"]["err_global_ha_pct"].abs().mean()), 1),
            "Mean_Plot_Error_Human_pct": round(float(res_5cm["df_plots"]["err_human_ha_pct"].abs().mean()), 1),
        },
        {
            "Resolution": "10 cm",
            "Approx_Point_Density": "~150 pts/m²",
            "Matched_Trees": res_10cm["n_trees"],
            "Optimal_Model": res_10cm["best_cfg"],
            "Tree_Volume_RMSE_m3": round(res_10cm["tree_rmse_global"], 4),
            "Tree_Volume_Bias_pct": round(res_10cm["tree_bias_global_pct"], 1),
            "Mean_Plot_Error_Global_pct": round(float(res_10cm["df_plots"]["err_global_ha_pct"].abs().mean()), 1),
            "Mean_Plot_Error_Human_pct": round(float(res_10cm["df_plots"]["err_human_ha_pct"].abs().mean()), 1),
        }
    ])

    summary_csv = os.path.join(export_dir, "resolution_comparison_summary.csv")
    summary_table.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("RESOLUTION COMPARISON SUMMARY (TALHÃO 62):")
    print("=" * 80)
    print(summary_table.to_string(index=False))

    exp3_meta = {
        "experiment": "Exp3_Resolution_Comparison_5vs10cm",
        "stand": "Talhao_62",
        "resolution_summary": summary_table.to_dict(orient="records"),
        "plots_summary": df_plots_all.to_dict(orient="records")
    }
    meta_json = os.path.join(export_dir, "exp3_summary.json")
    with open(meta_json, "w") as f:
        json.dump(exp3_meta, f, indent=2)

    print(f"\n  Exported Exp3 summary to: {meta_json}")
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
