"""
Experiment 1 Runner: Talhão 37 Cross-Validation & Sampling Saturation Curve.
Zero data leakage evaluation over 11 field plots in sparse 10cm flight.
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd

# Path routing (relative to repository structure so any team member can run it)
RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(RESEARCH_DIR))
sys.path.append(BASE_DIR)
sys.path.append(RESEARCH_DIR)

from src.data_loader import load_inventory_data, load_classified_las
from src.stem_enhancement import enhance_stem_points, extract_qc_trees
from src.spatial_matcher import align_and_match_plot
from src.biometrics import compute_tree_biometrics_grid
from src.cross_validation import run_leave_one_plot_out, run_sampling_saturation_curve


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 1: TALHÃO 37 CROSS-VALIDATION & SAMPLING SATURATION")
    print("=" * 80)

    config_path = os.path.join(RESEARCH_DIR, "configs/config_t37_cv.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    export_dir = cfg["validation"]["export_dir"]
    if not os.path.isabs(export_dir):
        export_dir = os.path.join(RESEARCH_DIR, export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # 1. Load Field Inventory
    print("\n[1/5] Loading field inventory for Talhão 37...")
    df_field_all, df_parc, df_tal = load_inventory_data(
        cfg["dataset"]["excel_path"], cfg["dataset"]["talhao_id"]
    )
    print(f"  Field trees: {len(df_field_all)} | Target plots: {len(df_parc)}")

    # 2. Extract, Enhance and Match Trees for each plot
    print("\n[2/5] Processing point clouds, enhancing stems and performing spatial matching...")
    plots = cfg["dataset"]["plots"]
    cloud_dir = cfg["dataset"]["cloud_dir"]
    pattern = cfg["dataset"]["cloud_pattern"]

    tree_records = []
    qc_params = cfg["qc"]
    enh_params = cfg["enhancement"]
    mat_params = cfg["matching"]

    for pid in sorted(plots):
        cpath = os.path.join(cloud_dir, pattern.format(plot_id=pid))
        if not os.path.exists(cpath):
            print(f"  [WARN] Skipping missing plot file: {cpath}")
            continue

        pts, inst, sem = load_classified_las(cpath)
        inst_enh, n_reassigned = enhance_stem_points(pts, inst, sem, **enh_params)
        qc_trees = extract_qc_trees(pts, inst_enh, sem, plot_id=pid, **qc_params)

        p_df = df_field_all[df_field_all["PAR_CODIGO"] == pid].reset_index(drop=True)
        matches = align_and_match_plot(qc_trees, p_df, **mat_params)

        n_gt = sum(1 for m in matches if m["has_gt_volume"])
        print(f"  Plot {pid}: {len(qc_trees)} QC trees (reassigned {n_reassigned} pts) -> {len(matches)} matched ({n_gt} with GT vol)")

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
            tree_records.append(rec)

    df_trees = pd.DataFrame(tree_records)
    print(f"\n  -> Total validated trees with GT across T37: {len(df_trees)}")

    trees_csv = os.path.join(export_dir, "df_trees_t37_matched.csv")
    df_trees.to_csv(trees_csv, index=False)
    print(f"  Exported matched trees dataset to: {trees_csv}")

    cfg_cols = [c for c in df_trees.columns if c.startswith(("voxel_", "axis_", "cyl_"))]

    # 3. Leave-One-Plot-Out (LOPO) 11 Folds
    print("\n[3/5] Running Leave-One-Plot-Out (LOPO - 11 Folds) validation...")
    df_lopo = run_leave_one_plot_out(df_trees, df_field_all, df_parc, cfg_cols)
    lopo_csv = os.path.join(export_dir, "lopo_results_t37.csv")
    df_lopo.to_csv(lopo_csv, index=False)

    print("\n" + "=" * 65)
    print("LOPO SUMMARY (100% Blind Predictions on Unseen Plots):")
    print("=" * 65)
    mean_abs_err_global = float(df_lopo["err_global_ha_pct"].abs().mean())
    mean_abs_err_bias_cal = float(df_lopo["err_bias_cal_ha_pct"].abs().mean())
    mean_abs_err_human = float(df_lopo["err_human_ha_pct"].abs().mean())
    print(f"  Mode 1A (Tree-RMSE Calibrated) Mean Abs Error : {mean_abs_err_global:.2f}% (Bias: {df_lopo['err_global_ha_pct'].mean():+.2f}%)")
    print(f"  Mode 1B (Plot-Bias Calibrated) Mean Abs Error : {mean_abs_err_bias_cal:.2f}% (Bias: {df_lopo['err_bias_cal_ha_pct'].mean():+.2f}%)")
    print(f"  Mode 2  (Human in the Loop)    Mean Abs Error : {mean_abs_err_human:.2f}% (Bias: {df_lopo['err_human_ha_pct'].mean():+.2f}%)")
    print(f"  Results saved to: {lopo_csv}")

    # 4. Sampling Saturation Curve (Diminishing Returns Analysis)
    print("\n[4/5] Computing Sampling Saturation Learning Curves (varying N train plots)...")
    sample_sizes = cfg["validation"]["saturation_sample_sizes"]
    n_repeats = cfg["validation"]["monte_carlo_repeats"]
    seed = cfg["validation"]["random_seed"]

    df_saturation = run_sampling_saturation_curve(
        df_trees, df_field_all, df_parc, cfg_cols,
        sample_sizes=sample_sizes, n_repeats=n_repeats, random_seed=seed
    )
    sat_csv = os.path.join(export_dir, "sampling_saturation_curve.csv")
    df_saturation.to_csv(sat_csv, index=False)

    # Summarize saturation curve
    sat_summary = df_saturation.groupby("n_train_plots").agg(
        global_err_mean=("mean_abs_plot_err_global", "mean"),
        global_err_std=("mean_abs_plot_err_global", "std"),
        bias_cal_err_mean=("mean_abs_plot_err_bias_cal", "mean"),
        bias_cal_err_std=("mean_abs_plot_err_bias_cal", "std"),
        human_err_mean=("mean_abs_plot_err_human", "mean"),
        human_err_std=("mean_abs_plot_err_human", "std"),
    ).reset_index()
    sat_summary_csv = os.path.join(export_dir, "sampling_saturation_summary.csv")
    sat_summary.to_csv(sat_summary_csv, index=False)

    print("\n" + "=" * 65)
    print("SAMPLING SATURATION CURVE (N Train Plots vs Held-Out Error %):")
    print("=" * 65)
    for _, r in sat_summary.iterrows():
        print(
            f"  N = {int(r['n_train_plots']):2d} train plots -> "
            f"Tree-RMSE Cal: {r['global_err_mean']:.1f}% (±{r['global_err_std']:.1f}%) | "
            f"Bias Cal: {r['bias_cal_err_mean']:.1f}% (±{r['bias_cal_err_std']:.1f}%) | "
            f"Human: {r['human_err_mean']:.1f}% (±{r['human_err_std']:.1f}%)"
        )

    # 5. Export Master Summary JSON
    print("\n[5/5] Exporting master experiment summary JSON...")
    exp1_meta = {
        "experiment": "Exp1_T37_CrossValidation_Saturation",
        "talhao_id": 37,
        "resolution": "10cm",
        "total_trees_matched": len(df_trees),
        "lopo_mean_abs_err_global_pct": mean_abs_err_global,
        "lopo_mean_abs_err_bias_cal_pct": mean_abs_err_bias_cal,
        "lopo_mean_abs_err_human_pct": mean_abs_err_human,
        "saturation_summary": sat_summary.to_dict(orient="records"),
        "lopo_details": df_lopo.to_dict(orient="records")
    }
    meta_json = os.path.join(export_dir, "exp1_summary.json")
    with open(meta_json, "w") as f:
        json.dump(exp1_meta, f, indent=2)

    print(f"  Master JSON exported to: {meta_json}")
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
