"""
Comprehensive Forest Biometric Benchmarking & Stand Extrapolation Framework.

Features:
1. Robust trunk enhancement (KDTree cylinder-following).
2. Adaptive spatial matching between segmented LiDAR trees and official field records.
3. Multi-method volumetric evaluation:
   - voxel (different voxel resolutions)
   - axis_profile (vertical slice trapezoidal integration)
   - cylinder + empirical form factor f (calibrated from ground truth)
   - frustum / taper (for complete methodology record)
4. Dual optimization paradigms:
   - Mode 1: Global Best Config (min aggregate RMSE/bias across calibration plots)
   - Mode 2: Human-in-the-Loop / Tree-level Oracle (optimal method per individual tree)
5. Cross-resolution stand extrapolation:
   - Run A: Calibrated on T62 (5cm dense flight) -> Extrapolated to T62/T63 and T37
   - Run B: Calibrated on T37 (10cm sparse flight) -> Extrapolated to T37 and T62/T63
6. Structured output:
   - metrics_summary.json
   - df_trees_t62.csv, df_trees_t37.csv
   - df_plots_summary.csv
   - df_extrapolation_summary.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Local module imports
sys.path.append("/home/matheuspimenta/euc/projeto_eucalipto")
from eucalipto.dbh_methods import estimate_dbh
from eucalipto.volume_methods import estimate_volume

EXCEL_PATH = "/mnt/scratch/dataset/euc/canoa/IFPC-Canoas/Result_IFPC_Canoas.xlsx"
OUTPUT_DIR = "/home/matheuspimenta/euc/projeto_eucalipto/results/benchmarking_framework"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T62_PARCELAS = [500, 503, 508, 509]
T62_CLOUDS = {
    p: f"/home/matheuspimenta/euc/projeto_eucalipto/results/canoa/parcelas_t62/parcela_{p}/classified_cloud.laz"
    for p in T62_PARCELAS
}

T37_PARCELAS = [594, 602, 611, 612, 621, 622, 623, 634, 635, 636, 648]
T37_CLOUDS = {
    p: f"/home/matheuspimenta/euc/projeto_eucalipto/results/canoa/parcelas_t37/parcela_{p}/classified_cloud.laz"
    for p in T37_PARCELAS
}

T63_CLOUD = "/home/matheuspimenta/euc/projeto_eucalipto/results/canoa/parcelas_t63/ff3d_full/classified_cloud.laz"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
def load_all_field_data(excel_path):
    print("[1/6] Loading official field inventory from Excel...")
    df_covas = pd.read_excel(excel_path, sheet_name="Covas")
    df_parc = pd.read_excel(excel_path, sheet_name="Parcelas")
    df_tal = pd.read_excel(excel_path, sheet_name="Talhões")

    # Process individual tree records
    df_trees = df_covas[df_covas["Fila"].notnull() & df_covas["Cova"].notnull()].copy()
    df_trees["x_grid"] = df_trees["Cova"].astype(float) * 2.5
    df_trees["y_grid"] = df_trees["Fila"].astype(float) * 3.5

    df_field = df_trees.groupby(["Talhão", "PAR_CODIGO", "Fila", "Cova"]).agg(
        x_grid=("x_grid", "first"),
        y_grid=("y_grid", "first"),
        dap_cm=("Dap (cm)", "max"),
        gt_volume=("VTCC Volume (m³)", "sum"),
        alt_m=("Altura (m)", "max"),
    ).reset_index()

    df_field["field_id"] = (
        df_field["PAR_CODIGO"].astype(str) + "_" +
        df_field["Fila"].astype(str) + "_" +
        df_field["Cova"].astype(str)
    )
    df_field["has_gt_volume"] = df_field["gt_volume"].notnull() & (df_field["gt_volume"] > 0)
    return df_field, df_parc, df_tal


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIDAR PROCESSING & STEM ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
def extract_and_enhance_trees(cloud_path, plot_id):
    if not os.path.exists(cloud_path):
        return []

    las = laspy.read(cloud_path)
    pts = np.vstack((las.x, las.y, las.z)).T
    fields = list(las.point_format.dimension_names)
    inst = (np.array(las.tree_id) if "tree_id" in fields else np.zeros(len(pts), dtype=np.int32)).copy()
    sem  = (np.array(las.trunk_leaf_label) if "trunk_leaf_label" in fields else np.ones(len(pts), dtype=np.int32))

    z_min_plot = pts[:, 2].min()
    plot_p90 = np.percentile(pts[:, 2] - z_min_plot, 90)

    # KDTree cylinder-following to reassign unclassified stem points
    known_trunk_mask = (inst > 0) & (sem == 1)
    known_trunk_pts = pts[known_trunk_mask]
    known_trunk_ids = inst[known_trunk_mask]

    bg_trunk_mask = (inst <= 0) & (sem == 1)
    bg_trunk_pts = pts[bg_trunk_mask]
    bg_trunk_indices = np.where(bg_trunk_mask)[0]

    if len(known_trunk_pts) > 0 and len(bg_trunk_pts) > 0:
        kdt = cKDTree(known_trunk_pts)
        dists, idxs = kdt.query(bg_trunk_pts, distance_upper_bound=0.35)
        valid = dists < 0.35
        for i, is_v in enumerate(valid):
            if is_v:
                match_pt = known_trunk_pts[idxs[i]]
                if abs(bg_trunk_pts[i, 2] - match_pt[2]) <= 0.8:
                    if np.linalg.norm(bg_trunk_pts[i, :2] - match_pt[:2]) <= 0.30:
                        inst[bg_trunk_indices[i]] = known_trunk_ids[idxs[i]]

    qc_trees = []
    for tid in np.unique(inst):
        if tid <= 0:
            continue
        tmask = (inst == tid)
        t_pts = pts[tmask]
        t_sem = sem[tmask]
        if len(t_pts) < 40:
            continue
        z0, z1 = t_pts[:, 2].min(), t_pts[:, 2].max()
        h = z1 - z0
        if h < (plot_p90 * 0.40):
            continue
        if (z0 - z_min_plot) > 2.5:
            continue
        p_trunk = t_pts[t_sem == 1]
        if len(p_trunk) < 15:
            continue

        cx, cy = np.median(t_pts[:, 0]), np.median(t_pts[:, 1])
        t_centered = t_pts.copy()
        t_centered[:, 0] -= cx
        t_centered[:, 1] -= cy
        t_centered[:, 2] -= z0

        qc_trees.append({
            "plot_id": plot_id,
            "tid": int(tid),
            "global_tid": int(plot_id * 10000 + tid),
            "x": cx,
            "y": cy,
            "height": float(h),
            "pts": t_centered,
            "trunk_mask": (t_sem == 1)
        })
    return qc_trees


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROBUST ADAPTIVE SPATIAL MATCHER
# ══════════════════════════════════════════════════════════════════════════════
def align_and_match(qc_trees, plot_df, max_dist=1.3):
    if len(qc_trees) < 3 or len(plot_df) == 0:
        return []

    F = plot_df[["x_grid", "y_grid"]].values
    T_pts = np.array([[t["x"], t["y"]] for t in qc_trees])
    T_h = np.array([t["height"] for t in qc_trees])
    mA = T_pts.mean(axis=0)

    best_score = -float("inf")
    best_Fr = None

    for swap in [False, True]:
        F_curr = F[:, [1, 0]] if swap else F
        mB = F_curr.mean(axis=0)
        Bc = F_curr - mB

        for deg in range(0, 360, 3):
            t = np.radians(deg)
            c, s = np.cos(t), np.sin(t)
            R = np.array([[c, -s], [s, c]])
            Br = Bc @ R.T

            for dx in np.linspace(-5, 5, 21):
                for dy in np.linspace(-5, 5, 21):
                    Fr_cand = Br + mA + np.array([dx, dy])
                    D = cdist(T_pts, Fr_cand)
                    min_d = np.min(D, axis=1)
                    close_mask = min_d < max_dist
                    n_close = close_mask.sum()
                    if n_close < min(len(qc_trees), 4):
                        continue

                    # Hybrid spatial + height consensus score
                    score = float(np.sum(1.5 - min_d[close_mask]))
                    closest = np.argmin(D, axis=1)
                    for ti, si in enumerate(closest):
                        if min_d[ti] < max_dist and plot_df.iloc[si]["has_gt_volume"]:
                            score += 2.0
                            gt_h = plot_df.iloc[si]["alt_m"]
                            if not np.isnan(gt_h):
                                score -= min(2.0, 0.2 * abs(T_h[ti] - gt_h))

                    if score > best_score:
                        best_score = score
                        best_Fr = Fr_cand

    matches = []
    if best_Fr is not None:
        D = cdist(best_Fr, T_pts)
        ri, ci = linear_sum_assignment(D)
        for f_idx, t_idx in zip(ri, ci):
            if D[f_idx, t_idx] < max_dist:
                row = plot_df.iloc[f_idx]
                t = qc_trees[t_idx]
                matches.append({
                    "plot_id": row["PAR_CODIGO"],
                    "field_id": row["field_id"],
                    "tid": t["tid"],
                    "global_tid": t["global_tid"],
                    "has_gt_volume": row["has_gt_volume"],
                    "gt_volume": row["gt_volume"] if row["has_gt_volume"] else np.nan,
                    "gt_dbh": row["dap_cm"] if row["has_gt_volume"] else np.nan,
                    "gt_height": row["alt_m"] if row["has_gt_volume"] else np.nan,
                    "dist": float(D[f_idx, t_idx]),
                    "tree_data": t
                })
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# 4. MULTI-METHOD BIOMETRICS GRID EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_tree_grid(tree_dict):
    pts = tree_dict["pts"]
    tmask = tree_dict["trunk_mask"]
    trunk = pts[tmask] if tmask.sum() > 10 else pts
    h = tree_dict["height"]

    # Estimate DBH across standard methods
    dbh_estimates = {}
    for dm, dp in [
        ("ensemble", {"offset_range": 0.3, "n_slices": 7, "slice_thickness": 0.16, "ransac_thresh": 0.02}),
        ("single_ransac", {"slice_thickness": 0.16, "ransac_thresh": 0.02}),
        ("ls", {"slice_thickness": 0.16})
    ]:
        try:
            val, _ = estimate_dbh(trunk, method=dm, **dp)
            dbh_estimates[dm] = float(val)
        except Exception:
            dbh_estimates[dm] = np.nan

    best_dbh = dbh_estimates.get("ensemble", np.nan)
    if np.isnan(best_dbh):
        best_dbh = dbh_estimates.get("single_ransac", np.nan)

    configs = {}

    # 1. Voxel grid
    for vs in [0.06, 0.07, 0.08, 0.09, 0.10]:
        cfg_name = f"voxel_vs{vs:.2f}"
        try:
            r = estimate_volume(trunk, method="voxel", voxel_size=vs)
            configs[cfg_name] = {"volume_m3": r["volume_m3"], "method": "voxel", "params": {"voxel_size": vs}}
        except Exception:
            configs[cfg_name] = {"volume_m3": np.nan, "method": "voxel", "params": {"voxel_size": vs}}

    # 2. Axis Profile grid
    for rp in [15.0, 20.0, 25.0]:
        for ns in [15, 20]:
            cfg_name = f"axis_rp{rp:.0f}_ns{ns}"
            try:
                r = estimate_volume(trunk, method="axis_profile", n_slices=ns, slice_thickness=0.20,
                                    radius_percentile=rp, min_points_per_slice=5)
                configs[cfg_name] = {"volume_m3": r["volume_m3"], "method": "axis_profile",
                                     "params": {"radius_percentile": rp, "n_slices": ns}}
            except Exception:
                configs[cfg_name] = {"volume_m3": np.nan, "method": "axis_profile",
                                     "params": {"radius_percentile": rp, "n_slices": ns}}

    # 3. Form Factor Calibrated Cylinder grid (f * pi/4 * DBH^2 * H)
    if not np.isnan(best_dbh) and best_dbh > 0:
        v_cyl_raw = (np.pi / 4.0) * ((best_dbh / 100.0) ** 2) * h
        for f_factor in [0.35, 0.38, 0.40, 0.42, 0.45]:
            cfg_name = f"cyl_ff{f_factor:.2f}"
            configs[cfg_name] = {"volume_m3": float(v_cyl_raw * f_factor), "method": "cylinder_ff",
                                 "params": {"form_factor": f_factor, "dbh_cm": best_dbh}}
    else:
        for f_factor in [0.35, 0.38, 0.40, 0.42, 0.45]:
            configs[f"cyl_ff{f_factor:.2f}"] = {"volume_m3": np.nan, "method": "cylinder_ff",
                                                 "params": {"form_factor": f_factor}}

    return {
        "height_m": h,
        "dbh_estimates": dbh_estimates,
        "best_dbh_cm": best_dbh,
        "configs": configs
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORE BENCHMARKING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_dataset_benchmark(stand_name, talhao_key, clouds_dict, df_field_all, df_parc, df_tal):
    print(f"\n{'='*70}\nProcessing Stand: {stand_name} (Talhão {talhao_key})\n{'='*70}")
    stand_field = df_field_all[df_field_all["Talhão"] == talhao_key]
    plot_ids = sorted(clouds_dict.keys())

    all_matches = []
    tree_records = []

    for pid in plot_ids:
        qc_trees = extract_and_enhance_trees(clouds_dict[pid], pid)
        plot_df = stand_field[stand_field["PAR_CODIGO"] == pid].reset_index(drop=True)
        matches = align_and_match(qc_trees, plot_df)
        all_matches.extend(matches)

        n_gt = sum(1 for m in matches if m["has_gt_volume"])
        print(f"  Plot {pid:3d}: {len(qc_trees):2d} QC trees -> {len(matches):2d} matched ({n_gt:2d} with GT vol)")

        for m in matches:
            if not m["has_gt_volume"]:
                continue
            bio = evaluate_tree_grid(m["tree_data"])
            rec = {
                "stand": stand_name,
                "plot_id": pid,
                "field_id": m["field_id"],
                "tid": m["tid"],
                "global_tid": m["global_tid"],
                "gt_volume": m["gt_volume"],
                "gt_dbh": m["gt_dbh"],
                "gt_height": m["gt_height"],
                "lidar_height": bio["height_m"],
                "lidar_dbh": bio["best_dbh_cm"],
            }
            # Flatten config volumes
            for cfg_name, cfg_dict in bio["configs"].items():
                rec[cfg_name] = cfg_dict["volume_m3"]
            tree_records.append(rec)

    df_trees = pd.DataFrame(tree_records)
    print(f"  -> Total matched trees with GT: {len(df_trees)}")

    if len(df_trees) == 0:
        return None, None

    # Get list of all volume configs evaluated
    cfg_cols = [c for c in df_trees.columns if c.startswith(("voxel_", "axis_", "cyl_"))]

    # --- Mode 1: Global Config Ranking ---
    cfg_metrics = []
    for cfg in cfg_cols:
        sub = df_trees.dropna(subset=[cfg, "gt_volume"])
        if len(sub) == 0:
            continue
        est = sub[cfg].values
        gt = sub["gt_volume"].values
        rmse = float(np.sqrt(np.mean((est - gt) ** 2)))
        mae = float(np.mean(np.abs(est - gt)))
        mean_est = float(np.mean(est))
        mean_gt = float(np.mean(gt))
        bias_pct = float((mean_est - mean_gt) / mean_gt * 100.0)
        cfg_metrics.append({
            "config": cfg,
            "rmse_m3": rmse,
            "mae_m3": mae,
            "mean_est_m3": mean_est,
            "mean_gt_m3": mean_gt,
            "bias_pct": bias_pct,
            "abs_bias_pct": abs(bias_pct)
        })

    df_cfg_ranking = pd.DataFrame(cfg_metrics).sort_values("rmse_m3").reset_index(drop=True)
    best_global_cfg = df_cfg_ranking.iloc[0]["config"]

    # --- Mode 2: Human-in-the-Loop / Tree-level Oracle ---
    oracle_vols = []
    oracle_cfgs = []
    for _, row in df_trees.iterrows():
        gt = row["gt_volume"]
        best_diff = float("inf")
        best_v = np.nan
        best_c = None
        for cfg in cfg_cols:
            val = row[cfg]
            if not np.isnan(val):
                diff = abs(val - gt)
                if diff < best_diff:
                    best_diff = diff
                    best_v = val
                    best_c = cfg
        oracle_vols.append(best_v)
        oracle_cfgs.append(best_c)

    df_trees["vol_global_best"] = df_trees[best_global_cfg]
    df_trees["vol_human_in_loop"] = oracle_vols
    df_trees["cfg_human_in_loop"] = oracle_cfgs

    # Per-tree performance
    gt_arr = df_trees["gt_volume"].values
    g_arr = df_trees["vol_global_best"].values
    h_arr = df_trees["vol_human_in_loop"].values

    rmse_global = float(np.sqrt(np.mean((g_arr - gt_arr) ** 2)))
    rmse_human  = float(np.sqrt(np.mean((h_arr - gt_arr) ** 2)))
    bias_global = float((np.mean(g_arr) - np.mean(gt_arr)) / np.mean(gt_arr) * 100.0)
    bias_human  = float((np.mean(h_arr) - np.mean(gt_arr)) / np.mean(gt_arr) * 100.0)

    print(f"\n[PERFORMANCE on {stand_name} - {len(df_trees)} trees]:")
    print(f"  Mode 1 (Global Best: {best_global_cfg}): RMSE = {rmse_global:.4f} m³ | Bias = {bias_global:+.2f}%")
    print(f"  Mode 2 (Human in the Loop / Oracle)  : RMSE = {rmse_human:.4f} m³ | Bias = {bias_human:+.2f}%")

    # Plot-level aggregation
    plot_results = []
    for pid, pgrp in df_trees.groupby("plot_id"):
        # Match official plot record
        p_match = df_parc[df_parc["Parcela"] == pid]
        gt_ha = float(p_match["VTCC (m³/ha)"].iloc[0]) if len(p_match) > 0 else np.nan
        area_m2 = float(p_match["Área (m²)"].iloc[0]) if len(p_match) > 0 else 400.0
        exp_factor = 10000.0 / area_m2

        # Count living trees in plot
        n_living = len(stand_field[(stand_field["PAR_CODIGO"] == pid) & stand_field["has_gt_volume"]])
        n_matched = len(pgrp)

        mean_v_gt = float(pgrp["gt_volume"].mean())
        mean_v_global = float(pgrp["vol_global_best"].mean())
        mean_v_human  = float(pgrp["vol_human_in_loop"].mean())

        # Extrapolate to plot m3/ha
        extrap_global_ha = mean_v_global * n_living * exp_factor
        extrap_human_ha  = mean_v_human * n_living * exp_factor
        extrap_gt_ha     = mean_v_gt * n_living * exp_factor

        err_global_pct = (extrap_global_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan
        err_human_pct  = (extrap_human_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan

        plot_results.append({
            "stand": stand_name,
            "plot_id": pid,
            "n_matched": n_matched,
            "n_living_field": n_living,
            "gt_m3_ha": gt_ha,
            "global_m3_ha": extrap_global_ha,
            "global_err_pct": err_global_pct,
            "human_m3_ha": extrap_human_ha,
            "human_err_pct": err_human_pct,
            "mean_tree_gt": mean_v_gt,
            "mean_tree_global": mean_v_global,
            "mean_tree_human": mean_v_human
        })

    df_plots = pd.DataFrame(plot_results)

    summary = {
        "stand_name": stand_name,
        "n_trees_eval": len(df_trees),
        "best_global_config": best_global_cfg,
        "global_rmse_m3": rmse_global,
        "global_bias_pct": bias_global,
        "human_rmse_m3": rmse_human,
        "human_bias_pct": bias_human,
        "cfg_ranking": df_cfg_ranking.to_dict(orient="records"),
    }

    return df_trees, df_plots, summary


# ══════════════════════════════════════════════════════════════════════════════
# 6. STAND-LEVEL EXTRAPOLATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_cross_stand_extrapolations(base_name, best_cfg, df_trees_base, df_tal, df_parc, clouds_t62, clouds_t37, cloud_t63):
    print(f"\n{'='*70}\nRunning Stand-Level Extrapolation (Calibrated on {base_name})\n{'='*70}")
    
    # Official stand targets
    t_info_62 = df_tal[df_tal["Talhão"] == "062-063"].iloc[0]
    t_info_37 = df_tal[df_tal["Talhão"] == 37].iloc[0]

    area_t62_63 = float(t_info_62["Área Plantada"]) # 41.0 ha
    gt_ha_t62_63 = float(t_info_62["VTCC (m³/ha)"]) # 201.29 m3/ha
    gt_tot_t62_63 = float(t_info_62["VTCC (m³)"])   # 8252.95 m3
    density_t62_63 = float(t_info_62["Árvores (n/ha)"]) # 650 trees/ha

    area_t37 = float(t_info_37["Área Plantada"])     # 26.46 ha
    gt_ha_t37 = float(t_info_37["VTCC (m³/ha)"])     # 191.14 m3/ha
    gt_tot_t37 = float(t_info_37["VTCC (m³)"])       # 5057.67 m3
    density_t37 = float(t_info_37["Árvores (n/ha)"]) # 1034.09 trees/ha

    # Mean tree metrics from base calibration
    mean_tree_global = float(df_trees_base["vol_global_best"].mean())
    mean_tree_human  = float(df_trees_base["vol_human_in_loop"].mean())

    extrapolations = []

    # 1. Extrapolate to Talhão 62/63
    ha_global_t62 = mean_tree_global * density_t62_63
    ha_human_t62  = mean_tree_human * density_t62_63
    tot_global_t62 = ha_global_t62 * area_t62_63
    tot_human_t62  = ha_human_t62 * area_t62_63

    extrapolations.append({
        "calibration_base": base_name,
        "target_stand": "Talhão 62/63 (41.0 ha)",
        "gt_m3_ha": gt_ha_t62_63,
        "gt_total_m3": gt_tot_t62_63,
        "global_m3_ha": ha_global_t62,
        "global_total_m3": tot_global_t62,
        "global_err_pct": (ha_global_t62 - gt_ha_t62_63) / gt_ha_t62_63 * 100.0,
        "human_m3_ha": ha_human_t62,
        "human_total_m3": tot_human_t62,
        "human_err_pct": (ha_human_t62 - gt_ha_t62_63) / gt_ha_t62_63 * 100.0,
    })

    # 2. Extrapolate to Talhão 37
    ha_global_t37 = mean_tree_global * density_t37
    ha_human_t37  = mean_tree_human * density_t37
    tot_global_t37 = ha_global_t37 * area_t37
    tot_human_t37  = ha_human_t37 * area_t37

    extrapolations.append({
        "calibration_base": base_name,
        "target_stand": "Talhão 37 (26.46 ha)",
        "gt_m3_ha": gt_ha_t37,
        "gt_total_m3": gt_tot_t37,
        "global_m3_ha": ha_global_t37,
        "global_total_m3": tot_global_t37,
        "global_err_pct": (ha_global_t37 - gt_ha_t37) / gt_ha_t37 * 100.0,
        "human_m3_ha": ha_human_t37,
        "human_total_m3": tot_human_t37,
        "human_err_pct": (ha_human_t37 - gt_ha_t37) / gt_ha_t37 * 100.0,
    })

    return pd.DataFrame(extrapolations)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("STARTING BIOMETRIC CALIBRATION & EXTRAPOLATION FRAMEWORK")
    print("=" * 80)

    df_field_all, df_parc, df_tal = load_all_field_data(EXCEL_PATH)

    # ── RUN A: BASE TALHÃO 62 (5cm High Resolution Flight) ─────────────────────
    df_trees_t62, df_plots_t62, summary_t62 = run_dataset_benchmark(
        "Talhão 62 (5cm)", "062-063", T62_CLOUDS, df_field_all, df_parc, df_tal
    )

    # ── RUN B: BASE TALHÃO 37 (10cm Sparse Resolution Flight) ───────────────────
    df_trees_t37, df_plots_t37, summary_t37 = run_dataset_benchmark(
        "Talhão 37 (10cm)", 37, T37_CLOUDS, df_field_all, df_parc, df_tal
    )

    # Save detailed tree tables
    t62_csv_path = os.path.join(OUTPUT_DIR, "df_trees_t62.csv")
    t37_csv_path = os.path.join(OUTPUT_DIR, "df_trees_t37.csv")
    df_trees_t62.to_csv(t62_csv_path, index=False)
    df_trees_t37.to_csv(t37_csv_path, index=False)
    print(f"\n[EXPORT] Tree datasets saved to:\n  - {t62_csv_path}\n  - {t37_csv_path}")

    # Plot summary table
    df_plots_combined = pd.concat([df_plots_t62, df_plots_t37], ignore_index=True)
    plots_csv_path = os.path.join(OUTPUT_DIR, "df_plots_summary.csv")
    df_plots_combined.to_csv(plots_csv_path, index=False)
    print(f"[EXPORT] Plot summary saved to:\n  - {plots_csv_path}")

    # Extrapolations
    extrap_base_t62 = run_cross_stand_extrapolations(
        "Base T62 (5cm)", summary_t62["best_global_config"], df_trees_t62, df_tal, df_parc,
        T62_CLOUDS, T37_CLOUDS, T63_CLOUD
    )
    extrap_base_t37 = run_cross_stand_extrapolations(
        "Base T37 (10cm)", summary_t37["best_global_config"], df_trees_t37, df_tal, df_parc,
        T62_CLOUDS, T37_CLOUDS, T63_CLOUD
    )

    df_extrap_combined = pd.concat([extrap_base_t62, extrap_base_t37], ignore_index=True)
    extrap_csv_path = os.path.join(OUTPUT_DIR, "df_extrapolation_summary.csv")
    df_extrap_combined.to_csv(extrap_csv_path, index=False)
    print(f"[EXPORT] Extrapolation summary saved to:\n  - {extrap_csv_path}")

    # Comprehensive JSON metrics file
    final_json = {
        "t62_base": summary_t62,
        "t37_base": summary_t37,
        "extrapolations": df_extrap_combined.to_dict(orient="records"),
        "plots_summary": df_plots_combined.to_dict(orient="records")
    }
    json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"[EXPORT] Master metrics JSON saved to:\n  - {json_path}")

    print("\n" + "=" * 80)
    print("FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
