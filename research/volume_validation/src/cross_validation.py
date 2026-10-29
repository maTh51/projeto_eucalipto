"""
Cross-validation engines: Leave-One-Plot-Out (LOPO) and Sampling Saturation Learning Curves.
"""

import numpy as np
import pandas as pd
from itertools import combinations


def evaluate_config_on_subset(df_trees: pd.DataFrame, cfg_col: str):
    """
    Compute RMSE, MAE, and relative bias for a specific config on a tree subset.
    """
    sub = df_trees.dropna(subset=[cfg_col, "gt_volume"])
    if len(sub) == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias_pct": np.nan}

    est = sub[cfg_col].values
    gt = sub["gt_volume"].values
    rmse = float(np.sqrt(np.mean((est - gt) ** 2)))
    mae = float(np.mean(np.abs(est - gt)))
    bias_pct = float((np.mean(est) - np.mean(gt)) / np.mean(gt) * 100.0)
    return {"rmse": rmse, "mae": mae, "bias_pct": bias_pct}


def find_best_global_config(df_train_trees: pd.DataFrame, cfg_cols: list):
    """
    Find the configuration that minimizes individual tree RMSE on the training tree set.
    """
    best_cfg = None
    best_rmse = float("inf")
    for cfg in cfg_cols:
        metrics = evaluate_config_on_subset(df_train_trees, cfg)
        if not np.isnan(metrics["rmse"]) and metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_cfg = cfg
    return best_cfg, best_rmse


def find_best_unbiased_config(df_train_trees: pd.DataFrame, cfg_cols: list, rmse_tolerance: float = 1.35):
    """
    Find the configuration that minimizes absolute relative bias |bias_pct| on the training tree set,
    restricted to stable configurations whose tree RMSE is within `rmse_tolerance` of the minimum RMSE
    (preventing high-variance methods with cancelling outliers from being selected).
    """
    _, min_rmse = find_best_global_config(df_train_trees, cfg_cols)
    best_cfg = None
    best_abs_bias = float("inf")
    for cfg in cfg_cols:
        metrics = evaluate_config_on_subset(df_train_trees, cfg)
        if np.isnan(metrics["bias_pct"]) or np.isnan(metrics["rmse"]):
            continue
        if metrics["rmse"] <= min_rmse * rmse_tolerance and abs(metrics["bias_pct"]) < best_abs_bias:
            best_abs_bias = abs(metrics["bias_pct"])
            best_cfg = cfg
    return best_cfg, best_abs_bias


def run_leave_one_plot_out(df_trees: pd.DataFrame, df_field_all: pd.DataFrame,
                           df_parc: pd.DataFrame, cfg_cols: list):
    """
    Execute 100% blind Leave-One-Plot-Out validation across all plots,
    evaluating side-by-side:
      - Mode 1A: Calibrated by Tree RMSE (min tree RMSE on train plots)
      - Mode 1B: Calibrated by Volume Bias (min |bias_pct| on train plots)
      - Mode 2:  Human-in-the-Loop / Oracle (best config per tree)
    """
    plot_ids = sorted(df_trees["plot_id"].unique())
    lopo_results = []

    for test_pid in plot_ids:
        df_train = df_trees[df_trees["plot_id"] != test_pid]
        df_test  = df_trees[df_trees["plot_id"] == test_pid]

        if len(df_test) == 0:
            continue

        # 1A. Calibrate optimal config by Tree RMSE on training plots
        best_cfg, train_rmse = find_best_global_config(df_train, cfg_cols)
        test_metrics_global = evaluate_config_on_subset(df_test, best_cfg)

        # 1B. Calibrate optimal config by Volume Bias on training plots
        best_bias_cfg, train_abs_bias = find_best_unbiased_config(df_train, cfg_cols)
        test_metrics_bias = evaluate_config_on_subset(df_test, best_bias_cfg)

        # 2. Mode 2: Human-in-the-Loop on test plot (oracle best per tree)
        human_vols = []
        for _, row in df_test.iterrows():
            gt = row["gt_volume"]
            best_diff = float("inf")
            best_v = np.nan
            for c in cfg_cols:
                v = row[c]
                if not np.isnan(v) and abs(v - gt) < best_diff:
                    best_diff = abs(v - gt)
                    best_v = v
            human_vols.append(best_v)

        gt_test = df_test["gt_volume"].values
        h_arr = np.array(human_vols)
        rmse_human = float(np.sqrt(np.mean((h_arr - gt_test) ** 2)))
        bias_human = float((np.mean(h_arr) - np.mean(gt_test)) / np.mean(gt_test) * 100.0)

        # 3. Plot-level volume extrapolation (m3/ha)
        p_row = df_parc[df_parc["Parcela"] == test_pid]
        gt_ha = float(p_row["VTCC (m³/ha)"].iloc[0]) if len(p_row) > 0 else np.nan
        area_m2 = float(p_row["Área (m²)"].iloc[0]) if len(p_row) > 0 else 400.0
        exp_factor = 10000.0 / area_m2

        # Count living trees in plot from field inventory
        n_living = len(df_field_all[(df_field_all["PAR_CODIGO"] == test_pid) & df_field_all["has_gt_volume"]])
        mean_v_global = float(df_test[best_cfg].mean())
        mean_v_bias   = float(df_test[best_bias_cfg].mean())
        mean_v_human  = float(np.mean(human_vols))

        extrap_global_ha = mean_v_global * n_living * exp_factor
        extrap_bias_ha   = mean_v_bias * n_living * exp_factor
        extrap_human_ha  = mean_v_human * n_living * exp_factor

        err_global_ha_pct = (extrap_global_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan
        err_bias_ha_pct   = (extrap_bias_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan
        err_human_ha_pct  = (extrap_human_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan

        lopo_results.append({
            "test_plot_id": test_pid,
            "calibrated_config": best_cfg,
            "calibrated_bias_config": best_bias_cfg,
            "train_rmse": train_rmse,
            "train_abs_bias_pct": train_abs_bias,
            "n_test_trees": len(df_test),
            "n_living_field": n_living,
            "tree_rmse_global": test_metrics_global["rmse"],
            "tree_bias_global_pct": test_metrics_global["bias_pct"],
            "tree_rmse_bias_cal": test_metrics_bias["rmse"],
            "tree_bias_bias_cal_pct": test_metrics_bias["bias_pct"],
            "tree_rmse_human": rmse_human,
            "tree_bias_human_pct": bias_human,
            "gt_m3_ha": gt_ha,
            "pred_global_m3_ha": extrap_global_ha,
            "err_global_ha_pct": err_global_ha_pct,
            "pred_bias_cal_m3_ha": extrap_bias_ha,
            "err_bias_cal_ha_pct": err_bias_ha_pct,
            "pred_human_m3_ha": extrap_human_ha,
            "err_human_ha_pct": err_human_ha_pct
        })

    return pd.DataFrame(lopo_results)


def run_sampling_saturation_curve(df_trees: pd.DataFrame, df_field_all: pd.DataFrame,
                                   df_parc: pd.DataFrame, cfg_cols: list,
                                   sample_sizes: list = [2, 4, 6, 8, 10],
                                   n_repeats: int = 20, random_seed: int = 42):
    """
    Subsample field plots at different sample sizes to determine the point of saturation
    where additional field plots yield diminishing accuracy returns.
    Uses unique combinations without replacement when C(N, k) <= n_repeats.
    """
    rng = np.random.default_rng(random_seed)
    all_plots = np.array(sorted(df_trees["plot_id"].unique()))
    saturation_records = []

    for k in sample_sizes:
        if k >= len(all_plots):
            continue

        all_combos = list(combinations(all_plots, k))
        if len(all_combos) <= n_repeats:
            selected_combos = all_combos
        else:
            idxs = rng.choice(len(all_combos), size=n_repeats, replace=False)
            selected_combos = [all_combos[i] for i in idxs]

        for rep, train_tuple in enumerate(selected_combos):
            train_plots = np.array(train_tuple)
            test_plots  = np.setdiff1d(all_plots, train_plots)

            df_train = df_trees[df_trees["plot_id"].isin(train_plots)]
            df_test  = df_trees[df_trees["plot_id"].isin(test_plots)]

            if len(df_train) == 0 or len(df_test) == 0:
                continue

            best_cfg, _ = find_best_global_config(df_train, cfg_cols)
            best_bias_cfg, _ = find_best_unbiased_config(df_train, cfg_cols)

            # Test plot predictions
            test_plot_errors_global = []
            test_plot_errors_bias_cal = []
            test_plot_errors_human = []

            for pid in test_plots:
                sub_t = df_test[df_test["plot_id"] == pid]
                if len(sub_t) == 0:
                    continue

                p_row = df_parc[df_parc["Parcela"] == pid]
                gt_ha = float(p_row["VTCC (m³/ha)"].iloc[0]) if len(p_row) > 0 else np.nan
                area_m2 = float(p_row["Área (m²)"].iloc[0]) if len(p_row) > 0 else 400.0
                exp_factor = 10000.0 / area_m2
                n_living = len(df_field_all[(df_field_all["PAR_CODIGO"] == pid) & df_field_all["has_gt_volume"]])

                # Global prediction (Tree-RMSE calibrated)
                m_v_g = float(sub_t[best_cfg].mean())
                pred_g_ha = m_v_g * n_living * exp_factor
                err_g = abs(pred_g_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan
                test_plot_errors_global.append(err_g)

                # Bias-calibrated prediction (Plot/Bias calibrated)
                m_v_b = float(sub_t[best_bias_cfg].mean())
                pred_b_ha = m_v_b * n_living * exp_factor
                err_b = abs(pred_b_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan
                test_plot_errors_bias_cal.append(err_b)

                # Human prediction
                h_vols = [row[cfg_cols].iloc[(row[cfg_cols] - row["gt_volume"]).abs().argmin()] for _, row in sub_t.iterrows()]
                m_v_h = float(np.mean(h_vols))
                pred_h_ha = m_v_h * n_living * exp_factor
                err_h = abs(pred_h_ha - gt_ha) / gt_ha * 100.0 if not np.isnan(gt_ha) else np.nan
                test_plot_errors_human.append(err_h)

            saturation_records.append({
                "n_train_plots": k,
                "repeat_idx": rep,
                "best_cfg": best_cfg,
                "best_bias_cfg": best_bias_cfg,
                "mean_abs_plot_err_global": float(np.mean(test_plot_errors_global)),
                "mean_abs_plot_err_bias_cal": float(np.mean(test_plot_errors_bias_cal)),
                "mean_abs_plot_err_human": float(np.mean(test_plot_errors_human))
            })

    return pd.DataFrame(saturation_records)
