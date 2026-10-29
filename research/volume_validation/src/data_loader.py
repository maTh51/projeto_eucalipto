"""
Data loading and preprocessing utilities for forest inventory and LiDAR point clouds.
"""

import os
import laspy
import numpy as np
import pandas as pd


def load_inventory_data(excel_path: str, talhao_id):
    """
    Load official field measurements, plot summary and stand-level targets from Excel.
    """
    df_covas = pd.read_excel(excel_path, sheet_name="Covas")
    df_parc = pd.read_excel(excel_path, sheet_name="Parcelas")
    df_tal = pd.read_excel(excel_path, sheet_name="Talhões")

    # Filter trees for the specified stand
    df_trees = df_covas[df_covas["Talhão"] == talhao_id].copy()
    df_trees = df_trees[df_trees["Fila"].notnull() & df_trees["Cova"].notnull()].copy()
    
    # Reconstruct local theoretical grid coordinates
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

    # Filter plot and stand summaries
    df_parc_filtered = df_parc[df_parc["Talhão"] == talhao_id].copy()
    df_tal_filtered = df_tal[df_tal["Talhão"] == talhao_id].copy()

    return df_field, df_parc_filtered, df_tal_filtered


def load_classified_las(cloud_path: str):
    """
    Load classified point cloud and extract coordinates and labels.
    """
    if not os.path.exists(cloud_path):
        raise FileNotFoundError(f"Cloud file not found: {cloud_path}")

    las = laspy.read(cloud_path)
    pts = np.vstack((las.x, las.y, las.z)).T
    fields = list(las.point_format.dimension_names)

    inst = (np.array(las.tree_id) if "tree_id" in fields else np.zeros(len(pts), dtype=np.int32)).copy()
    sem = (np.array(las.trunk_leaf_label) if "trunk_leaf_label" in fields else np.ones(len(pts), dtype=np.int32))

    return pts, inst, sem
