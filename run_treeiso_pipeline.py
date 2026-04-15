"""Pipeline baseado em treeiso + heurística de tronco.

Fluxo:

1. Executa o algoritmo original do treeiso (artemis_treeiso) sobre
   todos os arquivos LAS/LAZ em um diretório de entrada, gerando
   arquivos "*_treeiso.laz" com o campo "final_segs".
2. Usa "final_segs" como identificador de árvore/segmento para separar
   as nuvens por árvore.
3. Aplica a heurística de tronco em cada árvore (eucalipto.trunk_heuristic).
4. Calcula DAP (ensemble) e volume (cilindro) para cada tronco.
5. Salva um CSV resumo em um diretório de resultados.

Uso:

    python run_treeiso_pipeline.py

Edite a seção CONFIG abaixo antes de rodar.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict

import numpy as np

from eucalipto import isolation_treeiso
from eucalipto import trunk_heuristic
from eucalipto import dbh_methods
from eucalipto import volume_methods
from eucalipto import io as eio
from eucalipto import treeiso_exports
from eucalipto.preprocess_treeiso import prepare_treeiso_input


# ==========================
# CONFIGURATION (EDIT HERE)
# ==========================

# Caminho para o repositório artemis_treeiso
TREEISO_REPO_DIR = "/home/matheuspimenta/Jobs/Eucalipto/artemis_treeiso"

# Diretório ou arquivo LAS/LAZ (sem processamento treeiso ainda)
INPUT_DIR = "/home/matheuspimenta/Jobs/Eucalipto/projeto_eucalipto/data/CULS_CULS_plot_1_annotated_train.ply"  # TODO: ajustar

# Campo de ID de árvore/segmento na saída do treeiso
TREE_ID_DIM = "final_segs"

# Em alguns ambientes, o backend C++ do cut-pursuit pode colapsar tudo em 1 componente.
# Neste caso, forçamos o backend Python para manter segmentação estável.
TREEISO_FORCE_PYTHON_CUT_PURSUIT = True

# Filtro prévio de chão antes do treeiso
# Exemplo atual: treeID == 0 representa chão.
PRE_FILTER_ENABLED = True
PRE_FILTER_DIM = "treeID"
PRE_FILTER_GROUND_VALUE = 0

# Diretório de saída (dentro deste repositório)
RESULTS_DIR = "results_treeiso"

# Parâmetros da heurística de tronco
TRUNK_PARAMS = {
    "base_percentile": 20.0,
    "k_neighbors": 25,
    "linearity_threshold": 0.4,
    "scattering_threshold": 0.4,
    "dist_percentile": 30.0,
    "expansion_radius": 0.2,
    "component_radius": 0.2,
}

# Densidade da madeira (kg/m^3) para estimar massa seca (opcional)
WOOD_DENSITY = 900.0

# Exportações extras para validação/apresentação
EXPORT_CLASSIFIED_CLOUD = True
EXPORT_VOLUME_REPRESENTATION = True


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if PRE_FILTER_ENABLED:
        print(
            f"Pré-filtro habilitado: manter pontos com {PRE_FILTER_DIM} != {PRE_FILTER_GROUND_VALUE}"
        )
    input_for_treeiso = prepare_treeiso_input(
        INPUT_DIR,
        pre_filter_enabled=PRE_FILTER_ENABLED,
        pre_filter_dim=PRE_FILTER_DIM,
        pre_filter_ground_value=PRE_FILTER_GROUND_VALUE,
        output_root_dir=RESULTS_DIR,
    )

    # 1) Rodar treeiso nos LAS/LAZ do diretório de entrada
    treeiso_outputs = isolation_treeiso.run_treeiso_on_dir(
        TREEISO_REPO_DIR,
        input_for_treeiso,
        force_python_cut_pursuit=TREEISO_FORCE_PYTHON_CUT_PURSUIT,
    )

    if len(treeiso_outputs) == 0:
        raise RuntimeError(
            "Nenhum arquivo LAS/LAZ foi processado. "
            "Verifique INPUT_DIR (diretório com *.las/*.laz ou arquivo único .las/.laz)."
        )

    classified_dir = Path(RESULTS_DIR) / "classified_clouds"
    volume_dir = Path(RESULTS_DIR) / "volume_representation"
    classified_dir.mkdir(parents=True, exist_ok=True)
    volume_dir.mkdir(parents=True, exist_ok=True)

    # 2) Construir métricas por árvore e exportar nuvens classificadas
    metrics: Dict[str, dict] = {}
    cylinders = []

    for out_path in treeiso_outputs:
        points, extras = eio.load_laz(out_path)

        if TREE_ID_DIM not in extras:
            raise ValueError(f"Campo '{TREE_ID_DIM}' não encontrado em {out_path}")

        seg_ids = np.asarray(extras[TREE_ID_DIM]).astype(np.int32)
        unique_ids = np.unique(seg_ids)
        stem = Path(out_path).stem

        trunk_mask_all = np.zeros(points.shape[0], dtype=np.uint8)
        wood_score_all = np.full(points.shape[0], np.nan, dtype=np.float32)
        dist_axis_all = np.full(points.shape[0], np.nan, dtype=np.float32)
        dbh_cm_tree_all = np.full(points.shape[0], np.nan, dtype=np.float32)
        volume_m3_tree_all = np.full(points.shape[0], np.nan, dtype=np.float32)

        for seg_id in unique_ids:
            mask = seg_ids == int(seg_id)
            seg_idx = np.where(mask)[0]
            pts_seg = points[seg_idx]
            if pts_seg.shape[0] < 3:
                continue

            wood_mask, wood_score, dist_axis = trunk_heuristic.extract_trunk(
            pts_seg,
            **TRUNK_PARAMS,
        )
            trunk_mask_all[seg_idx] = wood_mask.astype(np.uint8)
            wood_score_all[seg_idx] = wood_score.astype(np.float32)
            dist_axis_all[seg_idx] = dist_axis.astype(np.float32)

            tid = f"{stem}_seg{int(seg_id)}"
            trunk_pts = pts_seg[wood_mask]
            if trunk_pts.shape[0] < 3:
                continue

            dbh_cm, dbh_info = dbh_methods.estimate_dbh(trunk_pts, method="ensemble")
            if dbh_cm is None:
                metrics[tid] = {"dbh_cm": None, "volume_m3": None, "dbh_info": dbh_info}
                continue

            height_m = eio.tree_height(trunk_pts)
            vol_info = volume_methods.estimate_volume(
                trunk_pts,
                dbh_cm=dbh_cm,
                height_m=height_m,
                method="cylinder",
                wood_density_kg_m3=WOOD_DENSITY,
            )

            res = {"dbh_cm": dbh_cm, "dbh_info": dbh_info}
            res.update(vol_info)
            metrics[tid] = res

            dbh_cm_tree_all[seg_idx] = np.float32(dbh_cm)
            volume_m3_tree_all[seg_idx] = np.float32(vol_info.get("volume_m3", np.nan))

            radius_m = float(dbh_cm) / 200.0
            cylinders.append({
                "tree_id": int(seg_id),
                "center_x": float(np.mean(trunk_pts[:, 0])),
                "center_y": float(np.mean(trunk_pts[:, 1])),
                "z_min": float(np.min(trunk_pts[:, 2])),
                "height_m": float(height_m),
                "radius_m": float(radius_m),
                "volume_m3": float(vol_info.get("volume_m3", np.nan)),
            })

        if EXPORT_CLASSIFIED_CLOUD:
            classified_path = classified_dir / f"{stem}_classified.laz"
            treeiso_exports.save_classified_cloud_laz(
                output_path=str(classified_path),
                points=points,
                tree_segment_id=seg_ids,
                trunk_mask=trunk_mask_all,
                wood_score=wood_score_all,
                dist_axis=dist_axis_all,
                dbh_cm_tree=dbh_cm_tree_all,
                volume_m3_tree=volume_m3_tree_all,
            )
            print(f"Nuvem classificada salva em {classified_path}")

    # 3) Escrever CSV resumo
    csv_path = Path(RESULTS_DIR) / "treeiso_metrics_summary.csv"
    fieldnames = [
        "tree_id",
        "dbh_cm",
        "height_m",
        "volume_m3",
        "volume_liters",
        "mass_kg",
        "dbh_method",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tid, res in sorted(metrics.items()):
            row = {
                "tree_id": tid,
                "dbh_cm": res.get("dbh_cm"),
                "height_m": res.get("height_m"),
                "volume_m3": res.get("volume_m3"),
                "volume_liters": res.get("volume_liters"),
                "mass_kg": res.get("mass_kg"),
                "dbh_method": res.get("dbh_info", {}).get("method"),
            }
            writer.writerow(row)

    if EXPORT_VOLUME_REPRESENTATION:
        cyl_csv = volume_dir / "treeiso_volume_cylinders.csv"
        cyl_ply = volume_dir / "treeiso_volume_cylinders.ply"
        treeiso_exports.save_cylinder_primitives_csv(str(cyl_csv), cylinders)
        treeiso_exports.save_cylinder_representation_ply(str(cyl_ply), cylinders)
        print(f"Representação de volume salva em {cyl_csv} e {cyl_ply}")

    print(f"Resumo salvo em {csv_path}")


if __name__ == "__main__":
    main()
