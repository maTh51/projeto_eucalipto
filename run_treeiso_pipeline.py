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


# ==========================
# CONFIGURATION (EDIT HERE)
# ==========================

# Caminho para o repositório artemis_treeiso
TREEISO_REPO_DIR = "/home/matheuspimenta/Jobs/Eucalipto/artemis_treeiso"

# Diretório contendo os arquivos LAS/LAZ (sem processamento treeiso ainda)
INPUT_DIR = "/path/to/your/las_directory"  # TODO: ajustar

# Campo de ID de árvore/segmento na saída do treeiso
TREE_ID_DIM = "final_segs"

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


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Rodar treeiso nos LAS/LAZ do diretório de entrada
    treeiso_outputs = isolation_treeiso.run_treeiso_on_dir(
        TREEISO_REPO_DIR,
        INPUT_DIR,
    )

    # 2) Construir dicionário árvore -> nuvem de pontos (a partir de final_segs)
    per_tree_points: Dict[str, np.ndarray] = {}

    for out_path in treeiso_outputs:
        points, extras = eio.load_laz(out_path)

        if TREE_ID_DIM not in extras:
            raise ValueError(f"Campo '{TREE_ID_DIM}' não encontrado em {out_path}")

        seg_ids = extras[TREE_ID_DIM]
        unique_ids = np.unique(seg_ids)
        stem = Path(out_path).stem

        for seg_id in unique_ids:
            mask = seg_ids == seg_id
            pts_seg = points[mask]
            if pts_seg.shape[0] < 3:
                continue
            key = f"{stem}_seg{int(seg_id)}"
            per_tree_points[key] = pts_seg

    # 3) Heurística de tronco por árvore
    per_tree_trunks: Dict[str, np.ndarray] = {}
    for tid, pts in per_tree_points.items():
        wood_mask, wood_score, dist_axis = trunk_heuristic.extract_trunk(
            pts,
            **TRUNK_PARAMS,
        )
        trunk_pts = pts[wood_mask]
        if trunk_pts.shape[0] < 3:
            continue
        per_tree_trunks[tid] = trunk_pts

    # 4) Cálculo de DAP (ensemble) e volume (cilindro)
    metrics: Dict[str, dict] = {}
    for tid, trunk_pts in per_tree_trunks.items():
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

    # 5) Escrever CSV resumo
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

    print(f"Resumo salvo em {csv_path}")


if __name__ == "__main__":
    main()
