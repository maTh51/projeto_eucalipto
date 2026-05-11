"""Pipeline baseado em treeiso + segmentação de tronco.

Fluxo:

1. Executa o algoritmo original do treeiso (artemis_treeiso) sobre
   todos os arquivos LAS/LAZ em um diretório de entrada, gerando
   arquivos "*_treeiso.laz" com o campo "final_segs".
2. Usa "final_segs" como identificador de árvore/segmento para separar
   as nuvens por árvore.
3. Extrai tronco em cada árvore, preferencialmente com a rede leaf-wood
    via Docker (fallback: heurística local).
4. Calcula DAP (ensemble) e volume (cilindro) para cada tronco.
5. Salva um CSV resumo em um diretório de resultados.

Uso:

    python run_treeiso_pipeline.py

Edite a seção CONFIG abaixo antes de rodar.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from eucalipto import isolation_treeiso
from eucalipto import trunk_heuristic
from eucalipto import trunk_heuristic_v2
from eucalipto import trunk_heuristic_v2_plus
from eucalipto import trunk_heuristic_v3
from eucalipto import trunk_cleanup
from eucalipto import dbh_methods
from eucalipto import volume_methods
from eucalipto import io as eio
from eucalipto import treeiso_exports
from eucalipto import leafwood_docker
from eucalipto.preprocess_treeiso import prepare_treeiso_input

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


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

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = logging.INFO

# Fonte de segmentação de tronco:
# - "leafwood_docker": usa rede leaf-wood (recomendado)
# - "heuristic": usa heurísticas locais (v1/v2/v2_plus/v3)
TRUNK_SOURCE = "leafwood_docker"

# Estratégia heurística (usada somente quando TRUNK_SOURCE == "heuristic"):
# "v1" (thresholds), "v2" (score-based), "v2_plus" (local percentiles), "v3" (distance-based)
TRUNK_EXTRACTION_METHOD = "v2"

# Configuração da integração leaf-wood via Docker
LEAFWOOD_REPO_DIR = "/home/matheuspimenta/Jobs/Eucalipto/leaf-wood-segmentation-with-deep-learning"
LEAFWOOD_DOCKER_SUBDIR = "docker"
LEAFWOOD_DOCKER_SERVICE = "open3dml"
LEAFWOOD_MODEL_CKPT = "/project/model_weights/weights_randlanet.pth"
LEAFWOOD_DEVICE = "cuda"
LEAFWOOD_BATCH_SIZE = 1
LEAFWOOD_WOOD_LABEL = 1

# Parâmetros da heurística de tronco
# AJUSTADO para dados ULS de grandes árvores (30k-105k pontos/árvore)
TRUNK_PARAMS = {
    "base_percentile": 25.0,       # Pega mais pontos da base (menos topo da copa)
    "k_neighbors": 50,             # Mais vizinhos = mais suavização do ruído ULS
    "linearity_threshold": 0.08,   # EXTREMAMENTE relaxado
    "scattering_threshold": 0.8,   # EXTREMAMENTE relaxado
    "verticality_threshold": 0.60, # Usado apenas para scoring, não hard criterion
    "max_trunk_radius": 1.0,       # Aumentado para 100cm
    "expansion_radius": 0.3,      
    "component_radius": 0.4,      
    "use_verticality_as_hard_criterion": False,  # ← KEY: Verticality is soft, not hard!
}

# Parâmetros específ icos para v2 (score-based)
TRUNK_PARAMS_V2 = {
    "base_percentile": 25.0,
    "k_neighbors": 50,
    "linearity_threshold": 0.08,
    "scattering_threshold": 0.8,
    "max_trunk_radius": 1.0,
    "expansion_radius": 0.3,
    "component_radius": 0.4,
    "score_percentile": 45.0,        # ORIGINAL: top 45% (best for seg 0,3)
    "max_height_trunk_pct": 0.85,
}

# Parâmetros específicos para v2_plus (local percentiles per height band) - NOVO!
TRUNK_PARAMS_V2_PLUS = {
    "base_percentile": 25.0,
    "k_neighbors": 50,
    "linearity_threshold": 0.08,
    "scattering_threshold": 0.8,
    "max_trunk_radius": 1.0,
    "expansion_radius": 0.3,
    "component_radius": 0.4,
    "score_percentile": 25.0,        # Apply per height band: top 25% (more aggressive)
    "max_height_trunk_pct": 0.85,
    "n_height_bands": 10,            # More granular: 10 bands
}

# Parâmetros específicos para v3 (distance-based) - NOVO
TRUNK_PARAMS_V3 = {
    "base_percentile": 25.0,
    "k_neighbors": 50,
    "linearity_threshold": 0.08,
    "scattering_threshold": 0.8,
    "max_trunk_radius": 1.0,         # More permissive: 100cm to capture trunk spread
    "expansion_radius": 0.3,
    "component_radius": 0.4,
    "max_height_trunk_pct": 0.80,    # Remove top 20% (pure foliage)
    "min_density_neighbors": 3,      # Less selective: accept sparser points
}

# Densidade da madeira (kg/m^3) para estimar massa seca (opcional)
WOOD_DENSITY = 900.0

# Método de volume:
# - "cylinder": usa DBH + altura
# - "voxel": conta voxels ocupados no tronco
# - "axis_profile": reconstrói tronco por perfil radial robusto em fatias
VOLUME_METHOD = "axis_profile"

VOLUME_PARAMS_VOXEL = {
    "voxel_size": 0.05,
}

VOLUME_PARAMS_AXIS_PROFILE = {
    "n_slices": 20,
    "slice_thickness": 0.30,
    "radius_percentile": 85.0,
    "min_points_per_slice": 20,
}

# Se True, calcula e reporta os volumes auxiliares (cylinder, voxel e axis_profile)
# independentemente do método selecionado em VOLUME_METHOD.
REPORT_VOLUME_COMPARISON = True

# Exportações extras para validação/apresentação
EXPORT_CLASSIFIED_CLOUD = True
EXPORT_VOLUME_REPRESENTATION = True
EXPORT_TRUNK_DIAGNOSTICS = True  # Novo: exportar diagnósticos JSON por árvore


def main() -> None:
    # Configure logging level
    logging.getLogger().setLevel(LOG_LEVEL)
    trunk_heuristic_logger = logging.getLogger('eucalipto.trunk_heuristic')
    trunk_heuristic_logger.setLevel(LOG_LEVEL)
    
    logger.info("="*70)
    logger.info("PIPELINE TREEISO + SEGMENTAÇÃO DE TRONCO")
    logger.info("="*70)
    logger.info(f"Fonte de tronco: {TRUNK_SOURCE}")
    logger.info(f"Método de volume: {VOLUME_METHOD}")
    if TRUNK_SOURCE == "leafwood_docker":
        logger.info(f"Leaf-wood repo: {LEAFWOOD_REPO_DIR}")
        logger.info(f"Leaf-wood device: {LEAFWOOD_DEVICE}")
        logger.info(f"Leaf-wood checkpoint: {LEAFWOOD_MODEL_CKPT}")
    else:
        logger.info(f"Método heurístico: {TRUNK_EXTRACTION_METHOD.upper()}")
        if TRUNK_EXTRACTION_METHOD == "v3":
            logger.info(f"Max trunk radius: {TRUNK_PARAMS_V3.get('max_trunk_radius', 'N/A')}m")
            logger.info(f"Max height: {TRUNK_PARAMS_V3.get('max_height_trunk_pct', 'N/A')}%")
        elif TRUNK_EXTRACTION_METHOD == "v2_plus":
            logger.info(f"Score percentile: {TRUNK_PARAMS_V2_PLUS.get('score_percentile', 'N/A')}")
            logger.info(f"Height bands: {TRUNK_PARAMS_V2_PLUS.get('n_height_bands', 'N/A')}")
            logger.info(f"Max height: {TRUNK_PARAMS_V2_PLUS.get('max_height_trunk_pct', 'N/A')}")
        else:
            logger.info(f"Score percentile: {TRUNK_PARAMS_V2.get('score_percentile', 'N/A')}")
            logger.info(f"Max height: {TRUNK_PARAMS_V2.get('max_height_trunk_pct', 'N/A')}")
    logger.info("="*70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if PRE_FILTER_ENABLED:
        logger.info(
            f"Pré-filtro habilitado: manter pontos com {PRE_FILTER_DIM} != {PRE_FILTER_GROUND_VALUE}"
        )
    
    logger.info("Preparando entrada para treeiso...")
    input_for_treeiso = prepare_treeiso_input(
        INPUT_DIR,
        pre_filter_enabled=PRE_FILTER_ENABLED,
        pre_filter_dim=PRE_FILTER_DIM,
        pre_filter_ground_value=PRE_FILTER_GROUND_VALUE,
        output_root_dir=RESULTS_DIR,
    )

    # 1) Rodar treeiso nos LAS/LAZ do diretório de entrada
    logger.info(f"Executando treeiso...")
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
    
    logger.info(f"TreeISO processou {len(treeiso_outputs)} arquivo(s)")

    classified_dir = Path(RESULTS_DIR) / "classified_clouds"
    volume_dir = Path(RESULTS_DIR) / "volume_representation"
    diagnostics_dir = Path(RESULTS_DIR) / "trunk_diagnostics" if EXPORT_TRUNK_DIAGNOSTICS else None
    
    classified_dir.mkdir(parents=True, exist_ok=True)
    volume_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_dir:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # 2) Construir métricas por árvore e exportar nuvens classificadas
    metrics: Dict[str, dict] = {}
    diagnostics_by_tree: Dict[str, dict] = {}
    cylinders = []
    
    tree_count = 0
    tree_warnings = 0
    tree_errors = 0

    for out_path in treeiso_outputs:
        logger.info(f"\nProcessando arquivo: {Path(out_path).name}")
        points, extras = eio.load_laz(out_path)

        if TREE_ID_DIM not in extras:
            raise ValueError(f"Campo '{TREE_ID_DIM}' não encontrado em {out_path}")

        seg_ids = np.asarray(extras[TREE_ID_DIM]).astype(np.int32)
        unique_ids = np.unique(seg_ids)
        stem = Path(out_path).stem
        
        logger.info(f"  {len(unique_ids)} segmentos detectados")

        leafwood_by_segment: Dict[int, np.ndarray] = {}
        if TRUNK_SOURCE == "leafwood_docker":
            job_name = f"{stem}_leafwood"
            logger.info("  Rodando inferência leaf-wood via Docker...")
            leafwood_by_segment = leafwood_docker.run_leafwood_for_treeiso_segments(
                points=points,
                seg_ids=seg_ids,
                segment_ids=unique_ids,
                leafwood_repo_dir=LEAFWOOD_REPO_DIR,
                docker_subdir=LEAFWOOD_DOCKER_SUBDIR,
                docker_service=LEAFWOOD_DOCKER_SERVICE,
                model_ckpt=LEAFWOOD_MODEL_CKPT,
                device=LEAFWOOD_DEVICE,
                batch_size=LEAFWOOD_BATCH_SIZE,
                job_name=job_name,
            )
            logger.info(f"  Inferência leaf-wood concluída para {len(leafwood_by_segment)} segmentos")

        trunk_mask_all = np.zeros(points.shape[0], dtype=np.uint8)
        trunk_mask_raw_all = np.zeros(points.shape[0], dtype=np.uint8)
        trunk_mask_clean_all = np.zeros(points.shape[0], dtype=np.uint8)
        outlier_removed_mask_all = np.zeros(points.shape[0], dtype=np.uint8)
        leafwood_pred_all = np.full(points.shape[0], 255, dtype=np.uint8)
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

            tree_count += 1
            tid = f"{stem}_seg{int(seg_id)}"
            
            logger.debug(f"  Árvore {tid}: {pts_seg.shape[0]} pontos")

            if TRUNK_SOURCE == "leafwood_docker":
                pred_labels = leafwood_by_segment.get(int(seg_id))
                if pred_labels is None:
                    logger.warning(f"  {tid}: Sem predição leaf-wood para o segmento")
                    tree_errors += 1
                    continue

                if pred_labels.shape[0] != pts_seg.shape[0]:
                    logger.warning(
                        f"  {tid}: tamanho de predição incompatível "
                        f"({pred_labels.shape[0]} vs {pts_seg.shape[0]})"
                    )
                    tree_errors += 1
                    continue

                wood_mask_raw = pred_labels == LEAFWOOD_WOOD_LABEL
                wood_score = pred_labels.astype(np.float32)
                dist_axis = np.full(pred_labels.shape[0], np.nan, dtype=np.float32)
                leafwood_pred_all[seg_idx] = pred_labels.astype(np.uint8)

                n_trunk = int(np.sum(wood_mask_raw))
                trunk_ratio = n_trunk / max(1, int(pts_seg.shape[0]))
                trunk_diag = {
                    "method": "leafwood_docker",
                    "n_total_points": int(pts_seg.shape[0]),
                    "n_final_trunk": n_trunk,
                    "trunk_percentage": float(100.0 * trunk_ratio),
                    "has_warnings": bool(n_trunk < 3),
                }
            else:
                # Extract trunk with diagnostics (heuristic mode)
                if TRUNK_EXTRACTION_METHOD == "v2":
                    wood_mask, wood_score, dist_axis, trunk_diag = trunk_heuristic_v2.extract_trunk_v2(
                        pts_seg,
                        **TRUNK_PARAMS_V2,
                    )
                elif TRUNK_EXTRACTION_METHOD == "v2_plus":
                    wood_mask, wood_score, dist_axis, trunk_diag = trunk_heuristic_v2_plus.extract_trunk_v2_plus(
                        pts_seg,
                        **TRUNK_PARAMS_V2_PLUS,
                    )
                elif TRUNK_EXTRACTION_METHOD == "v3":
                    wood_mask, wood_score, dist_axis, trunk_diag = trunk_heuristic_v3.extract_trunk_v3(
                        pts_seg,
                        **TRUNK_PARAMS_V3,
                    )
                else:
                    wood_mask, wood_score, dist_axis, trunk_diag = trunk_heuristic.extract_trunk(
                        pts_seg,
                        **TRUNK_PARAMS,
                    )
                wood_mask_raw = wood_mask
            
            trunk_mask_raw_all[seg_idx] = wood_mask_raw.astype(np.uint8)
            wood_score_all[seg_idx] = wood_score.astype(np.float32)
            dist_axis_all[seg_idx] = dist_axis.astype(np.float32)
            
            # Save diagnostics
            diagnostics_by_tree[tid] = {
                "n_total_points": int(pts_seg.shape[0]),
                **trunk_diag,
            }
            
            if trunk_diag.get("has_warnings", False):
                tree_warnings += 1
                logger.warning(f"  ⚠️  {tid}: Possível problema na extração "
                              f"({trunk_diag['n_final_trunk']} pts, "
                              f"{trunk_diag['trunk_percentage']:.1f}%)")

            trunk_pts_raw = pts_seg[wood_mask_raw]
            n_raw = int(trunk_pts_raw.shape[0])
            if n_raw < 3:
                logger.warning(f"  {tid}: Insuficientes pontos de tronco para DBH")
                tree_errors += 1
                continue
            
            # Clean trunk points of outliers before DBH
            trunk_pts, keep_mask_raw = trunk_cleanup.clean_trunk_points(
                trunk_pts_raw,
                return_mask=True,
            )
            n_before_clean = n_raw
            n_after_clean = len(trunk_pts)
            n_removed = n_before_clean - n_after_clean
            if n_after_clean < 20:
                logger.warning(f"  {tid}: Tronco ficou muito pequeno após limpeza ({n_after_clean} pts)")
                tree_errors += 1
                continue

            raw_idx_local = np.where(wood_mask_raw)[0]
            clean_mask_seg = np.zeros(pts_seg.shape[0], dtype=bool)
            clean_mask_seg[raw_idx_local[keep_mask_raw]] = True
            removed_mask_seg = wood_mask_raw & (~clean_mask_seg)

            trunk_mask_clean_all[seg_idx] = clean_mask_seg.astype(np.uint8)
            outlier_removed_mask_all[seg_idx] = removed_mask_seg.astype(np.uint8)
            trunk_mask_all[seg_idx] = clean_mask_seg.astype(np.uint8)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Limpeza de tronco: {n_before_clean} → {n_after_clean} pts")

            # Calculate DBH
            dbh_cm, dbh_info = dbh_methods.estimate_dbh(trunk_pts, method="ensemble")
            if dbh_cm is None:
                logger.warning(f"  {tid}: Falha no cálculo de DBH")
                metrics[tid] = {
                    "dbh_cm": None,
                    "volume_m3": None,
                    "dbh_info": dbh_info,
                    "height_m": eio.tree_height(trunk_pts),
                    "n_points_segment": int(pts_seg.shape[0]),
                    "n_points_trunk_raw": int(n_before_clean),
                    "n_points_trunk_clean": int(n_after_clean),
                    "n_points_outliers_removed": int(n_removed),
                    "outliers_removed_pct": float(100.0 * n_removed / max(1, n_before_clean)),
                    "volume_method_selected": VOLUME_METHOD,
                    "volume_cylinder_m3": np.nan,
                    "volume_voxel_m3": np.nan,
                    "volume_axis_profile_m3": np.nan,
                    "volume_delta_axis_minus_cylinder_m3": np.nan,
                    "volume_delta_axis_minus_cylinder_pct": np.nan,
                }
                tree_errors += 1
                continue

            # Calculate volume
            height_m = eio.tree_height(trunk_pts)
            vol_cylinder_info = None
            vol_voxel_info = None
            vol_axis_info = None

            try:
                vol_cylinder_info = volume_methods.estimate_volume(
                    trunk_pts,
                    dbh_cm=dbh_cm,
                    height_m=height_m,
                    method="cylinder",
                    wood_density_kg_m3=WOOD_DENSITY,
                )
            except Exception as exc:
                logger.warning(f"  {tid}: Falha volume cylinder: {exc}")

            try:
                vol_voxel_info = volume_methods.estimate_volume(
                    trunk_pts,
                    method="voxel",
                    wood_density_kg_m3=WOOD_DENSITY,
                    **VOLUME_PARAMS_VOXEL,
                )
            except Exception as exc:
                logger.warning(f"  {tid}: Falha volume voxel: {exc}")

            try:
                vol_axis_info = volume_methods.estimate_volume(
                    trunk_pts,
                    method="axis_profile",
                    wood_density_kg_m3=WOOD_DENSITY,
                    **VOLUME_PARAMS_AXIS_PROFILE,
                )
            except Exception as exc:
                logger.warning(f"  {tid}: Falha volume axis_profile: {exc}")

            if VOLUME_METHOD == "cylinder":
                vol_info = vol_cylinder_info if vol_cylinder_info is not None else vol_axis_info
            elif VOLUME_METHOD == "voxel":
                vol_info = vol_voxel_info if vol_voxel_info is not None else vol_axis_info
            elif VOLUME_METHOD == "axis_profile":
                vol_info = vol_axis_info if vol_axis_info is not None else vol_cylinder_info
            else:
                raise ValueError(f"Método de volume não suportado no pipeline: {VOLUME_METHOD}")

            if vol_info is None:
                logger.warning(f"  {tid}: Falha no cálculo de volume em todos os métodos")
                tree_errors += 1
                continue

            if vol_info.get("height_m") is None:
                vol_info["height_m"] = float(height_m)

            vol_cyl_m3 = (
                float(vol_cylinder_info.get("volume_m3"))
                if vol_cylinder_info and vol_cylinder_info.get("volume_m3") is not None
                else np.nan
            )
            vol_voxel_m3 = (
                float(vol_voxel_info.get("volume_m3"))
                if vol_voxel_info and vol_voxel_info.get("volume_m3") is not None
                else np.nan
            )
            vol_axis_m3 = (
                float(vol_axis_info.get("volume_m3"))
                if vol_axis_info and vol_axis_info.get("volume_m3") is not None
                else np.nan
            )

            if np.isfinite(vol_cyl_m3) and np.isfinite(vol_axis_m3) and vol_cyl_m3 > 0:
                vol_delta_m3 = float(vol_axis_m3 - vol_cyl_m3)
                vol_delta_pct = float(100.0 * vol_delta_m3 / vol_cyl_m3)
            else:
                vol_delta_m3 = np.nan
                vol_delta_pct = np.nan

            res = {
                "dbh_cm": dbh_cm,
                "dbh_info": dbh_info,
                "height_m": height_m,
                "n_points_segment": int(pts_seg.shape[0]),
                "n_points_trunk_raw": int(n_before_clean),
                "n_points_trunk_clean": int(n_after_clean),
                "n_points_outliers_removed": int(n_removed),
                "outliers_removed_pct": float(100.0 * n_removed / max(1, n_before_clean)),
                "volume_method_selected": VOLUME_METHOD,
                "volume_cylinder_m3": vol_cyl_m3,
                "volume_voxel_m3": vol_voxel_m3,
                "volume_axis_profile_m3": vol_axis_m3,
                "volume_delta_axis_minus_cylinder_m3": vol_delta_m3,
                "volume_delta_axis_minus_cylinder_pct": vol_delta_pct,
            }
            res.update(vol_info)
            metrics[tid] = res

            diagnostics_by_tree[tid].update({
                "n_points_trunk_raw": int(n_before_clean),
                "n_points_trunk_clean": int(n_after_clean),
                "n_points_outliers_removed": int(n_removed),
                "outliers_removed_pct": float(100.0 * n_removed / max(1, n_before_clean)),
                "volume_cylinder_m3": None if not np.isfinite(vol_cyl_m3) else float(vol_cyl_m3),
                "volume_voxel_m3": None if not np.isfinite(vol_voxel_m3) else float(vol_voxel_m3),
                "volume_axis_profile_m3": None if not np.isfinite(vol_axis_m3) else float(vol_axis_m3),
                "volume_method_selected": VOLUME_METHOD,
            })
            
            logger.info(f"  ✓ {tid}: DBH={dbh_cm:.1f}cm, Vol={vol_info.get('volume_m3', 0):.3f}m³")

            dbh_cm_tree_all[seg_idx] = np.float32(dbh_cm)
            volume_m3_tree_all[seg_idx] = np.float32(vol_info.get("volume_m3", np.nan))

            if dbh_cm is not None:
                radius_m = float(dbh_cm) / 200.0
            else:
                vol_h = float(vol_info.get("height_m", height_m) or height_m)
                vol_v = float(vol_info.get("volume_m3", np.nan))
                if vol_h > 0 and np.isfinite(vol_v) and vol_v >= 0:
                    radius_m = float(np.sqrt(vol_v / (np.pi * vol_h)))
                else:
                    radius_m = float("nan")
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
                extra_fields={
                    "leafwood_pred": leafwood_pred_all,
                    "trunk_mask_raw": trunk_mask_raw_all,
                    "trunk_mask_clean": trunk_mask_clean_all,
                    "trunk_outlier_removed": outlier_removed_mask_all,
                },
            )
            logger.info(f"  Nuvem classificada: {classified_path}")

    # 3) Save diagnostics JSON
    if EXPORT_TRUNK_DIAGNOSTICS:
        diag_path = diagnostics_dir / "trunk_extraction_diagnostics.json"
        with diag_path.open("w") as f:
            json.dump(diagnostics_by_tree, f, indent=2)
        logger.info(f"Diagnósticos salvos em {diag_path}")

    # 4) Escrever CSV resumo
    csv_path = Path(RESULTS_DIR) / "treeiso_metrics_summary.csv"
    fieldnames = [
        "tree_id",
        "n_points_segment",
        "n_points_trunk_raw",
        "n_points_trunk_clean",
        "n_points_outliers_removed",
        "outliers_removed_pct",
        "dbh_cm",
        "height_m",
        "volume_method_selected",
        "volume_m3",
        "volume_cylinder_m3",
        "volume_axis_profile_m3",
        "volume_delta_axis_minus_cylinder_m3",
        "volume_delta_axis_minus_cylinder_pct",
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
                "n_points_segment": res.get("n_points_segment"),
                "n_points_trunk_raw": res.get("n_points_trunk_raw"),
                "n_points_trunk_clean": res.get("n_points_trunk_clean"),
                "n_points_outliers_removed": res.get("n_points_outliers_removed"),
                "outliers_removed_pct": res.get("outliers_removed_pct"),
                "dbh_cm": res.get("dbh_cm"),
                "height_m": res.get("height_m"),
                "volume_method_selected": res.get("volume_method_selected"),
                "volume_m3": res.get("volume_m3"),
                "volume_cylinder_m3": res.get("volume_cylinder_m3"),
                "volume_axis_profile_m3": res.get("volume_axis_profile_m3"),
                "volume_delta_axis_minus_cylinder_m3": res.get("volume_delta_axis_minus_cylinder_m3"),
                "volume_delta_axis_minus_cylinder_pct": res.get("volume_delta_axis_minus_cylinder_pct"),
                "volume_liters": res.get("volume_liters"),
                "mass_kg": res.get("mass_kg"),
                "dbh_method": res.get("dbh_info", {}).get("method"),
            }
            writer.writerow(row)

    if REPORT_VOLUME_COMPARISON:
        volume_cmp_path = Path(RESULTS_DIR) / "treeiso_volume_comparison.csv"
        cmp_fields = [
            "tree_id",
            "volume_cylinder_m3",
            "volume_axis_profile_m3",
            "volume_delta_axis_minus_cylinder_m3",
            "volume_delta_axis_minus_cylinder_pct",
        ]
        with volume_cmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cmp_fields)
            writer.writeheader()
            for tid, res in sorted(metrics.items()):
                writer.writerow({
                    "tree_id": tid,
                    "volume_cylinder_m3": res.get("volume_cylinder_m3"),
                    "volume_axis_profile_m3": res.get("volume_axis_profile_m3"),
                    "volume_delta_axis_minus_cylinder_m3": res.get("volume_delta_axis_minus_cylinder_m3"),
                    "volume_delta_axis_minus_cylinder_pct": res.get("volume_delta_axis_minus_cylinder_pct"),
                })
        logger.info(f"Comparação de volumes salva em: {volume_cmp_path}")

    if EXPORT_VOLUME_REPRESENTATION:
        cyl_csv = volume_dir / "treeiso_volume_cylinders.csv"
        cyl_ply = volume_dir / "treeiso_volume_cylinders.ply"
        treeiso_exports.save_cylinder_primitives_csv(str(cyl_csv), cylinders)
        treeiso_exports.save_cylinder_representation_ply(str(cyl_ply), cylinders)
        logger.info(f"Representação de volume: {cyl_csv}, {cyl_ply}")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info("="*70)
    logger.info(f"Total de árvores processadas: {tree_count}")
    logger.info(f"Sucessos: {len(metrics) - tree_errors}")
    logger.info(f"Warnings: {tree_warnings}")
    logger.info(f"Erros: {tree_errors}")
    logger.info(f"Resumo salvo em: {csv_path}")
    logger.info("="*70)



if __name__ == "__main__":
    main()
