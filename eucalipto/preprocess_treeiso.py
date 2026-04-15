"""Preprocessing helpers for treeiso input preparation.

Handles three input modes:
- directory with LAS/LAZ files
- single LAS/LAZ file
- single PLY file (converted to LAZ)

Optionally applies a pre-segmentation ground filter before treeiso.
"""

from __future__ import annotations

import glob
import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import io as eio


def _preprocessed_dir(output_root_dir: str) -> str:
    out_dir = os.path.join(output_root_dir, "preprocessed_inputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _append_report(output_root_dir: str, line: str) -> None:
    out_dir = _preprocessed_dir(output_root_dir)
    report_path = os.path.join(out_dir, "preprocess_report.txt")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _class_distribution(values: np.ndarray, max_items: int = 20) -> str:
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    pairs = []
    for idx in order[:max_items]:
        pairs.append(f"{unique[idx]}:{counts[idx]}")
    return ", ".join(pairs)


def _report_filter_stats(source: str,
                         total_points: int,
                         kept_points: int,
                         pre_filter_dim: str,
                         pre_filter_ground_value: int,
                         values: np.ndarray,
                         output_root_dir: str) -> None:
    removed_points = total_points - kept_points
    keep_ratio = kept_points / total_points if total_points > 0 else 0.0
    dist = _class_distribution(values)
    line = (
        f"source={source} | field={pre_filter_dim} | ground={pre_filter_ground_value} | "
        f"total={total_points} | kept={kept_points} | removed={removed_points} | "
        f"keep_ratio={keep_ratio:.6f} | top_counts={dist}"
    )
    print("Resumo pre-filtro:", line)
    _append_report(output_root_dir, line)


def convert_ply_to_laz(input_ply: str,
                       pre_filter_enabled: bool = False,
                       pre_filter_dim: str = "treeID",
                       pre_filter_ground_value: int = 0,
                       output_root_dir: str = "results_treeiso") -> str:
    """Convert a .ply file to .laz, optionally filtering ground points."""
    try:
        plyfile_mod = importlib.import_module("plyfile")
    except ImportError as exc:
        raise ImportError(
            "Dependência 'plyfile' não encontrada. Instale com: pip install plyfile"
        ) from exc

    ply_data = plyfile_mod.PlyData.read(input_ply)

    if "vertex" not in ply_data:
        raise ValueError(f"Elemento 'vertex' não encontrado em {input_ply}")

    vertex = ply_data["vertex"].data
    required_fields = {"x", "y", "z"}
    if not required_fields.issubset(vertex.dtype.names or ()):  # pragma: no cover
        raise ValueError(f"PLY sem colunas x,y,z: {input_ply}")

    points = np.column_stack([
        np.asarray(vertex["x"], dtype=np.float64),
        np.asarray(vertex["y"], dtype=np.float64),
        np.asarray(vertex["z"], dtype=np.float64),
    ])

    if pre_filter_enabled:
        names = set(vertex.dtype.names or ())
        if pre_filter_dim not in names:
            raise ValueError(
                f"Campo de pré-segmentação '{pre_filter_dim}' não encontrado no PLY: {input_ply}"
            )

        preseg = np.asarray(vertex[pre_filter_dim])
        keep_mask = preseg != pre_filter_ground_value
        kept_points = int(np.count_nonzero(keep_mask))
        total_points = int(len(keep_mask))
        if kept_points == 0:
            raise ValueError(
                "Filtro de chão removeu todos os pontos. "
                f"Campo={pre_filter_dim}, valor_chao={pre_filter_ground_value}"
            )

        points = points[keep_mask]
        print(
            f"Pré-filtro aplicado em PLY ({pre_filter_dim}!={pre_filter_ground_value}): "
            f"{kept_points}/{len(keep_mask)} pontos mantidos"
        )
        _report_filter_stats(
            source=input_ply,
            total_points=total_points,
            kept_points=kept_points,
            pre_filter_dim=pre_filter_dim,
            pre_filter_ground_value=pre_filter_ground_value,
            values=preseg,
            output_root_dir=output_root_dir,
        )

    out_dir = _preprocessed_dir(output_root_dir)
    output_laz = os.path.join(out_dir, Path(input_ply).stem + "_preprocessed.laz")
    eio.save_laz(output_laz, points)
    return output_laz


def filter_laz_by_preseg(input_laz: str,
                         pre_filter_dim: str,
                         pre_filter_ground_value: int,
                         output_root_dir: str = "results_treeiso") -> str:
    """Filter ground points from a single LAS/LAZ using a pre-seg field."""
    points, extras = eio.load_laz(input_laz)

    if pre_filter_dim not in extras:
        raise ValueError(
            f"Campo de pré-segmentação '{pre_filter_dim}' não encontrado em {input_laz}"
        )

    preseg = np.asarray(extras[pre_filter_dim])
    keep_mask = preseg != pre_filter_ground_value
    kept_points = int(np.count_nonzero(keep_mask))
    total_points = int(len(keep_mask))
    if kept_points == 0:
        raise ValueError(
            "Filtro de chão removeu todos os pontos. "
            f"Campo={pre_filter_dim}, valor_chao={pre_filter_ground_value}"
        )

    _report_filter_stats(
        source=input_laz,
        total_points=total_points,
        kept_points=kept_points,
        pre_filter_dim=pre_filter_dim,
        pre_filter_ground_value=pre_filter_ground_value,
        values=preseg,
        output_root_dir=output_root_dir,
    )

    out_dir = _preprocessed_dir(output_root_dir)
    output_laz = os.path.join(out_dir, Path(input_laz).stem + "_preprocessed.laz")
    eio.save_laz(output_laz, points[keep_mask])
    print(
        f"Pré-filtro aplicado em LAS/LAZ ({pre_filter_dim}!={pre_filter_ground_value}): "
        f"{kept_points}/{len(keep_mask)} pontos mantidos"
    )
    return output_laz


def filter_laz_dir_by_preseg(input_dir: str,
                             pre_filter_dim: str,
                             pre_filter_ground_value: int,
                             output_root_dir: str) -> str:
    """Filter ground points from all LAS/LAZ in a directory."""
    patterns = [os.path.join(input_dir, "*.las"), os.path.join(input_dir, "*.laz")]
    input_paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if len(input_paths) == 0:
        return input_dir

    out_dir = _preprocessed_dir(output_root_dir)

    for src in input_paths:
        points, extras = eio.load_laz(src)
        if pre_filter_dim not in extras:
            raise ValueError(
                f"Campo de pré-segmentação '{pre_filter_dim}' não encontrado em {src}"
            )

        preseg = np.asarray(extras[pre_filter_dim])
        keep_mask = preseg != pre_filter_ground_value
        kept_points = int(np.count_nonzero(keep_mask))
        total_points = int(len(keep_mask))
        if kept_points == 0:
            raise ValueError(
                "Filtro de chão removeu todos os pontos. "
                f"Campo={pre_filter_dim}, valor_chao={pre_filter_ground_value}, arquivo={src}"
            )

        _report_filter_stats(
            source=src,
            total_points=total_points,
            kept_points=kept_points,
            pre_filter_dim=pre_filter_dim,
            pre_filter_ground_value=pre_filter_ground_value,
            values=preseg,
            output_root_dir=output_root_dir,
        )

        dst = os.path.join(out_dir, Path(src).stem + "_preprocessed.laz")
        eio.save_laz(dst, points[keep_mask])
        print(
            f"Pré-filtro em {Path(src).name}: {kept_points}/{len(keep_mask)} pontos mantidos"
        )

    return out_dir


def prepare_treeiso_input(input_path: str,
                          pre_filter_enabled: bool = False,
                          pre_filter_dim: str = "treeID",
                          pre_filter_ground_value: int = 0,
                          output_root_dir: str = "results_treeiso") -> str:
    """Normalize treeiso input to a directory or single LAS/LAZ path."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"INPUT_DIR não encontrado: {input_path}")

    if os.path.isdir(input_path):
        if pre_filter_enabled:
            return filter_laz_dir_by_preseg(
                input_dir=input_path,
                pre_filter_dim=pre_filter_dim,
                pre_filter_ground_value=pre_filter_ground_value,
                output_root_dir=output_root_dir,
            )
        return input_path

    ext = Path(input_path).suffix.lower()
    if ext in {".las", ".laz"}:
        if pre_filter_enabled:
            return filter_laz_by_preseg(
                input_laz=input_path,
                pre_filter_dim=pre_filter_dim,
                pre_filter_ground_value=pre_filter_ground_value,
                output_root_dir=output_root_dir,
            )
        return input_path

    if ext == ".ply":
        converted_path = convert_ply_to_laz(
            input_ply=input_path,
            pre_filter_enabled=pre_filter_enabled,
            pre_filter_dim=pre_filter_dim,
            pre_filter_ground_value=pre_filter_ground_value,
            output_root_dir=output_root_dir,
        )
        print(f"PLY convertido para LAZ: {converted_path}")
        return converted_path

    raise ValueError(
        "INPUT_DIR deve ser diretório, arquivo .las/.laz, ou arquivo .ply. "
        f"Recebido: {input_path}"
    )
