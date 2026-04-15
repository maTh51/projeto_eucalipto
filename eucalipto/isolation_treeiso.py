"""Integração com treeiso (artemis_treeiso).

Fornece duas funcionalidades principais:

1) Executar o algoritmo original do treeiso em um diretório de
    arquivos LAS/LAZ, chamando ``process_las_file`` do próprio
    repositório artemis_treeiso.
2) A partir de um LAS/LAZ que já possua um identificador de árvore
    (por exemplo, ``treeID`` ou ``final_segs``), separar a nuvem em
    uma nuvem por árvore/segmento.
"""

from __future__ import annotations

from typing import Dict, List

import glob
import importlib
import os
import sys

import numpy as np

from .io import load_laz


def split_by_tree_id(path: str,
                     tree_id_dim: str = "treeID") -> Dict[int, np.ndarray]:
    """Split an input LAS/LAZ into per-tree point clouds using tree IDs.

    Parameters
    ----------
    path : str
        LAS/LAZ file path.
    tree_id_dim : str
        Name of the dimension holding tree identifiers.

    Returns
    -------
    dict
        Mapping tree_id -> (N_i, 3) array of points.
    """
    points, extras = load_laz(path)

    if tree_id_dim not in extras:
        raise ValueError(f"Dimension '{tree_id_dim}' not found in file {path}")

    tree_ids = extras[tree_id_dim]
    per_tree: Dict[int, list] = {}

    for p, tid in zip(points, tree_ids):
        per_tree.setdefault(int(tid), []).append(p)

    return {tid: np.vstack(pts) for tid, pts in per_tree.items() if len(pts) > 0}


def run_treeiso_on_dir(treeiso_repo_dir: str,
                       input_dir: str,
                       force_python_cut_pursuit: bool = False) -> List[str]:
    """Executa o treeiso sobre todos os LAS/LAZ de um diretório.

    Esta função insere o diretório PythonCpp do repositório
    artemis_treeiso no ``sys.path`` e importa o módulo ``treeiso``
    original, chamando ``process_las_file`` para cada arquivo
    ``*.las``/``*.laz`` encontrado em ``input_dir``.

    Retorna a lista de caminhos dos arquivos gerados com sufixo
    ``*_treeiso.laz``.
    """

    pythoncpp_dir = os.path.join(treeiso_repo_dir, "PythonCpp")
    if pythoncpp_dir not in sys.path:
        sys.path.insert(0, pythoncpp_dir)

    treeiso_mod = importlib.import_module("treeiso")

    if force_python_cut_pursuit:
        try:
            cut_pursuit_py_mod = importlib.import_module("cut_pursuit_L0")
            treeiso_mod.perform_cut_pursuit = cut_pursuit_py_mod.perform_cut_pursuit
            if hasattr(treeiso_mod, "USE_CPP"):
                treeiso_mod.USE_CPP = False
            print("Forçando backend Python do cut-pursuit (cut_pursuit_L0).")
        except Exception as exc:
            raise RuntimeError(
                "Falha ao forçar backend Python do cut-pursuit. "
                "Verifique se cut_pursuit_L0.py está disponível no diretório PythonCpp."
            ) from exc

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input path not found: {input_dir}")

    if os.path.isdir(input_dir):
        pattern = os.path.join(input_dir, "*.la[sz]")
        input_paths = sorted(glob.glob(pattern))
    else:
        ext = os.path.splitext(input_dir)[1].lower()
        if ext not in {".las", ".laz"}:
            raise ValueError(
                "Input path must be a directory with LAS/LAZ files or a single .las/.laz file. "
                f"Received: {input_dir}"
            )
        input_paths = [input_dir]

    if len(input_paths) == 0:
        print(f"No LAS/LAZ files found in input path: {input_dir}")
        return []

    output_paths: List[str] = []
    for path_to_las in input_paths:
        treeiso_mod.process_las_file(path_to_las)
        out_path = path_to_las[:-4] + "_treeiso.laz"
        if os.path.exists(out_path):
            output_paths.append(out_path)

    return output_paths
