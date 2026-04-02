"""Tree isolation helpers for treeiso (artemis_treeiso).

At this stage we do not call the full treeiso framework directly.
Instead, this module focuses on splitting an input cloud by an
existing tree identifier dimension (e.g. treeID from FOR-Instance).

Integration with artemis_treeiso's algorithms can be added here later
when a stable programmatic API or CLI contract is defined.
"""

from __future__ import annotations

from typing import Dict

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
