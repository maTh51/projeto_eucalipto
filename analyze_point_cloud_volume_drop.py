#!/usr/bin/env python3
"""Diagnose why point-cloud volume runs produce fewer trees than expected.

This script reports three counts that usually explain the drop:

1. Trees present in the input cloud by `tree_id`.
2. Trees that survive the trunk-point filter used by `compare_method_combinations`.
3. Result rows in the generated CSV, including method-combination expansion.

Typical usage:
    python analyze_point_cloud_volume_drop.py \
        --cloud /path/to/cloud.ply \
        --results-csv point_cloud_volume_results.csv

If you only have the CSV, pass `--results-csv` and the script will still
summarize the method-combination structure of the output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
from typing import Sequence


DEFAULT_DBH_METHODS = ["ensemble", "single_ransac", "ls"]
DEFAULT_VOLUME_METHODS = ["cylinder", "voxel", "taper", "frustum", "axis_profile"]


DEFAULT_TREE_ID_CANDIDATES = ["tree_id", "instance_pred", "treeID", "final_segs"]
DEFAULT_TRUNK_CANDIDATES = ["trunk_leaf_label", "semantic_seg", "semantic_pred", "leafwood_pred"]
DEFAULT_TRUNK_LABEL = 1


def resolve_column(extras: dict[str, object], candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in extras:
            return candidate
    available = ", ".join(sorted(extras.keys()))
    raise KeyError(f"No {label} column found. Tried {list(candidates)}. Available: {available}")


def load_cloud(path: Path):
    import numpy as np

    ext = path.suffix.lower()
    if ext == ".ply":
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError("plyfile is required to read .ply clouds") from exc

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        points = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T.astype(np.float64)
        extras = {}
        for name in vertex.data.dtype.names:
            if name in {"x", "y", "z"}:
                continue
            extras[name] = np.asarray(vertex[name])
        return points, extras
    if ext in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError("laspy is required to read .las/.laz clouds") from exc

        las = laspy.read(str(path))
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        extras = {}
        for dim in las.point_format.extra_dimension_names:
            extras[dim] = np.asarray(getattr(las, dim))
        for dim in ("classification", "intensity"):
            if hasattr(las, dim):
                extras[dim] = np.asarray(getattr(las, dim))
        return points, extras
    raise ValueError(f"Unsupported cloud format: {path}")


def summarize_cloud(
    cloud_path: Path,
    tree_id_candidates: Sequence[str],
    trunk_candidates: Sequence[str],
    trunk_label: int,
) -> list[dict[str, object]]:
    import numpy as np

    points, extras = load_cloud(cloud_path)
    tree_id_col = resolve_column(extras, tree_id_candidates, "tree id")
    trunk_col = resolve_column(extras, trunk_candidates, "trunk/leaf")

    tree_ids = np.asarray(extras[tree_id_col]).astype(np.int64)
    trunk_flags = np.asarray(extras[trunk_col]).astype(np.int64)

    rows: list[dict[str, object]] = []
    for tree_id in sorted(np.unique(tree_ids)):
        mask_tree = tree_ids == tree_id
        mask_trunk = mask_tree & (trunk_flags == trunk_label)

        rows.append(
            {
                "tree_id": int(tree_id),
                "total_points": int(mask_tree.sum()),
                "trunk_points": int(mask_trunk.sum()),
                "has_enough_trunk_points": bool(mask_trunk.sum() >= 3),
            }
        )

    return rows


def summarize_results(results_csv: Path) -> None:
    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print(f"Results CSV is empty: {results_csv}")
        return

    print("RESULT CSV SUMMARY")
    print(f"  rows: {len(rows)}")
    print(f"  unique tree_id: {len({row['tree_id'] for row in rows})}")
    print(f"  expected combinations per tree: {len(DEFAULT_DBH_METHODS) * len(DEFAULT_VOLUME_METHODS)}")
    print(f"  dbh methods: {list(DEFAULT_DBH_METHODS)}")
    print(f"  volume methods: {list(DEFAULT_VOLUME_METHODS)}")

    status_counts = Counter(row["status"] for row in rows)
    print("  status counts:")
    for status, count in status_counts.most_common():
        print(f"    {status}: {count}")

    combo_per_tree = Counter(row["tree_id"] for row in rows)
    counts = list(combo_per_tree.values())
    print(f"  combinations per tree: min={min(counts)}, max={max(counts)}")
    print(f"  trees with 15 rows: {sum(1 for count in counts if count == 15)}")
    print(f"  trees with <15 rows: {sum(1 for count in counts if count < 15)}")


def summarize_cloud_and_compare(cloud_df: list[dict[str, object]], results_csv: Path | None) -> None:
    total_trees = len(cloud_df)
    trunk_ok = sum(1 for row in cloud_df if row["has_enough_trunk_points"])
    trunk_fail = total_trees - trunk_ok

    print("CLOUD SUMMARY")
    print(f"  trees in cloud: {total_trees}")
    print(f"  trees with >=3 trunk points: {trunk_ok}")
    print(f"  trees filtered out by trunk-point rule: {trunk_fail}")
    print("  trunk-point distribution:")
    counts = Counter(row["trunk_points"] for row in cloud_df)
    for trunk_points, count in counts.most_common(10):
        print(f"    {trunk_points} trunk points: {count} trees")

    if results_csv is not None and results_csv.exists():
        print()
        summarize_results(results_csv)

        expected_rows = trunk_ok * len(DEFAULT_DBH_METHODS) * len(DEFAULT_VOLUME_METHODS)
        print()
        print("CROSS-CHECK")
        print(f"  expected rows from surviving trees only: {expected_rows}")
        with results_csv.open(newline="") as handle:
            observed_rows = sum(1 for _ in csv.DictReader(handle))
        print(f"  observed rows in CSV: {observed_rows}")
        print(
            "  note: the CSV may include error rows for method failures, "
            "but trees with fewer than 3 trunk points never enter the combination loop."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose where point-cloud trees are dropped before volume combinations are generated."
    )
    parser.add_argument("--cloud", type=Path, help="Input cloud used by FF3D/TreeISO (.ply/.las/.laz)")
    parser.add_argument("--results-csv", type=Path, help="CSV generated by the volume notebook")
    parser.add_argument(
        "--tree-id-candidate",
        action="append",
        dest="tree_id_candidates",
        help="Candidate column name for tree IDs. Repeat to add more candidates.",
    )
    parser.add_argument(
        "--trunk-candidate",
        action="append",
        dest="trunk_candidates",
        help="Candidate column name for trunk/leaf labels. Repeat to add more candidates.",
    )
    parser.add_argument(
        "--trunk-label",
        type=int,
        default=DEFAULT_TRUNK_LABEL,
        help="Label value that identifies trunk points (default: 1)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cloud is None and args.results_csv is None:
        parser.error("provide at least --cloud or --results-csv")

    tree_id_candidates = args.tree_id_candidates or DEFAULT_TREE_ID_CANDIDATES
    trunk_candidates = args.trunk_candidates or DEFAULT_TRUNK_CANDIDATES

    if args.cloud is not None:
        if not args.cloud.exists():
            raise FileNotFoundError(f"Cloud not found: {args.cloud}")
        cloud_df = summarize_cloud(args.cloud, tree_id_candidates, trunk_candidates, args.trunk_label)
        summarize_cloud_and_compare(cloud_df, args.results_csv)

    if args.cloud is None and args.results_csv is not None:
        summarize_results(args.results_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())