#!/usr/bin/env python3
"""
Count trees in an instance-segmented point cloud (LAZ/LAS with a `predInstance`
extra attribute) and optionally cut the cloud to keep only trees below (or
above) a size threshold.

Requires:  pip install "laspy[lazrs]" numpy

Size metrics (--metric):
  height  : z-extent of the instance (z_max - z_min)                   [default]
  crown   : largest XY extent of the instance (max of x-range, y-range)
  dbh     : diameter from a circle fit on a thin horizontal slice at
            breast height (--dbh-height above the tree base)

The threshold is given in centimetres (--threshold). The cloud coordinates are
assumed to be in metres (use --cloud-unit cm|m|mm otherwise).

Output files use the same format as the input unless -o has a different
extension (.laz = compressed, .las = uncompressed).

Examples
--------
# Just count trees and write a per-tree CSV
python3 count_and_filter_trees.py -i result.laz --csv trees.csv --no-write

# Keep only trees with height < 10 cm
python3 count_and_filter_trees.py -i result.laz -o result_small.laz --threshold 10

# Same, and also save each small tree as its own file
python3 count_and_filter_trees.py -i result.laz -o result_small.laz --threshold 10 --per-tree-dir small_trees/

# Keep only trees with DBH < 10 cm
python3 count_and_filter_trees.py -i result.laz -o result_thin.laz --metric dbh --threshold 10

# Keep trees LARGER than the threshold instead
python3 count_and_filter_trees.py -i result.laz -o out.laz --threshold 10 --keep larger
"""

import argparse
import csv
import sys
from pathlib import Path

import laspy
import numpy as np

UNIT_TO_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1}


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Input LAZ/LAS with the instance attribute")
    p.add_argument("-o", "--output", default=None, help="Output LAZ/LAS (default: <input>_filtered.<ext>)")
    p.add_argument("--no-write", action="store_true", help="Only count/report, do not write a filtered file")
    p.add_argument("--csv", default=None, help="Write per-tree statistics to this CSV file")
    p.add_argument("--per-tree-dir", default=None,
                   help="Also save one file per kept tree in this folder (<input stem>_tree_<id>.<ext>)")

    p.add_argument("--attr", default="PredInstance", help="Instance attribute name (default: predInstance)")
    p.add_argument("--background-ids", type=int, nargs="*", default=[0],
                   help="Instance ids that are NOT trees (default: 0). Use '--background-ids' with no values to disable.")
    p.add_argument("--min-points", type=int, default=1,
                   help="Ignore instances with fewer points than this when counting trees (default: 1)")

    p.add_argument("--metric", choices=["height", "crown", "dbh"], default="height", help="Size metric (default: height)")
    p.add_argument("--threshold", type=float, default=None, help="Size threshold in cm. Required for filtering")
    p.add_argument("--keep", choices=["smaller", "larger"], default="smaller",
                   help="Keep trees smaller (default) or larger than the threshold")
    p.add_argument("--cloud-unit", choices=list(UNIT_TO_CM), default="m", help="Units of the cloud coordinates (default: m)")

    p.add_argument("--dbh-height", type=float, default=1.3, help="Breast height above tree base, in metres (default: 1.3)")
    p.add_argument("--dbh-slice", type=float, default=0.10, help="Slice thickness for DBH fit, in metres (default: 0.10)")
    p.add_argument("--dbh-min-points", type=int, default=10, help="Minimum points in the slice to fit DBH (default: 10)")
    return p.parse_args()


# --------------------------------------------------------------------------- geometry
def fit_circle_diameter(x, y):
    """Algebraic (Kasa) circle fit. Returns diameter, or NaN if degenerate."""
    x0, y0 = x.mean(), y.mean()
    xs, ys = x - x0, y - y0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    cx, cy, c = sol[0] / 2, sol[1] / 2, sol[2]
    r2 = c + cx ** 2 + cy ** 2
    return 2 * np.sqrt(r2) if r2 > 0 else np.nan


def per_tree_stats(xyz, labels, args):
    """Group points by instance id and compute per-tree statistics."""
    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    xs, ys, zs = xyz[order, 0], xyz[order, 1], xyz[order, 2]
    zmin = np.minimum.reduceat(zs, starts)
    zmax = np.maximum.reduceat(zs, starts)
    xrng = np.maximum.reduceat(xs, starts) - np.minimum.reduceat(xs, starts)
    yrng = np.maximum.reduceat(ys, starts) - np.minimum.reduceat(ys, starts)
    cx = np.add.reduceat(xs, starts) / counts
    cy = np.add.reduceat(ys, starts) / counts

    stats = {
        "id": uniq,
        "n_points": counts,
        "z_min": zmin,
        "z_max": zmax,
        "height": zmax - zmin,
        "crown": np.maximum(xrng, yrng),
        "x": cx,
        "y": cy,
        "dbh": np.full(len(uniq), np.nan),
    }

    if args.metric == "dbh":
        # circle fit on a slice around base_z + dbh_height, per tree
        for k in range(len(uniq)):
            seg = slice(starts[k], starts[k] + counts[k])
            z_target = zmin[k] + args.dbh_height
            m = np.abs(zs[seg] - z_target) <= args.dbh_slice / 2
            if m.sum() >= args.dbh_min_points:
                stats["dbh"][k] = fit_circle_diameter(xs[seg][m], ys[seg][m])
    return stats


# --------------------------------------------------------------------------- io
def write_subset(las, selector, path):
    """Write the points selected by a boolean mask / index array to a LAZ/LAS file."""
    out = laspy.LasData(las.header)
    out.points = las.points[selector]
    out.update_header()
    out.write(str(path))


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    in_path = Path(args.input)
    ext = in_path.suffix if in_path.suffix.lower() in (".laz", ".las") else ".laz"
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_filtered" + ext)
    to_cm = UNIT_TO_CM[args.cloud_unit]

    # ---- read
    print(f"Reading {in_path} ...")
    las = laspy.read(str(in_path))
    dims = list(las.point_format.dimension_names)
    if args.attr not in dims:
        sys.exit(f"Attribute '{args.attr}' not found. Available: {dims}")

    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    labels = np.asarray(las[args.attr])
    if not np.issubdtype(labels.dtype, np.integer):
        labels = np.rint(labels)
    labels = labels.astype(np.int64)
    print(f"{len(las.points):,} points, attribute '{args.attr}'")

    # ---- per-tree stats (background removed)
    is_tree_pt = ~np.isin(labels, args.background_ids)
    if not is_tree_pt.any():
        sys.exit("No tree points found (all points are background ids).")
    stats = per_tree_stats(xyz[is_tree_pt], labels[is_tree_pt], args)

    valid = stats["n_points"] >= args.min_points
    n_trees = int(valid.sum())
    n_dropped_small = int((~valid).sum())

    print("\n=== Tree count ===")
    print(f"Trees (instances with >= {args.min_points} points): {n_trees}")
    if n_dropped_small:
        print(f"Ignored instances with < {args.min_points} points: {n_dropped_small}")
    print(f"Background points (ids {args.background_ids}): {int((~is_tree_pt).sum()):,}")

    metric_cm = stats[args.metric] * to_cm
    m = metric_cm[valid]
    m = m[~np.isnan(m)]
    if len(m):
        print(f"\n{args.metric} (cm): min={m.min():.1f}  median={np.median(m):.1f}  "
              f"mean={m.mean():.1f}  max={m.max():.1f}")
    if args.metric == "dbh":
        print(f"Trees without a valid DBH fit: {int(np.isnan(metric_cm[valid]).sum())}")

    # ---- CSV
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "n_points", "x", "y", "z_min", "z_max", "height_cm", "crown_cm", "dbh_cm"])
            for k in range(len(stats["id"])):
                w.writerow([stats["id"][k], stats["n_points"][k],
                            f"{stats['x'][k]:.4f}", f"{stats['y'][k]:.4f}",
                            f"{stats['z_min'][k]:.4f}", f"{stats['z_max'][k]:.4f}",
                            f"{stats['height'][k] * to_cm:.2f}", f"{stats['crown'][k] * to_cm:.2f}",
                            "" if np.isnan(stats["dbh"][k]) else f"{stats['dbh'][k] * to_cm:.2f}"])
        print(f"\nPer-tree stats written to {args.csv}")

    # ---- filter + write
    if args.no_write:
        return
    if args.threshold is None:
        sys.exit("--threshold is required to filter (or use --no-write to only count).")

    with np.errstate(invalid="ignore"):
        if args.keep == "smaller":
            keep_tree = metric_cm < args.threshold
        else:
            keep_tree = metric_cm > args.threshold
    keep_tree &= valid  # NaN metrics and ignored tiny instances are dropped
    kept_ids = stats["id"][keep_tree]

    point_mask = np.isin(labels, kept_ids)
    print(f"\n=== Filter: {args.metric} {'<' if args.keep == 'smaller' else '>'} {args.threshold} cm ===")
    print(f"Trees kept: {len(kept_ids)} / {n_trees}")
    print(f"Points kept: {int(point_mask.sum()):,} / {len(las.points):,}")

    if not point_mask.any():
        sys.exit("No points left after filtering; nothing written.")

    write_subset(las, point_mask, out_path)
    print(f"Filtered cloud written to {out_path}")

    # ---- one file per kept tree
    if args.per_tree_dir:
        tree_dir = Path(args.per_tree_dir)
        tree_dir.mkdir(parents=True, exist_ok=True)

        kept_idx = np.flatnonzero(point_mask)
        order = np.argsort(labels[kept_idx], kind="stable")  # stable: keeps original point order per tree
        sorted_idx = kept_idx[order]
        ids, starts = np.unique(labels[sorted_idx], return_index=True)
        ends = np.append(starts[1:], len(sorted_idx))

        for tree_id, s, e in zip(ids, starts, ends):
            write_subset(las, sorted_idx[s:e], tree_dir / f"{in_path.stem}_tree_{tree_id}{ext}")
        print(f"{len(ids)} per-tree files written to {tree_dir}")


if __name__ == "__main__":
    main()
