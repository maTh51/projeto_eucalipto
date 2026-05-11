#!/usr/bin/env python3
"""Map leaf-wood inference labels back to a segmented LAS/LAZ cloud.

The leaf-wood pipeline predicts labels for a single extracted segment (e.g.
"tree_segment_id == 3") and saves results as TXT/NPY. This script injects
those labels into the original segmented LAZ/LAS as an extra scalar field,
so the result can be inspected in CloudCompare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eucalipto import io as eio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map leaf-wood predictions from TXT/NPY back to an original "
            "segmented LAS/LAZ file."
        )
    )
    parser.add_argument(
        "--input-laz",
        required=True,
        help="Path to the original segmented LAS/LAZ file.",
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Prediction file (.txt/.npy). Accepts labels-only or XYZ+label.",
    )
    parser.add_argument(
        "--segment-field",
        default="tree_segment_id",
        help="Name of the segment id field in input cloud.",
    )
    parser.add_argument(
        "--segment-value",
        type=int,
        default=3,
        help="Segment value that was used to create the inference input TXT.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output LAZ/LAS path. If omitted, defaults to "
            "<input_stem>_leafwood_segment_<segment_value>.laz"
        ),
    )
    parser.add_argument(
        "--label-field",
        default="leafwood_pred",
        help="Name of the output scalar field containing predicted labels.",
    )
    parser.add_argument(
        "--outside-value",
        type=int,
        default=255,
        help="Value used for points outside the selected segment.",
    )
    return parser.parse_args()


def _load_prediction_labels(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path)

    arr = np.asarray(arr)
    if arr.ndim == 1:
        labels = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        labels = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] >= 4:
        labels = arr[:, -1]
    else:
        raise ValueError(
            "Prediction format not recognized. Expected labels-only (N,), "
            "(N,1), or XYZ+label with >=4 columns."
        )

    labels = np.rint(labels).astype(np.int32)
    return labels


def _default_output_path(input_path: Path, segment_value: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_leafwood_segment_{segment_value}.laz")


def main() -> int:
    args = parse_args()

    input_laz = Path(args.input_laz).expanduser().resolve()
    pred_path = Path(args.prediction).expanduser().resolve()

    if not input_laz.exists():
        raise FileNotFoundError(f"Input LAS/LAZ not found: {input_laz}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    points, extras = eio.load_laz(str(input_laz))
    if args.segment_field not in extras:
        available = ", ".join(sorted(extras.keys())) if extras else "(none)"
        raise KeyError(
            f"Field '{args.segment_field}' not found in input. "
            f"Available fields: {available}"
        )

    segment_values = np.asarray(extras[args.segment_field])
    segment_mask = segment_values == args.segment_value
    segment_count = int(np.count_nonzero(segment_mask))

    if segment_count == 0:
        raise ValueError(
            f"No points found for {args.segment_field} == {args.segment_value}."
        )

    pred_labels = _load_prediction_labels(pred_path)

    if pred_labels.shape[0] != segment_count:
        raise ValueError(
            "Prediction length mismatch. "
            f"predictions={pred_labels.shape[0]}, segment_points={segment_count}. "
            "Make sure prediction file comes from the same extracted segment."
        )

    full_labels = np.full(points.shape[0], args.outside_value, dtype=np.uint8)
    full_labels[segment_mask] = pred_labels.astype(np.uint8)

    extras_out = {k: np.asarray(v) for k, v in extras.items()}
    extras_out[args.label_field] = full_labels

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(input_laz, args.segment_value)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eio.save_laz(
        str(output_path),
        points,
        extras=extras_out,
        like_path=str(input_laz),
    )

    n_leaf = int(np.sum(pred_labels == 0))
    n_wood = int(np.sum(pred_labels == 1))

    print(f"Input cloud points: {points.shape[0]}")
    print(f"Selected segment points: {segment_count}")
    print(f"Predictions loaded: {pred_labels.shape[0]}")
    print(f"Leaf (0): {n_leaf}")
    print(f"Wood (1): {n_wood}")
    print(f"Saved: {output_path}")
    print(f"Scalar field for CloudCompare: {args.label_field}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
