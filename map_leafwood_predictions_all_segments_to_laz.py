#!/usr/bin/env python3
"""Map per-segment leaf-wood predictions back to one LAS/LAZ cloud.

Expected prediction files:
  <pred_dir>/<prefix><segment_id><suffix>
Default example:
  data/results_leafwood/tree_segment_id_3.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eucalipto import io as eio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject leaf-wood predictions for all segment IDs into one LAZ/LAS."
    )
    parser.add_argument("--input-laz", required=True, help="Input segmented LAS/LAZ.")
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory with per-segment prediction files.",
    )
    parser.add_argument(
        "--segment-field",
        default="tree_segment_id",
        help="Segment id field name in input cloud.",
    )
    parser.add_argument(
        "--pred-prefix",
        default="tree_segment_id_",
        help="Prediction filename prefix.",
    )
    parser.add_argument(
        "--pred-suffix",
        default=".txt",
        help="Prediction filename suffix.",
    )
    parser.add_argument(
        "--label-field",
        default="leafwood_pred",
        help="Output scalar field name.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output LAZ/LAS path. Defaults to <input_stem>_leafwood_all_segments.laz",
    )
    return parser.parse_args()


def _load_pred_labels(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
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
        raise ValueError(f"Unsupported prediction format: {path}")

    return np.rint(labels).astype(np.uint8)


def main() -> int:
    args = parse_args()

    input_laz = Path(args.input_laz).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve()

    if not input_laz.exists():
        raise FileNotFoundError(f"Input LAZ/LAS not found: {input_laz}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    points, extras = eio.load_laz(str(input_laz))
    if args.segment_field not in extras:
        available = ", ".join(sorted(extras.keys())) if extras else "(none)"
        raise KeyError(
            f"Field '{args.segment_field}' not found in input. Available: {available}"
        )

    seg = np.asarray(extras[args.segment_field])
    seg_ids = np.unique(seg)

    full_labels = np.full(points.shape[0], 255, dtype=np.uint8)

    for sid in seg_ids:
        pred_path = pred_dir / f"{args.pred_prefix}{int(sid)}{args.pred_suffix}"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file for segment {int(sid)}: {pred_path}")

        labels = _load_pred_labels(pred_path)
        mask = seg == sid
        n_pts = int(np.count_nonzero(mask))

        if labels.shape[0] != n_pts:
            raise ValueError(
                f"Size mismatch for segment {int(sid)}: pred={labels.shape[0]} vs cloud={n_pts}"
            )

        full_labels[mask] = labels

    extras_out = {k: np.asarray(v) for k, v in extras.items()}
    extras_out[args.label_field] = full_labels

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_laz.with_name(f"{input_laz.stem}_leafwood_all_segments.laz")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eio.save_laz(
        str(output_path),
        points,
        extras=extras_out,
        like_path=str(input_laz),
    )

    u, c = np.unique(full_labels, return_counts=True)
    print(f"Input points: {points.shape[0]}")
    print(f"Segments processed: {len(seg_ids)}")
    print("Label counts:")
    for lbl, cnt in zip(u.tolist(), c.tolist()):
        print(f"  {lbl}: {cnt}")
    print(f"Saved: {output_path}")
    print(f"Scalar field: {args.label_field}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
