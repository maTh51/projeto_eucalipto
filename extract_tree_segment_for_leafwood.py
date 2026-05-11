#!/usr/bin/env python3
"""Extract one tree segment from a classified LAZ/LAS file for leaf-wood inference.

The leaf-wood inference script accepts .txt/.npy/.ply and only uses the first
three columns as XYZ coordinates. This utility exports XYZ-only point clouds in
.txt (default) or .npy format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eucalipto import io as eio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one tree segment from LAZ/LAS and export XYZ for leaf-wood inference."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the classified input LAZ/LAS file.",
    )
    parser.add_argument(
        "--segment-field",
        default="tree_segment_id",
        help="Name of the dimension holding the segment identifier.",
    )
    parser.add_argument(
        "--segment-value",
        type=int,
        default=3,
        help="Segment value to extract.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "npy"],
        default="txt",
        help="Output format accepted by leaf-wood inference.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path. If omitted, defaults to "
            "<input_stem>_<segment_field>_<segment_value>.<format>."
        ),
    )
    parser.add_argument(
        "--txt-precision",
        type=int,
        default=3,
        help="Decimal precision for TXT output.",
    )
    return parser.parse_args()


def build_default_output_path(
    input_path: Path,
    segment_field: str,
    segment_value: int,
    output_format: str,
) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_{segment_field}_{segment_value}.{output_format}"
    )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    points, extras = eio.load_laz(str(input_path))

    if args.segment_field not in extras:
        available = ", ".join(sorted(extras.keys())) if extras else "(none)"
        raise KeyError(
            f"Dimension '{args.segment_field}' not found. Available dimensions: {available}"
        )

    segment_values = np.asarray(extras[args.segment_field])
    mask = segment_values == args.segment_value
    selected = int(np.sum(mask))

    if selected == 0:
        raise ValueError(
            f"No points found for {args.segment_field} == {args.segment_value}."
        )

    xyz = points[mask, :3].astype(np.float32)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else build_default_output_path(
            input_path,
            args.segment_field,
            args.segment_value,
            args.format,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "txt":
        np.savetxt(output_path, xyz, fmt=f"%.{args.txt_precision}f")
    else:
        np.save(output_path, xyz)

    print(f"Input points: {points.shape[0]}")
    print(f"Selected points: {selected}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
