#!/usr/bin/env python3
"""Count disconnected objects in a closed mesh stored in a PLY file.

The script treats each connected component of faces as one object.

It uses only the Python standard library, so it can run even when optional
mesh libraries are not installed.

Usage:
    python count_mesh_objects.py /path/to/mesh.ply
    python count_mesh_objects.py /path/to/mesh.ply --top 10
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import BinaryIO


PLY_SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass(frozen=True)
class PropertySpec:
    kind: str
    name: str
    scalar_type: str | None = None
    list_count_type: str | None = None
    list_item_type: str | None = None


@dataclass(frozen=True)
class ElementSpec:
    name: str
    count: int
    properties: list[PropertySpec]


@dataclass(frozen=True)
class PlySpec:
    format: str
    elements: list[ElementSpec]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count connected objects in a mesh.ply file."
    )
    parser.add_argument("mesh_path", type=Path, help="Input PLY mesh file")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many largest objects to list by face count (default: 10)",
    )
    return parser


def parse_ply_header(mesh_path: Path) -> tuple[PlySpec, int]:
    with mesh_path.open("rb") as handle:
        first_line = handle.readline().decode("ascii", errors="strict").strip()
        if first_line != "ply":
            raise ValueError(f"Not a PLY file: {mesh_path}")

        format_name = None
        elements: list[ElementSpec] = []
        current_element: ElementSpec | None = None
        header_bytes = len(first_line.encode("ascii")) + 1

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Unexpected end of file while reading header: {mesh_path}")
            header_bytes += len(raw_line)
            line = raw_line.decode("ascii", errors="strict").strip()

            if line == "end_header":
                if format_name is None:
                    raise ValueError(f"PLY file is missing a format declaration: {mesh_path}")
                if current_element is not None:
                    elements.append(current_element)
                return PlySpec(format_name, elements), header_bytes

            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue

            tokens = line.split()
            if tokens[0] == "format":
                if len(tokens) < 2:
                    raise ValueError(f"Invalid format line in {mesh_path}: {line}")
                format_name = tokens[1]
                continue

            if tokens[0] == "element":
                if len(tokens) != 3:
                    raise ValueError(f"Invalid element line in {mesh_path}: {line}")
                if current_element is not None:
                    elements.append(current_element)
                current_element = ElementSpec(tokens[1], int(tokens[2]), [])
                continue

            if tokens[0] == "property":
                if current_element is None:
                    raise ValueError(f"Property declared before any element in {mesh_path}: {line}")

                if tokens[1] == "list":
                    if len(tokens) != 5:
                        raise ValueError(f"Invalid list property in {mesh_path}: {line}")
                    prop = PropertySpec(
                        kind="list",
                        name=tokens[4],
                        list_count_type=tokens[2],
                        list_item_type=tokens[3],
                    )
                else:
                    if len(tokens) != 3:
                        raise ValueError(f"Invalid scalar property in {mesh_path}: {line}")
                    prop = PropertySpec(kind="scalar", name=tokens[2], scalar_type=tokens[1])

                current_element.properties.append(prop)
                continue

            raise ValueError(f"Unsupported PLY header line in {mesh_path}: {line}")


def read_scalar(handle: BinaryIO, scalar_type: str, endian: str) -> int | float:
    try:
        fmt, size = PLY_SCALAR_TYPES[scalar_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY scalar type: {scalar_type}") from exc

    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("Unexpected end of file while reading PLY data")
    return struct.unpack(endian + fmt, payload)[0]


def skip_element_record(handle: BinaryIO, element: ElementSpec, endian: str) -> None:
    for prop in element.properties:
        if prop.kind == "scalar":
            if prop.scalar_type is None:
                raise ValueError(f"Missing scalar type for property {prop.name}")
            read_scalar(handle, prop.scalar_type, endian)
            continue

        if prop.list_count_type is None or prop.list_item_type is None:
            raise ValueError(f"Missing list types for property {prop.name}")
        count = int(read_scalar(handle, prop.list_count_type, endian))
        item_size = PLY_SCALAR_TYPES[prop.list_item_type][1]
        payload = handle.read(item_size * count)
        if len(payload) != item_size * count:
            raise ValueError("Unexpected end of file while skipping PLY list property")


def read_face_record(handle: BinaryIO, element: ElementSpec, endian: str) -> list[int]:
    face_vertices: list[int] | None = None
    for prop in element.properties:
        if prop.kind == "scalar":
            if prop.scalar_type is None:
                raise ValueError(f"Missing scalar type for property {prop.name}")
            read_scalar(handle, prop.scalar_type, endian)
            continue

        if prop.list_count_type is None or prop.list_item_type is None:
            raise ValueError(f"Missing list types for property {prop.name}")
        count = int(read_scalar(handle, prop.list_count_type, endian))
        values = [int(read_scalar(handle, prop.list_item_type, endian)) for _ in range(count)]
        if face_vertices is None:
            face_vertices = values

    if face_vertices is None:
        raise ValueError("Face element did not contain a list property with vertex indices")
    return face_vertices


def load_faces(mesh_path: Path) -> tuple[list[list[int]], int, int]:
    spec, header_bytes = parse_ply_header(mesh_path)

    vertex_element = next((element for element in spec.elements if element.name == "vertex"), None)
    face_element = next((element for element in spec.elements if element.name == "face"), None)

    if vertex_element is None:
        raise ValueError(f"PLY file has no vertex element: {mesh_path}")
    if face_element is None:
        raise ValueError(f"PLY file has no face element: {mesh_path}")

    if spec.format == "ascii":
        faces, vertex_count = load_faces_ascii(mesh_path, spec)
        return faces, face_element.count, vertex_count

    if spec.format == "binary_little_endian":
        endian = "<"
    elif spec.format == "binary_big_endian":
        endian = ">"
    else:
        raise ValueError(f"Unsupported PLY format: {spec.format}")

    faces, vertex_count = load_faces_binary(mesh_path, spec, endian, header_bytes)
    return faces, face_element.count, vertex_count


def load_faces_ascii(mesh_path: Path, spec: PlySpec) -> tuple[list[list[int]], int]:
    faces: list[list[int]] = []
    vertex_count = 0

    with mesh_path.open("r", encoding="ascii", errors="strict") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break

        for element in spec.elements:
            if element.name == "vertex":
                vertex_count = element.count
                for _ in range(element.count):
                    row = handle.readline()
                    if not row:
                        raise ValueError("Unexpected end of file while reading vertex records")
            elif element.name == "face":
                for _ in range(element.count):
                    row = handle.readline()
                    if not row:
                        raise ValueError("Unexpected end of file while reading face records")
                    tokens = row.split()
                    if not tokens:
                        raise ValueError("Empty face record encountered")

                    face_vertices: list[int] | None = None
                    index = 0
                    for prop in element.properties:
                        if prop.kind == "scalar":
                            index += 1
                            continue

                        count = int(tokens[index])
                        index += 1
                        values = [int(tokens[index + offset]) for offset in range(count)]
                        index += count
                        if face_vertices is None:
                            face_vertices = values

                    if face_vertices is None:
                        raise ValueError("Face element did not contain a list property with vertex indices")
                    faces.append(face_vertices)
            else:
                for _ in range(element.count):
                    row = handle.readline()
                    if not row:
                        raise ValueError(f"Unexpected end of file while reading {element.name} records")

    return faces, vertex_count


def load_faces_binary(
    mesh_path: Path,
    spec: PlySpec,
    endian: str,
    header_bytes: int,
) -> tuple[list[list[int]], int]:
    faces: list[list[int]] = []
    vertex_count = 0

    with mesh_path.open("rb") as handle:
        handle.seek(header_bytes)

        for element in spec.elements:
            if element.name == "vertex":
                vertex_count = element.count
                for _ in range(element.count):
                    skip_element_record(handle, element, endian)
            elif element.name == "face":
                for _ in range(element.count):
                    faces.append(read_face_record(handle, element, endian))
            else:
                for _ in range(element.count):
                    skip_element_record(handle, element, endian)

    return faces, vertex_count


def union_find_components(faces: list[list[int]]) -> Counter:
    parent = list(range(len(faces)))
    rank = [0] * len(faces)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces[vertex_index].append(face_index)

    for incident_faces in vertex_to_faces.values():
        if len(incident_faces) < 2:
            continue
        anchor = incident_faces[0]
        for other in incident_faces[1:]:
            union(anchor, other)

    component_sizes = Counter(find(face_index) for face_index in range(len(faces)))
    return component_sizes


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mesh_path = args.mesh_path
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    faces, face_count, vertex_count = load_faces(mesh_path)
    if not faces:
        print(f"No faces found in {mesh_path.name}; object count is 0.")
        return 0

    component_sizes = union_find_components(faces)
    object_count = len(component_sizes)

    print(f"mesh: {mesh_path}")
    print(f"vertices: {vertex_count}")
    print(f"faces: {face_count}")
    print(f"objects: {object_count}")

    top_n = max(args.top, 0)
    if top_n > 0:
        print(f"largest objects by face count (top {top_n}):")
        for index, (_, size) in enumerate(component_sizes.most_common(top_n), start=1):
            print(f"  {index}. object_{index}: {size} faces")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())