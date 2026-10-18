"""Reader for the binary PCL PLY files of CowDatabase / CowDatabase2.

Owns: parsing the PLY header and the vertex block into an OrganizedCloud.
Must not know about datasets, segmentation or features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cattle3d.types import DEPTH_HEIGHT, DEPTH_WIDTH, OrganizedCloud

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "short": "<i2",
    "ushort": "<u2",
}


class PlyFormatError(ValueError):
    """The file is not a binary little-endian PLY with an x/y/z vertex block."""


def _parse_header(raw: bytes) -> tuple[int, np.dtype, int]:
    marker = raw.find(b"end_header")
    if marker < 0:
        raise PlyFormatError("no end_header in PLY")
    body_start = raw.index(b"\n", marker) + 1
    lines = raw[:marker].decode("ascii", errors="strict").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise PlyFormatError("missing 'ply' magic")
    if not any(line.strip() == "format binary_little_endian 1.0" for line in lines):
        raise PlyFormatError("only binary_little_endian PLY is supported")

    n_vertices = -1
    fields: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n_vertices = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise PlyFormatError("list properties in vertex element are not supported")
            ply_type, name = parts[1], parts[2]
            if ply_type not in _PLY_TYPES:
                raise PlyFormatError(f"unsupported PLY type {ply_type!r}")
            fields.append((name, _PLY_TYPES[ply_type]))
    names = [name for name, _ in fields]
    if n_vertices < 0 or not {"x", "y", "z"} <= set(names):
        raise PlyFormatError("vertex element with x, y, z is required")
    return n_vertices, np.dtype(fields), body_start


def read_points(path: Path) -> np.ndarray:
    """Return the (N, 3) float64 x/y/z array of a PLY file, in file order."""
    raw = Path(path).read_bytes()
    n_vertices, dtype, start = _parse_header(raw)
    if start + n_vertices * dtype.itemsize > len(raw):
        raise PlyFormatError(f"{path}: file shorter than its vertex block")
    vertices = np.frombuffer(raw, dtype=dtype, count=n_vertices, offset=start)
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)


def read_organized_cloud(
    path: Path, height: int = DEPTH_HEIGHT, width: int = DEPTH_WIDTH
) -> OrganizedCloud:
    """Read a dense depth-frame cloud and restore its (H, W) pixel layout."""
    points = read_points(path)
    if points.shape[0] != height * width:
        raise PlyFormatError(
            f"{path}: {points.shape[0]} points, expected a dense {height}x{width} frame"
        )
    return OrganizedCloud(points.reshape(height, width, 3))
