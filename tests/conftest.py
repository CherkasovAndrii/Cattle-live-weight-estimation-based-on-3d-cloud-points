"""Synthetic fixtures: the tests never need the real datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cattle3d.types import DEPTH_HEIGHT, DEPTH_WIDTH

FX = FY = 365.0
CX, CY = DEPTH_WIDTH / 2, DEPTH_HEIGHT / 2


def write_pcl_ply(path: Path, xyz: np.ndarray) -> None:
    """Write a binary PLY laid out like the Kinect/PCL export of the datasets."""
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment PCL generated\n"
        f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        "element camera 1\nproperty float view_px\nproperty float view_py\nproperty float view_pz\n"
        "end_header\n"
    )
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
    v = np.zeros(n, dtype=dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    path.write_bytes(header.encode("ascii") + v.tobytes() + np.zeros(3, "<f4").tobytes())


def render_depth(surfaces, height: int = DEPTH_HEIGHT, width: int = DEPTH_WIDTH) -> np.ndarray:
    """Ray-cast simple surfaces into an organized (H, W, 3) cloud.

    ``surfaces`` are callables mapping unit ray directions (H, W, 3) to the
    distance along each ray (np.inf where missed); the nearest hit wins.
    """
    v, u = np.mgrid[0:height, 0:width].astype(float)
    rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=2)
    t = np.full((height, width), np.inf)
    for surface in surfaces:
        t = np.minimum(t, surface(rays))
    xyz = rays * t[..., None]
    xyz[~np.isfinite(t)] = 0.0
    return xyz


def plane_hit(normal, d):
    normal = np.asarray(normal, float)

    def hit(rays):
        denom = rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -d / denom
        return np.where((t > 0) & np.isfinite(t), t, np.inf)

    return hit


def box_hit(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def hit(rays):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = lo / rays, hi / rays
        tmin = np.nanmax(np.minimum(t1, t2), axis=2)
        tmax = np.nanmin(np.maximum(t1, t2), axis=2)
        return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)

    return hit


@pytest.fixture
def side_scene() -> np.ndarray:
    """A side camera 1.2 m above the floor (camera y points down), a 1.6 x 0.5 x 0.8 m
    box standing on the floor 2.0-2.5 m away, and a wall 6 m away."""
    floor = plane_hit([0, -1, 0], 1.2)  # floor is y = +1.2 (below the camera)
    animal = box_hit([-0.8, 1.2 - 1.3, 2.0], [0.8, 1.2 - 0.5, 2.5])  # height 0.5-1.3 m
    wall = plane_hit([0, 0, -1], 6.0)
    return render_depth([floor, animal, wall])


@pytest.fixture
def top_scene() -> np.ndarray:
    """A top camera 3.0 m above the floor looking down (+z), a box 1.5 m long,
    0.5 m wide, 1.2 m tall."""
    floor = plane_hit([0, 0, -1], 3.0)
    animal = box_hit([-0.75, -0.25, 3.0 - 1.2], [0.75, 0.25, 3.0])
    return render_depth([floor, animal])
