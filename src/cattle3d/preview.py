"""A compact visual summary of one scan, for screens that show the operator
what the cameras saw and where the volume came from.

Owns: depth thumbnails with the animal mask, the body points (all three views
merged when the station calibration is known, else the top view) and the
trunk cross-sections that the volume feature integrates. Must not know about
HTTP or storage. Everything is plain JSON (images as PNG data URLs, points as
base64 int16 millimetres) so a browser can draw it without libraries.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from cattle3d.features.geometry_features import N_SLICES, trunk_interval
from cattle3d.geometry.rig import MergedViews
from cattle3d.geometry.segment import SegmentationTrace
from cattle3d.types import View

MAX_POINTS = 24_000
VIEW_ORDER = (View.LEFT, View.RIGHT, View.TOP)
THUMB_STEP = 2  # 512×424 Kinect frame -> 256×212 thumbnail


def png_data_url(rgb: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 image as a PNG data URL (no imaging library)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def depth_thumbnail(trace: SegmentationTrace) -> str:
    """Background in grey by height above the floor, the animal in green."""
    step = THUMB_STEP
    valid = trace.valid[::step, ::step]
    animal = trace.animal[::step, ::step]
    height = np.clip(trace.height[::step, ::step] / 2.0, 0, 1)
    img = np.full(valid.shape + (3,), 22, np.uint8)
    grey = (60 + 120 * height).astype(np.uint8)
    img[valid] = np.stack([grey, grey, grey], axis=-1)[valid]
    shade = 0.55 + 0.45 * height
    img[animal] = np.stack([60 * shade, 200 * shade, 120 * shade], axis=-1)[animal].astype(np.uint8)
    return png_data_url(img)


def _hull(section: np.ndarray) -> list[list[float]]:
    """Convex outline of one slice in the (lateral, height) plane, with the
    floor projections the volume feature adds, as [y, z] pairs in metres."""
    pts = np.vstack([section, np.column_stack([section[:, 0], np.zeros(len(section))])])
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return []
    return [[round(float(x), 3), round(float(z), 3)] for x, z in pts[hull.vertices]]


def scan_preview(
    traces: dict[View, SegmentationTrace], merged: MergedViews | None = None, seed: int = 0,
) -> dict[str, object]:
    """With the views merged (needs the station calibration) the whole animal
    is shown; without it only the top view's points."""
    top = traces[View.TOP].result.body_points
    lo, hi = trunk_interval(top)
    centre = 0.5 * (lo + hi)
    if merged is not None:
        pts, view_of = merged.points, merged.view_index
    else:
        pts, view_of = top, np.full(len(top), VIEW_ORDER.index(View.TOP), np.uint8)
    pts = pts.copy()
    pts[:, 0] -= centre
    if len(pts) > MAX_POINTS:
        keep = np.random.default_rng(seed).choice(len(pts), MAX_POINTS, replace=False)
        pts, view_of = pts[keep], view_of[keep]
    mm = np.clip(np.round(pts * 1000), -32768, 32767).astype("<i2")

    trunk = top[(top[:, 0] >= lo) & (top[:, 0] <= hi)]
    edges = np.linspace(*np.percentile(trunk[:, 0], [2, 98]), N_SLICES + 1)
    which = np.digitize(trunk[:, 0], edges) - 1
    slices = []
    for k in range(N_SLICES):
        section = trunk[which == k][:, 1:3]
        if len(section) >= 3:
            outline = _hull(section)
            if outline:
                slices.append({"x": round(float(0.5 * (edges[k] + edges[k + 1]) - centre), 3), "outline": outline})

    return {
        "views": {
            v.value: {"image": depth_thumbnail(t), "animal_points": int(t.animal.sum())} for v, t in traces.items()
        },
        "points_mm": base64.b64encode(mm.tobytes()).decode(),
        "point_view": base64.b64encode(view_of.astype(np.uint8).tobytes()).decode(),
        "view_order": [v.value for v in VIEW_ORDER],
        "n_points": int(len(mm)),
        "views_merged": merged is not None,
        "view_agreement": None if merged is None else {
            "same_animal": merged.same_animal,
            "footprint_share": round(merged.footprint_share, 3),
            "back_gap_m": round(merged.back_gap_m, 3),
            "shift_m": round(merged.shift_m, 3),
        },
        "trunk_m": [round(float(lo - centre), 3), round(float(hi - centre), 3)],
        "slices": slices,
    }
