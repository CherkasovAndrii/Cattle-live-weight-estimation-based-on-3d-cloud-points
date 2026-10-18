"""Body-size and volume features from segmented views.

Owns: the feature definitions, per view and per animal. Must not know about
files, datasets or models.

All lengths are metres, areas m², volumes m³. Percentiles rather than min/max
are used everywhere so a few stray points do not move a feature.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from cattle3d.types import SegmentedView, View

N_SLICES = 64
_LO, _HI = 2.0, 98.0
TRUNK_REL_WIDTH = 0.65
MID_TRUNK = (0.2, 0.8)  # share of the trunk used for the mid-trunk features


def _slices(along: np.ndarray, n: int = N_SLICES) -> tuple[np.ndarray, float]:
    lo, hi = np.percentile(along, [_LO, _HI])
    edges = np.linspace(lo, hi, n + 1)
    return edges, float((hi - lo) / n)


def _hull_area(points_2d: np.ndarray) -> float:
    if points_2d.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(points_2d).volume)  # "volume" of a 2-D hull is its area
    except QhullError:
        return 0.0


def cross_section_volume(body: np.ndarray, n_slices: int = N_SLICES) -> float:
    """Integral of per-slice cross-section areas along the body axis.

    Each slice's cross-section is the convex hull of its points in the
    (lateral, height) plane together with their projections on the floor, so a
    top view gives the area under the back profile. Convex hulls span holes in
    the depth data, which makes the estimate tolerant of missing pixels.
    """
    edges, step = _slices(body[:, 0], n_slices)
    which = np.digitize(body[:, 0], edges) - 1
    total = 0.0
    for k in range(n_slices):
        section = body[which == k][:, 1:3]
        if section.shape[0] < 3:
            continue
        floor = np.column_stack([section[:, 0], np.zeros(section.shape[0])])
        total += _hull_area(np.vstack([section, floor])) * step
    return total


def profile_area(body: np.ndarray, value_axis: int, n_slices: int = N_SLICES) -> float:
    """Area swept by the robust extent of one axis along the body axis."""
    edges, step = _slices(body[:, 0], n_slices)
    which = np.digitize(body[:, 0], edges) - 1
    total = 0.0
    for k in range(n_slices):
        values = body[which == k, value_axis]
        if values.size < 5:
            continue
        lo, hi = np.percentile(values, [1, 99])
        total += (hi - lo) * step
    return total


def width_profile(body: np.ndarray, n_slices: int = N_SLICES) -> tuple[np.ndarray, np.ndarray]:
    """Lateral width (5–95 percentile) of each slice along the body axis, lightly smoothed.

    Returns the slice edges (n_slices + 1) and the widths (n_slices).
    """
    edges = np.linspace(*np.percentile(body[:, 0], [0.5, 99.5]), n_slices + 1)
    which = np.digitize(body[:, 0], edges) - 1
    widths = np.zeros(n_slices)
    for k in range(n_slices):
        lateral = body[which == k, 1]
        if lateral.size >= 10:
            widths[k] = np.subtract(*np.percentile(lateral, [95, 5]))
    return edges, np.convolve(widths, np.ones(3) / 3, mode="same")


def trunk_interval(body: np.ndarray, n_slices: int = N_SLICES, rel_width: float = TRUNK_REL_WIDTH) -> tuple[float, float]:
    """Along-axis span of the trunk in a top view.

    The head and neck are much narrower than the barrel, so the trunk is the
    longest run of slices whose lateral width exceeds ``rel_width`` of the
    widest part (ears and horns can be wide but form short runs).
    """
    edges, widths = width_profile(body, n_slices)
    wide = widths >= rel_width * np.percentile(widths, 95)
    best_start, best_len, start = 0, 0, None
    for k, flag in enumerate(np.append(wide, False)):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            if k - start > best_len:
                best_start, best_len = start, k - start
            start = None
    if best_len == 0:
        return float(edges[0]), float(edges[-1])
    return float(edges[best_start]), float(edges[best_start + best_len])


def _central(body: np.ndarray, keep: float = 0.5) -> np.ndarray:
    lo, hi = np.percentile(body[:, 0], [50 * (1 - keep), 100 - 50 * (1 - keep)])
    return body[(body[:, 0] >= lo) & (body[:, 0] <= hi)]


def view_features(seg: SegmentedView) -> dict[str, float]:
    body = seg.body_points
    if body.shape[0] < 100:
        raise ValueError(f"{seg.view.value}: only {body.shape[0]} animal points")
    along, lateral, height = body[:, 0], body[:, 1], body[:, 2]
    feats = {
        "n_points": float(body.shape[0]),
        "length": float(np.subtract(*np.percentile(along, [_HI, _LO]))),
        "width": float(np.subtract(*np.percentile(lateral, [_HI, _LO]))),
        "height": float(np.percentile(height, _HI)),
        "floor_inlier_ratio": seg.floor_inlier_ratio,
        "n_components": float(seg.n_components),
        "border_fraction": seg.border_fraction,
    }
    if seg.view is View.TOP:
        feats["cs_volume"] = cross_section_volume(body)
        feats["footprint_area"] = profile_area(body, value_axis=1)
        lo, hi = trunk_interval(body)
        trunk = body[(along >= lo) & (along <= hi)]
        feats["trunk_length"] = hi - lo
        feats["trunk_cs_volume"] = cross_section_volume(trunk)
        feats["trunk_footprint_area"] = profile_area(trunk, value_axis=1)
        feats["trunk_width"] = float(np.subtract(*np.percentile(trunk[:, 1], [_HI, _LO])))
        feats["trunk_height"] = float(np.percentile(trunk[:, 2], 90))
        # The central 60 % of the trunk stays in frame even when the camera
        # cuts off the rump or the head, so it is the most robust size signal.
        a, b = lo + MID_TRUNK[0] * (hi - lo), lo + MID_TRUNK[1] * (hi - lo)
        mid = trunk[(trunk[:, 0] >= a) & (trunk[:, 0] <= b)]
        feats["mid_cs_area"] = cross_section_volume(mid) / max(b - a, 1e-6)
        feats["mid_width"] = float(np.subtract(*np.percentile(mid[:, 1], [_HI, _LO])))
        feats["mid_height"] = float(np.percentile(mid[:, 2], 95))
    else:
        feats["side_area"] = profile_area(body, value_axis=2)
        mid = _central(body)
        feats["mid_side_area"] = profile_area(mid, value_axis=2)
        feats["mid_height"] = float(np.percentile(mid[:, 2], _HI))
    return feats


def animal_features(per_view: dict[View, dict[str, float]]) -> dict[str, float]:
    """Flatten per-view features and add combinations of the side views."""
    out: dict[str, float] = {}
    for view, feats in per_view.items():
        for name, value in feats.items():
            out[f"{view.value}_{name}"] = value
    left, right, top = per_view[View.LEFT], per_view[View.RIGHT], per_view[View.TOP]
    for name in ("length", "height", "side_area", "mid_side_area", "mid_height"):
        out[f"side_mean_{name}"] = 0.5 * (left[name] + right[name])
        out[f"side_max_{name}"] = max(left[name], right[name])
    out["lr_side_area_disagreement"] = abs(left["side_area"] - right["side_area"]) / max(
        out["side_mean_side_area"], 1e-9
    )
    # A box-like body proxy from orthogonal views: top footprint times side height.
    out["footprint_x_height"] = top["footprint_area"] * out["side_mean_height"]
    out["trunk_box_volume"] = top["trunk_footprint_area"] * top["trunk_height"]
    return out
