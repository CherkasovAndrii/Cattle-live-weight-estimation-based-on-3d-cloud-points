"""Floor removal, animal segmentation and the metric body frame.

Owns: turning one organized depth cloud into the animal's points in a
gravity-aligned body frame. Must not know about datasets, labels or models.

Steps for one view:
1. Keep pixels with a depth return inside the working distance.
2. Fit the floor with RANSAC, constrained to planes whose normal is close to
   the expected "up" direction of that camera, so a wall or a fence panel is
   never taken for the floor.
3. Keep pixels between ``min_height`` and ``max_height`` above the floor.
4. Split them into connected surfaces on the pixel grid, breaking links at
   depth jumps, and keep the largest surface near the image centre. Fences in
   front of or behind the animal are separate surfaces because of the jump.
   Fragments of the same animal (legs cut off by a rail, speckle on dark hide)
   are merged back when they touch it in 3-D.
5. Express the points in a body frame: height = distance above the floor,
   body axis = main horizontal direction of the points. Metres are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from cattle3d.types import OrganizedCloud, Plane, SegmentedView, View


class SegmentationError(RuntimeError):
    """No floor or no animal could be found in a view."""


@dataclass(frozen=True)
class SegmentationConfig:
    min_range_m: float = 0.5
    max_range_m: float = 4.0
    floor_inlier_m: float = 0.025
    floor_max_tilt_deg: float = 40.0
    ransac_iterations: int = 400
    ransac_sample: int = 30_000
    min_height_m: float = 0.06
    max_height_m: float = 2.2
    depth_jump_m: float = 0.06
    min_component_pixels: int = 1_500
    min_fragment_pixels: int = 40
    merge_distance_m: float = 0.10
    seed: int = 0


# Expected floor normal in camera coordinates, up to sign. Side cameras are
# mounted level, so the image vertical (camera y) is gravity; the top camera
# looks down its optical axis (camera z).
_EXPECTED_UP = {
    View.LEFT: np.array([0.0, 1.0, 0.0]),
    View.RIGHT: np.array([0.0, 1.0, 0.0]),
    View.TOP: np.array([0.0, 0.0, 1.0]),
}


def _plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    return normal, float(-normal @ centroid)


def fit_floor(
    points: np.ndarray, expected_up: np.ndarray, config: SegmentationConfig
) -> tuple[Plane, float]:
    """RANSAC plane with a tilt constraint; returns the plane and its inlier ratio.

    The normal is oriented so that the camera origin is on the positive side,
    which makes "height above the floor" positive for everything in view.
    """
    if points.shape[0] < 3:
        raise SegmentationError("not enough points to fit a floor")
    rng = np.random.default_rng(config.seed)
    sample = points
    if points.shape[0] > config.ransac_sample:
        sample = points[rng.choice(points.shape[0], config.ransac_sample, replace=False)]

    cos_limit = np.cos(np.deg2rad(config.floor_max_tilt_deg))
    best_count, best = 0, None
    for _ in range(config.ransac_iterations):
        a, b, c = sample[rng.choice(sample.shape[0], 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if abs(normal @ expected_up) < cos_limit:
            continue
        d = -normal @ a
        count = int((np.abs(sample @ normal + d) < config.floor_inlier_m).sum())
        if count > best_count:
            best_count, best = count, (normal, d)
    if best is None:
        raise SegmentationError("no floor plane within the allowed tilt")

    normal, d = best
    inliers = np.abs(points @ normal + d) < config.floor_inlier_m
    normal, d = _plane_from_points(points[inliers])
    if d < 0:  # camera origin (0,0,0) must be on the positive side
        normal, d = -normal, -d
    inliers = np.abs(points @ normal + d) < config.floor_inlier_m
    return Plane(normal=normal, d=d), float(inliers.mean())


def _grid_components(mask: np.ndarray, depth: np.ndarray, jump: float) -> np.ndarray:
    """Label 4-connected pixels of ``mask`` whose depths differ by < ``jump``."""
    h, w = mask.shape
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(int(mask.sum()))
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = mask[: h - dy, : w - dx] & mask[dy:, dx:]
        close = np.abs(depth[: h - dy, : w - dx] - depth[dy:, dx:]) < jump
        link = a & close
        rows.append(index[: h - dy, : w - dx][link])
        cols.append(index[dy:, dx:][link])
    r, c = np.concatenate(rows), np.concatenate(cols)
    n = int(mask.sum())
    graph = coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out = np.full(mask.shape, -1, dtype=np.int64)
    out[mask] = labels
    return out


def _pick_animal(labels: np.ndarray, min_pixels: int) -> tuple[np.ndarray, int]:
    """Largest component, weighted towards the image centre columns."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    keep = counts >= min_pixels
    if not keep.any():
        raise SegmentationError("no surface large enough to be an animal")
    ids, counts = ids[keep], counts[keep]
    w = labels.shape[1]
    cols = np.broadcast_to(np.arange(w), labels.shape)
    scores = []
    for label, count in zip(ids, counts):
        centre = cols[labels == label].mean() / w
        scores.append(count * (1.0 - abs(centre - 0.5)))
    return labels == ids[int(np.argmax(scores))], int(ids.size)


def _merge_fragments(
    labels: np.ndarray, animal: np.ndarray, xyz: np.ndarray, config: SegmentationConfig
) -> np.ndarray:
    """Add surfaces that touch the animal in 3-D and lie within its columns.

    Dark hide and fence rails split the animal on the pixel grid (a rail across
    the legs cuts them off the body). A fragment is merged when its closest
    point is within ``merge_distance_m`` of the animal in 3-D and nearly all of
    its pixels are inside the animal's column span, which keeps out rails that
    run across the whole frame.
    """
    cols = np.broadcast_to(np.arange(labels.shape[1]), labels.shape)
    merged = animal.copy()
    for _ in range(2):  # a second pass picks up fragments touching merged ones
        col_lo, col_hi = np.percentile(cols[merged], [0.5, 99.5])
        tree = cKDTree(xyz[merged][:: max(1, int(merged.sum()) // 20_000)])
        candidates = np.unique(labels[(labels >= 0) & ~merged])
        added = False
        for label in candidates:
            part = labels == label
            if part.sum() < config.min_fragment_pixels:
                continue
            inside = ((cols[part] >= col_lo) & (cols[part] <= col_hi)).mean()
            if inside < 0.9:
                continue
            gap, _ = tree.query(xyz[part][:: max(1, int(part.sum()) // 500)], k=1)
            if gap.min() < config.merge_distance_m:
                merged |= part
                added = True
        if not added:
            break
    return merged


def body_frame(points: np.ndarray, floor: Plane) -> np.ndarray:
    """Rotate floor-relative coordinates so axis 0 is the body's long axis."""
    up = floor.normal
    height = floor.signed_distance(points)
    on_floor = points - np.outer(height, up)
    # Any orthonormal basis of the floor plane, then PCA inside it.
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    planar = np.stack([on_floor @ e1, on_floor @ e2], axis=1)
    planar -= planar.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(planar.T))
    axis = vecs[:, -1]
    lateral = np.array([-axis[1], axis[0]])
    return np.stack([planar @ axis, planar @ lateral, height], axis=1)


@dataclass(frozen=True)
class SegmentationTrace:
    """Every intermediate mask of one view, for inspection and figures.

    ``segment_view`` is the trace without the masks; both run the same code.
    """

    view: View
    valid: np.ndarray  # (H, W) depth return inside the working distance
    height: np.ndarray  # (H, W) metres above the fitted floor, 0 where invalid
    above: np.ndarray  # (H, W) between min_height and max_height
    labels: np.ndarray  # (H, W) connected-surface id, -1 outside ``above``
    picked: np.ndarray  # (H, W) the surface chosen as the animal before merging
    animal: np.ndarray  # (H, W) final animal mask after fragment merging
    result: SegmentedView


def trace_view(
    cloud: OrganizedCloud, view: View, config: SegmentationConfig = SegmentationConfig()
) -> SegmentationTrace:
    xyz = cloud.xyz
    rng_m = np.linalg.norm(xyz, axis=2)
    valid = cloud.valid & (rng_m > config.min_range_m) & (rng_m < config.max_range_m)
    if valid.sum() < config.min_component_pixels:
        raise SegmentationError(f"{view.value}: too few depth returns in range")

    floor, inlier_ratio = fit_floor(xyz[valid], _EXPECTED_UP[view], config)
    height = np.zeros(valid.shape)
    height[valid] = floor.signed_distance(xyz[valid])
    above = valid & (height > config.min_height_m) & (height < config.max_height_m)

    labels = _grid_components(above, xyz[..., 2], config.depth_jump_m)
    picked, n_components = _pick_animal(labels, config.min_component_pixels)
    animal = _merge_fragments(labels, picked, xyz, config)
    points = xyz[animal]
    edge = np.zeros_like(animal)
    edge[:, :3] = edge[:, -3:] = edge[:3, :] = edge[-3:, :] = True
    rows_with_animal = max(int(animal.any(axis=1).sum()), 1)
    cols_with_animal = max(int(animal.any(axis=0).sum()), 1)
    # Edge pixels relative to the animal's extent: ~0 when fully in frame.
    border = float((animal & edge).sum()) / (3 * (rows_with_animal + cols_with_animal))
    result = SegmentedView(
        view=view,
        body_points=body_frame(points, floor),
        floor=floor,
        n_valid_pixels=int(valid.sum()),
        n_components=n_components,
        floor_inlier_ratio=inlier_ratio,
        border_fraction=border,
    )
    return SegmentationTrace(view, valid, height, above, labels, picked, animal, result)


def segment_view(
    cloud: OrganizedCloud, view: View, config: SegmentationConfig = SegmentationConfig()
) -> SegmentedView:
    return trace_view(cloud, view, config).result
