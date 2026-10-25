"""Station calibration: where the side cameras sit relative to the top camera.

Owns: the rigid maps left/right camera -> top camera (estimated once per
scanning station from aligned captures or the dataset's matrices) and merging
the three segmented views into one point set for display. Must not know about
datasets, HTTP or models.

The weight features never need this: every feature is computed inside one
view. The merge only serves to show the operator a whole animal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cattle3d.geometry.segment import SegmentationTrace, body_axes
from cattle3d.types import OrganizedCloud, View

SIDE_VIEWS = (View.LEFT, View.RIGHT)
MAX_REFINE_M = 0.4  # larger corrections mean the calibration is wrong, not slightly off


@dataclass(frozen=True)
class Rig:
    """``to_top[view] = (R, t)`` maps that camera's points into the top camera
    frame: ``p_top = p @ R.T + t``."""

    to_top: dict[View, tuple[np.ndarray, np.ndarray]]

    def to_json(self) -> dict[str, list[list[float]]]:
        """``{"left": [[R row 0], [R row 1], [R row 2], [t]], "right": ...}``"""
        return {v.value: [*r.round(6).tolist(), t.round(6).tolist()] for v, (r, t) in self.to_top.items()}

    @classmethod
    def from_json(cls, data: dict[str, list[list[float]]]) -> "Rig":
        out = {}
        for view in SIDE_VIEWS:
            m = np.asarray(data[view.value], float)
            if m.shape != (4, 3) or not np.isfinite(m).all():
                raise ValueError(f"{view.value}: expected 3 rotation rows and a translation of 3 numbers")
            r = m[:3]
            if not np.allclose(r @ r.T, np.eye(3), atol=0.02) or np.linalg.det(r) < 0:
                raise ValueError(f"{view.value}: the rotation is not a proper rotation matrix")
            out[view] = (_orthonormal(r), m[3])
        return cls(out)


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation and translation taking points ``a`` onto ``b``."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def _orthonormal(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def rig_from_aligned(raw: dict[View, np.ndarray], aligned: np.ndarray, step: int = 50) -> Rig:
    """Calibration from one capture whose dataset also gives the three views
    already merged in the top camera frame, as organized blocks (CowDatabase2
    ``aligned.ply``). Pixel i of a raw view and pixel i of its block are the
    same point, so the fit is exact; the block order is found by residual."""
    n = next(iter(raw.values())).shape[0]
    blocks = [aligned[i * n:(i + 1) * n] for i in range(len(aligned) // n)]
    out = {}
    for view in SIDE_VIEWS:
        p = raw[view]
        ok = np.isfinite(p).all(axis=1) & (np.abs(p).sum(axis=1) > 0)
        best = None
        for block in blocks:
            sel = ok & np.isfinite(block).all(axis=1)
            a, b = p[sel][::step], block[sel][::step]
            r, t = kabsch(a, b)
            residual = float(np.median(np.linalg.norm(a @ r.T + t - b, axis=1)))
            if best is None or residual < best[0]:
                best = (residual, r, t)
        assert best is not None
        out[view] = (best[1], best[2])
    return Rig(out)


def read_matrices(path: Path) -> Rig:
    """Calibration from a CowDatabase ``Matrices/<file>.matrices`` text file
    ("Left -> Top" followed by a 4×4 matrix, then "Right -> Top")."""
    found: dict[View, tuple[np.ndarray, np.ndarray]] = {}
    name, rows = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if "->" in line:
            name, rows = line.split("->")[0].strip().lower(), []
        elif line and name is not None:
            rows.append([float(v) for v in line.split()])
            if len(rows) == 4:
                m = np.array(rows)
                if name in ("left", "right"):
                    found[View(name)] = (_orthonormal(m[:3, :3]), m[:3, 3])
    if set(found) != set(SIDE_VIEWS):
        raise ValueError(f"{path}: expected Left -> Top and Right -> Top matrices")
    return Rig(found)


def median_rig(rigs: list[Rig]) -> Rig:
    """One station calibration from several captures of the same station."""
    out = {}
    for view in SIDE_VIEWS:
        r = _orthonormal(np.median([rig.to_top[view][0] for rig in rigs], axis=0))
        t = np.median([rig.to_top[view][1] for rig in rigs], axis=0)
        out[view] = (r, t)
    return Rig(out)


def _refine(side: np.ndarray, tree: cKDTree, target: np.ndarray) -> np.ndarray:
    """Translation-only ICP onto the top view where the surfaces overlap (the
    upper flanks), so a station whose cameras moved a little still lines up."""
    shift = np.zeros(3)
    for radius in (0.25, 0.18, 0.12, 0.08, 0.06, 0.05, 0.05, 0.05):
        dist, idx = tree.query(side + shift, distance_upper_bound=radius)
        hit = np.isfinite(dist)
        if hit.sum() < 200:
            break
        shift += np.median(target[idx[hit]] - (side[hit] + shift), axis=0)
    return shift if np.linalg.norm(shift) <= MAX_REFINE_M else np.zeros(3)


def merged_body_points(
    clouds: dict[View, OrganizedCloud], traces: dict[View, SegmentationTrace], rig: Rig, refine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """All three views' animal points in the top view's body frame, and the
    view index of each point (order of ``(LEFT, RIGHT, TOP)``)."""
    top_cam = clouds[View.TOP].xyz[traces[View.TOP].animal]
    rot, shift = body_axes(top_cam, traces[View.TOP].result.floor)
    tree = cKDTree(top_cam[::2]) if refine else None
    parts, ids = [], []
    for i, view in enumerate((View.LEFT, View.RIGHT, View.TOP)):
        p = clouds[view].xyz[traces[view].animal]
        if view is not View.TOP:
            r, t = rig.to_top[view]
            p = p @ r.T + t
            if tree is not None:
                p = p + _refine(p[::3], tree, top_cam[::2])
        parts.append(p @ rot.T + shift)
        ids.append(np.full(len(p), i, np.uint8))
    return np.concatenate(parts), np.concatenate(ids)
