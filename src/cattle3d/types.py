"""Domain value objects shared by every layer.

Owns: the typed records that flow through the pipeline (views, animals,
segmented clouds, predictions). Must not know about file formats, paths of
specific datasets, or models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

# Kinect v2 depth frame shape; every cloud in both public datasets is the dense
# row-major back-projection of one such frame.
DEPTH_HEIGHT = 424
DEPTH_WIDTH = 512


class View(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"

    @property
    def is_side(self) -> bool:
        return self is not View.TOP


ALL_VIEWS: tuple[View, ...] = (View.LEFT, View.RIGHT, View.TOP)


@dataclass(frozen=True)
class OrganizedCloud:
    """A dense depth-camera cloud in camera coordinates (metres).

    ``xyz`` has shape (H, W, 3). Pixels without a depth return are all-zero,
    which is how the Kinect/PCL export marks them.
    """

    xyz: np.ndarray

    def __post_init__(self) -> None:
        if self.xyz.ndim != 3 or self.xyz.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) cloud, got {self.xyz.shape}")

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.xyz).all(axis=2) & (self.xyz[..., 2] > 0)


@dataclass(frozen=True)
class AnimalRecord:
    """One animal of one herd, with its label and where its views live."""

    herd: str
    animal_id: str
    weight_kg: float
    view_paths: dict[View, Path]
    manual: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.herd}/{self.animal_id}"


@dataclass(frozen=True)
class Plane:
    """Plane n·p + d = 0 with unit normal oriented towards the camera."""

    normal: np.ndarray
    d: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d


@dataclass(frozen=True)
class SegmentedView:
    """The animal's points from one view, expressed in a body frame.

    Body frame: axis 0 runs along the body (head/tail sign is arbitrary),
    axis 1 is lateral, axis 2 is height above the floor. Metric scale is kept;
    nothing here is normalised, because size is the signal for mass.
    """

    view: View
    body_points: np.ndarray  # (N, 3) in the body frame
    floor: Plane
    n_valid_pixels: int
    n_components: int
    floor_inlier_ratio: float
    border_fraction: float  # share of animal pixels touching the frame edge (cut-off body)


@dataclass(frozen=True)
class Prediction:
    """What the model returns for one animal."""

    weight_kg: float
    lower_kg: float
    upper_kg: float
    in_operating_range: bool
    size_score: float
    reason: str | None = None
