"""Is a depth camera recording as well as the cameras the model was trained on?

Owns: two cheap per-view measures of recording quality, taken on every scan,
and the limits beyond which a weight should not be trusted. Must not know
about models or HTTP.

- Floor noise: the floor is flat, so for three horizontally adjacent floor
  pixels the middle range minus the mean of its neighbours is ~0; its robust
  spread over 1.5^0.5 estimates the per-pixel depth noise. Reported as a share
  of the distance, because time-of-flight noise grows with distance.
- Edge density: valid/missing 4-neighbour pixel pairs per valid pixel. Missing
  returns of a working camera come in patches (dark hide, far background), so
  this is low; scattered single missing pixels raise it fast.

Limits come from the sensor study (``cattle3d sensor``, reports/sensor_limits.md)
on the 195 animals of the two public herds: the recordings measured at most
0.33 % floor noise and 0.74 edge density in any view. The model tolerates added
random noise of 0.5 % but not 1 % (measured floor noise then ~1 % in the side
views), and losing 10 % of pixels but not 20 % (edge density 0.89 or more).
The limits sit between, so no recorded scan is flagged. In-sample: two herds,
one camera model, one barn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cattle3d.types import OrganizedCloud, Plane, View

MAX_FLOOR_NOISE_REL = 0.0075
MAX_EDGE_DENSITY = 0.85
MIN_FLOOR_PIXELS = 1_000


@dataclass(frozen=True)
class FloorNoise:
    """Depth noise measured on the floor of one view (metres)."""

    range_m: float  # median distance of the floor pixels
    plane_sigma_m: float  # robust spread around the plane (includes floor unevenness)
    local_sigma_m: float  # pixel-level noise from second differences

    @property
    def plane_rel(self) -> float:
        return self.plane_sigma_m / self.range_m

    @property
    def local_rel(self) -> float:
        return self.local_sigma_m / self.range_m


def _robust_sigma(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def floor_noise(
    cloud: OrganizedCloud, floor: Plane, inlier_m: float = 0.025, min_range_m: float = 0.5, max_range_m: float = 4.0,
) -> FloorNoise | None:
    """Noise on the floor pixels within ``inlier_m`` of ``floor``; None if too
    little floor is in view."""
    rng_m = np.linalg.norm(cloud.xyz, axis=2)
    usable = cloud.valid & (rng_m > min_range_m) & (rng_m < max_range_m)
    dist = np.where(usable, cloud.xyz @ floor.normal + floor.d, np.inf)
    on_floor = usable & (np.abs(dist) < inlier_m)
    trio = on_floor[:, :-2] & on_floor[:, 1:-1] & on_floor[:, 2:]
    if trio.sum() < MIN_FLOOR_PIXELS:
        return None
    second = rng_m[:, 1:-1] - 0.5 * (rng_m[:, :-2] + rng_m[:, 2:])
    return FloorNoise(
        range_m=float(np.median(rng_m[on_floor])),
        plane_sigma_m=_robust_sigma(dist[on_floor]),
        local_sigma_m=_robust_sigma(second[trio]) / np.sqrt(1.5),
    )


def edge_density(cloud: OrganizedCloud) -> float:
    valid = cloud.valid
    pairs = int((valid[:, 1:] != valid[:, :-1]).sum() + (valid[1:, :] != valid[:-1, :]).sum())
    return pairs / max(1, int(valid.sum()))


@dataclass(frozen=True)
class ViewQuality:
    view: View
    floor_noise_rel: float | None  # None: too little floor in view to measure
    edge_density: float

    @property
    def noisy(self) -> bool:
        return self.floor_noise_rel is not None and self.floor_noise_rel > MAX_FLOOR_NOISE_REL

    @property
    def scattered_loss(self) -> bool:
        return self.edge_density > MAX_EDGE_DENSITY


def view_quality(cloud: OrganizedCloud, view: View, floor: Plane, inlier_m: float = 0.025) -> ViewQuality:
    noise = floor_noise(cloud, floor, inlier_m)
    return ViewQuality(view, None if noise is None else noise.local_rel, edge_density(cloud))


def quality_reasons(qualities: list[ViewQuality]) -> list[str]:
    """Why the recording itself is not good enough to trust a weight."""
    reasons = []
    for q in qualities:
        if q.noisy:
            reasons.append(f"depth noise in {q.view.value} view {q.floor_noise_rel * 100:.1f} % of range "
                           f"(limit {MAX_FLOOR_NOISE_REL * 100:g} %); check the camera")
        if q.scattered_loss:
            reasons.append(f"many scattered missing pixels in {q.view.value} view; clean the camera window")
    return reasons
