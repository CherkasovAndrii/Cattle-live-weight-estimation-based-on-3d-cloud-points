"""What the farm can do with the model: scan, weigh, calibrate, check drift.

Owns: turning three clouds into a stored, herd-corrected decision (trusted,
flagged or refused), choosing reference weighings, and the herd's status.
Must not know about HTTP; the API, a website or a tablet backend call this.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cattle3d.api.store import Calibration, Herd, NewScan, Scan, Store
from cattle3d.geometry.quality import ViewQuality, quality_reasons, view_quality
from cattle3d.geometry.rig import MergedViews, Rig, merge_views
from cattle3d.geometry.segment import SegmentationError, SegmentationTrace
from cattle3d.modeling.calibration import (
    RECOMMENDED_CALIBRATION_ANIMALS,
    DriftCheck,
    HerdCorrection,
    check_drift,
    fit_correction,
)
from cattle3d.modeling.weight_model import WeightModel
from cattle3d.pipeline import analyse_clouds, features_from_clouds
from cattle3d.preview import scan_preview
from cattle3d.types import OrganizedCloud, View

Extractor = Callable[[dict[View, OrganizedCloud]], dict[str, float]]

UNCALIBRATED = (
    f"herd not calibrated: expect a systematic shift; weigh {RECOMMENDED_CALIBRATION_ANIMALS} "
    "animals and calibrate"
)
MIXED_VIEWS = (
    "the three views do not seem to show the same animal (they do not line up with the station "
    "calibration); rescan before using this weight"
)
CONTRADICTORY_VIEWS = (
    "the three views contradict each other: a cloud is in the wrong slot or shows another animal"
)
# Body measures shown next to a weighing, from the same features the model used.
PREVIEW_MEASURES = {
    "trunk_volume_m3": "top_trunk_cs_volume", "trunk_length_m": "top_trunk_length",
    "trunk_width_m": "top_trunk_width", "height_m": "top_height",
}
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 20.0, 2_000.0  # plausibility of a scale reading


class NotFoundError(LookupError):
    pass


class InvalidRequestError(ValueError):
    pass


@dataclass(frozen=True)
class HerdStatus:
    herd: Herd
    calibration: Calibration | None
    n_reference_animals: int
    recommended_reference_animals: int
    scans_by_status: dict[str, int]
    drift: DriftCheck | None


@dataclass(frozen=True)
class _Analysis:
    """What one scan's clouds gave: the features and, when the views were
    segmented here, what is needed to explain the decision to the operator."""

    features: dict[str, float]
    traces: dict[View, SegmentationTrace] | None = None
    quality: list[ViewQuality] = field(default_factory=list)
    merged: MergedViews | None = None
    analyse_ms: float = 0.0

    @property
    def reasons(self) -> list[str]:
        """Is the camera as good as the training cameras, and is it one animal?"""
        reasons = quality_reasons(self.quality)
        if self.merged is not None and not self.merged.same_animal:
            reasons.append(MIXED_VIEWS)
        return reasons


def _analyse(clouds: dict[View, OrganizedCloud], station: Rig | None) -> _Analysis:
    t0 = time.perf_counter()
    features, traces = analyse_clouds(clouds)
    analyse_ms = 1000 * (time.perf_counter() - t0)
    quality = [view_quality(clouds[v], v, t.result.floor) for v, t in traces.items()]
    merged = None
    if station is not None:
        merged = merge_views(clouds, traces, station)
        if merged.contradictory:
            raise ValueError(CONTRADICTORY_VIEWS)
    return _Analysis(features, traces, quality, merged, analyse_ms)


def _correction(c: Calibration | None) -> HerdCorrection | None:
    return None if c is None else HerdCorrection(c.slope, c.intercept_kg, c.n_reference, c.half_width_kg)


def _check_weight(weight_kg: float) -> float:
    if not (MIN_WEIGHT_KG <= weight_kg <= MAX_WEIGHT_KG):
        raise InvalidRequestError(f"scale weight {weight_kg} kg is outside {MIN_WEIGHT_KG:g}-{MAX_WEIGHT_KG:g} kg")
    return float(weight_kg)


class WeightService:
    def __init__(self, model: WeightModel, store: Store, extract: Extractor = features_from_clouds) -> None:
        self.model = model
        self.store = store
        self.extract = extract
        # Station calibration per herd (camera extrinsics), used only to show
        # the three views as one animal. Set at start-up; not stored yet.
        self.station_rigs: dict[int, Rig] = {}

    # ---- herds -------------------------------------------------------------------
    def create_herd(self, name: str) -> Herd:
        name = name.strip()
        if not name:
            raise InvalidRequestError("herd name must not be empty")
        return self.store.add_herd(name)

    def herd(self, herd_id: int) -> Herd:
        herd = self.store.herd(herd_id)
        if herd is None:
            raise NotFoundError(f"herd {herd_id} not found")
        return herd

    def herd_status(self, herd_id: int) -> HerdStatus:
        herd = self.herd(herd_id)
        calibration = self.store.current_calibration(herd_id)
        drift = None
        if calibration is not None:
            spot = self.store.spot_weighings(calibration.id)
            drift = check_drift(np.array([s.scale_weight_kg - s.weight_kg for s in spot], float))
        return HerdStatus(
            herd=herd,
            calibration=calibration,
            n_reference_animals=len(self.store.reference_weighings(herd_id)),
            recommended_reference_animals=RECOMMENDED_CALIBRATION_ANIMALS,
            scans_by_status=self.store.count_scans(herd_id),
            drift=drift,
        )

    # ---- scans -------------------------------------------------------------------
    def scan(
        self, herd_id: int, animal_tag: str, clouds: dict[View, OrganizedCloud], scale_weight_kg: float | None = None,
        rig: Rig | None = None,
    ) -> Scan:
        """Predict one animal and store the decision.

        A view without a usable animal is a refusal ("rescan"), not an error:
        it is stored so the refusal rate of a station can be monitored. With
        the station calibration (``rig``, else the herd's) the three views are
        also checked to show one animal; if not, the weight is flagged.
        """
        return self._scan_clouds(herd_id, animal_tag, clouds, scale_weight_kg, rig, preview=False)[0]

    def scan_with_preview(
        self, herd_id: int, animal_tag: str, clouds: dict[View, OrganizedCloud], scale_weight_kg: float | None = None,
        rig: Rig | None = None,
    ) -> tuple[Scan, dict[str, object] | None]:
        """``scan`` plus what the cameras saw (see ``cattle3d.preview``) and how
        long each stage took. The preview is None when the scan was refused."""
        t0 = time.perf_counter()
        stored, analysis = self._scan_clouds(herd_id, animal_tag, clouds, scale_weight_kg, rig, preview=True)
        total_ms = 1000 * (time.perf_counter() - t0)
        if analysis is None or analysis.traces is None or stored.status == "refused":
            return stored, None
        preview = scan_preview(analysis.traces, analysis.merged)
        preview["measures"] = {k: round(float(analysis.features[f]), 3) for k, f in PREVIEW_MEASURES.items()}
        preview["recording_quality"] = {  # see cattle3d.geometry.quality
            q.view.value: {"floor_noise_rel": None if q.floor_noise_rel is None else round(q.floor_noise_rel, 5),
                           "edge_density": round(q.edge_density, 3)}
            for q in analysis.quality
        }
        preview["timing_ms"] = {"segmentation_and_features_ms": round(analysis.analyse_ms), "total_ms": round(total_ms)}
        return stored, preview

    def scan_from_features(
        self, herd_id: int, animal_tag: str, features: dict[str, float], scale_weight_kg: float | None = None
    ) -> Scan:
        """Store a scan whose features were computed elsewhere (e.g. imported
        from an earlier extraction), with the same decision rules."""
        return self._scan(herd_id, animal_tag, lambda: _Analysis(dict(features)), scale_weight_kg)[0]

    def _scan_clouds(
        self, herd_id: int, animal_tag: str, clouds: dict[View, OrganizedCloud], scale_weight_kg: float | None,
        rig: Rig | None, preview: bool,
    ) -> tuple[Scan, _Analysis | None]:
        if self.extract is not features_from_clouds and not preview:
            # An injected extractor (tests, other sensors) has no views to compare.
            return self._scan(herd_id, animal_tag, lambda: _Analysis(self.extract(clouds)), scale_weight_kg)
        station = rig or self.station_rigs.get(herd_id)
        return self._scan(herd_id, animal_tag, lambda: _analyse(clouds, station), scale_weight_kg)

    def _scan(
        self, herd_id: int, animal_tag: str, analyse: Callable[[], _Analysis], scale_weight_kg: float | None,
    ) -> tuple[Scan, _Analysis | None]:
        """Validate the request, analyse, predict and store; the analysis is
        None when the scan was refused."""
        self.herd(herd_id)
        animal_tag = animal_tag.strip()
        if not animal_tag:
            raise InvalidRequestError("animal tag must not be empty")
        if scale_weight_kg is not None:
            _check_weight(scale_weight_kg)
        calibration = self.store.current_calibration(herd_id)
        calibration_id = None if calibration is None else calibration.id
        version = self.model.package_version
        analysis: _Analysis | None
        try:
            analysis = analyse()
        except (SegmentationError, ValueError) as exc:
            analysis = None
            new = NewScan(herd_id, animal_tag, "refused", [f"rescan: {exc}"], None, None, None, None,
                          calibration_id, None, version)
        else:
            row = pd.DataFrame([analysis.features])
            raw = float(self.model.raw_predict(row)[0])
            prediction = self.model.predict(row, correction=_correction(calibration))[0]
            reasons = prediction.reason.split("; ") if prediction.reason else []
            reasons += analysis.reasons
            if calibration is None:
                reasons.append(UNCALIBRATED)
            new = NewScan(herd_id, animal_tag, "flagged" if reasons else "trusted", reasons, raw,
                          prediction.weight_kg, prediction.lower_kg, prediction.upper_kg,
                          calibration_id, analysis.features, version)
        stored = self.store.add_scan(new)
        if scale_weight_kg is not None:
            stored = self.record_weighing(stored.id, scale_weight_kg)
        return stored, analysis

    def get_scan(self, scan_id: int) -> Scan:
        scan = self.store.scan(scan_id)
        if scan is None:
            raise NotFoundError(f"scan {scan_id} not found")
        return scan

    def record_weighing(self, scan_id: int, weight_kg: float) -> Scan:
        """Attach a scale reading to a scan: a reference for calibration, and
        after calibration also a spot check for drift."""
        self.get_scan(scan_id)
        updated = self.store.set_scale_weight(scan_id, _check_weight(weight_kg))
        assert updated is not None
        return updated

    def herd_scans(self, herd_id: int, limit: int = 50) -> list[Scan]:
        self.herd(herd_id)
        return self.store.herd_scans(herd_id, limit)

    def animal_history(self, herd_id: int, animal_tag: str) -> list[Scan]:
        self.herd(herd_id)
        return self.store.animal_scans(herd_id, animal_tag)

    # ---- calibration ---------------------------------------------------------------
    def calibrate(self, herd_id: int) -> Calibration:
        """Fit the herd correction from the latest weighed scan of each animal."""
        self.herd(herd_id)
        reference = self.store.reference_weighings(herd_id)
        raw = np.array([s.raw_weight_kg for s in reference], float)
        weight = np.array([s.scale_weight_kg for s in reference], float)
        try:
            c = fit_correction(raw, weight, self.model.alpha, self.model.interval_half_width_kg)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        return self.store.add_calibration(herd_id, c.slope, c.intercept_kg, c.n_reference, c.half_width_kg)
