"""What the farm can do with the model: scan, weigh, calibrate, check drift.

Owns: turning three clouds into a stored, herd-corrected decision (trusted,
flagged or refused), choosing reference weighings, and the herd's status.
Must not know about HTTP; the API, a website or a tablet backend call this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cattle3d.api.store import Calibration, Herd, NewScan, Scan, Store
from cattle3d.geometry.segment import SegmentationError
from cattle3d.modeling.calibration import (
    RECOMMENDED_CALIBRATION_ANIMALS,
    DriftCheck,
    HerdCorrection,
    check_drift,
    fit_correction,
)
from cattle3d.modeling.weight_model import WeightModel
from cattle3d.pipeline import features_from_clouds
from cattle3d.types import OrganizedCloud, View

Extractor = Callable[[dict[View, OrganizedCloud]], dict[str, float]]

UNCALIBRATED = (
    f"herd not calibrated: expect a systematic shift; weigh {RECOMMENDED_CALIBRATION_ANIMALS} "
    "animals and calibrate"
)
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
        self, herd_id: int, animal_tag: str, clouds: dict[View, OrganizedCloud], scale_weight_kg: float | None = None
    ) -> Scan:
        """Predict one animal and store the decision.

        A view without a usable animal is a refusal ("rescan"), not an error:
        it is stored so the refusal rate of a station can be monitored.
        """
        self.herd(herd_id)
        animal_tag = animal_tag.strip()
        if not animal_tag:
            raise InvalidRequestError("animal tag must not be empty")
        if scale_weight_kg is not None:
            _check_weight(scale_weight_kg)
        calibration = self.store.current_calibration(herd_id)
        version = self.model.package_version
        try:
            features = self.extract(clouds)
        except (SegmentationError, ValueError) as exc:
            stored = self.store.add_scan(
                NewScan(herd_id, animal_tag, "refused", [f"rescan: {exc}"], None, None, None, None,
                        None if calibration is None else calibration.id, None, version)
            )
        else:
            row = pd.DataFrame([features])
            raw = float(self.model.raw_predict(row)[0])
            prediction = self.model.predict(row, correction=_correction(calibration))[0]
            reasons = prediction.reason.split("; ") if prediction.reason else []
            if calibration is None:
                reasons.append(UNCALIBRATED)
            stored = self.store.add_scan(
                NewScan(
                    herd_id, animal_tag, "flagged" if reasons else "trusted", reasons, raw,
                    prediction.weight_kg, prediction.lower_kg, prediction.upper_kg,
                    None if calibration is None else calibration.id, features, version,
                )
            )
        if scale_weight_kg is not None:
            return self.record_weighing(stored.id, scale_weight_kg)
        return stored

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
