"""Herd calibration and drift: the per-farm layer on top of the shared model.

Owns: fitting a linear correction from a herd's reference weighings, the
prediction interval that goes with it, and the check that a calibrated herd
has drifted since. Must not know about storage, files or HTTP; it takes arrays.

The study (H3) found that a model moved to a new herd is off mostly by a
shift and a slope, and that a slope + intercept fit on 20 weighed animals
brings the error back to within 1 MAPE point of training on that herd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from cattle3d.evaluation.experiments import conformal_quantile

# Below MIN no correction is fitted; from SLOPE on, slope and intercept are
# fitted (with fewer, a slope is noise and only the intercept moves).
MIN_CALIBRATION_ANIMALS = 3
SLOPE_CALIBRATION_ANIMALS = 10
# H3b: with 20 reference animals 90-96 % of random reference sets met the margin.
RECOMMENDED_CALIBRATION_ANIMALS = 20

# Drift: spot weighings after calibration, scored on the most recent window.
# DRIFT_MIN_KG is an engineering default (about half the within-herd MAE),
# not a validated threshold: a smaller shift is inside the model's own error.
DRIFT_MIN_ANIMALS = 5
DRIFT_WINDOW = 20
DRIFT_MIN_KG = 10.0


@dataclass(frozen=True)
class HerdCorrection:
    """weight = slope × model output + intercept, with its own interval."""

    slope: float
    intercept_kg: float
    n_reference: int
    half_width_kg: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(raw, float) + self.intercept_kg


def _fit(raw: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    if raw.size >= SLOPE_CALIBRATION_ANIMALS and np.ptp(raw) > 0:
        slope, intercept = np.polyfit(raw, weight, 1)
        return float(slope), float(intercept)
    return 1.0, float(np.mean(weight - raw))


def fit_correction(
    raw: np.ndarray, weight: np.ndarray, alpha: float = 0.1, min_half_width_kg: float = 0.0
) -> HerdCorrection:
    """Fit a herd correction from model outputs and scale weights of the same animals.

    The interval is split-conformal on leave-one-out residuals of the
    correction (in-sample residuals would be too optimistic with 20 animals),
    and never narrower than ``min_half_width_kg``, the model's own interval.
    """
    raw, weight = np.asarray(raw, float), np.asarray(weight, float)
    if raw.shape != weight.shape or raw.ndim != 1:
        raise ValueError("raw predictions and weights must be 1-D arrays of equal length")
    if raw.size < MIN_CALIBRATION_ANIMALS:
        raise ValueError(
            f"at least {MIN_CALIBRATION_ANIMALS} reference weighings are needed to calibrate a herd, "
            f"got {raw.size}"
        )
    slope, intercept = _fit(raw, weight)
    loo = np.empty(raw.size)
    for i in range(raw.size):
        keep = np.arange(raw.size) != i
        s, b = _fit(raw[keep], weight[keep])
        loo[i] = weight[i] - (s * raw[i] + b)
    half_width = max(min_half_width_kg, conformal_quantile(np.abs(loo), alpha))
    return HerdCorrection(slope, intercept, int(raw.size), float(half_width))


@dataclass(frozen=True)
class DriftCheck:
    n_weighings: int
    mean_residual_kg: float | None  # scale weight minus prediction
    ci_low_kg: float | None
    ci_high_kg: float | None
    drifted: bool
    message: str


def check_drift(residuals: np.ndarray) -> DriftCheck:
    """Has the herd moved away from its calibration?

    ``residuals`` are scale weight minus calibrated prediction, oldest first,
    for spot weighings made after the calibration. Drift is declared when the
    95 % t-interval of the recent mean excludes zero and the mean is at least
    ``DRIFT_MIN_KG``.
    """
    r = np.asarray(residuals, float)[-DRIFT_WINDOW:]
    if r.size < DRIFT_MIN_ANIMALS:
        mean = float(r.mean()) if r.size else None
        return DriftCheck(
            int(r.size), mean, None, None, False,
            f"{DRIFT_MIN_ANIMALS} spot weighings after calibration are needed for a drift check, "
            f"have {r.size}",
        )
    mean = float(r.mean())
    half = float(stats.t.ppf(0.975, r.size - 1) * r.std(ddof=1) / np.sqrt(r.size))
    lo, hi = mean - half, mean + half
    drifted = (lo > 0 or hi < 0) and abs(mean) >= DRIFT_MIN_KG
    message = (
        f"predictions are off by {mean:+.1f} kg on the last {r.size} weighings; recalibrate"
        if drifted
        else f"no drift on the last {r.size} weighings (mean {mean:+.1f} kg)"
    )
    return DriftCheck(int(r.size), mean, lo, hi, drifted, message)
