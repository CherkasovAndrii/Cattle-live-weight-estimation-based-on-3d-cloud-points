"""The deployable weight model: one saved artifact with prediction, interval
and operating-range check.

Owns: fitting the deployable ensemble, its conformal interval, the training
range it may be trusted in, and the quality gates on input views. Must not
know about files of specific datasets; it takes feature rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cattle3d import __version__
from cattle3d.evaluation.experiments import conformal_quantile, oof_predictions
from cattle3d.evaluation.stats import metrics
from cattle3d.modeling.models import PHYSICS_FEATURE, VOLUME_FEATURES, ensemble_model
from cattle3d.types import Prediction

# Quality gates: below these the segmentation is too thin to trust.
MIN_POINTS = {"left_n_points": 3_000, "right_n_points": 3_000, "top_n_points": 8_000}
# Animals whose trunk volume lies outside the training range by more than
# this fraction of the range are flagged; out-of-range error is mostly bias.
RANGE_TOLERANCE = 0.05
# Herd calibration: below this many reference weighings the herd is not
# calibrated; from SLOPE_CALIBRATION_ANIMALS on, slope and intercept are fitted.
MIN_CALIBRATION_ANIMALS = 3
SLOPE_CALIBRATION_ANIMALS = 10


class ModelInputError(ValueError):
    """The feature row is missing required features."""


@dataclass
class WeightModel:
    features: tuple[str, ...]
    estimator: object
    interval_half_width_kg: float
    alpha: float
    volume_range: tuple[float, float]
    volume_median: float
    volume_iqr: float
    weight_range: tuple[float, float]
    herd_corrections: dict[str, tuple[float, float]] = field(default_factory=dict)  # slope, intercept
    training_herds: tuple[str, ...] = ()
    n_training_animals: int = 0
    cv_metrics: dict[str, float] = field(default_factory=dict)
    created_utc: str = ""
    package_version: str = __version__

    # ---- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, table: pd.DataFrame, alpha: float = 0.1, seed: int = 0) -> "WeightModel":
        features = VOLUME_FEATURES
        X = table[list(features)].to_numpy(float)
        y = table["weight_kg"].to_numpy(float)
        oof = oof_predictions(lambda s: ensemble_model(features, s), X, y, seeds=(seed,))[0]
        volume = table[PHYSICS_FEATURE].to_numpy(float)
        q75, q25 = np.percentile(volume, [75, 25])
        return cls(
            features=features,
            estimator=ensemble_model(features, seed).fit(X, y),
            interval_half_width_kg=conformal_quantile(np.abs(oof - y), alpha),
            alpha=alpha,
            volume_range=(float(volume.min()), float(volume.max())),
            volume_median=float(np.median(volume)),
            volume_iqr=float(q75 - q25),
            weight_range=(float(y.min()), float(y.max())),
            training_herds=tuple(sorted(table["herd"].unique())),
            n_training_animals=int(y.size),
            cv_metrics={k: float(v) for k, v in metrics(y, oof).as_row().items()},
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # ---- herd calibration ---------------------------------------------------
    def calibrate_herd(self, herd: str, rows: pd.DataFrame) -> tuple[float, float]:
        """Store a linear correction for a herd from its reference weighings.

        ``rows`` needs the model features and ``weight_kg``. With 10 or more
        animals both slope and intercept are refitted (transfer between herds
        changes both); with fewer, only the intercept, because a slope from a
        handful of animals is noise. Returns (slope, intercept).
        """
        if len(rows) < MIN_CALIBRATION_ANIMALS:
            raise ValueError(
                f"at least {MIN_CALIBRATION_ANIMALS} reference weighings are needed to calibrate a herd"
            )
        raw = self._raw_predict(rows)
        y = rows["weight_kg"].to_numpy(float)
        if len(rows) >= SLOPE_CALIBRATION_ANIMALS and np.ptp(raw) > 0:
            slope, intercept = np.polyfit(raw, y, 1)
        else:
            slope, intercept = 1.0, float(np.mean(y - raw))
        self.herd_corrections[herd] = (float(slope), float(intercept))
        return self.herd_corrections[herd]

    # ---- prediction ----------------------------------------------------------
    def _raw_predict(self, rows: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in rows.columns]
        if missing:
            raise ModelInputError(f"missing features: {missing}")
        X = rows[list(self.features)].to_numpy(float)
        if not np.isfinite(X).all():
            raise ModelInputError("features contain NaN or infinite values")
        return self.estimator.predict(X)

    def predict_row(self, row: dict[str, float], herd: str | None = None) -> Prediction:
        return self.predict(pd.DataFrame([row]), herd)[0]

    def predict(self, rows: pd.DataFrame, herd: str | None = None) -> list[Prediction]:
        slope, intercept = self.herd_corrections.get(herd or "", (1.0, 0.0))
        raw = slope * self._raw_predict(rows) + intercept
        lo, hi = self.volume_range
        tolerance = RANGE_TOLERANCE * (hi - lo)
        out: list[Prediction] = []
        for (_, row), weight in zip(rows.iterrows(), raw):
            volume = float(row[PHYSICS_FEATURE])
            score = (volume - self.volume_median) / self.volume_iqr if self.volume_iqr else 0.0
            reasons = []
            if not (lo - tolerance <= volume <= hi + tolerance):
                reasons.append("body volume outside the training range")
            for name, minimum in MIN_POINTS.items():
                if name in row and row[name] < minimum:
                    reasons.append(f"too few points in {name.split('_')[0]} view")
            out.append(
                Prediction(
                    weight_kg=float(weight),
                    lower_kg=float(weight - self.interval_half_width_kg),
                    upper_kg=float(weight + self.interval_half_width_kg),
                    in_operating_range=not reasons,
                    size_score=float(score),
                    reason="; ".join(reasons) or None,
                )
            )
        return out

    # ---- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "WeightModel":
        model = joblib.load(path)
        if not isinstance(model, WeightModel):
            raise TypeError(f"{path} does not contain a WeightModel")
        return model

    def card(self) -> dict[str, object]:
        """Human-readable description of what this model was trained on."""
        return {
            "package_version": self.package_version,
            "created_utc": self.created_utc,
            "training_herds": list(self.training_herds),
            "n_training_animals": self.n_training_animals,
            "weight_range_kg": list(self.weight_range),
            "trunk_volume_range_m3": list(self.volume_range),
            "interval": f"{int((1 - self.alpha) * 100)}% conformal, ±{self.interval_half_width_kg:.1f} kg",
            "cross_validated": {k: round(v, 2) for k, v in self.cv_metrics.items()},
            "herd_corrections": {h: {"slope": a, "intercept_kg": b} for h, (a, b) in self.herd_corrections.items()},
        }
