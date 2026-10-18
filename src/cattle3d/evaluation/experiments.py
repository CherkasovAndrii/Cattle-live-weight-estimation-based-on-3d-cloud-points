"""Cross-validation, transfer and recalibration experiments.

Owns: how models are trained and scored so that no animal is ever in both
the training and the scoring set. One row of a feature table is one animal,
so K-fold over rows is grouped by animal. Must not know about files or plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from cattle3d.modeling.models import ModelFactory


def oof_predictions(
    factory: ModelFactory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_jobs: int = -1,
) -> np.ndarray:
    """Out-of-fold predictions, shape (len(seeds), n_animals).

    Any tuning inside the estimator (GridSearchCV) runs on the training folds
    only, so the outer score is a nested-CV estimate.
    """
    jobs = [
        (s, seed, train, test)
        for s, seed in enumerate(seeds)
        for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(X)
    ]

    def run(seed: int, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        return factory(seed).fit(X[train], y[train]).predict(X[test])

    preds = Parallel(n_jobs=n_jobs)(delayed(run)(seed, tr, te) for _, seed, tr, te in jobs)
    out = np.empty((len(seeds), y.size))
    for (s, _, _, test), pred in zip(jobs, preds):
        out[s, test] = pred
    return out


def fit_predict(
    factory: ModelFactory, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int = 0
) -> np.ndarray:
    return factory(seed).fit(X_train, y_train).predict(X_test)


@dataclass(frozen=True)
class RecalibrationPoint:
    k: int
    mode: str  # "intercept" or "intercept+slope"
    mae_mean: float
    mae_low: float
    mae_high: float
    mape_mean: float
    mape_low: float
    mape_high: float
    bias_mean: float
    share_within: float  # share of draws whose MAPE is at most ``mape_threshold``


def recalibration_curve(
    zero_shot_pred: np.ndarray,
    y_target: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20, 30),
    repetitions: int = 500,
    seed: int = 0,
    mape_threshold: float = float("inf"),
) -> list[RecalibrationPoint]:
    """Correct a transferred model with k reference weighings of the new herd.

    For each repetition, k target animals are drawn as the reference set; a
    correction is fitted on them and scored on the remaining target animals.
    """
    rng = np.random.default_rng(seed)
    n = y_target.size
    points: list[RecalibrationPoint] = []
    for mode in ("intercept", "intercept+slope"):
        for k in ks:
            maes, mapes, biases = [], [], []
            for _ in range(repetitions):
                ref = rng.choice(n, k, replace=False)
                rest = np.setdiff1d(np.arange(n), ref)
                if mode == "intercept":
                    corrected = zero_shot_pred[rest] + (y_target[ref] - zero_shot_pred[ref]).mean()
                else:
                    slope, intercept = np.polyfit(zero_shot_pred[ref], y_target[ref], 1)
                    corrected = slope * zero_shot_pred[rest] + intercept
                err = corrected - y_target[rest]
                maes.append(np.abs(err).mean())
                mapes.append((np.abs(err) / y_target[rest]).mean() * 100)
                biases.append(err.mean())
            lo, hi = np.percentile(maes, [2.5, 97.5])
            plo, phi = np.percentile(mapes, [2.5, 97.5])
            points.append(
                RecalibrationPoint(
                    k, mode, float(np.mean(maes)), float(lo), float(hi),
                    float(np.mean(mapes)), float(plo), float(phi), float(np.mean(biases)),
                    float(np.mean(np.array(mapes) <= mape_threshold)),
                )
            )
    return points


def conformal_quantile(abs_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal half-width: the ceil((n+1)(1-alpha))-th smallest residual."""
    r = np.sort(np.asarray(abs_residuals, float))
    rank = int(np.ceil((r.size + 1) * (1 - alpha)))
    return float(r[min(rank, r.size) - 1])


def conformal_coverage(
    pred: np.ndarray, y: np.ndarray, alpha: float = 0.1, calib_size: int = 30,
    repetitions: int = 500, seed: int = 0,
) -> tuple[float, float, float, float]:
    """Repeated split: calibrate on ``calib_size`` animals, measure coverage on the rest.

    Returns mean coverage, its 2.5/97.5 percentiles over repetitions, and the
    mean interval half-width in kg.
    """
    rng = np.random.default_rng(seed)
    abs_res = np.abs(pred - y)
    covs, widths = [], []
    for _ in range(repetitions):
        cal = rng.choice(y.size, calib_size, replace=False)
        rest = np.setdiff1d(np.arange(y.size), cal)
        q = conformal_quantile(abs_res[cal], alpha)
        covs.append((abs_res[rest] <= q).mean())
        widths.append(q)
    lo, hi = np.percentile(covs, [2.5, 97.5])
    return float(np.mean(covs)), float(lo), float(hi), float(np.mean(widths))


def frame(df: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    sub = df.dropna(subset=list(columns) + ["weight_kg"])
    if len(sub) != len(df):
        raise ValueError(f"{len(df) - len(sub)} animals have missing values in {columns}")
    return sub[list(columns)].to_numpy(float), sub["weight_kg"].to_numpy(float)

