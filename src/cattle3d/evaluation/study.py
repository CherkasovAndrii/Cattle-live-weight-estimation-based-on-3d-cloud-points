"""The full study on the two public herds: H1, H3a-d and H4.

Owns: which experiments are run, on which feature sets, and what is
collected from each. Returns plain data; rendering is in ``report``.
Must not read files or draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from cattle3d.evaluation.experiments import (
    conformal_coverage,
    fit_predict,
    oof_predictions,
    recalibration_curve,
)
from cattle3d.evaluation.stats import (
    Interval,
    PairedComparison,
    bootstrap_ci,
    holm,
    metrics,
    paired_comparison,
)
from cattle3d.io.datasets import COMMON_MANUAL, HERD_A, HERD_B
from cattle3d.modeling.models import (
    PHYSICS_FEATURE,
    VOLUME_FEATURES,
    ColumnSubset,
    ModelFactory,
    ensemble_model,
    lgbm_model,
    physics_model,
    robust_linear_model,
)

HERD_NAMES = {HERD_A: "CowDatabase2 (Black Angus)", HERD_B: "CowDatabase (Hereford)"}
MANUAL_COLUMNS = tuple(f"manual_{m}" for m in COMMON_MANUAL)
# Pre-registered margin for H3b: recalibrated transfer is "as good" as
# within-herd CV when its MAPE is at most this many points higher.
H3B_MARGIN_MAPE = 1.0
RECAL_KS = (5, 10, 20, 30)


def _physics(seed: int):
    return ColumnSubset(physics_model(seed), (VOLUME_FEATURES.index(PHYSICS_FEATURE),))


MODELS: dict[str, tuple[ModelFactory, tuple[str, ...]]] = {
    "predict-mean": (lambda s: DummyRegressor(), VOLUME_FEATURES),
    "manual measurements (robust linear)": (robust_linear_model, MANUAL_COLUMNS),
    "physics: a·volume + b": (_physics, VOLUME_FEATURES),
    "3-D robust linear": (robust_linear_model, VOLUME_FEATURES),
    "3-D LightGBM (nested tuning)": (lgbm_model, VOLUME_FEATURES),
    "3-D ensemble (deployable)": (lambda s: ensemble_model(VOLUME_FEATURES, s), VOLUME_FEATURES),
}
ENSEMBLE = "3-D ensemble (deployable)"
MANUAL = "manual measurements (robust linear)"
PHYSICS = "physics: a·volume + b"


@dataclass
class HerdResult:
    herd: str
    n: int
    weight_kg: np.ndarray
    animal_ids: list[str]
    oof: dict[str, np.ndarray]  # seed-averaged OOF prediction per model
    seed_mae_sd: dict[str, float]
    mae_ci: dict[str, Interval]
    h1: dict[str, PairedComparison]
    conformal: tuple[float, float, float, float]


@dataclass
class TransferResult:
    source: str
    target: str
    zero_shot: dict[str, np.ndarray]
    bias_ci: dict[str, Interval]
    physics_vs_ensemble: PairedComparison
    recalibration: list
    within_target_mape: float
    recalibrated_conformal: tuple[float, float, float, float]


@dataclass
class StudyResult:
    herds: dict[str, HerdResult]
    transfers: list[TransferResult]
    pooled: dict[str, dict[str, float]]
    h1_holm: dict[str, float] = field(default_factory=dict)
    h3_holm: dict[str, float] = field(default_factory=dict)


def _xy(df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    return df[list(cols)].to_numpy(float), df["weight_kg"].to_numpy(float)


def within_herd(df: pd.DataFrame, herd: str, seeds: tuple[int, ...]) -> HerdResult:
    y = df["weight_kg"].to_numpy(float)
    oof, sd, ci = {}, {}, {}
    for name, (factory, cols) in MODELS.items():
        X, _ = _xy(df, cols)
        per_seed = oof_predictions(factory, X, y, seeds=seeds)
        oof[name] = per_seed.mean(axis=0)
        sd[name] = float(np.std([np.abs(p - y).mean() for p in per_seed], ddof=1)) if len(seeds) > 1 else 0.0
        ci[name] = bootstrap_ci(np.abs(oof[name] - y))
    h1 = {
        "ensemble vs manual": paired_comparison(y, oof[ENSEMBLE], oof[MANUAL]),
        "physics vs manual": paired_comparison(y, oof[PHYSICS], oof[MANUAL]),
        "ensemble vs physics": paired_comparison(y, oof[ENSEMBLE], oof[PHYSICS]),
    }
    return HerdResult(
        herd=herd, n=int(y.size), weight_kg=y, animal_ids=df["animal_id"].astype(str).tolist(),
        oof=oof, seed_mae_sd=sd, mae_ci=ci, h1=h1,
        conformal=conformal_coverage(oof[ENSEMBLE], y, alpha=0.1, calib_size=30),
    )


def transfer(source: pd.DataFrame, target: pd.DataFrame, target_within: HerdResult) -> TransferResult:
    Xs, ys = _xy(source, VOLUME_FEATURES)
    Xt, yt = _xy(target, VOLUME_FEATURES)
    zero = {
        PHYSICS: fit_predict(_physics, Xs, ys, Xt),
        "3-D robust linear": fit_predict(robust_linear_model, Xs, ys, Xt),
        ENSEMBLE: fit_predict(lambda s: ensemble_model(VOLUME_FEATURES, s), Xs, ys, Xt),
    }
    bias = {name: bootstrap_ci(pred - yt) for name, pred in zero.items()}
    within_mape = metrics(yt, target_within.oof[ENSEMBLE]).mape

    # Recalibrated transfer conformal check: calibrate offset and interval on
    # 20 target animals, measure coverage on the rest.
    rng = np.random.default_rng(0)
    covs, widths = [], []
    for _ in range(500):
        ref = rng.choice(yt.size, 20, replace=False)
        rest = np.setdiff1d(np.arange(yt.size), ref)
        corrected = zero[ENSEMBLE] + (yt[ref] - zero[ENSEMBLE][ref]).mean()
        res = np.abs(corrected - yt)
        r = np.sort(res[ref])
        q = r[min(int(np.ceil(21 * 0.9)), 20) - 1]
        covs.append((res[rest] <= q).mean())
        widths.append(q)
    lo, hi = np.percentile(covs, [2.5, 97.5])
    return TransferResult(
        source=str(source["herd"].iloc[0]),
        target=str(target["herd"].iloc[0]),
        zero_shot=zero,
        bias_ci=bias,
        physics_vs_ensemble=paired_comparison(yt, zero[PHYSICS], zero[ENSEMBLE]),
        recalibration=recalibration_curve(
            zero[ENSEMBLE], yt, RECAL_KS, mape_threshold=within_mape + H3B_MARGIN_MAPE
        ),
        within_target_mape=within_mape,
        recalibrated_conformal=(float(np.mean(covs)), float(lo), float(hi), float(np.mean(widths))),
    )


def pooled(table: pd.DataFrame, seeds: tuple[int, ...]) -> dict[str, dict[str, float]]:
    """Grouped CV on both herds together, scored per herd (H3d)."""
    X, y = _xy(table, VOLUME_FEATURES)
    pred = oof_predictions(lambda s: ensemble_model(VOLUME_FEATURES, s), X, y, seeds=seeds).mean(axis=0)
    out = {}
    for herd in table["herd"].unique():
        mask = (table["herd"] == herd).to_numpy()
        out[herd] = metrics(y[mask], pred[mask]).as_row()
    out["both"] = metrics(y, pred).as_row()
    return out


def run_study(table: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> StudyResult:
    herds = {}
    for herd in (HERD_A, HERD_B):
        sub = table[table["herd"] == herd].reset_index(drop=True)
        herds[herd] = within_herd(sub, herd, seeds)
    a = table[table["herd"] == HERD_A].reset_index(drop=True)
    b = table[table["herd"] == HERD_B].reset_index(drop=True)
    transfers = [transfer(a, b, herds[HERD_B]), transfer(b, a, herds[HERD_A])]

    h1_p = {f"{h}: {k}": v.wilcoxon_p for h, r in herds.items() for k, v in r.h1.items()}
    h3_p = {f"{t.source}->{t.target}: physics vs ensemble": t.physics_vs_ensemble.wilcoxon_p for t in transfers}
    return StudyResult(herds, transfers, pooled(table, seeds), holm(h1_p), holm(h3_p))
