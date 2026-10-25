"""How much depth-sensor error the method tolerates.

Owns: degrading the recorded clouds the way a worse or miscalibrated depth
camera would, measuring the real Kinect v2 noise in the recordings, and
scoring the deployable model on degraded features. Must not read files or
draw; the CLI does that.

Three kinds of error, each applied to all three views of an animal:

- ``noise``: independent random depth error per pixel, a share of the range
  (Kinect v2 random error grows with distance). A pixel moves along its own
  ray, so it stays in the same image cell.
- ``scale``: one systematic depth scale error for the whole scan, as from a
  miscalibrated camera. Every distance is off by the same share.
- ``dropout``: a share of the valid pixels lose their return, as on dark or
  wet hide.

The model is always trained on clean features (cross-validated by animal) and
scored on degraded ones: the question is what happens to a trained model when
the camera in the field is worse than the one it was trained with. Each level
is also scored after the herd recalibration a farm does anyway (slope and
intercept on 20 scale weighings), which absorbs part of a systematic error.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from cattle3d.evaluation.stats import Interval, bootstrap_ci
from cattle3d.geometry.quality import (
    MAX_EDGE_DENSITY,
    MAX_FLOOR_NOISE_REL,
    FloorNoise,
    edge_density,
    floor_noise,
    quality_reasons,
    view_quality,
)
from cattle3d.geometry.segment import _EXPECTED_UP, SegmentationConfig, SegmentationError, fit_floor
from cattle3d.io.ply import read_organized_cloud
from cattle3d.modeling.calibration import fit_correction
from cattle3d.pipeline import analyse_clouds, features_from_clouds
from cattle3d.types import AnimalRecord, OrganizedCloud, View

KINDS = ("noise", "scale", "dropout")
# Practical tolerance: the largest level whose MAE is at most this much worse
# than on clean clouds (upper bound of the paired bootstrap CI).
TOLERANCE_KG = 2.5
RECAL_ANIMALS = 20
CLEAN_KIND = "noise"  # level 0 of any kind is the clean recording
DEFAULT_LEVELS: dict[str, tuple[float, ...]] = {
    "noise": (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03),
    "scale": (-0.05, -0.03, -0.01, 0.01, 0.03, 0.05),
    "dropout": (0.1, 0.2, 0.3, 0.4, 0.5),
}


@dataclass(frozen=True)
class Perturbation:
    kind: str
    level: float  # noise: relative sigma; scale: relative error; dropout: share of pixels

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation {self.kind!r}; expected one of {KINDS}")
        if self.kind == "dropout" and not 0 <= self.level < 1:
            raise ValueError("dropout share must be in [0, 1)")
        if self.kind == "noise" and self.level < 0:
            raise ValueError("noise sigma must not be negative")

    @property
    def label(self) -> str:
        return f"{self.kind} {self.level * 100:+g} %" if self.kind == "scale" else f"{self.kind} {self.level * 100:g} %"


def default_perturbations() -> list[Perturbation]:
    return [Perturbation(kind, level) for kind, levels in DEFAULT_LEVELS.items() for level in levels]


def cache_name(herd: str, p: Perturbation, seed: int = 0) -> str:
    return f"{herd}_{p.kind}_{p.level:+.4f}_s{seed}.csv"


def rng_for(key: str, p: Perturbation, seed: int = 0) -> np.random.Generator:
    """Same animal, same perturbation, same seed: same degraded clouds."""
    return np.random.default_rng([seed, zlib.crc32(f"{key}|{p.kind}|{p.level:.6f}".encode())])


def perturb(cloud: OrganizedCloud, p: Perturbation, rng: np.random.Generator) -> OrganizedCloud:
    xyz = cloud.xyz.astype(float, copy=True)
    valid = cloud.valid
    if p.kind == "noise" and p.level > 0:
        factor = 1.0 + rng.normal(0.0, p.level, size=valid.sum())
        xyz[valid] *= np.maximum(factor, 0.05)[:, None]
    elif p.kind == "scale":
        xyz[valid] *= 1.0 + p.level
    elif p.kind == "dropout" and p.level > 0:
        drop = valid.copy()
        drop[valid] = rng.random(valid.sum()) < p.level
        xyz[drop] = 0.0
    return OrganizedCloud(xyz)


def perturb_views(
    clouds: dict[View, OrganizedCloud], p: Perturbation, key: str, seed: int = 0
) -> dict[View, OrganizedCloud]:
    rng = rng_for(key, p, seed)
    return {view: perturb(clouds[view], p, rng) for view in sorted(clouds, key=lambda v: v.value)}


def _degraded_row(job: tuple[AnimalRecord, Perturbation, int]) -> dict[str, object]:
    record, p, seed = job
    row: dict[str, object] = {"herd": record.herd, "animal_id": record.animal_id, "weight_kg": record.weight_kg}
    try:
        clouds = {view: read_organized_cloud(path) for view, path in record.view_paths.items()}
        row.update(features_from_clouds(perturb_views(clouds, p, record.key, seed)))
        row["error"] = ""
    except Exception as exc:  # a degraded view that cannot be segmented is a result, not a crash
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def extract_degraded(
    records: tuple[AnimalRecord, ...], p: Perturbation, seed: int = 0, workers: int = 4
) -> pd.DataFrame:
    """Feature table of a herd recorded with a degraded camera, one row per animal."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_degraded_row, [(r, p, seed) for r in records], chunksize=2))
    return pd.DataFrame(rows)


# ---- the real noise of the recordings ------------------------------------------------
def recorded_quality(
    cloud: OrganizedCloud, view: View, config: SegmentationConfig = SegmentationConfig()
) -> tuple[FloorNoise, float]:
    """Floor noise and edge density of one recorded view (see ``geometry.quality``)."""
    rng_m = np.linalg.norm(cloud.xyz, axis=2)
    usable = cloud.valid & (rng_m > config.min_range_m) & (rng_m < config.max_range_m)
    plane, _ = fit_floor(cloud.xyz[usable], _EXPECTED_UP[view], config)
    noise = floor_noise(cloud, plane, config.floor_inlier_m, config.min_range_m, config.max_range_m)
    if noise is None:
        raise SegmentationError("too few floor pixels to measure noise")
    return noise, edge_density(cloud)


# ---- scoring degraded features -------------------------------------------------------
@dataclass(frozen=True)
class LevelResult:
    herd: str
    perturbation: Perturbation
    n: int  # animals scored
    n_failed: int  # animals whose degraded clouds could not be segmented
    mae: Interval  # trained on clean, scored on degraded, no herd correction
    delta_mae: Interval  # paired against the same animals on clean clouds
    bias_kg: float  # mean prediction minus scale weight
    mae_recal: float  # after a slope+intercept on RECAL_ANIMALS reference animals
    mae_recal_clean: float  # the same reference draws on clean clouds
    mean_baseline_mae: float  # predicting the herd mean


def clean_oof_models(
    factory: Callable[[int], object], X: np.ndarray, y: np.ndarray, seeds: tuple[int, ...], n_splits: int = 5
) -> list[tuple[int, np.ndarray, object]]:
    """Models fitted on clean training folds: (seed, test indices, model)."""
    out = []
    for seed in seeds:
        for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(X):
            out.append((seed, test, factory(seed).fit(X[train], y[train])))
    return out


def oof_on(models: list[tuple[int, np.ndarray, object]], X: np.ndarray, seeds: tuple[int, ...]) -> np.ndarray:
    """Seed-averaged out-of-fold prediction of feature matrix ``X`` (rows may be NaN)."""
    pred = np.zeros((len(seeds), X.shape[0]))
    ok = np.isfinite(X).all(axis=1)
    for seed, test, model in models:
        rows = test[ok[test]]
        if rows.size:
            pred[seeds.index(seed), rows] = model.predict(X[rows])
    pred = pred.mean(axis=0)
    pred[~ok] = np.nan
    return pred


def recalibrated_mae(pred: np.ndarray, y: np.ndarray, draws: int = 200, k: int = RECAL_ANIMALS, seed: int = 0) -> float:
    """Mean MAE on the other animals after fitting slope and intercept on k
    random reference animals, as a farm does with its first scale weighings."""
    rng = np.random.default_rng(seed)
    n = y.size
    maes = []
    for _ in range(draws):
        ref = rng.choice(n, k, replace=False)
        rest = np.setdiff1d(np.arange(n), ref)
        c = fit_correction(pred[ref], y[ref])
        maes.append(float(np.mean(np.abs(c.slope * pred[rest] + c.intercept_kg - y[rest]))))
    return float(np.mean(maes))


def score_level(
    herd: str, p: Perturbation, y: np.ndarray, clean_pred: np.ndarray, degraded_pred: np.ndarray, mean_pred: np.ndarray,
) -> LevelResult:
    ok = np.isfinite(degraded_pred)
    if ok.sum() < RECAL_ANIMALS + 10:  # too few animals left to score: the level is a failure
        nan = Interval(float("nan"), float("nan"), float("nan"))
        return LevelResult(herd, p, int(ok.sum()), int((~ok).sum()), nan, nan, float("nan"), float("nan"),
                           float("nan"), float("nan"))
    yk, ck, dk = y[ok], clean_pred[ok], degraded_pred[ok]
    err_d, err_c = np.abs(dk - yk), np.abs(ck - yk)
    return LevelResult(
        herd=herd, perturbation=p, n=int(ok.sum()), n_failed=int((~ok).sum()),
        mae=bootstrap_ci(err_d), delta_mae=bootstrap_ci(err_d - err_c), bias_kg=float(np.mean(dk - yk)),
        mae_recal=recalibrated_mae(dk, yk), mae_recal_clean=recalibrated_mae(ck, yk),
        mean_baseline_mae=float(np.mean(np.abs(mean_pred[ok] - yk))),
    )


def tolerated_level(
    results: list[LevelResult], kind: str, tolerance_kg: float = TOLERANCE_KG, recalibrated: bool = False,
) -> float | None:
    """Largest level of ``kind`` (by magnitude) up to which every level keeps the
    upper CI bound of the MAE increase within ``tolerance_kg`` and loses no animal.
    With ``recalibrated``, the point estimate of the increase after the herd
    recalibration is compared instead (a farm calibrates with its own camera)."""
    levels = sorted((r for r in results if r.perturbation.kind == kind), key=lambda r: abs(r.perturbation.level))
    best = None
    for r in levels:
        worse = r.mae_recal - r.mae_recal_clean if recalibrated else r.delta_mae.high
        if r.n_failed or not worse <= tolerance_kg:
            break
        best = abs(r.perturbation.level)
    return best


def results_frame(results: list[LevelResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "herd": r.herd, "kind": r.perturbation.kind, "level": r.perturbation.level, "n": r.n, "n_failed": r.n_failed,
        "mae": r.mae.estimate, "mae_low": r.mae.low, "mae_high": r.mae.high,
        "delta_mae": r.delta_mae.estimate, "delta_low": r.delta_mae.low, "delta_high": r.delta_mae.high,
        "bias_kg": r.bias_kg, "mae_recal": r.mae_recal, "mae_recal_clean": r.mae_recal_clean,
        "mean_baseline_mae": r.mean_baseline_mae,
    } for r in results])


def score_herd(
    herd: str, clean: pd.DataFrame, degraded: dict[Perturbation, pd.DataFrame], columns: tuple[str, ...],
    factory: Callable[[int], object], baseline: Callable[[int], object], seeds: tuple[int, ...] = (0, 1, 2),
) -> list[LevelResult]:
    """All levels of one herd. Animals that fail on clean clouds are left out
    everywhere; animals that fail only on degraded clouds count as failures."""
    clean = clean[clean["error"].fillna("") == ""].reset_index(drop=True)
    y = clean["weight_kg"].to_numpy(float)
    Xc = clean[list(columns)].to_numpy(float)
    models = clean_oof_models(factory, Xc, y, seeds)
    clean_pred = oof_on(models, Xc, seeds)
    mean_pred = oof_on(clean_oof_models(baseline, Xc, y, seeds), Xc, seeds)
    out = []
    for p, table in degraded.items():
        rows = table.set_index(table["animal_id"].astype(str))
        rows = rows[rows["error"].fillna("") == ""]
        Xd = np.full_like(Xc, np.nan)
        ids = clean["animal_id"].astype(str)
        present = ids.isin(rows.index).to_numpy()
        if present.any():  # at the worst levels no animal is left, and neither are feature columns
            Xd[present] = rows.loc[ids[present], list(columns)].to_numpy(float)
        out.append(score_level(herd, p, y, clean_pred, oof_on(models, Xd, seeds), mean_pred))
    return out


def floor_noise_frame(records: tuple[AnimalRecord, ...]) -> pd.DataFrame:
    rows = []
    for record in records:
        for view, path in record.view_paths.items():
            try:
                f, edges = recorded_quality(read_organized_cloud(path), view)
            except SegmentationError:
                continue
            rows.append({"herd": record.herd, "animal_id": record.animal_id, "view": view.value, "range_m": f.range_m,
                         "plane_sigma_m": f.plane_sigma_m, "local_sigma_m": f.local_sigma_m,
                         "plane_rel": f.plane_rel, "local_rel": f.local_rel, "edge_density": edges})
    return pd.DataFrame(rows)


QUALITY_LEVELS = (Perturbation("noise", 0.0), Perturbation("noise", 0.005), Perturbation("noise", 0.01),
                  Perturbation("dropout", 0.1), Perturbation("dropout", 0.2))


def _quality_rows(record: AnimalRecord) -> list[dict[str, object]]:
    clouds = {view: read_organized_cloud(path) for view, path in record.view_paths.items()}
    rows = []
    for p in QUALITY_LEVELS:
        degraded = perturb_views(clouds, p, record.key)
        try:
            _, traces = analyse_clouds(degraded)
            reasons = quality_reasons([view_quality(degraded[v], v, t.result.floor) for v, t in traces.items()])
            outcome = "flagged" if reasons else "passed"
        except (SegmentationError, ValueError):
            outcome = "refused"
        rows.append({"herd": record.herd, "animal_id": record.animal_id, "kind": p.kind, "level": p.level,
                     "outcome": outcome})
    return rows


def quality_check_frame(records: tuple[AnimalRecord, ...], workers: int = 4) -> pd.DataFrame:
    """What the per-scan recording check (``geometry.quality``) says on clean
    and degraded clouds of every animal."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame([row for rows in pool.map(_quality_rows, records, chunksize=2) for row in rows])


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %"


def sensor_markdown(results: list[LevelResult], floor: pd.DataFrame, quality: pd.DataFrame | None = None) -> str:
    df = results_frame(results)
    lines = [
        "# Permissible depth-sensor error",
        "",
        "Generated by `cattle3d sensor`. The deployable 3-D ensemble is trained on clean clouds "
        "(5-fold CV by animal, 3 seeds) and scored on clouds degraded as a worse camera would record them. "
        f"Tolerance: the MAE increase over clean clouds, upper bound of the paired 95 % bootstrap CI, stays "
        f"within {TOLERANCE_KG:g} kg and no animal is lost. 'After recalibration' fits slope and intercept on "
        f"{RECAL_ANIMALS} random reference animals of the degraded herd (200 draws) and scores the rest.",
        "",
        "## Measured noise of the recordings (floor pixels)",
        "",
        "| herd | view | n | range, m | sigma around plane, mm | local sigma, mm | local, % of range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (herd, view), g in floor.groupby(["herd", "view"]):
        lines.append(f"| {herd} | {view} | {len(g)} | {g.range_m.median():.2f} | {g.plane_sigma_m.median() * 1000:.1f} "
                     f"| {g.local_sigma_m.median() * 1000:.1f} | {g.local_rel.median() * 100:.2f} |")
    lines += ["", "Medians over animals. The spread around the plane includes floor unevenness and bedding, so it "
              "is an upper bound; the local estimate (second differences of adjacent floor pixels) is closer to "
              "pixel noise but still includes fine floor texture.", ""]
    lines += ["## Tolerated error", "",
              "| herd | herd calibration | random noise (σ) | depth scale error | lost pixels |", "|---|---|---:|---:|---:|"]
    for herd in df.herd.unique():
        hr = [r for r in results if r.herd == herd]
        for recal in (False, True):
            cells = [tolerated_level(hr, k, recalibrated=recal) for k in KINDS]
            lines.append(f"| {herd} | {'after recalibration' if recal else 'none'} | " + " | ".join(
                "below the first level" if c is None else f"≤ {_pct(c, 2 if c < 0.01 else 1)}" for c in cells) + " |")
    if quality is not None and len(quality):
        lines += ["", "## Per-scan recording check", "",
                  f"Every scan now measures floor noise and edge density in each view (`cattle3d.geometry.quality`) and "
                  f"is flagged above {MAX_FLOOR_NOISE_REL * 100:g} % of range or {MAX_EDGE_DENSITY:g}. Recorded maxima over "
                  f"all views of both herds: floor noise {floor.local_rel.max() * 100:.2f} %, edge density "
                  f"{floor.edge_density.max():.2f}. Outcome per animal:", "",
                  "| herd | recording | passed | flagged | refused |", "|---|---|---:|---:|---:|"]
        for (herd, kind, level), g in quality.groupby(["herd", "kind", "level"], sort=False):
            c = g.outcome.value_counts()
            lines.append(f"| {herd} | {Perturbation(kind, level).label if level else 'as recorded'} | {c.get('passed', 0)} "
                         f"| {c.get('flagged', 0)} | {c.get('refused', 0)} |")
        lines += ["", "The limits were chosen on these same animals, between the tolerated and the first harmful "
                  "level, so the rates are in-sample."]
    lines += ["", "## All levels", "",
              "| herd | error | animals scored | lost | MAE, kg [95 % CI] | ΔMAE vs clean, kg [95 % CI] | bias, kg "
              "| MAE after recalibration (clean), kg | predict-mean MAE, kg |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r.herd} | {r.perturbation.label} | {r.n} | {r.n_failed} | {r.mae} | {r.delta_mae} | {r.bias_kg:+.1f} "
            f"| {r.mae_recal:.1f} ({r.mae_recal_clean:.1f}) | {r.mean_baseline_mae:.1f} |")
    return "\n".join(lines) + "\n"
