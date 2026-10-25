"""Figures of the study results, in either language.

``cattle3d evaluate`` writes the English set next to ``results.md``;
``cattle3d figures`` writes both sets for the thesis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cattle3d.evaluation.experiments import conformal_quantile
from cattle3d.evaluation.stats import metrics
from cattle3d.evaluation.study import (
    ENSEMBLE,
    H3B_MARGIN_MAPE,
    MANUAL,
    PHYSICS,
    StudyResult,
)
from cattle3d.figures.draw import Lang
from cattle3d.figures.style import (
    WIDTH_IN,
    apply_style,
    breed,
    dec,
    herd_name,
    panel_label,
    save,
)

MODEL_NAMES = {
    ENSEMBLE: ("3-D ансамбль", "3-D ensemble"),
    PHYSICS: ("фізична a·V + b", "physics a·V + b"),
    "3-D robust linear": ("3-D робастна лінійна", "3-D robust linear"),
    MANUAL: ("заміри рулеткою", "tape measurements"),
}


def model_name(name: str, t: Lang) -> str:
    uk, en = MODEL_NAMES.get(name, (name, name))
    return t(uk, en)


def within_herd(result: StudyResult, out: Path, t: Lang) -> Path:
    """Out-of-fold prediction against scale weight, 3-D ensemble and tape model."""
    apply_style()
    fig, axes = plt.subplots(1, len(result.herds), figsize=(WIDTH_IN, 3.1), layout="constrained")
    for ax, (herd, r), letter in zip(np.atleast_1d(axes), result.herds.items(), t.letters):
        y = r.weight_kg
        for name, color, marker in ((MANUAL, "#94a3b8", "x"), (ENSEMBLE, "#c2410c", "o")):
            m = metrics(y, r.oof[name])
            ax.scatter(y, r.oof[name], s=10 if marker == "o" else 12, color=color, marker=marker,
                       linewidths=0 if marker == "o" else 0.8, alpha=0.85,
                       label=f"{model_name(name, t)}: MAE {dec(m.mae, 1, t)} {t('кг', 'kg')}")
        lo, hi = y.min() - 20, y.max() + 20
        ax.plot([lo, hi], [lo, hi], color="#1f2937", lw=0.8)
        ax.set_title(herd_name(herd, t))
        ax.set_xlabel(t("маса на вагах, кг", "scale weight, kg"))
        ax.set_ylabel(t("прогноз (поза вибіркою), кг", "out-of-fold prediction, kg"))
        ax.legend(frameon=False, loc="upper left")
        panel_label(ax, letter)
    return save(fig, out)


def recalibration(result: StudyResult, out: Path, t: Lang) -> Path:
    """MAPE on the target herd against the number of reference weighings."""
    apply_style()
    fig, axes = plt.subplots(1, len(result.transfers), figsize=(WIDTH_IN, 2.8), layout="constrained",
                             sharey=False)
    labels = {"intercept": t("лише зсув", "intercept only"), "intercept+slope": t("нахил і зсув", "slope + intercept")}
    for ax, tr, letter in zip(np.atleast_1d(axes), result.transfers, t.letters):
        for mode, style in (("intercept", "-o"), ("intercept+slope", "--s")):
            pts = [p for p in tr.recalibration if p.mode == mode]
            ks = [p.k for p in pts]
            ax.plot(ks, [p.mape_mean for p in pts], style, ms=4, lw=1.2, label=labels[mode])
            ax.fill_between(ks, [p.mape_low for p in pts], [p.mape_high for p in pts], alpha=0.15)
        ax.axhline(tr.within_target_mape, color="#1f2937", lw=0.9, label=t("у межах стада (CV)", "within herd (CV)"))
        ax.axhline(tr.within_target_mape + H3B_MARGIN_MAPE, color="#1f2937", lw=0.9, ls=":",
                   label=t(f"межа +{dec(H3B_MARGIN_MAPE, 0, t)} в.п.", f"margin +{H3B_MARGIN_MAPE:g} pp"))
        ax.set_title(f"{tr.source} → {tr.target}".replace("cowdb2", "A").replace("cowdb1", "B")
                     + t(f" (ціль: {breed(tr.target, t)})",
                         f" (target: {breed(tr.target, t)})"))
        ax.set_ylim(bottom=tr.within_target_mape - 0.6)
        ax.set_xlabel(t("контрольних зважувань k", "reference weighings k"))
        ax.set_ylabel("MAPE, %")
        ax.set_xticks([p.k for p in tr.recalibration if p.mode == "intercept"])
        ax.legend(frameon=False, fontsize=6.5)
        panel_label(ax, letter)
    return save(fig, out)


def zero_shot(result: StudyResult, out: Path, t: Lang) -> Path:
    """Predictions for the other herd before any correction: the error is a shift."""
    apply_style()
    fig, axes = plt.subplots(1, len(result.transfers), figsize=(WIDTH_IN, 3.0), layout="constrained")
    for ax, tr, letter in zip(np.atleast_1d(axes), result.transfers, t.letters):
        y = result.herds[tr.target].weight_kg
        for name, pred in tr.zero_shot.items():
            bias = float(np.mean(pred - y))
            ax.scatter(y, pred, s=8, alpha=0.7, linewidths=0,
                       label=f"{model_name(name, t)}: {t('зсув', 'bias')} {bias:+.0f} {t('кг', 'kg')}")
        lo, hi = y.min() - 30, y.max() + 30
        ax.plot([lo, hi], [lo, hi], color="#1f2937", lw=0.8)
        ax.set_title(f"{tr.source} → {tr.target}".replace("cowdb2", "A").replace("cowdb1", "B")
                     + f" ({breed(tr.target, t)})")
        ax.set_xlabel(t("маса на вагах, кг", "scale weight, kg"))
        ax.set_ylabel(t("прогноз без адаптації, кг", "prediction as is, kg"))
        ax.legend(loc="upper left", fontsize=6.5, framealpha=0.85, edgecolor="none")
        panel_label(ax, letter)
    return save(fig, out)


def intervals(result: StudyResult, out: Path, t: Lang, alpha: float = 0.1) -> Path:
    """Every animal's 90 % interval against its true weight; misses in red."""
    apply_style()
    fig, axes = plt.subplots(len(result.herds), 1, figsize=(WIDTH_IN, 4.4), layout="constrained")
    for ax, (herd, r), letter in zip(np.atleast_1d(axes), result.herds.items(), t.letters):
        y, pred = r.weight_kg, r.oof[ENSEMBLE]
        q = conformal_quantile(np.abs(pred - y), alpha)
        order = np.argsort(pred)
        x = np.arange(y.size)
        ax.fill_between(x, pred[order] - q, pred[order] + q, color="#fdba74", alpha=0.5, lw=0,
                        label=t(f"90 % інтервал (±{q:.0f} кг)", f"90 % interval (±{q:.0f} kg)"))
        ax.plot(x, pred[order], color="#c2410c", lw=1, label=t("прогноз", "prediction"))
        miss = np.abs(pred - y)[order] > q
        ax.scatter(x[~miss], y[order][~miss], s=7, color="#1f2937", linewidths=0, label=t("маса на вагах", "scale weight"))
        ax.scatter(x[miss], y[order][miss], s=12, color="#dc2626", linewidths=0,
                   label=t(f"поза інтервалом: {miss.sum()} з {y.size}", f"outside: {miss.sum()} of {y.size}"))
        ax.set_title(herd_name(herd, t))
        ax.set_xlabel(t("тварини, впорядковані за прогнозом", "animals, sorted by prediction"))
        ax.set_ylabel(t("маса, кг", "weight, kg"))
        ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.5)
        panel_label(ax, letter)
    return save(fig, out)


def write_result_figures(result: StudyResult, out_dir: Path, t: Lang) -> list[Path]:
    return [
        within_herd(result, out_dir / "within_herd_oof.png", t),
        recalibration(result, out_dir / "recalibration_curve.png", t),
        zero_shot(result, out_dir / "zero_shot_transfer.png", t),
        intervals(result, out_dir / "prediction_intervals.png", t),
    ]
