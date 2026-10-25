"""Figure of the sensor-error study (``cattle3d sensor``), in either language."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cattle3d.evaluation.sensor import TOLERANCE_KG
from cattle3d.figures.draw import Lang
from cattle3d.figures.style import HERD_COLORS, WIDTH_IN, apply_style, breed, dec, panel_label, save


def sensor_tolerance(results: pd.DataFrame, floor: pd.DataFrame, out: Path, t: Lang) -> Path:
    """MAE increase over clean clouds against each kind of depth error.

    Open markers: some animals could not be segmented at that level."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.7), layout="constrained", sharey=True)
    titles = {
        "noise": t("випадковий шум глибини", "random depth noise"),
        "scale": t("похибка масштабу глибини", "depth scale error"),
        "dropout": t("втрачені пікселі", "lost pixels"),
    }
    xlabels = {
        "noise": t("σ, % відстані", "σ, % of range"),
        "scale": t("похибка, %", "error, %"),
        "dropout": t("частка пікселів, %", "share of pixels, %"),
    }
    for ax, kind, letter in zip(axes, ("noise", "scale", "dropout"), t.letters):
        for herd in results.herd.unique():
            # The clean recording (level 0) is the first point of every curve.
            h = results[results.herd == herd]
            g = pd.concat([h[(h.kind == "noise") & (h.level == 0)], h[(h.kind == kind) & (h.level != 0)]])
            g = g.sort_values("level")
            x = g.level.to_numpy() * 100
            color = HERD_COLORS[herd]
            ax.plot(x, g.delta_mae, "-", color=color, lw=1.1, label=breed(herd, t))
            ax.fill_between(x, g.delta_low, g.delta_high, color=color, alpha=0.12, lw=0)
            lost = g.n_failed.to_numpy() > 0
            ax.scatter(x[~lost], g.delta_mae[~lost], s=12, color=color, zorder=3)
            ax.scatter(x[lost], g.delta_mae[lost], s=14, facecolor="white", edgecolor=color, zorder=3)
            if kind == "scale":
                recal = (g.mae_recal - g.mae_recal_clean).to_numpy()
                ax.plot(x, recal, "--", color=color, lw=1.0,
                        label=t(f"{breed(herd, t)}, після калібрування", f"{breed(herd, t)}, recalibrated"))
        if kind == "noise" and len(floor):
            lo, hi = np.percentile(floor.local_rel * 100, [10, 90])
            ax.axvspan(lo, hi, color="#16a34a", alpha=0.15, lw=0, label=t("шум записів Kinect", "Kinect recordings"))
        ax.axhline(TOLERANCE_KG, color="#1f2937", lw=0.8, ls=":")
        ax.axhline(0, color="#94a3b8", lw=0.6)
        ax.set_title(titles[kind])
        ax.set_xlabel(xlabels[kind])
        panel_label(ax, letter)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    axes[0].set_ylabel(t("зростання MAE, кг", "MAE increase, kg"))
    axes[0].text(axes[0].get_xlim()[1], TOLERANCE_KG, t(f"допуск {dec(TOLERANCE_KG, 1, t)} кг", f"tolerance {TOLERANCE_KG:g} kg"),
                 ha="right", va="bottom", fontsize=6.5, color="#1f2937")
    return save(fig, out)
