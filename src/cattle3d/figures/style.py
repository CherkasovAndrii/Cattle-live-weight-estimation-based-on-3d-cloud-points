"""Shared look of the data figures: page width, fonts, saving, panel letters."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cattle3d.figures.draw import FONT, Lang
from cattle3d.types import View

WIDTH_IN = 6.7  # A4 portrait text width
HERD_COLORS = {"cowdb2": "#1f2937", "cowdb1": "#b45309"}


def herd_name(herd: str, t: Lang) -> str:
    return {
        "cowdb2": t("CowDatabase2, абердин-ангус", "CowDatabase2, Black Angus"),
        "cowdb1": t("CowDatabase, герефорд", "CowDatabase, Hereford"),
    }.get(herd, herd)


def breed(herd: str, t: Lang) -> str:
    return herd_name(herd, t).split(", ")[-1]


def view_name(view: View, t: Lang) -> str:
    return {
        View.LEFT: t("лівий ракурс", "left view"),
        View.RIGHT: t("правий ракурс", "right view"),
        View.TOP: t("верхній ракурс", "top view"),
    }[view]


def dec(x: float, digits: int, t: Lang) -> str:
    """A number with a decimal comma in Ukrainian."""
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if t.code == "uk" else s


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": FONT, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def image_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_label(ax: plt.Axes, letter: str, x: float = 0.0) -> None:
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom", ha="right")
