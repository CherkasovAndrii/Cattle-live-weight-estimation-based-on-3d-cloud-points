"""Figures of each pipeline stage on a real animal.

Every panel is produced by the library's own functions (``trace_view``,
``width_profile``, ``trunk_interval``, ``cross_section_volume``), so what the
figure shows is exactly what the model sees.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from cattle3d.features.geometry_features import (
    MID_TRUNK,
    TRUNK_REL_WIDTH,
    trunk_interval,
    width_profile,
)
from cattle3d.figures.draw import Lang
from cattle3d.figures.style import (
    HERD_COLORS,
    WIDTH_IN,
    apply_style,
    dec,
    herd_name,
    image_axes,
    panel_label,
    save,
    view_name,
)
from cattle3d.geometry.segment import SegmentationConfig, SegmentationTrace, trace_view
from cattle3d.io.datasets import audited_cowdb1_join, load_cowdatabase
from cattle3d.io.ply import read_organized_cloud
from cattle3d.modeling.models import PHYSICS_FEATURE
from cattle3d.types import ALL_VIEWS, AnimalRecord, View

ANIMAL = "#f97316"
FRAGMENT = "#dc2626"


def representative(records: tuple[AnimalRecord, ...]) -> AnimalRecord:
    """The animal with the median weight: a typical, not a cherry-picked, example."""
    weights = np.array([r.weight_kg for r in records])
    return records[int(np.argsort(weights)[len(weights) // 2])]


def traces(record: AnimalRecord, config: SegmentationConfig = SegmentationConfig()) -> dict[View, SegmentationTrace]:
    return {v: trace_view(read_organized_cloud(record.view_paths[v]), v, config) for v in ALL_VIEWS}


# ---------------------------------------------------------------------------
def raw_views(record: AnimalRecord, out: Path, t: Lang) -> Path:
    """The three depth images as the camera delivers them."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 1.95), layout="constrained")
    for ax, view, letter in zip(axes, ALL_VIEWS, t.letters):
        xyz = read_organized_cloud(record.view_paths[view]).xyz
        depth = np.linalg.norm(xyz, axis=2)
        depth[depth == 0] = np.nan
        im = ax.imshow(depth, cmap="viridis", vmin=0.5, vmax=4.0)
        ax.set_facecolor("black")
        ax.set_title(view_name(view, t))
        image_axes(ax)
        panel_label(ax, letter)
    bar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    bar.set_label(t("відстань від камери, м", "distance from camera, m"))
    return save(fig, out)


def segmentation_steps(record: AnimalRecord, out: Path, t: Lang, views: tuple[View, ...] = (View.LEFT, View.TOP)) -> Path:
    """Height above the fitted floor, the band kept, surfaces, and the animal."""
    apply_style()
    tr = traces(record)
    fig, axes = plt.subplots(len(views), 4, figsize=(WIDTH_IN, 1.45 * len(views) + 0.35), layout="constrained")
    titles = [
        t("висота над підлогою", "height above floor"),
        t("смуга 0,06–2,2 м", "band 0.06–2.2 m") if t.code == "en" else "смуга 0,06–2,2 м",
        t("зв’язні поверхні", "connected surfaces"),
        t("тварина + фрагменти", "animal + fragments"),
    ]
    rng = np.random.default_rng(3)
    for row, view in zip(np.atleast_2d(axes), views):
        s = tr[view]
        height = np.where(s.valid, s.height, np.nan)
        row[0].set_facecolor("white")
        im = row[0].imshow(height, cmap="magma", vmin=-0.05, vmax=2.0)
        band = np.ones(s.valid.shape + (3,))
        band[s.valid] = 0.82
        band[s.above] = (0.15, 0.35, 0.75)
        row[1].imshow(band)
        n = int(s.labels.max()) + 1
        colors = rng.uniform(0.25, 0.95, size=(max(n, 1), 3))
        comp = np.ones(s.valid.shape + (3,))
        comp[s.labels >= 0] = colors[s.labels[s.labels >= 0]]
        row[2].imshow(comp)
        final = np.ones(s.valid.shape + (3,))
        final[s.valid] = 0.85
        final[s.animal] = matplotlib.colors.to_rgb(FRAGMENT)
        final[s.picked] = matplotlib.colors.to_rgb(ANIMAL)
        row[3].imshow(final)
        for ax in row:
            image_axes(ax)
        row[0].set_ylabel(view_name(view, t))
    for ax, title in zip(np.atleast_2d(axes)[0], titles):
        ax.set_title(title)
    bar = fig.colorbar(im, ax=np.atleast_2d(axes)[-1, 0], shrink=0.9, location="bottom", pad=0.02, aspect=18)
    bar.set_label(t("висота, м", "height, m"))
    handles = [
        matplotlib.patches.Patch(color=ANIMAL, label=t("обрана поверхня", "picked surface")),
        matplotlib.patches.Patch(color=FRAGMENT, label=t("приєднані фрагменти", "merged fragments")),
    ]
    fig.legend(handles=handles, loc="outside lower right", ncol=2, frameon=False)
    return save(fig, out)


def body_frame(record: AnimalRecord, out: Path, t: Lang) -> Path:
    """The animal's points in metres: top view from above, side view from the side."""
    apply_style()
    tr = traces(record)
    top = tr[View.TOP].result.body_points
    side = tr[View.LEFT].result.body_points
    lo, hi = trunk_interval(top)
    a, b = lo + MID_TRUNK[0] * (hi - lo), lo + MID_TRUNK[1] * (hi - lo)

    fig = plt.figure(figsize=(WIDTH_IN, 4.3), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    sub = top[:: max(1, top.shape[0] // 6000)]
    ax3d.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], cmap="viridis", s=0.6, linewidths=0)
    ax3d.set_box_aspect((np.ptp(sub[:, 0]), np.ptp(sub[:, 1]), max(np.ptp(sub[:, 2]), 0.05) * 1.4))
    ax3d.view_init(elev=28, azim=-60)
    ax3d.set_xlabel(t("вздовж тіла, м", "along body, m"), labelpad=-6)
    ax3d.set_ylabel(t("поперек, м", "across, m"), labelpad=-6)
    ax3d.set_zlabel(t("висота, м", "height, m"), labelpad=-8)
    ax3d.set_xticks([-0.5, 0, 0.5])
    ax3d.set_yticks([-0.3, 0, 0.3])
    ax3d.set_zticks(np.unique(np.round(np.percentile(sub[:, 2], [5, 95]), 1)))
    ax3d.tick_params(pad=-3, labelsize=6)
    ax3d.set_title(t("верхній ракурс у системі тіла", "top view in the body frame"))
    ax3d.text2D(0.0, 0.97, t.letters[0], transform=ax3d.transAxes, fontsize=9, weight="bold")

    ax = fig.add_subplot(grid[0, 1])
    sc = ax.scatter(top[:, 0], top[:, 1], c=top[:, 2], cmap="viridis", s=0.3, linewidths=0, rasterized=True)
    ax.axvspan(lo, hi, color="#93c5fd", alpha=0.25, lw=0, label=t("тулуб", "trunk"))
    ax.axvspan(a, b, color="#2563eb", alpha=0.18, lw=0, label=t("середина тулуба", "mid-trunk"))
    ax.set_aspect("equal")
    ax.set_xlabel(t("вздовж тіла, м", "along body, m"))
    ax.set_ylabel(t("поперек, м", "across, m"))
    ax.set_title(t("вигляд згори: тулуб і його середина", "from above: trunk and mid-trunk"))
    ax.legend(loc="lower right", ncol=2, framealpha=0.85, edgecolor="none")
    fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.01).set_label(t("висота, м", "height, m"))
    panel_label(ax, t.letters[1])

    ax = fig.add_subplot(grid[1, 1])
    ax.scatter(side[:, 0], side[:, 2], c="#475569", s=0.2, linewidths=0, rasterized=True)
    ax.axhline(0, color="#a16207", lw=1)
    ax.text(side[:, 0].min(), 0.03, t("підлога", "floor"), fontsize=7, color="#a16207")
    ax.set_aspect("equal")
    ax.set_xlabel(t("вздовж тіла, м", "along body, m"))
    ax.set_ylabel(t("висота, м", "height, m"))
    ax.set_title(t("лівий ракурс: профіль тіла", "left view: body profile"))
    panel_label(ax, t.letters[2])
    return save(fig, out)


def cross_sections(record: AnimalRecord, out: Path, t: Lang) -> Path:
    """How the trunk is found and how its volume is integrated slice by slice."""
    apply_style()
    top = traces(record)[View.TOP].result.body_points
    edges, widths = width_profile(top)
    lo, hi = trunk_interval(top)
    centres = (edges[:-1] + edges[1:]) / 2
    threshold = TRUNK_REL_WIDTH * np.percentile(widths, 95)

    fig = plt.figure(figsize=(WIDTH_IN, 3.9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax = fig.add_subplot(grid[0, :])
    ax.plot(centres, widths, color="#1f2937", lw=1.2, label=t("ширина зрізу", "slice width"))
    ax.axhline(threshold, color="#b91c1c", lw=0.9, ls="--",
               label=t(f"поріг {TRUNK_REL_WIDTH:g}·p95".replace(".", ","), f"threshold {TRUNK_REL_WIDTH:g}·p95"))
    ax.axvspan(lo, hi, color="#93c5fd", alpha=0.3, lw=0, label=t("тулуб", "trunk"))
    ax.set_xlabel(t("вздовж тіла, м", "along body, m"))
    ax.set_ylabel(t("ширина, м", "width, m"))
    ax.set_title(t("профіль ширини: тулуб = найдовша серія широких зрізів",
                   "width profile: trunk = longest run of wide slices"))
    ax.legend(frameon=False, loc="lower center", ncol=3)
    panel_label(ax, t.letters[0])

    trunk = top[(top[:, 0] >= lo) & (top[:, 0] <= hi)]
    step = (hi - lo) / 64
    for i, frac in enumerate((0.25, 0.5, 0.75)):
        x = lo + frac * (hi - lo)
        ax.axvline(x, color="#2563eb", lw=0.8)
        section = trunk[np.abs(trunk[:, 0] - x) < step / 2][:, 1:3]
        pts = np.vstack([section, np.column_stack([section[:, 0], np.zeros(len(section))])])
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        a = fig.add_subplot(grid[1, i])
        a.fill(poly[:, 0], poly[:, 1], color="#fdba74", alpha=0.6, lw=0)
        a.plot(*np.vstack([poly, poly[:1]]).T, color="#c2410c", lw=0.9)
        a.scatter(section[:, 0], section[:, 1], s=1.5, color="#1f2937", linewidths=0)
        a.axhline(0, color="#a16207", lw=0.8)
        a.set_aspect("equal")
        a.set_xticks([-0.2, 0.2])
        a.set_xlabel(t("поперек, м", "across, m"))
        if i == 0:
            a.set_ylabel(t("висота, м", "height, m"))
        area = hull.volume
        letter = t.letters[i + 1]
        a.set_title(t(f"{letter}   {frac:.0%} тулуба: {dec(area, 2, t)} м²".replace("%", " %"),
                      f"{letter}   {frac:.0%} of trunk: {area:.2f} m²"), loc="left")
    return save(fig, out)


# ---------------------------------------------------------------------------
def feature_vs_weight(table: pd.DataFrame, out: Path, t: Lang) -> Path:
    """The strongest single 3-D signal against scale weight, per herd."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.8), layout="constrained")
    for ax, (feature, label) in zip(axes, [
        (PHYSICS_FEATURE, t("об’єм тулуба, м³", "trunk volume, m³")),
        ("side_mean_mid_side_area", t("площа середньої частини збоку, м²", "mid-body side area, m²")),
    ]):
        for herd, part in table.groupby("herd"):
            r = np.corrcoef(part[feature], part["weight_kg"])[0, 1]
            ax.scatter(part[feature], part["weight_kg"], s=9, alpha=0.75, color=HERD_COLORS.get(herd),
                       linewidths=0, label=f"{herd_name(herd, t)} (r = {dec(r, 2, t)})")
        ax.set_xlabel(label)
        ax.set_ylabel(t("маса на вагах, кг", "scale weight, kg"))
        ax.legend(frameon=False, loc="upper left")
    panel_label(axes[0], t.letters[0])
    panel_label(axes[1], t.letters[1])
    return save(fig, out)


def published_join_pairs(table: pd.DataFrame, cowdb1_root: Path) -> pd.DataFrame:
    """Weight of table row N next to the features of cloud file N, as the public release pairs them.

    ``table`` holds features under the audited join (row -> its true file), so
    the features of file N are found through the inverse of that join.
    """
    feats = table[table["herd"] == "cowdb1"].set_index("animal_id")
    row_for_file = {file_id: row for row, file_id in audited_cowdb1_join().items()}
    pairs = []
    for rec in load_cowdatabase(cowdb1_root, join="published").animals:  # animal_id = row = file
        row = row_for_file.get(rec.animal_id)
        if row in feats.index:
            pairs.append({"weight_kg": rec.weight_kg, PHYSICS_FEATURE: feats.at[row, PHYSICS_FEATURE]})
    return pd.DataFrame(pairs)


def join_audit(table: pd.DataFrame, published_pairs: pd.DataFrame, out: Path, t: Lang) -> Path:
    """CowDatabase: weight against volume with the published and the audited file-to-row join.

    ``published_pairs`` has columns weight_kg and the volume feature, one row
    per table row paired with file N as the release does.
    """
    apply_style()
    herd = table[table["herd"] == "cowdb1"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7), layout="constrained", sharey=True)
    for ax, data, title, letter in [
        (axes[0], published_pairs, t("опубліковане зіставлення (файл N = рядок N)",
                                     "published join (file N = row N)"), t.letters[0]),
        (axes[1], herd, t("перевірене зіставлення (лише за висотами)", "audited join (from heights only)"), t.letters[1]),
    ]:
        r = np.corrcoef(data[PHYSICS_FEATURE], data["weight_kg"])[0, 1]
        ax.scatter(data[PHYSICS_FEATURE], data["weight_kg"], s=9, alpha=0.75, color=HERD_COLORS["cowdb1"], linewidths=0)
        ax.set_title(title)
        ax.text(0.03, 0.95, f"r = {dec(r, 2, t)}", transform=ax.transAxes,
                va="top", fontsize=8.5, weight="bold")
        ax.set_xlabel(t("об’єм тулуба, м³", "trunk volume, m³"))
        panel_label(ax, letter)
    axes[0].set_ylabel(t("маса з таблиці, кг", "weight from the table, kg"))
    return save(fig, out)
