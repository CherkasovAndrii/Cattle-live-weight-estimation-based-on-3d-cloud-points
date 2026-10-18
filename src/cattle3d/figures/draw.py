"""Small box-and-arrow drawing kit on top of matplotlib.

Coordinates are abstract units with the origin at the bottom left; one unit
is ``UNIT_IN`` inches on paper, so fonts keep their size across figures.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

UNIT_IN = 0.42  # a 16-unit-wide scheme is 6.7 in: a portrait A4 text width
TITLE_PT = 9.0
SUB_PT = 7.8
FONT = "DejaVu Sans"  # has Cyrillic glyphs

STYLES: dict[str, dict[str, object]] = {
    "input": {"fc": "#dbeafe", "ec": "#1e40af"},
    "step": {"fc": "#ffffff", "ec": "#334155"},
    "model": {"fc": "#ffedd5", "ec": "#c2410c"},
    "output": {"fc": "#dcfce7", "ec": "#166534"},
    "check": {"fc": "#fef9c3", "ec": "#a16207"},
    "stop": {"fc": "#fee2e2", "ec": "#b91c1c"},
    "store": {"fc": "#f1f5f9", "ec": "#475569"},
    "planned": {"fc": "#ffffff", "ec": "#64748b", "ls": (0, (4, 3))},
}
INK = "#1f2937"
MUTED = "#475569"


class Lang:
    """Pick the Ukrainian or the English variant of a text."""

    def __init__(self, code: str) -> None:
        if code not in ("uk", "en"):
            raise ValueError(f"language must be 'uk' or 'en', not {code!r}")
        self.code = code

    def __call__(self, uk: str, en: str) -> str:
        return uk if self.code == "uk" else en

    @property
    def letters(self) -> str:
        """Panel letters: Cyrillic in the Ukrainian thesis (а, б, в, ...)."""
        return "абвгде" if self.code == "uk" else "abcdef"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    def top(self, f: float = 0.5) -> tuple[float, float]:
        return (self.x + f * self.w, self.y + self.h)

    def bottom(self, f: float = 0.5) -> tuple[float, float]:
        return (self.x + f * self.w, self.y)

    def left(self, f: float = 0.5) -> tuple[float, float]:
        return (self.x, self.y + f * self.h)

    def right(self, f: float = 0.5) -> tuple[float, float]:
        return (self.x + self.w, self.y + f * self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


class Canvas:
    def __init__(self, width: float, height: float) -> None:
        plt.rcParams["font.family"] = FONT
        self.fig = plt.figure(figsize=(width * UNIT_IN, height * UNIT_IN))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.width, self.height = width, height
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    def text(self, x: float, y: float, s: str, size: float = SUB_PT, **kw: object) -> None:
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        self.ax.text(x, y, s, fontsize=size, zorder=5, linespacing=1.3, **kw)

    @staticmethod
    def wrap(text: str, width_units: float, size: float, bold: bool = False) -> str:
        """Wrap each paragraph to the box width; manual line breaks are kept.

        Use a no-break space (\\u00a0) to keep a number with its unit.
        """
        chars = max(8, int(width_units * UNIT_IN * 72 / ((0.68 if bold else 0.6) * size)))
        return "\n".join(
            textwrap.fill(line, chars, break_long_words=False, subsequent_indent="   " if line.lstrip().startswith("•") else "")
            if line.strip() else ""
            for line in text.split("\n")
        )

    @staticmethod
    def line_h(size: float) -> float:
        return size * 1.3 / 72 / UNIT_IN

    def box(
        self, x: float, y: float, w: float, h: float, title: str, sub: str = "",
        style: str = "step", size: float = TITLE_PT, align: str = "center", dashed: bool = False,
    ) -> Box:
        """Rounded box; the title (bold) and the wrapped sub text are centred as one block.

        ``dashed`` marks a component that is planned, not built yet.
        """
        st = dict(STYLES[style])
        ls = st.pop("ls", (0, (4, 3)) if dashed else "-")
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                lw=1.0, ls=ls, zorder=3, **st,
            )
        )
        sub_size = size - (TITLE_PT - SUB_PT)
        title = self.wrap(title, w - 0.3, size, bold=True)
        sub = self.wrap(sub, w - 0.35, sub_size) if sub else ""
        n_t, n_s = title.count("\n") + 1, (sub.count("\n") + 1 if sub else 0)
        block = n_t * self.line_h(size) + (0.12 + n_s * self.line_h(sub_size) if sub else 0)
        top = y + h / 2 + block / 2
        tx = x + w / 2 if align == "center" else x + 0.2
        self.text(tx, top, title, size, weight="bold", va="top", ha=align)
        if sub:
            self.text(tx, top - n_t * self.line_h(size) - 0.12, sub, sub_size, va="top", ha=align,
                      color=MUTED)
        return Box(x, y, w, h)

    def diamond(self, cx: float, cy: float, w: float, h: float, text: str, size: float = SUB_PT) -> Box:
        st = STYLES["check"]
        pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=st["fc"], ec=st["ec"], lw=1.0, zorder=3))
        self.text(cx, cy, text, size)
        return Box(cx - w / 2, cy - h / 2, w, h)

    def group(self, x: float, y: float, w: float, h: float, title: str, color: str = "#94a3b8") -> Box:
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                lw=1.0, ls=(0, (2, 2)), fc="#f8fafc", ec=color, zorder=1,
            )
        )
        self.text(x + 0.2, y + h - 0.18, title, TITLE_PT, ha="left", va="top", weight="bold", color=MUTED)
        return Box(x, y, w, h)

    def arrow(
        self, *points: tuple[float, float], label: str = "", label_pos: float = 0.5,
        label_side: str = "right", color: str = INK, dashed: bool = False, both: bool = False,
    ) -> None:
        """Polyline through ``points`` with an arrowhead at the end."""
        ls = (0, (4, 3)) if dashed else "-"
        for a, b in pairwise(points[:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.0, ls=ls, zorder=2,
                         solid_capstyle="butt")
        style = "<|-|>" if both else "-|>"
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2], points[-1], arrowstyle=style, mutation_scale=9, lw=1.0,
                color=color, ls=ls, zorder=2, shrinkA=0, shrinkB=1,
            )
        )
        if label:
            # Label the longest segment so it never sits on a corner.
            segs = list(pairwise(points))
            a, b = max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
            mx, my = a[0] + label_pos * (b[0] - a[0]), a[1] + label_pos * (b[1] - a[1])
            vertical = abs(b[0] - a[0]) < abs(b[1] - a[1])
            if vertical:
                dx = 0.1 if label_side == "right" else -0.1
                self.text(mx + dx, my, label, 7, ha="left" if label_side == "right" else "right",
                          color=MUTED, style="italic")
            else:
                self.text(mx, my + 0.14, label, 7, va="bottom", color=MUTED, style="italic")

    def legend(self, x: float, y: float, items: list[tuple[str, str]], step: float = 4.0) -> None:
        """One row of sample boxes; a style name ending in ':dashed' is drawn dashed."""
        for i, (style, label) in enumerate(items):
            name, _, dashed = style.partition(":")
            st = dict(STYLES[name])
            ls = st.pop("ls", (0, (3, 2)) if dashed else "-")
            self.ax.add_patch(
                FancyBboxPatch((x + i * step, y), 0.55, 0.32, boxstyle="round,pad=0,rounding_size=0.06",
                               lw=1.0, ls=ls, zorder=3, **st)
            )
            self.text(x + i * step + 0.7, y + 0.16, label, SUB_PT, ha="left", color=MUTED)

    def crop_bottom(self, y: float) -> None:
        """Drop the empty space below ``y`` (for layouts whose length is known only at the end)."""
        self.ax.set_ylim(y, self.height)
        self.fig.set_size_inches(self.width * UNIT_IN, (self.height - y) * UNIT_IN)

    def save(self, path: Path, dpi: int = 300) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(self.fig)
        return path
