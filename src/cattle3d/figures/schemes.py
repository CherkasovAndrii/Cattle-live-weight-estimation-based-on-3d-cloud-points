"""Schemes of the method: drawn from the configuration the code actually uses.

Numbers on the schemes (thresholds, k values, gates) are read from the
library's constants, so a scheme cannot drift away from the code.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from cattle3d.evaluation.study import H3B_MARGIN_MAPE, RECAL_KS
from cattle3d.features.geometry_features import N_SLICES
from cattle3d.figures.draw import Canvas, Lang
from cattle3d.geometry.segment import SegmentationConfig
from cattle3d.modeling.models import VOLUME_FEATURES
from cattle3d.modeling.weight_model import (
    MIN_POINTS,
    RANGE_TOLERANCE,
    SLOPE_CALIBRATION_ANIMALS,
)

CFG = SegmentationConfig()


NB = "\u00a0"  # no-break space: keeps "90 %" and "3 000" on one line


def _int(x: int) -> str:
    return f"{x:,}".replace(",", NB)


def _pct(x: float) -> str:
    return f"{x * 100:g}{NB}%"


def _num(x: float, t: Lang) -> str:
    """Decimal comma in Ukrainian."""
    s = f"{x:g}"
    return s.replace(".", ",") if t.code == "uk" else s


def pipeline_scheme(out: Path, t: Lang) -> Path:
    """From three depth clouds to a weight with an interval and a trust flag."""
    c = Canvas(16, 18.8)
    n = lambda x: _num(x, t)

    inputs = [
        c.box(0.3 + i * 5.3, 17.0, 4.8, 1.5, name, t("хмара точок PLY, 512 × 424", "PLY point cloud, 512 × 424"),
              style="input")
        for i, name in enumerate((t("Ліва камера", "Left camera"), t("Права камера", "Right camera"),
                                  t("Верхня камера", "Top camera")))
    ]
    g = c.group(0.3, 8.9, 15.4, 7.4, t("Обробка кожного ракурсу окремо (× 3)", "Each view separately (× 3)"))
    for box in inputs:
        c.arrow(box.bottom(), (box.bottom()[0], g.y + g.h))

    steps = [
        (t("1. Фільтр дальності", "1. Range filter"),
         t(f"лишаються точки {n(CFG.min_range_m)}–{n(CFG.max_range_m)} м від камери; порожні пікселі відкинуто",
           f"keep points {n(CFG.min_range_m)}–{n(CFG.max_range_m)} m from the camera; empty pixels dropped")),
        (t("2. Площина підлоги (RANSAC)", "2. Floor plane (RANSAC)"),
         t(f"нахил до вертикалі камери ≤ {CFG.floor_max_tilt_deg:g}°, допуск {n(CFG.floor_inlier_m * 100)} см: "
           "стіну чи паркан не прийнято за підлогу",
           f"tilt from the camera's up ≤ {CFG.floor_max_tilt_deg:g}°, tolerance {n(CFG.floor_inlier_m * 100)} cm: "
           "a wall or fence is never taken for the floor")),
        (t("3. Висота над підлогою", "3. Height above floor"),
         t(f"лишаються точки на висоті {n(CFG.min_height_m)}–{n(CFG.max_height_m)} м; підлогу та бруд прибрано",
           f"keep points {n(CFG.min_height_m)}–{n(CFG.max_height_m)} m above the floor; floor and dirt removed")),
        (t("4. Зв’язні поверхні", "4. Connected surfaces"),
         t(f"сусідні пікселі з’єднуються, якщо глибина змінюється менше ніж на {n(CFG.depth_jump_m * 100)} см",
           f"neighbouring pixels join unless depth jumps by {n(CFG.depth_jump_m * 100)} cm or more")),
        (t("5. Вибір тварини", "5. Pick the animal"),
         t(f"найбільша поверхня біля центру кадру + фрагменти ближче {n(CFG.merge_distance_m * 100)} см "
           "(ноги за жердиною, темна шкіра)",
           f"largest surface near the frame centre + fragments within {n(CFG.merge_distance_m * 100)} cm "
           "(legs behind a rail, dark hide)")),
        (t("6. Система координат тіла", "6. Body frame"),
         t("висота = відстань до підлоги; вісь тіла = головна компонента (PCA); масштаб у метрах",
           "height = distance to the floor; body axis = principal component (PCA); metres kept")),
    ]
    boxes = [
        c.box(0.6 + (i % 3) * 5.1, 12.85 - (i // 3) * 3.6, 4.6, 2.55, title, sub)
        for i, (title, sub) in enumerate(steps)
    ]
    c.arrow(boxes[0].right(), boxes[1].left())
    c.arrow(boxes[1].right(), boxes[2].left())
    mid_y = (boxes[2].y + boxes[5].y + boxes[5].h) / 2
    c.arrow(boxes[2].bottom(), (boxes[2].bottom()[0], mid_y), (boxes[3].top()[0], mid_y), boxes[3].top())
    c.arrow(boxes[3].right(), boxes[4].left())
    c.arrow(boxes[4].right(), boxes[5].left())

    feats = c.box(0.3, 4.55, 15.4, 3.0, "", style="step")
    c.text(8.0, feats.y + feats.h - 0.3, t(f"Ознаки тварини: {len(VOLUME_FEATURES)} для моделі + контроль якості",
                                             f"Animal features: {len(VOLUME_FEATURES)} for the model + quality"),
           9.0, weight="bold")
    columns = [
        (t("Верхній ракурс", "Top view"),
         t(f"• об’єм тулуба ({N_SLICES} зрізи)\n• площа середнього перерізу\n"
           "• ширина, висота, довжина тулуба\n• площа проекції тулуба",
           f"• trunk volume ({N_SLICES} slices)\n• mid-trunk cross-section area\n"
           "• trunk width, height, length\n• trunk footprint area")),
        (t("Бокові ракурси (L і R)", "Side views (L and R)"),
         t("• площа середньої частини тіла\n• висота середньої частини\n• довжина тіла\n• об’єм «площа × висота»",
           "• mid-body side area\n• mid-body height\n• body length\n• box volume footprint × height")),
        (t("Контроль якості", "Quality"),
         t("• кількість точок у ракурсі\n• частка тварини на межі кадру\n• частка точок підлоги\n"
           "• розбіжність лівого/правого боку",
           "• points per view\n• share on the frame edge\n• floor inlier share\n"
           "• left/right disagreement")),
    ]
    for i, (head, body) in enumerate(columns):
        x = 0.6 + i * 5.1
        c.text(x, feats.y + feats.h - 0.85, head, 8.4, ha="left", va="top", weight="bold")
        c.text(x, feats.y + feats.h - 1.3, body, 7.8, ha="left", va="top", color="#475569")
    c.arrow(g.bottom(), feats.top())

    side_min, top_min = _int(MIN_POINTS["left_n_points"]), _int(MIN_POINTS["top_n_points"])
    pts = t(f"L, R ≥ {side_min} і T ≥ {top_min} точок", f"L, R ≥ {side_min} and T ≥ {top_min} points")
    row = [
        (t("Ансамбль моделей", "Model ensemble"),
         t("фізична a·V + b; робастна лінійна (Huber); LightGBM → середнє",
           "physics a·V + b; robust linear (Huber); LightGBM → mean"), "model"),
        (t("Калібрування стада", "Herd calibration"),
         t(f"ŷ′ = k·ŷ + b; k, b з {SLOPE_CALIBRATION_ANIMALS}–30 контрольних зважувань",
           f"ŷ′ = k·ŷ + b; k, b from {SLOPE_CALIBRATION_ANIMALS}–30 reference weighings"), "model"),
        (t("Перевірки довіри", "Trust checks"),
         t(f"{pts}; об’єм тулуба в діапазоні навчання ±{_pct(RANGE_TOLERANCE)}",
           f"{pts}; trunk volume within the training range ±{_pct(RANGE_TOLERANCE)}"), "check"),
        (t("Результат", "Result"),
         t(f"маса, кг; 90{NB}% інтервал ŷ′ ± Δ; прапорець довіри та причина",
           f"weight, kg; 90{NB}% interval ŷ′ ± Δ; trust flag and reason"), "output"),
    ]
    out_boxes = [
        c.box(0.3 + i * 3.95, 0.3, 3.55, 3.2, title, sub, style=style)
        for i, (title, sub, style) in enumerate(row)
    ]
    c.arrow(feats.bottom(0.115), out_boxes[0].top())
    for a, b in pairwise(out_boxes):
        c.arrow(a.right(), b.left())
    return c.save(out)


def evaluation_scheme(out: Path, t: Lang, n_a: int = 94, n_b: int = 101) -> Path:
    """Nested within-herd CV (H1) and transfer between herds with recalibration (H3)."""
    from matplotlib.patches import Rectangle

    c = Canvas(16, 17.4)
    table = c.box(0.3, 15.7, 15.4, 1.4, t("Таблиця ознак: 1 рядок = 1 тварина", "Feature table: 1 row = 1 animal"),
                  t(f"стадо A: CowDatabase2, {n_a} корів Black Angus · стадо B: CowDatabase, {n_b} корова Hereford",
                    f"herd A: CowDatabase2, {n_a} Black Angus cows · herd B: CowDatabase, {n_b} Hereford cows"),
                  style="input")
    left = c.group(0.3, 0.3, 7.6, 14.9, t("H1: точність у межах стада", "H1: accuracy within a herd"))
    right = c.group(8.1, 0.3, 7.6, 14.9, t("H3: перенесення на інше стадо", "H3: transfer to another herd"))
    c.arrow(table.bottom(0.25), (table.bottom(0.25)[0], left.y + left.h))
    c.arrow(table.bottom(0.75), (table.bottom(0.75)[0], right.y + right.h))

    # Outer folds: one row per fold, the scored block in orange.
    c.text(0.7, 14.25, t("Зовнішня 5-кратна CV × 5 випадкових розбиттів", "Outer 5-fold CV × 5 split seeds"),
           8.4, ha="left", weight="bold")
    x0, cw, ch = 1.3, 0.95, 0.42
    for r in range(5):
        y = 13.35 - r * 0.55
        c.text(x0 - 0.3, y + ch / 2, str(r + 1), 7.5, color="#475569")
        for k in range(5):
            test = k == r
            c.ax.add_patch(Rectangle((x0 + k * cw, y), cw - 0.06, ch, zorder=3, lw=0.6, ec="#334155",
                                     fc="#fdba74" if test else "#dbeafe"))
    for i, (fc, label) in enumerate((("#dbeafe", t("навчання", "train")), ("#fdba74", t("оцінка", "score")))):
        c.ax.add_patch(Rectangle((6.25, 12.9 - i * 0.6), 0.4, 0.3, fc=fc, ec="#334155", lw=0.6, zorder=3))
        c.text(6.75, 13.05 - i * 0.6, label, 7.5, ha="left", color="#475569")
    c.text(0.7, 10.75, t("тварина ніколи не буває і в навчанні, і в оцінці",
                         "an animal is never in training and scoring at once"), 7.5, ha="left", color="#475569",
           style="italic")
    inner = c.box(0.7, 9.0, 6.8, 1.3, t("Внутрішня 3-кратна CV", "Inner 3-fold CV"),
                  t("підбір гіперпараметрів LightGBM лише на тренувальних тваринах",
                    "LightGBM hyperparameters tuned on training animals only"))
    row4 = 13.35 - 3 * 0.55 + ch / 2  # a fold's training part goes to the inner CV
    c.arrow((x0 + 5 * cw - 0.06, row4), (7.15, row4), (7.15, inner.y + inner.h))
    steps = [
        (t("Позавибіркові прогнози", "Out-of-fold predictions"),
         t("для кожної тварини, середнє 5 розбиттів", "for every animal, mean of 5 seeds"), "step", 1.2),
        (t("Метрики", "Metrics"), t("MAE, RMSE, MAPE, зсув, p95, максимум", "MAE, RMSE, MAPE, bias, p95, max"),
         "step", 1.2),
        (t("95 % бутстреп-ДІ", "95 % bootstrap CI"),
         t("10 000 повторних вибірок тварин", "10 000 resamples of animals"), "step", 1.2),
        (t("3-D проти рулетки", "3-D vs tape measurements"),
         t("парний бутстреп + тест Вілкоксона; поправка Холма",
           "paired bootstrap + Wilcoxon test; Holm correction"), "check", 1.3),
        (t("Конформний 90 % інтервал", "Conformal 90 % interval"),
         t("поріг з 30 тварин, покриття на решті (500 повторів)",
           "threshold from 30 animals, coverage on the rest (500 repeats)"), "output", 1.3),
    ]
    prev, y = inner, 7.4
    for title, sub, style, h in steps:
        box = c.box(0.7, y, 6.8, h, title, sub, style=style)
        c.arrow(prev.bottom(), box.top())
        prev, y = box, y - h - 0.45

    ks = ", ".join(str(k) for k in RECAL_KS)
    src = c.box(8.5, 12.75, 6.8, 1.3, t("Навчання на стаді-джерелі", "Train on the source herd"),
                t("усі тварини A (і навпаки: B → A)", "all animals of A (and the reverse, B → A)"), style="model")
    zero = c.box(8.5, 10.95, 6.8, 1.3, t("Прогноз для стада-цілі без адаптації", "Predict the target herd as is"),
                 t("«нульове» перенесення: зсув чи точність?", "zero-shot transfer: bias or scatter?"))
    pick = c.box(8.5, 9.15, 6.8, 1.3, t("k контрольних зважувань", "k reference weighings"),
                 t(f"k = {ks}; 500 випадкових наборів тварин-цілей",
                   f"k = {ks}; 500 random sets of target animals"))
    off = c.box(8.5, 7.05, 3.25, 1.5, t("Лише зсув", "Intercept only"), "ŷ′ = ŷ + b", style="model")
    lin = c.box(12.05, 7.05, 3.25, 1.5, t("Нахил і зсув", "Slope + intercept"), "ŷ′ = k·ŷ + b", style="model")
    score = c.box(8.5, 5.0, 6.8, 1.3, t("Оцінка на решті тварин цілі", "Score on the remaining target animals"),
                  t("MAPE, MAE, зсув для кожного набору", "MAPE, MAE, bias for each set"))
    margin = _num(H3B_MARGIN_MAPE, t)
    dec = c.diamond(11.9, 3.25, 6.9, 1.9, t(f"MAPE ≤ MAPE у межах стада\n+ {margin} в.п.?",
                                             f"MAPE ≤ within-herd MAPE\n+ {margin} pp?"), size=8.2)
    res = c.box(8.5, 0.6, 6.8, 1.35, t("Скільки зважувань потрібно фермі", "How many weighings a farm needs"),
                t("частка наборів, що вкладаються в межу, для кожного k",
                  "share of sets within the margin, for every k"), style="output")
    c.arrow(src.bottom(), zero.top())
    c.arrow(zero.bottom(), pick.top())
    c.arrow(pick.bottom(0.25), off.top())
    c.arrow(pick.bottom(0.77), lin.top())
    c.arrow(off.bottom(), (off.bottom()[0], score.y + score.h))
    c.arrow(lin.bottom(), (lin.bottom()[0], score.y + score.h))
    c.arrow(score.bottom(0.5), dec.top())
    c.arrow(dec.bottom(), res.top())
    return c.save(out)


def integration_scheme(out: Path, t: Lang) -> Path:
    """Where the model sits: farm devices, the planned API and clients, research pipeline."""
    c = Canvas(16, 17.6)
    c.legend(0.4, 17.0, [
        ("step", t("наявне (бібліотека, CLI, ваги)", "exists (library, CLI, scale)")),
        ("step:dashed", t("заплановано", "planned")),
    ], step=5.2)

    c.group(0.3, 12.9, 10.3, 3.8, t("Ферма", "Farm"))
    station = c.box(0.6, 13.2, 3.0, 2.9, t("Станція сканування", "Scanning station"),
                    t("3 камери глибини обабіч і над проходом", "3 depth cameras beside and above a race"),
                    style="input", dashed=True)
    tablet = c.box(3.95, 13.2, 3.0, 2.9, t("Планшет", "Tablet app"),
                   t("тварина за біркою; маса й інтервал; «пересканувати», якщо не довіряємо",
                     "animal by ear tag; weight and interval; “rescan” when not trusted"),
                   style="output", dashed=True)
    scale = c.box(7.3, 13.2, 3.0, 2.9, t("Ваги", "Scale"),
                  t("контрольні зважування для калібрування", "reference weighings for calibration"),
                  style="input")
    c.arrow(station.right(), tablet.left())
    c.arrow(scale.left(), tablet.right())

    others = c.group(10.9, 12.9, 4.8, 3.8, t("Інші клієнти", "Other clients"))
    c.box(11.2, 14.55, 4.2, 1.55, t("Сайт-презентація", "Presentation website"),
          t("метод, демо, результати", "method, demo, results"), style="output", dashed=True)
    c.box(11.2, 13.2, 4.2, 1.15, t("Облікова система ферми", "Farm management software"),
          style="output", dashed=True, size=8.4)

    c.group(0.3, 9.5, 15.4, 3.0, t("Сервіс", "Service"))
    api = c.box(0.6, 9.75, 14.8, 2.1, t("REST API", "REST API"), t(
        "POST /v1/predictions — 3 хмари → маса, інтервал, довіра;   GET /v1/model — картка моделі\n"
        "POST /v1/herds/{id}/calibration — контрольні зважування;   GET /health\n"
        "GET /v1/animals/{id}/history — динаміка маси тварини",
        "POST /v1/predictions — 3 clouds → weight, interval, trust;   GET /v1/model — model card\n"
        "POST /v1/herds/{id}/calibration — reference weighings;   GET /health\n"
        "GET /v1/animals/{id}/history — an animal's weight over time"),
        align="left", dashed=True)
    c.arrow((tablet.bottom()[0], tablet.y), (tablet.bottom()[0], api.y + api.h), both=True)
    c.arrow((others.bottom()[0], others.y), (others.bottom()[0], api.y + api.h), both=True)

    c.group(0.3, 4.5, 10.85, 4.6, t("Ядро (цей репозиторій)", "Core (this repository)"))
    lib = c.box(0.6, 4.8, 6.2, 3.6, t("Бібліотека cattle3d", "cattle3d library"), t(
        "io → geometry → features → modeling\n\n"
        "одна функція ознак для навчання і прогнозу: features_from_clouds()",
        "io → geometry → features → modeling\n\n"
        "one feature function for training and prediction: features_from_clouds()"))
    art = c.box(7.3, 4.8, 3.55, 3.6, t("Артефакт моделі", "Model artifact"),
                t("weight_model.joblib\nансамбль, Δ, діапазон, калібрування стад",
                  "weight_model.joblib\nensemble, Δ, range, herd calibrations"), style="store")
    c.group(11.45, 4.5, 4.25, 4.6, t("Дані", "Data"))
    db = c.box(11.7, 6.45, 3.75, 1.95, t("База даних", "Database"),
               t("стада, тварини, прогнози, зважування", "herds, animals, predictions, weighings"),
               style="store", dashed=True)
    blobs = c.box(11.7, 4.8, 3.75, 1.35, t("Сховище хмар", "Cloud storage"), "PLY", style="store", dashed=True)
    def down(box, f: float = 0.5) -> tuple[tuple[float, float], tuple[float, float]]:
        """Vertical link from the API's bottom edge to ``box``."""
        return (box.top(f)[0], api.y), box.top(f)

    c.arrow(*down(lib, 0.8), label="predict / calibrate", label_pos=0.25)
    c.arrow(lib.right(), art.left(), both=True)
    c.arrow(*down(db), both=True)

    c.group(0.3, 0.2, 15.4, 3.9, t("Дослідження (офлайн, CLI cattle3d)", "Research (offline, cattle3d CLI)"))
    names = [
        (t("Публічні дані", "Public data"), "CowDatabase2,\nCowDatabase", "input"),
        ("extract", t("ознаки → features.csv", "features → features.csv"), "step"),
        ("evaluate", t("CV, перенесення, тести", "CV, transfer, tests"), "step"),
        (t("Звіти", "Reports"), t("results.md, рисунки дисертації", "results.md, thesis figures"), "output"),
        ("train", t("модель на обох стадах", "model on both herds"), "model"),
    ]
    rb = [c.box(0.6 + i * 3.0, 0.85, 2.7, 2.45, title, sub, style=style) for i, (title, sub, style) in enumerate(names)]
    for a, b in pairwise(rb[:4]):
        c.arrow(a.right(), b.left())
    train = rb[4]
    c.arrow(rb[1].bottom(), (rb[1].bottom()[0], 0.5), (train.bottom()[0], 0.5), train.bottom())
    c.arrow(train.top(0.15), (train.top(0.15)[0], 4.3), (art.bottom()[0], 4.3), art.bottom())
    x_new = train.top(0.85)[0]
    c.arrow((x_new, blobs.y), (x_new, train.y + train.h), dashed=True)
    c.text(x_new - 0.15, 3.7, t("нові дані →\nперенавчання", "new data →\nretraining"),
           6.8, ha="right", color="#475569", style="italic")
    return c.save(out)


def decision_scheme(out: Path, t: Lang, half_width_kg: float | None = None) -> Path:
    """What happens to one scan: refuse, flag or trust the prediction."""
    c = Canvas(16, 16.2)
    delta = f"Δ = {_num(round(half_width_kg, 1), t)}{NB}{t('кг', 'kg')}" if half_width_kg else "Δ"
    cx, w = 5.2, 6.4
    x = cx - w / 2

    start = c.box(x, 14.6, w, 1.25, t("3 хмари точок: L, R, T", "3 point clouds: L, R, T"),
                  t("одне сканування тварини", "one scan of an animal"), style="input")
    seg = c.box(x, 12.85, w, 1.1, t("Сегментація кожного ракурсу", "Segment each view"))
    d1 = c.diamond(cx, 11.0, 5.6, 1.6, t("Тварину знайдено\nв усіх трьох ракурсах?", "Animal found\nin all three views?"))
    refuse = c.box(10.2, 10.25, 5.5, 1.5, t("Відмова", "Refuse"),
                   t("назвати ракурс і причину; «пересканувати»", "name the view and the reason; “rescan”"),
                   style="stop")
    feats = c.box(x, 8.45, w, 1.2, t("Ознаки → ансамбль → ŷ", "Features → ensemble → ŷ"))
    d2 = c.diamond(cx, 6.75, 5.6, 1.5, t("Стадо відкалібровано?", "Herd calibrated?"))
    calib = c.box(10.2, 6.15, 5.5, 1.2, "ŷ′ = k·ŷ + b", t("k, b цього стада", "k, b of this herd"), style="model")
    interval = c.box(x, 4.3, w, 1.1, t(f"Інтервал ŷ′ ± Δ (90{NB}%)", f"Interval ŷ′ ± Δ (90{NB}%)"), delta)
    side_min, top_min = _int(MIN_POINTS["left_n_points"]), _int(MIN_POINTS["top_n_points"])
    d3 = c.diamond(cx, 2.55, 6.6, 1.9, t(
        f"L, R ≥ {side_min}, T ≥ {top_min} точок\nі об’єм у діапазоні ±{_pct(RANGE_TOLERANCE)}?",
        f"L, R ≥ {side_min}, T ≥ {top_min} points\nand volume within range ±{_pct(RANGE_TOLERANCE)}?"))
    flag = c.box(10.2, 1.8, 5.5, 1.5, t("Видати з позначкою", "Return with a flag"),
                 t("«поза робочим діапазоном» + причина: не для рішень",
                   "“outside the operating range” + reason: not for decisions"), style="check")
    ok = c.box(x, 0.2, w, 1.0, t("Довірений прогноз: маса та інтервал", "Trusted: weight and interval"),
               style="output")

    c.arrow(start.bottom(), seg.top())
    c.arrow(seg.bottom(), d1.top())
    c.arrow(d1.right(), refuse.left(), label=t("ні", "no"))
    c.arrow(d1.bottom(), feats.top(), label=t("так", "yes"))
    c.arrow(feats.bottom(), d2.top())
    c.arrow(d2.right(), calib.left(), label=t("так", "yes"))
    c.arrow(d2.bottom(), interval.top(), label=t("ні: ŷ′ = ŷ", "no: ŷ′ = ŷ"), label_side="left")
    c.arrow(calib.bottom(), (calib.bottom()[0], 5.75), (cx + 0.35, 5.75), (cx + 0.35, interval.y + interval.h))
    c.arrow(interval.bottom(), d3.top())
    c.arrow(d3.right(), flag.left(), label=t("ні", "no"))
    c.arrow(d3.bottom(), ok.top(), label=t("так", "yes"))
    return c.save(out)


def calibration_sequence(out: Path, t: Lang) -> Path:
    """A farm's life with the system: onboard a herd, daily scans, periodic checks."""
    c = Canvas(16, 18)
    lanes = {
        "tablet": (2.0, t("Планшет", "Tablet")),
        "api": (6.3, "REST API"),
        "core": (10.4, "cattle3d"),
        "db": (14.2, t("База даних", "Database")),
    }
    top = 16.9
    for key, (x, name) in lanes.items():
        c.box(x - 1.5, top, 3.0, 0.9, name, dashed=key != "core")
    y = [top - 0.4]
    size = 7.6

    def phase(label: str) -> None:
        y[0] -= 0.2
        c.ax.axhspan(y[0] - 0.45, y[0], color="#e2e8f0", zorder=0)
        c.text(0.3, y[0] - 0.225, label, 8.2, ha="left", weight="bold", color="#334155")
        y[0] -= 0.45

    def msg(a: str, b: str, label: str, reply: bool = False) -> None:
        xa, xb = lanes[a][0], lanes[b][0]
        text = c.wrap(label, abs(xb - xa) - 0.4, size)
        y[0] -= 0.25 + (text.count("\n") + 1) * c.line_h(size)
        c.arrow((xa, y[0]), (xb, y[0]), dashed=reply)
        c.text((xa + xb) / 2, y[0] + 0.08, text, size, va="bottom")
        y[0] -= 0.15

    def note(lane: str, label: str) -> None:
        x, w = lanes[lane][0], 3.8
        text = c.wrap(label, w - 0.3, size)
        h = (text.count("\n") + 1) * c.line_h(size) + 0.3
        y[0] -= 0.25
        c.box(x - w / 2, y[0] - h, w, h, "", style="model")
        c.text(x, y[0] - h / 2, text, size)
        y[0] -= h + 0.1

    phase(t("1. Підключення стада (один раз)", "1. Onboard a herd (once)"))
    msg("tablet", "api", t(f"POST /herds/{{id}}/calibration: {SLOPE_CALIBRATION_ANIMALS}–30 тварин, 3 хмари + маса з ваг",
                           f"POST /herds/{{id}}/calibration: {SLOPE_CALIBRATION_ANIMALS}–30 animals, 3 clouds + scale weight"))
    msg("api", "core", t("ознаки → сирий прогноз ŷ", "features → raw prediction ŷ"))
    note("core", t("МНК: маса ≈ k·ŷ + b", "least squares: weight ≈ k·ŷ + b"))
    msg("core", "api", "k, b", reply=True)
    msg("api", "db", t("зберегти калібрування стада", "store the herd calibration"))

    phase(t("2. Щоденна робота", "2. Daily use"))
    msg("tablet", "api", t("POST /predictions?herd=id: бірка + 3 хмари", "POST /predictions?herd=id: ear tag + 3 clouds"))
    msg("api", "core", t("ознаки → ŷ → k·ŷ + b → перевірки", "features → ŷ → k·ŷ + b → checks"))
    msg("core", "api", t("маса, інтервал, довіра", "weight, interval, trust"), reply=True)
    msg("api", "db", t("історія маси тварини", "the animal's weight history"))
    msg("api", "tablet", t("маса ± Δ або «пересканувати»", "weight ± Δ or “rescan”"), reply=True)

    phase(t("3. Контроль дрейфу (періодично)", "3. Drift check (periodically)"))
    msg("tablet", "api", t("3–5 контрольних зважувань", "3–5 reference weighings"))
    msg("api", "db", t("порівняти з прогнозами цих тварин", "compare with these animals' predictions"))
    note("api", t("зсув > Δ/2 → повторне калібрування (крок 1)", "bias > Δ/2 → recalibrate (step 1)"))

    bottom = y[0] - 0.3
    for x, _ in lanes.values():
        c.ax.plot([x, x], [bottom, top], color="#94a3b8", lw=0.9, ls=(0, (3, 3)), zorder=1)
    c.crop_bottom(bottom - 0.2)
    return c.save(out)
