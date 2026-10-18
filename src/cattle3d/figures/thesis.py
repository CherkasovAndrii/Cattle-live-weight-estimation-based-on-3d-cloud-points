"""All thesis figures in reading order, in Ukrainian and English.

Schemes need nothing; data figures need the feature table and the dataset
checkouts; result figures run the study (the same one ``evaluate`` runs).
Whatever input is missing, the figures that need it are skipped, never faked.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cattle3d.evaluation.study import StudyResult
from cattle3d.figures import results as R
from cattle3d.figures import schemes as S
from cattle3d.figures import stages as D
from cattle3d.figures.draw import Lang


def write_thesis_figures(
    out_dir: Path,
    langs: tuple[str, ...] = ("uk", "en"),
    table: pd.DataFrame | None = None,
    cowdb2_root: Path | None = None,
    cowdb1_root: Path | None = None,
    study: StudyResult | None = None,
    interval_half_width_kg: float | None = None,
) -> list[Path]:
    from cattle3d.io.datasets import load_cowdatabase2

    example = D.representative(load_cowdatabase2(cowdb2_root).animals) if cowdb2_root else None
    pairs = D.published_join_pairs(table, cowdb1_root) if cowdb1_root and table is not None else None
    n = {h: int((table["herd"] == h).sum()) for h in ("cowdb2", "cowdb1")} if table is not None else {}
    written: list[Path] = []
    for code in langs:
        t, d = Lang(code), out_dir / code
        written.append(S.pipeline_scheme(d / "01_pipeline_scheme.png", t))
        if example is not None:
            written.append(D.raw_views(example, d / "02_raw_views.png", t))
            written.append(D.segmentation_steps(example, d / "03_segmentation_steps.png", t))
            written.append(D.body_frame(example, d / "04_body_frame.png", t))
            written.append(D.cross_sections(example, d / "05_cross_sections.png", t))
        if table is not None:
            written.append(D.feature_vs_weight(table, d / "06_feature_vs_weight.png", t))
        if pairs is not None:
            written.append(D.join_audit(table, pairs, d / "07_join_audit.png", t))
        written.append(S.evaluation_scheme(d / "08_evaluation_scheme.png", t,
                                           n.get("cowdb2", 94), n.get("cowdb1", 101)))
        if study is not None:
            written.append(R.within_herd(study, d / "09_within_herd_oof.png", t))
            written.append(R.intervals(study, d / "10_prediction_intervals.png", t))
            written.append(R.zero_shot(study, d / "11_zero_shot_transfer.png", t))
            written.append(R.recalibration(study, d / "12_recalibration_curve.png", t))
        written.append(S.decision_scheme(d / "13_decision_flow.png", t, interval_half_width_kg))
        written.append(S.integration_scheme(d / "14_integration_architecture.png", t))
        written.append(S.calibration_sequence(d / "15_calibration_sequence.png", t))
    return written
