"""Audit of the CowDatabase file-to-row join.

Owns: deciding which cloud file belongs to which table row, from extracted
heights and manual heights only, and summarising the evidence. Must not use
weight to decide the join; weight is used only afterwards, as an independent
check that is reported, never optimised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cattle3d.io.alignment import monotone_alignment, zscore

# Gap cost in squared-z units. The alignment is identical for every value from
# 4 to 12 on CowDatabase; 6 sits in the middle of that plateau.
DEFAULT_GAP_COST = 6.0


@dataclass(frozen=True)
class JoinAudit:
    mapping: pd.DataFrame  # columns: row_n, file_id
    height_corr_published: float
    height_corr_audited: float
    weight_volume_corr_published: float
    weight_volume_corr_audited: float
    segments: list[tuple[int, int]]  # (first file of the run, row offset)
    unmatched_rows: list[str]
    unmatched_files: list[str]


def _height_signals(per_file: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    measured = np.column_stack(
        [
            per_file["top_trunk_height"] + per_file["side_mean_mid_height"],
            per_file["top_height"] + per_file["side_mean_height"],
        ]
    )
    manual = np.column_stack([per_file["manual_withers_height"], per_file["manual_hip_height"]])
    return zscore(measured), zscore(manual)


def audit_join(per_file: pd.DataFrame, gap_cost: float = DEFAULT_GAP_COST) -> JoinAudit:
    """``per_file``: features extracted with the published join (file n == row n)."""
    table = per_file.assign(n=per_file["animal_id"].astype(int)).sort_values("n")
    table = table.reset_index(drop=True)
    measured, manual = _height_signals(table)
    pairs = monotone_alignment(measured, manual, gap_cost)

    n = table["n"].to_numpy()
    files = np.array([i for i, _ in pairs])
    rows = np.array([j for _, j in pairs])
    mapping = pd.DataFrame({"row_n": n[rows].astype(str), "file_id": n[files].astype(str)})

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    volume = table["top_trunk_cs_volume"].to_numpy()
    weight = table["weight_kg"].to_numpy()
    segments: list[tuple[int, int]] = []
    for i, j in pairs:
        offset = int(n[j] - n[i])
        if not segments or segments[-1][1] != offset:
            segments.append((int(n[i]), offset))
    return JoinAudit(
        mapping=mapping,
        height_corr_published=corr(measured.sum(1), manual.sum(1)),
        height_corr_audited=corr(measured[files].sum(1), manual[rows].sum(1)),
        weight_volume_corr_published=corr(volume, weight),
        weight_volume_corr_audited=corr(volume[files], weight[rows]),
        segments=segments,
        unmatched_rows=sorted(set(n.astype(str)) - set(mapping["row_n"]), key=int),
        unmatched_files=sorted(set(n.astype(str)) - set(mapping["file_id"]), key=int),
    )
