"""Monotone alignment of capture files to measurement-table rows.

Owns: the sequence alignment used to audit a file-to-label join. Must not
know about weights: the alignment may only use measurements that are not the
prediction target, so that it cannot leak the label it is meant to protect.

Why this exists: in the public CowDatabase release the cloud files are
numbered 1..103 and the table rows are numbered 1..103, but for a block of
animals the file number is one or two places off the row number (a skipped or
repeated capture). Captures and rows are both in time order, so the correct
join is a monotone alignment with occasional gaps, which is what
Needleman–Wunsch dynamic programming finds.
"""

from __future__ import annotations

import numpy as np


def monotone_alignment(
    file_features: np.ndarray, row_features: np.ndarray, gap_cost: float
) -> list[tuple[int, int]]:
    """Return (file_index, row_index) pairs of the minimum-cost monotone alignment.

    ``file_features`` (n, d) and ``row_features`` (m, d) must be on a common
    standardised scale. A match costs the squared distance of the two vectors;
    leaving a file or a row unmatched costs ``gap_cost``.
    """
    if file_features.ndim != 2 or row_features.ndim != 2:
        raise ValueError("features must be 2-D arrays")
    if file_features.shape[1] != row_features.shape[1]:
        raise ValueError("file and row features must have the same width")
    n, m = len(file_features), len(row_features)
    cost = ((file_features[:, None, :] - row_features[None, :, :]) ** 2).sum(axis=2)

    total = np.zeros((n + 1, m + 1))
    total[0, :] = np.arange(m + 1) * gap_cost
    total[:, 0] = np.arange(n + 1) * gap_cost
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 match, 1 skip file, 2 skip row
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                total[i - 1, j - 1] + cost[i - 1, j - 1],
                total[i - 1, j] + gap_cost,
                total[i, j - 1] + gap_cost,
            )
            k = int(np.argmin(options))
            total[i, j] = options[k]
            move[i, j] = k

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if move[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
