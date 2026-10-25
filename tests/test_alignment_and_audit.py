import numpy as np
import pandas as pd

from cattle3d.audit import audit_join
from cattle3d.io.alignment import monotone_alignment


def test_alignment_recovers_a_skipped_row_and_an_extra_file():
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(40, 2))
    # Row 10 was never captured; file 30 is an extra capture of nobody.
    files = np.vstack([rows[:10], rows[11:30], rng.normal(size=(1, 2)) * 5, rows[30:]])
    files = files + rng.normal(scale=0.05, size=files.shape)
    pairs = monotone_alignment(files, rows, gap_cost=2.0)
    mapping = dict(pairs)
    assert mapping[9] == 9 and mapping[10] == 11  # shift after the skipped row
    assert 29 not in mapping or mapping.get(29) == 29
    assert all(j2 > j1 for (_, j1), (_, j2) in zip(pairs, pairs[1:]))


def test_identity_when_nothing_is_wrong():
    x = np.random.default_rng(2).normal(size=(25, 2))
    assert monotone_alignment(x, x, gap_cost=4.0) == [(i, i) for i in range(25)]


def _table(n, shift_from=None):
    rng = np.random.default_rng(3)
    withers = rng.normal(120, 6, n)
    hip = withers + rng.normal(3, 1, n)
    weight = 4 * withers + rng.normal(0, 5, n)
    measured = withers / 100 + rng.normal(0, 0.005, n)
    if shift_from is not None:  # file i carries the animal of row i+1
        measured[shift_from:-1] = measured[shift_from + 1:]
    return pd.DataFrame({
        "animal_id": [str(i + 1) for i in range(n)],
        "manual_withers_height": withers, "manual_hip_height": hip, "weight_kg": weight,
        "top_trunk_height": measured, "side_mean_mid_height": measured,
        "top_height": measured + 0.03, "side_mean_height": measured + 0.03,
        "top_trunk_cs_volume": measured ** 3,
    })


def test_audit_finds_a_shifted_block_without_using_weight():
    table = _table(60, shift_from=20)
    audit = audit_join(table)
    assert audit.height_corr_audited > audit.height_corr_published
    assert any(offset == 1 for _, offset in audit.segments)
    # Scrambling weight must not change the join: weight is not an input.
    scrambled = table.assign(weight_kg=np.random.default_rng(0).permutation(table["weight_kg"]))
    pd.testing.assert_frame_equal(audit_join(scrambled).mapping, audit.mapping)
