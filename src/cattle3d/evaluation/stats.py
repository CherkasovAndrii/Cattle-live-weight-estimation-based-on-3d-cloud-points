"""Error metrics, bootstrap intervals and paired tests.

Owns: the statistics reported for every experiment. The animal is the unit:
every resampling draws animals, never views or frames. Must not know about
models or files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float
    rmse: float
    mape: float
    bias: float
    p95_abs: float
    max_abs: float
    r2: float
    weight_sd: float

    def as_row(self) -> dict[str, float]:
        return self.__dict__.copy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return Metrics(
        n=int(y_true.size),
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt((err**2).mean())),
        mape=float((abs_err / y_true).mean() * 100),
        bias=float(err.mean()),
        p95_abs=float(np.percentile(abs_err, 95)),
        max_abs=float(abs_err.max()),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        weight_sd=float(y_true.std(ddof=1)),
    )


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.2f} [{self.low:.2f}, {self.high:.2f}]"


def bootstrap_ci(
    values: np.ndarray, statistic=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap over animals."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = np.apply_along_axis(statistic, 1, values[idx]) if statistic is not np.mean else values[idx].mean(1)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(values)), float(low), float(high))


@dataclass(frozen=True)
class PairedComparison:
    """B minus A in absolute error; positive means model A is better."""

    mae_difference: Interval
    wilcoxon_p: float
    n: int

    @property
    def a_better(self) -> bool:
        return self.mae_difference.low > 0

    @property
    def b_better(self) -> bool:
        return self.mae_difference.high < 0


def paired_comparison(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> PairedComparison:
    abs_a = np.abs(np.asarray(pred_a) - y_true)
    abs_b = np.abs(np.asarray(pred_b) - y_true)
    diff = abs_b - abs_a
    p = float(wilcoxon(abs_a, abs_b).pvalue) if np.any(diff != 0) else 1.0
    return PairedComparison(bootstrap_ci(diff, n_boot=n_boot, seed=seed), p, int(diff.size))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    names = sorted(p_values, key=p_values.get)
    m = len(names)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted
