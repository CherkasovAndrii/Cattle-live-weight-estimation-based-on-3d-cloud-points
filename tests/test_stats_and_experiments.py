import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from cattle3d.evaluation.experiments import conformal_quantile, oof_predictions, recalibration_curve
from cattle3d.evaluation.stats import bootstrap_ci, holm, metrics, paired_comparison


def test_metrics_basic():
    m = metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    assert m.mae == 10 and m.bias == 0 and m.max_abs == 10
    assert m.mape == pytest.approx(7.5)


def test_bootstrap_ci_contains_the_estimate():
    ci = bootstrap_ci(np.random.default_rng(0).normal(5, 1, 200))
    assert ci.low < ci.estimate < ci.high
    assert ci.estimate == pytest.approx(5, abs=0.2)


def test_paired_comparison_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(500, 50, 100)
    good = y + rng.normal(0, 5, 100)
    bad = y + rng.normal(0, 30, 100)
    c = paired_comparison(y, good, bad)
    assert c.a_better and c.wilcoxon_p < 1e-6


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.08)
    assert adj["c"] == 1.0 or adj["c"] == pytest.approx(0.5)


class Memoriser(BaseEstimator, RegressorMixin):
    """Predicts the label of an identical training row, or -1: detects leakage."""

    def fit(self, X, y):
        self.lookup_ = {tuple(row): label for row, label in zip(X, y)}
        return self

    def predict(self, X):
        return np.array([self.lookup_.get(tuple(row), -1.0) for row in X])


def test_oof_never_scores_an_animal_the_model_was_trained_on():
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.arange(30, dtype=float)
    oof = oof_predictions(lambda s: Memoriser(), X, y, seeds=(0, 1))
    assert np.all(oof == -1.0)


def test_conformal_quantile_rank():
    assert conformal_quantile(np.arange(1, 10), alpha=0.1) == 9
    assert conformal_quantile(np.arange(1, 20), alpha=0.1) == 18


def test_recalibration_removes_a_constant_bias():
    rng = np.random.default_rng(0)
    y = rng.normal(450, 70, 100)
    pred = y + 120 + rng.normal(0, 10, 100)
    curve = recalibration_curve(pred, y, ks=(5, 30), repetitions=100)
    intercept = {p.k: p for p in curve if p.mode == "intercept"}
    assert abs(intercept[30].bias_mean) < 5
    assert intercept[30].mae_mean < 15 < np.abs(pred - y).mean()
