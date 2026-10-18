"""Weight regressors and the feature sets they use.

Owns: model definitions as sklearn estimators. Every model is a Pipeline so
scaling and tuning are fitted inside each training fold, never on test data.
Must not know about files, herds or reports.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# The single feature of the physics model: mass is close to proportional to
# trunk volume. Two parameters, so it extrapolates linearly.
PHYSICS_FEATURE = "top_trunk_cs_volume"

# Geometry features used by the multi-feature models. Quality signals
# (point counts, floor inliers, component counts) are deliberately left out:
# they depend on the capture setup and would not transfer between farms.
VOLUME_FEATURES = (
    "top_trunk_cs_volume",
    "top_mid_cs_area",
    "top_mid_width",
    "top_mid_height",
    "top_cs_volume",
    "top_trunk_footprint_area",
    "top_trunk_length",
    "top_trunk_width",
    "top_trunk_height",
    "top_height",
    "trunk_box_volume",
    "side_mean_mid_side_area",
    "side_mean_mid_height",
    "side_max_length",
)

# A factory builds a fresh, unfitted estimator for a given random seed.
ModelFactory = Callable[[int], BaseEstimator]


def physics_model(seed: int = 0) -> Pipeline:
    return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.5, alpha=0.0, max_iter=1000))


def robust_linear_model(seed: int = 0) -> Pipeline:
    return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.5, alpha=1.0, max_iter=2000))


def lgbm_model(seed: int = 0) -> GridSearchCV:
    """Small-sample LightGBM, tuned by an inner 3-fold CV (nested when cross-validated)."""
    base = LGBMRegressor(
        objective="huber",
        alpha=20.0,  # Huber transition in kg; the LightGBM default (0.9) makes it L1
        n_estimators=600,
        learning_rate=0.02,
        min_child_samples=8,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=1,  # tiny data: extra threads only add overhead and contention
    )
    grid = {"num_leaves": [4, 8], "n_estimators": [300, 600], "reg_lambda": [0.0, 5.0]}
    return GridSearchCV(
        base,
        grid,
        cv=KFold(3, shuffle=True, random_state=seed),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )


class ColumnSubset(BaseEstimator, RegressorMixin):
    """Fit ``estimator`` on a fixed subset of columns of a 2-D array."""

    def __init__(self, estimator: BaseEstimator, columns: tuple[int, ...]):
        self.estimator = estimator
        self.columns = columns

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ColumnSubset":
        self.estimator_ = clone(self.estimator).fit(X[:, list(self.columns)], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X[:, list(self.columns)])


class MeanEnsemble(BaseEstimator, RegressorMixin):
    """Unweighted mean of member predictions; members are refitted on each fit."""

    def __init__(self, members: tuple[BaseEstimator, ...]):
        self.members = members

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanEnsemble":
        self.members_ = [clone(m).fit(X, y) for m in self.members]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.members_], axis=0)


def ensemble_model(feature_names: tuple[str, ...], seed: int = 0) -> MeanEnsemble:
    """Physics + robust linear + LightGBM, the deployable model."""
    physics_col = (feature_names.index(PHYSICS_FEATURE),)
    return MeanEnsemble(
        (
            ColumnSubset(physics_model(), physics_col),
            robust_linear_model(),
            lgbm_model(seed),
        )
    )
