import numpy as np
import pandas as pd
import pytest

from cattle3d.modeling.models import PHYSICS_FEATURE, VOLUME_FEATURES
from cattle3d.modeling.weight_model import ModelInputError, WeightModel


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    n = 60
    volume = rng.uniform(0.6, 1.1, n)
    data = {name: volume * rng.uniform(0.9, 1.1) + rng.normal(0, 0.01, n) for name in VOLUME_FEATURES}
    data[PHYSICS_FEATURE] = volume
    data.update({"left_n_points": 20_000, "right_n_points": 20_000, "top_n_points": 30_000})
    df = pd.DataFrame(data)
    df["weight_kg"] = 600 * volume + rng.normal(0, 10, n)
    df["herd"] = "synthetic"
    return df


@pytest.fixture(scope="module")
def model(table):
    return WeightModel.fit(table)


def test_predicts_close_to_truth_inside_the_range(model, table):
    preds = model.predict(table)
    err = np.abs([p.weight_kg for p in preds] - table["weight_kg"].to_numpy())
    assert err.mean() < 20
    assert all(p.lower_kg < p.weight_kg < p.upper_kg for p in preds)
    assert all(p.in_operating_range for p in preds)


def test_flags_an_animal_outside_the_training_range(model, table):
    row = table.iloc[0].to_dict()
    row[PHYSICS_FEATURE] = 2.0
    pred = model.predict_row(row)
    assert not pred.in_operating_range
    assert "outside the training range" in pred.reason


def test_flags_a_thin_segmentation(model, table):
    row = table.iloc[0].to_dict() | {"top_n_points": 100}
    assert "too few points in top view" in model.predict_row(row).reason


def test_missing_feature_fails_loudly(model, table):
    with pytest.raises(ModelInputError):
        model.predict(table.drop(columns=[PHYSICS_FEATURE]))


def test_few_reference_weighings_shift_only_the_intercept(model, table):
    heavier = table.assign(weight_kg=table["weight_kg"] + 50)
    slope, intercept = model.calibrate_herd("farm-1", heavier.iloc[:5])
    assert slope == 1.0 and intercept == pytest.approx(50, abs=20)
    base = model.predict_row(table.iloc[20].to_dict()).weight_kg
    assert model.predict_row(table.iloc[20].to_dict(), herd="farm-1").weight_kg == pytest.approx(base + intercept)
    model.herd_corrections.pop("farm-1")


def test_ten_or_more_reference_weighings_also_fit_the_slope(model, table):
    scaled = table.assign(weight_kg=table["weight_kg"] * 0.7)
    slope, _ = model.calibrate_herd("farm-2", scaled.iloc[:30])
    assert slope == pytest.approx(0.7, abs=0.12)
    model.herd_corrections.pop("farm-2")


def test_calibration_needs_a_minimum_of_animals(model, table):
    with pytest.raises(ValueError):
        model.calibrate_herd("farm-3", table.iloc[:2])


def test_save_and_load_round_trip(model, table, tmp_path):
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = WeightModel.load(path)
    assert loaded.predict_row(table.iloc[3].to_dict()) == model.predict_row(table.iloc[3].to_dict())
    assert loaded.card()["n_training_animals"] == 60
