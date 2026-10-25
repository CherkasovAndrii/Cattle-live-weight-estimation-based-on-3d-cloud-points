"""The sensor-error study: degradations, the floor noise estimate, scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from cattle3d.evaluation.sensor import (
    LevelResult,
    Perturbation,
    recorded_quality,
    perturb,
    perturb_views,
    score_herd,
    tolerated_level,
)
from cattle3d.evaluation.stats import Interval
from cattle3d.types import OrganizedCloud, View


def _floor_cloud(sigma_m: float = 0.0, seed: int = 0, h: int = 120, w: int = 160) -> OrganizedCloud:
    """Top camera 2.5 m above a flat floor; range noise of ``sigma_m``."""
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - w / 2) / 365.0, (v - h / 2) / 365.0, np.ones((h, w))], axis=2)
    xyz = rays * (2.5 / rays[..., 2:3])
    r = np.linalg.norm(xyz, axis=2, keepdims=True)
    noise = np.random.default_rng(seed).normal(0, sigma_m, size=(h, w, 1))
    return OrganizedCloud(xyz * (1 + noise / r))


def test_zero_level_and_scale_are_exact():
    cloud = _floor_cloud()
    rng = np.random.default_rng(0)
    assert np.array_equal(perturb(cloud, Perturbation("noise", 0.0), rng).xyz, cloud.xyz)
    scaled = perturb(cloud, Perturbation("scale", -0.03), rng).xyz
    assert np.allclose(scaled, cloud.xyz * 0.97)


def test_noise_moves_points_along_their_rays():
    cloud = _floor_cloud()
    noisy = perturb(cloud, Perturbation("noise", 0.02), np.random.default_rng(1)).xyz
    a, b = cloud.xyz.reshape(-1, 3), noisy.reshape(-1, 3)
    assert np.allclose(np.cross(a, b), 0, atol=1e-9)  # same direction from the camera
    rel = np.linalg.norm(b, axis=1) / np.linalg.norm(a, axis=1) - 1
    assert rel.std() == pytest.approx(0.02, rel=0.05)


def test_dropout_removes_that_share_of_valid_pixels():
    xyz = _floor_cloud().xyz.copy()
    xyz[:10] = 0.0  # already missing
    cloud = OrganizedCloud(xyz)
    out = perturb(cloud, Perturbation("dropout", 0.3), np.random.default_rng(2))
    assert not out.valid[:10].any()
    assert 1 - out.valid.sum() / cloud.valid.sum() == pytest.approx(0.3, abs=0.01)


def test_degradation_is_reproducible_per_animal():
    clouds = {View.TOP: _floor_cloud()}
    p = Perturbation("noise", 0.01)
    a = perturb_views(clouds, p, "cowdb2/7")[View.TOP].xyz
    assert np.array_equal(a, perturb_views(clouds, p, "cowdb2/7")[View.TOP].xyz)
    assert not np.array_equal(a, perturb_views(clouds, p, "cowdb2/8")[View.TOP].xyz)


def test_bad_levels_are_rejected():
    with pytest.raises(ValueError):
        Perturbation("dropout", 1.0)
    with pytest.raises(ValueError):
        Perturbation("blur", 0.1)


def test_floor_noise_recovers_known_sigma():
    f, edges = recorded_quality(_floor_cloud(sigma_m=0.005), View.TOP)
    assert f.range_m == pytest.approx(2.55, abs=0.1)
    assert f.local_sigma_m == pytest.approx(0.005, rel=0.15)
    assert edges == 0
    assert recorded_quality(_floor_cloud(), View.TOP)[0].local_sigma_m < 1e-6


def _level(kind: str, level: float, delta_high: float, failed: int = 0) -> LevelResult:
    iv = Interval(0.0, 0.0, delta_high)
    return LevelResult("h", Perturbation(kind, level), 50, failed, iv, iv, 0.0, 0.0, 0.0, 0.0)


def test_tolerated_level_stops_at_the_first_breach():
    results = [_level("noise", 0.005, 1.0), _level("noise", 0.01, 2.0), _level("noise", 0.02, 3.0),
               _level("noise", 0.03, 1.0)]
    assert tolerated_level(results, "noise") == 0.01
    assert tolerated_level([_level("dropout", 0.1, 0.5, failed=1)], "dropout") is None
    # Scale errors of both signs: tolerated up to the smallest magnitude that breaches.
    scale = [_level("scale", -0.03, 3.0), _level("scale", -0.01, 1.0), _level("scale", 0.01, 1.0)]
    assert tolerated_level(scale, "scale") == 0.01


def test_score_herd_pairs_degraded_with_clean_animals():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.6, 1.0, 60)
    clean = pd.DataFrame({"animal_id": [str(i) for i in range(60)], "weight_kg": 700 * v + rng.normal(0, 5, 60),
                          "vol": v, "error": ""})
    shifted = clean.assign(vol=clean.vol * 1.05)
    broken = clean.assign(error=["SegmentationError: x" if i < 6 else "" for i in range(60)])
    same, scaled, lost = (Perturbation("noise", 0.0), Perturbation("scale", 0.05), Perturbation("dropout", 0.5))
    out = score_herd("h", clean, {same: clean, scaled: shifted, lost: broken}, ("vol",),
                     lambda s: LinearRegression(), lambda s: DummyRegressor(), seeds=(0,))
    by = {r.perturbation: r for r in out}
    assert by[same].delta_mae.estimate == 0 and by[same].n == 60
    assert by[scaled].bias_kg > 20 and by[scaled].delta_mae.estimate > 10
    assert by[scaled].mae_recal < by[scaled].mae.estimate  # the herd correction absorbs a scale error
    assert by[lost].n_failed == 6 and by[lost].n == 54


def test_quality_limits_separate_a_degraded_camera():
    from cattle3d.geometry.quality import view_quality
    from cattle3d.types import Plane

    floor = Plane(np.array([0.0, 0.0, -1.0]), 2.5)
    cloud = _floor_cloud(sigma_m=0.003)
    good = view_quality(cloud, View.TOP, floor)
    assert not good.noisy and not good.scattered_loss
    noisy = view_quality(perturb(cloud, Perturbation("noise", 0.012), np.random.default_rng(0)), View.TOP, floor, 0.1)
    assert noisy.noisy and noisy.floor_noise_rel == pytest.approx(0.012, rel=0.2)
    holes = view_quality(perturb(cloud, Perturbation("dropout", 0.25), np.random.default_rng(0)), View.TOP, floor)
    assert holes.scattered_loss and not holes.noisy
