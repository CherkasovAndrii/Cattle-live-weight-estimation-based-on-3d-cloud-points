import numpy as np
import pytest

from cattle3d.modeling.calibration import (
    DRIFT_MIN_ANIMALS,
    check_drift,
    fit_correction,
)


def test_slope_and_intercept_are_recovered_from_enough_animals():
    rng = np.random.default_rng(0)
    raw = rng.uniform(450, 650, 30)
    weight = 0.7 * raw + 60 + rng.normal(0, 5, raw.size)
    c = fit_correction(raw, weight)
    assert c.slope == pytest.approx(0.7, abs=0.05)
    assert c.n_reference == 30
    np.testing.assert_allclose(c.apply(raw), 0.7 * raw + 60, atol=10)


def test_few_animals_move_only_the_intercept():
    raw = np.array([500.0, 550.0, 600.0, 650.0])
    c = fit_correction(raw, raw + 40)
    assert (c.slope, c.intercept_kg) == (1.0, pytest.approx(40))


def test_interval_is_never_narrower_than_the_model_interval():
    raw = np.linspace(400, 600, 20)
    assert fit_correction(raw, raw + 30, min_half_width_kg=42.0).half_width_kg == 42.0


def test_interval_widens_with_noisy_reference_weighings():
    rng = np.random.default_rng(1)
    raw = rng.uniform(450, 650, 20)
    noisy = fit_correction(raw, raw + rng.normal(0, 40, raw.size))
    clean = fit_correction(raw, raw + rng.normal(0, 5, raw.size))
    assert noisy.half_width_kg > 3 * clean.half_width_kg


def test_too_few_reference_weighings_are_refused():
    with pytest.raises(ValueError, match="at least 3"):
        fit_correction(np.array([500.0, 600.0]), np.array([510.0, 590.0]))


def test_no_drift_check_before_enough_spot_weighings():
    d = check_drift(np.full(DRIFT_MIN_ANIMALS - 1, 50.0))
    assert not d.drifted and d.ci_low_kg is None and "needed" in d.message


def test_a_consistent_shift_is_drift_and_noise_is_not():
    rng = np.random.default_rng(2)
    assert check_drift(25 + rng.normal(0, 10, 15)).drifted
    assert not check_drift(rng.normal(0, 20, 15)).drifted


def test_a_significant_but_small_shift_is_not_drift():
    rng = np.random.default_rng(3)
    d = check_drift(4 + rng.normal(0, 1, 20))
    assert d.ci_low_kg > 0 and not d.drifted
