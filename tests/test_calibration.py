"""Calibration diagnostics, tested against ensembles whose true calibration is known.

The method throughout: build a forecast that is deliberately correct, deliberately too
narrow, deliberately too wide, or deliberately shifted, then assert each diagnostic
reports what it should. A diagnostic that cannot detect a fault planted on purpose will
not detect one that arrives by accident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kronosprior.calibration import (
    DEFAULT_LEVELS,
    assess,
    bias,
    compare,
    coverage,
    crps,
    fit_rescale,
    flatness,
    pit_values,
    rank_histogram,
    ranks,
    rescale,
    spread_skill_ratio,
)

N_CASES = 4000
N_SAMPLES = 200


def make(sigma_forecast: float, mu_forecast: float = 0.0, seed: int = 0):
    """Truth is standard normal. The forecast is N(mu_forecast, sigma_forecast)."""
    rng = np.random.default_rng(seed)
    realized = rng.normal(0.0, 1.0, size=N_CASES)
    samples = rng.normal(mu_forecast, sigma_forecast, size=(N_CASES, N_SAMPLES))
    return samples, realized


@pytest.fixture(scope="module")
def calibrated():
    return make(1.0, seed=1)


@pytest.fixture(scope="module")
def too_narrow():
    return make(0.5, seed=2)


@pytest.fixture(scope="module")
def too_wide():
    return make(2.0, seed=3)


@pytest.fixture(scope="module")
def shifted():
    return make(1.0, mu_forecast=1.0, seed=4)


# --------------------------------------------------------------------------------------


class TestPit:
    def test_uniform_when_calibrated(self, calibrated):
        pit = pit_values(*calibrated)
        assert pit.min() >= 0 and pit.max() <= 1
        assert abs(pit.mean() - 0.5) < 0.02
        # A uniform variable has variance 1/12.
        assert abs(pit.var() - 1 / 12) < 0.01

    def test_piles_at_the_edges_when_too_narrow(self, too_narrow):
        """An overconfident forecast puts the truth outside its range too often."""
        pit = pit_values(*too_narrow)
        extreme = ((pit < 0.05) | (pit > 0.95)).mean()
        assert extreme > 0.25, extreme

    def test_piles_in_the_middle_when_too_wide(self, too_wide):
        pit = pit_values(*too_wide)
        middle = ((pit > 0.35) & (pit < 0.65)).mean()
        assert middle > 0.45, middle

    def test_shifts_when_biased(self, shifted):
        """A forecast centred too high leaves the truth low in its distribution."""
        assert pit_values(*shifted).mean() < 0.35


class TestRanks:
    def test_within_bounds(self, calibrated):
        r = ranks(*calibrated, rng=np.random.default_rng(0))
        assert r.min() >= 1
        assert r.max() <= N_SAMPLES + 1

    def test_histogram_is_flat_when_calibrated(self, calibrated):
        counts = rank_histogram(*calibrated, n_bins=20)
        _, ri = flatness(counts)
        assert ri < 0.15, ri

    def test_histogram_is_not_flat_when_miscalibrated(self, too_narrow, too_wide, shifted):
        for data in (too_narrow, too_wide, shifted):
            _, ri = flatness(rank_histogram(*data, n_bins=20))
            assert ri > 0.40, ri

    def test_rejects_more_bins_than_ranks(self):
        samples = np.random.default_rng(0).normal(size=(10, 5))
        with pytest.raises(ValueError, match="exceeds the 6 possible ranks"):
            rank_histogram(samples, np.zeros(10), n_bins=20)

    def test_degenerate_ensemble_ranks_are_randomised(self):
        """With every sample equal, ties must scatter, or the histogram lies."""
        samples = np.zeros((500, 10))
        realized = np.zeros(500)
        r = ranks(samples, realized, rng=np.random.default_rng(0))
        assert len(np.unique(r)) > 5


class TestCoverage:
    def test_matches_nominal_when_calibrated(self, calibrated):
        table = coverage(*calibrated)
        assert (table["error"].abs() < 0.02).all(), table

    def test_falls_short_when_too_narrow(self, too_narrow):
        table = coverage(*too_narrow)
        assert (table["error"] < -0.05).all(), table

    def test_overshoots_when_too_wide(self, too_wide):
        table = coverage(*too_wide)
        # A 95% interval cannot overshoot by much, so check the levels with headroom.
        assert (table.loc[table["nominal"] <= 0.8, "error"] > 0.05).all(), table

    def test_width_grows_with_level(self, calibrated):
        table = coverage(*calibrated, levels=DEFAULT_LEVELS)
        assert table["mean_width"].is_monotonic_increasing


class TestCrps:
    def test_matches_the_analytic_gaussian_value(self):
        """CRPS(N(mu, sigma), y) has a closed form. The sample estimator must find it."""
        rng = np.random.default_rng(7)
        mu, sigma, y = 0.3, 1.7, 1.1
        samples = rng.normal(mu, sigma, size=(1, 20000))

        z = (y - mu) / sigma
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        pdf = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
        expected = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))

        got = float(crps(samples, np.array([y]))[0])
        assert abs(got - expected) < 0.02, (got, expected)

    def test_calibrated_beats_both_miscalibrations(self, calibrated, too_narrow, too_wide):
        """CRPS is a proper scoring rule, so the truthful forecast must win."""
        best = crps(*calibrated).mean()
        assert best < crps(*too_narrow).mean()
        assert best < crps(*too_wide).mean()

    def test_calibrated_beats_a_biased_forecast(self, calibrated, shifted):
        assert crps(*calibrated).mean() < crps(*shifted).mean()

    def test_zero_for_a_perfect_point_forecast(self):
        samples = np.full((3, 8), 2.0)
        assert np.allclose(crps(samples, np.full(3, 2.0)), 0.0)

    def test_reduces_to_absolute_error_without_spread(self):
        samples = np.full((4, 16), 1.0)
        realized = np.array([1.0, 2.0, 0.0, 4.0])
        assert np.allclose(crps(samples, realized), np.abs(realized - 1.0))

    def test_fair_estimator_needs_two_samples(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            crps(np.zeros((2, 1)), np.zeros(2), fair=True)

    def test_fair_estimator_removes_the_small_ensemble_penalty(self):
        """The naive term divides by n^2, counting the n zero-distance i == j pairs.

        That understates E|X - X'| and so overstates CRPS, penalising small ensembles
        for being small. The fair estimator divides by n(n-1) and scores lower.
        """
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(200, 4))
        realized = rng.normal(size=200)
        assert crps(samples, realized, fair=True).mean() < crps(
            samples, realized, fair=False
        ).mean()

    def test_the_two_estimators_converge_on_large_ensembles(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(200, 2000))
        realized = rng.normal(size=200)
        fair = crps(samples, realized, fair=True).mean()
        naive = crps(samples, realized, fair=False).mean()
        assert abs(fair - naive) < 0.002


class TestSpreadSkill:
    def test_near_one_when_calibrated(self, calibrated):
        assert abs(spread_skill_ratio(*calibrated) - 1.0) < 0.05

    def test_below_one_when_too_narrow(self, too_narrow):
        assert spread_skill_ratio(*too_narrow) < 0.65

    def test_above_one_when_too_wide(self, too_wide):
        assert spread_skill_ratio(*too_wide) > 1.5

    def test_corrected_for_ensemble_size(self):
        """A small ensemble must not look under-dispersed purely for being small."""
        rng = np.random.default_rng(11)
        realized = rng.normal(size=6000)
        for n in (4, 16, 512):
            samples = rng.normal(size=(6000, n))
            assert abs(spread_skill_ratio(samples, realized) - 1.0) < 0.12, n


class TestBias:
    def test_zero_when_centred(self, calibrated):
        assert abs(bias(*calibrated)) < 0.05

    def test_detects_a_shift(self, shifted):
        assert abs(bias(*shifted) - 1.0) < 0.05


class TestRescale:
    def test_preserves_the_mean(self):
        rng = np.random.default_rng(5)
        samples = rng.normal(size=(50, 32))
        widened = rescale(samples, 2.5)
        assert np.allclose(samples.mean(axis=1), widened.mean(axis=1))

    def test_scales_the_spread(self):
        rng = np.random.default_rng(5)
        samples = rng.normal(size=(50, 64))
        assert np.allclose(rescale(samples, 3.0).std(axis=1), 3.0 * samples.std(axis=1))

    def test_fit_recovers_a_known_narrowing(self, too_narrow):
        """Forecast sigma is 0.5 against truth sigma 1.0, so the fix is a factor of 2."""
        assert abs(fit_rescale(*too_narrow) - 2.0) < 0.25

    def test_fit_recovers_a_known_widening(self, too_wide):
        assert abs(fit_rescale(*too_wide) - 0.5) < 0.15

    def test_fit_leaves_a_calibrated_forecast_alone(self, calibrated):
        assert abs(fit_rescale(*calibrated) - 1.0) < 0.15

    def test_correction_actually_improves_crps(self, too_narrow):
        samples, realized = too_narrow
        before = crps(samples, realized).mean()
        after = crps(rescale(samples, fit_rescale(samples, realized)), realized).mean()
        assert after < before


class TestReport:
    def test_labels_dispersion_correctly(self, calibrated, too_narrow, too_wide):
        assert assess(*calibrated, label="ok").dispersion == "calibrated"
        assert "overconfident" in assess(*too_narrow, label="narrow").dispersion
        assert "underconfident" in assess(*too_wide, label="wide").dispersion

    def test_summary_and_str_render(self, calibrated):
        report = assess(*calibrated, label="calibrated")
        assert report.summary()["label"] == "calibrated"
        text = str(report)
        assert "CRPS" in text and "coverage" in text and "50% nominal" in text

    def test_compare_sorts_by_crps(self, calibrated, too_narrow, too_wide):
        table = compare(
            [
                assess(*too_narrow, label="narrow"),
                assess(*calibrated, label="good"),
                assess(*too_wide, label="wide"),
            ]
        )
        assert table.loc[0, "label"] == "good"
        assert table["crps"].is_monotonic_increasing


class TestInputValidation:
    def test_rejects_mismatched_case_counts(self):
        with pytest.raises(ValueError, match="case count mismatch"):
            crps(np.zeros((5, 10)), np.zeros(4))

    def test_rejects_non_finite(self):
        samples = np.zeros((3, 10))
        samples[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            crps(samples, np.zeros(3))

    def test_accepts_a_single_case_as_1d(self):
        rng = np.random.default_rng(0)
        assert crps(rng.normal(size=64), 0.0).shape == (1,)
