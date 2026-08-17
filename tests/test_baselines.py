"""Baseline forecasters.

Two things get checked. That each one honours the `SampleForecaster` contract, so it can
flow through the same cache and diagnostics as Kronos. And that each one has the
statistical property it was built to have, measured against synthetic data whose
generating process is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from kronosprior.baselines import (
    BASELINES,
    ClimatologyForecaster,
    EwmaGaussianForecaster,
    PersistenceForecaster,
    build_baseline,
    terminal_closes,
)
from kronosprior.calibration import assess, coverage, crps, spread_skill_ratio
from kronosprior.data import BAR_COLUMNS, symbol_frame, synthetic_panel
from kronosprior.forecast import SampleForecaster

HORIZON = 24
N_SAMPLES = 256


@pytest.fixture(scope="module")
def history():
    panel = synthetic_panel(["AAAUSDT"], n_bars=600, seed=42)
    return symbol_frame(panel, "AAAUSDT")


@pytest.fixture(scope="module")
def future(history):
    step = history.index[1] - history.index[0]
    return history.index[-1] + step * np.arange(1, HORIZON + 1)


ALL = [PersistenceForecaster(), EwmaGaussianForecaster(), ClimatologyForecaster()]


class TestContract:
    @pytest.mark.parametrize("fc", ALL, ids=lambda f: f.label)
    def test_satisfies_the_forecaster_protocol(self, fc):
        assert isinstance(fc, SampleForecaster)

    @pytest.mark.parametrize("fc", ALL, ids=lambda f: f.label)
    def test_shape_and_dtype(self, fc, history, future):
        out = fc.sample(history, future, N_SAMPLES, seed=0)
        assert out.shape == (N_SAMPLES, HORIZON, len(BAR_COLUMNS))
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("fc", ALL, ids=lambda f: f.label)
    def test_seeded_and_reproducible(self, fc, history, future):
        a = fc.sample(history, future, N_SAMPLES, seed=7)
        b = fc.sample(history, future, N_SAMPLES, seed=7)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("fc", ALL[1:], ids=lambda f: f.label)
    def test_different_seeds_diverge(self, fc, history, future):
        a = fc.sample(history, future, N_SAMPLES, seed=1)
        b = fc.sample(history, future, N_SAMPLES, seed=2)
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize("fc", ALL, ids=lambda f: f.label)
    def test_ohlc_bounds_hold(self, fc, history, future):
        out = fc.sample(history, future, 64, seed=3)
        o, h, low, c = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
        assert (h >= np.maximum(o, c)).all()
        assert (low <= np.minimum(o, c)).all()
        assert (c > 0).all()

    @pytest.mark.parametrize("fc", ALL, ids=lambda f: f.label)
    def test_first_open_is_the_last_observed_close(self, fc, history, future):
        out = fc.sample(history, future, 16, seed=0)
        assert np.allclose(out[:, 0, 0], history["close"].to_numpy()[-1], rtol=1e-6)


class TestPersistence:
    def test_has_no_spread(self, history, future):
        closes = terminal_closes(PersistenceForecaster().sample(history, future, 64, 0))
        assert closes.std() == 0

    def test_crps_reduces_to_absolute_error(self, history, future):
        last = float(history["close"].to_numpy()[-1])
        closes = terminal_closes(PersistenceForecaster().sample(history, future, 64, 0))
        realized = last * 1.05
        assert np.allclose(crps(closes[None, :], [realized]), abs(realized - last))

    def test_covers_nothing(self, history, future):
        last = float(history["close"].to_numpy()[-1])
        closes = terminal_closes(PersistenceForecaster().sample(history, future, 64, 0))
        table = coverage(closes[None, :], [last * 1.02])
        assert (table["empirical"] == 0).all()


class TestEwmaGaussian:
    def test_recovers_a_known_volatility(self):
        """Feed it a walk with a known sigma and it should estimate that sigma."""
        rng = np.random.default_rng(0)
        true_sigma = 0.013
        n = 4000
        import pandas as pd

        steps = rng.normal(0, true_sigma, n)
        close = 100 * np.exp(np.cumsum(steps))
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
                "amount": close,
            },
            index=idx,
        )
        est = EwmaGaussianForecaster(lam=0.97).volatility(frame)
        assert abs(est - true_sigma) / true_sigma < 0.25, (est, true_sigma)

    def test_spread_grows_with_horizon(self, history, future):
        out = EwmaGaussianForecaster().sample(history, future, 512, seed=0)
        spread = out[:, :, BAR_COLUMNS.index("close")].std(axis=0)
        assert spread[-1] > spread[0]

    def test_random_walk_spread_scales_with_sqrt_time(self, history, future):
        """Variance accumulates linearly under a driftless walk, so spread goes as sqrt(h)."""
        out = EwmaGaussianForecaster().sample(history, future, 4000, seed=0)
        logs = np.log(out[:, :, BAR_COLUMNS.index("close")])
        spread = logs.std(axis=0)
        ratio = spread[-1] / spread[0]
        assert abs(ratio - np.sqrt(HORIZON)) / np.sqrt(HORIZON) < 0.1, ratio

    def test_is_well_calibrated_on_data_matching_its_assumptions(self, history, future):
        """On a driftless walk the EWMA Gaussian is the correct model, so it must pass."""
        rng = np.random.default_rng(1)
        fc = EwmaGaussianForecaster()
        sigma = fc.volatility(history)
        last = float(history["close"].to_numpy()[-1])

        # A large ensemble, because the interval endpoints are themselves estimated.
        # At 400 members the 50% quantiles carry enough Monte Carlo error to move
        # empirical coverage by several points, which reads as miscalibration.
        samples = terminal_closes(fc.sample(history, future, 8000, seed=0))
        truth = last * np.exp(rng.normal(0, sigma * np.sqrt(HORIZON), size=3000))

        # Compare the same predictive distribution against many draws from the truth.
        report = assess(np.tile(samples, (3000, 1)), truth, label="ewma")
        assert 0.9 < report.spread_skill < 1.1, report.spread_skill
        assert (report.coverage["error"].abs() < 0.04).all(), report.coverage

    def test_rejects_a_bad_decay(self):
        for lam in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="lam must be"):
                EwmaGaussianForecaster(lam=lam)

    def test_requires_enough_history(self, future):
        panel = synthetic_panel(["AAAUSDT"], n_bars=20, seed=0)
        with pytest.raises(ValueError, match="need at least"):
            EwmaGaussianForecaster(min_obs=32).sample(
                symbol_frame(panel, "AAAUSDT"), future, 8, seed=0
            )


class TestClimatology:
    def test_spread_is_in_the_right_ballpark(self, history, future):
        """Resampled history should not be wildly narrower or wider than a walk."""
        clim = terminal_closes(ClimatologyForecaster().sample(history, future, 2000, seed=0))
        ewma = terminal_closes(EwmaGaussianForecaster().sample(history, future, 2000, seed=0))
        assert 0.4 < clim.std() / ewma.std() < 2.5

    def test_block_size_is_honoured(self, history, future):
        fc = ClimatologyForecaster(block=6)
        out = fc.sample(history, future, 128, seed=0)
        assert out.shape[1] == HORIZON

    def test_rejects_a_bad_block(self):
        with pytest.raises(ValueError, match="block must be"):
            ClimatologyForecaster(block=0)

    def test_requires_enough_history(self, future):
        panel = synthetic_panel(["AAAUSDT"], n_bars=4, seed=0)
        with pytest.raises(ValueError, match="need at least"):
            ClimatologyForecaster(block=12).sample(
                symbol_frame(panel, "AAAUSDT"), future, 8, seed=0
            )


class TestRegistry:
    def test_builds_every_registered_name(self):
        for name in BASELINES:
            assert build_baseline(name).label == name

    def test_rejects_an_unknown_name(self):
        with pytest.raises(KeyError, match="unknown baseline"):
            build_baseline("magic")


class TestRelativeOrdering:
    def test_a_spreadless_forecast_loses_to_one_with_spread(self, history, future):
        """The floor must actually be a floor when the truth moves at all."""
        rng = np.random.default_rng(2)
        last = float(history["close"].to_numpy()[-1])
        sigma = EwmaGaussianForecaster().volatility(history)
        truth = last * np.exp(rng.normal(0, sigma * np.sqrt(HORIZON), size=2000))

        flat = terminal_closes(PersistenceForecaster().sample(history, future, 64, 0))
        walk = terminal_closes(EwmaGaussianForecaster().sample(history, future, 400, 0))

        flat_score = crps(np.tile(flat, (2000, 1)), truth).mean()
        walk_score = crps(np.tile(walk, (2000, 1)), truth).mean()
        assert walk_score < flat_score

    def test_persistence_spread_skill_is_degenerate(self, history, future):
        rng = np.random.default_rng(2)
        last = float(history["close"].to_numpy()[-1])
        truth = last * np.exp(rng.normal(0, 0.05, size=500))
        flat = terminal_closes(PersistenceForecaster().sample(history, future, 32, 0))
        assert spread_skill_ratio(np.tile(flat, (500, 1)), truth) == 0.0
