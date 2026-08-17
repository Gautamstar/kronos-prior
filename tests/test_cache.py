"""Cache behaviour, and the Phase 0 determinism gate exercised through the stub."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kronosprior.cache import ForecastCache
from kronosprior.cli import main
from kronosprior.config import RunConfig
from kronosprior.data import BAR_COLUMNS, symbol_frame, synthetic_panel
from kronosprior.forecast import StubForecaster, rebalance_dates, window_for

SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]


@pytest.fixture
def setup(tmp_path):
    cfg = RunConfig(
        symbols=tuple(SYMBOLS), lookback=64, horizon=8, n_samples=32, root=tmp_path
    )
    panel = synthetic_panel(SYMBOLS, n_bars=400)
    return cfg, panel, ForecastCache.for_config(cfg)


def _fill(cfg, panel, cache, n_dates=3):
    fc = StubForecaster()
    cache.write_manifest(forecaster=fc)
    dates = rebalance_dates(panel.index, cfg)[:n_dates]
    for asof in dates:
        ctx, future = window_for(panel.index, asof, cfg)
        for sym in cfg.symbols:
            history = symbol_frame(panel, sym).iloc[ctx]
            cache.put(sym, asof, fc.sample(history, future, cfg.n_samples, cfg.seed_for(sym, asof)))
    return dates


class TestDeterminism:
    def test_same_seed_gives_identical_samples(self, setup):
        cfg, panel, _ = setup
        fc = StubForecaster()
        asof = rebalance_dates(panel.index, cfg)[0]
        ctx, future = window_for(panel.index, asof, cfg)
        history = symbol_frame(panel, "AAAUSDT").iloc[ctx]
        seed = cfg.seed_for("AAAUSDT", asof)

        a = fc.sample(history, future, cfg.n_samples, seed)
        b = fc.sample(history, future, cfg.n_samples, seed)
        assert np.array_equal(a, b)

    def test_different_seed_gives_different_samples(self, setup):
        cfg, panel, _ = setup
        fc = StubForecaster()
        asof = rebalance_dates(panel.index, cfg)[0]
        ctx, future = window_for(panel.index, asof, cfg)
        history = symbol_frame(panel, "AAAUSDT").iloc[ctx]
        a = fc.sample(history, future, cfg.n_samples, 1)
        b = fc.sample(history, future, cfg.n_samples, 2)
        assert not np.array_equal(a, b)

    def test_samples_do_not_collapse(self, setup):
        """A forecaster returning one repeated path has no distribution to carry."""
        cfg, panel, _ = setup
        fc = StubForecaster()
        asof = rebalance_dates(panel.index, cfg)[0]
        ctx, future = window_for(panel.index, asof, cfg)
        samples = fc.sample(symbol_frame(panel, "AAAUSDT").iloc[ctx], future, 64, 7)
        assert samples.std(axis=0).max() > 0

    def test_shape_contract(self, setup):
        cfg, panel, _ = setup
        asof = rebalance_dates(panel.index, cfg)[0]
        ctx, future = window_for(panel.index, asof, cfg)
        samples = StubForecaster().sample(
            symbol_frame(panel, "AAAUSDT").iloc[ctx], future, cfg.n_samples, 0
        )
        assert samples.shape == (cfg.n_samples, cfg.horizon, len(BAR_COLUMNS))


class TestCache:
    def test_round_trip(self, setup):
        cfg, panel, cache = setup
        dates = _fill(cfg, panel, cache, n_dates=2)
        got = cache.get("AAAUSDT", dates[0])
        assert got.shape == (cfg.n_samples, cfg.horizon, len(BAR_COLUMNS))
        assert got.dtype == np.float32

    def test_rejects_wrong_shape(self, setup):
        cfg, _, cache = setup
        with pytest.raises(ValueError, match="expected samples of shape"):
            cache.put("AAAUSDT", pd.Timestamp("2024-01-01", tz="UTC"), np.zeros((3, 3, 3)))

    def test_rejects_non_finite(self, setup):
        cfg, _, cache = setup
        bad = np.zeros((cfg.n_samples, cfg.horizon, len(BAR_COLUMNS)))
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            cache.put("AAAUSDT", pd.Timestamp("2024-01-01", tz="UTC"), bad)

    def test_missing_key_raises(self, setup):
        _, _, cache = setup
        with pytest.raises(KeyError):
            cache.get("AAAUSDT", pd.Timestamp("2030-01-01", tz="UTC"))

    def test_dates_round_trip_through_filenames(self, setup):
        cfg, panel, cache = setup
        written = _fill(cfg, panel, cache, n_dates=3)
        assert cache.dates("AAAUSDT") == list(written)

    def test_coverage_counts_every_symbol(self, setup):
        cfg, panel, cache = setup
        _fill(cfg, panel, cache, n_dates=3)
        cov = cache.coverage()
        assert set(cov.index) == set(SYMBOLS)
        assert (cov["n_dates"] == 3).all()

    def test_fingerprint_isolates_configurations(self, tmp_path):
        a = RunConfig(symbols=("AAAUSDT",), horizon=8, root=tmp_path)
        b = RunConfig(symbols=("AAAUSDT",), horizon=24, root=tmp_path)
        assert ForecastCache.for_config(a).root != ForecastCache.for_config(b).root


class TestManifest:
    def test_records_stubness_so_results_cannot_be_confused(self, setup):
        cfg, panel, cache = setup
        _fill(cfg, panel, cache, n_dates=1)
        manifest = cache.read_manifest()
        assert manifest["is_stub"] is True
        assert cache.is_stub is True
        assert manifest["forecaster"] == "StubForecaster"

    def test_records_the_config_fingerprint_and_universe_freeze(self, setup):
        cfg, panel, cache = setup
        _fill(cfg, panel, cache, n_dates=1)
        manifest = cache.read_manifest()
        assert manifest["fingerprint"] == cfg.fingerprint()
        assert manifest["universe_frozen_on"]
        assert "numpy" in manifest["versions"]

    def test_missing_manifest_is_an_error(self, setup):
        _, _, cache = setup
        with pytest.raises(FileNotFoundError):
            cache.read_manifest()


class TestHorizonReturns:
    def test_measures_from_the_realised_anchor_not_the_first_forecast_bar(self, setup):
        cfg, panel, cache = setup
        dates = _fill(cfg, panel, cache, n_dates=1)
        asof = dates[0]
        anchor = panel.xs("close", axis=1, level="field").loc[asof]

        rets = cache.horizon_returns(asof, anchor, list(cfg.symbols))
        assert rets.shape == (cfg.n_samples, len(SYMBOLS))

        close = BAR_COLUMNS.index("close")
        expected = cache.get("AAAUSDT", asof)[:, -1, close] / float(anchor["AAAUSDT"]) - 1.0
        assert np.allclose(rets[:, 0], expected)

    def test_column_order_follows_the_requested_symbols(self, setup):
        cfg, panel, cache = setup
        asof = _fill(cfg, panel, cache, n_dates=1)[0]
        anchor = panel.xs("close", axis=1, level="field").loc[asof]
        forward = cache.horizon_returns(asof, anchor, ["AAAUSDT", "BBBUSDT"])
        reverse = cache.horizon_returns(asof, anchor, ["BBBUSDT", "AAAUSDT"])
        assert np.allclose(forward[:, 0], reverse[:, 1])

    def test_missing_anchor_is_an_error(self, setup):
        cfg, panel, cache = setup
        asof = _fill(cfg, panel, cache, n_dates=1)[0]
        with pytest.raises(KeyError, match="anchor prices missing"):
            cache.horizon_returns(asof, pd.Series({"AAAUSDT": 100.0}), list(cfg.symbols))

    def test_independent_sampling_leaves_scenarios_uncorrelated(self, setup):
        """The defect the whole project exists to fix, pinned as a test.

        Kronos samples each asset independently, so stacking its paths into a scenario
        matrix produces near-zero cross-asset correlation regardless of how correlated
        the assets really are. Phase 2's coupling is what fixes this; if this test ever
        starts failing on its own, the sampling path changed underneath us.
        """
        cfg, panel, cache = setup
        asof = _fill(cfg, panel, cache, n_dates=1)[0]
        anchor = panel.xs("close", axis=1, level="field").loc[asof]

        scen = cache.horizon_returns(asof, anchor, list(cfg.symbols))
        corr = np.corrcoef(scen, rowvar=False)
        off = corr[~np.eye(len(SYMBOLS), dtype=bool)]

        realised = np.log(panel.xs("close", axis=1, level="field")).diff().dropna().corr()
        realised_off = realised.to_numpy()[~np.eye(len(SYMBOLS), dtype=bool)]

        assert np.abs(off).max() < 0.25, "scenarios should be ~uncorrelated before coupling"
        assert realised_off.mean() > 0.5, "the underlying assets are genuinely correlated"


class TestCliGate:
    def test_verify_passes_on_the_stub(self, tmp_path, capsys):
        code = main(
            [
                "--root", str(tmp_path), "verify",
                "--symbols", "AAAUSDT", "BBBUSDT",
                "--lookback", "64", "--horizon", "8", "--n-samples", "16",
                "--synthetic-bars", "300", "--stub",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "deterministic: PASS" in out
        assert "distribution: PASS" in out

    def test_forecast_then_reforecast_is_idempotent(self, tmp_path, capsys):
        argv = [
            "--root", str(tmp_path), "forecast",
            "--symbols", "AAAUSDT", "BBBUSDT",
            "--lookback", "64", "--horizon", "8", "--n-samples", "16",
            "--synthetic-bars", "300", "--stub", "--limit", "2",
        ]
        assert main(argv) == 0
        first = capsys.readouterr().out
        assert "wrote 4 shards" in first

        assert main(argv) == 0
        second = capsys.readouterr().out
        assert "wrote 0 shards, skipped 4" in second
