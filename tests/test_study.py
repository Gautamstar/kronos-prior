"""The Phase 1 study end to end, over a stub cache.

The load-bearing assertion is the last one: on data whose generating process is known,
the ranking has to recover it. A study that cannot order forecasters correctly when the
answer is known will not be trusted when it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from kronosprior.baselines import EwmaGaussianForecaster
from kronosprior.cache import ForecastCache
from kronosprior.cli import main
from kronosprior.config import RunConfig
from kronosprior.data import symbol_frame, synthetic_panel
from kronosprior.forecast import StubForecaster, rebalance_dates, window_for
from kronosprior.study import (
    gather_baseline,
    gather_cached,
    per_symbol_table,
    pool,
    realized_terminal,
    run_study,
)

SYMBOLS = ["AAAUSDT", "BBBUSDT"]


@pytest.fixture
def filled(tmp_path):
    cfg = RunConfig(
        symbols=tuple(SYMBOLS), lookback=128, horizon=12, n_samples=128, root=tmp_path
    )
    panel = synthetic_panel(SYMBOLS, n_bars=900, seed=5)
    cache = ForecastCache.for_config(cfg)
    fc = StubForecaster()
    cache.write_manifest(forecaster=fc)
    for asof in rebalance_dates(panel.index, cfg):
        ctx, future = window_for(panel.index, asof, cfg)
        for sym in SYMBOLS:
            history = symbol_frame(panel, sym).iloc[ctx]
            cache.put(sym, asof, fc.sample(history, future, cfg.n_samples,
                                           cfg.seed_for(sym, asof)))
    return cfg, panel, cache


class TestGather:
    def test_shapes_line_up(self, filled):
        cfg, panel, cache = filled
        got = gather_cached(cache, panel, SYMBOLS)
        assert set(got) == set(SYMBOLS)
        for samples, truth in got.values():
            assert samples.ndim == 2
            assert samples.shape[1] == cfg.n_samples
            assert samples.shape[0] == truth.shape[0]

    def test_realized_is_the_close_at_the_horizon_end(self, filled):
        cfg, panel, cache = filled
        asof = rebalance_dates(panel.index, cfg)[0]
        _, future = window_for(panel.index, asof, cfg)
        expected = panel.loc[future[-1], (SYMBOLS[0], "close")]
        assert realized_terminal(panel, asof, cfg, SYMBOLS)[SYMBOLS[0]] == expected

    def test_realized_never_reads_before_the_horizon_ends(self, filled):
        """The truth must come from the last bar of the window, never an earlier one."""
        cfg, panel, cache = filled
        samples, truth = gather_cached(cache, panel, [SYMBOLS[0]])[SYMBOLS[0]]
        dates = cache.dates(SYMBOLS[0])
        closes = panel[(SYMBOLS[0], "close")]
        for i, asof in enumerate(dates[: len(truth)]):
            _, future = window_for(panel.index, asof, cfg)
            assert truth[i] == closes.loc[future[-1]]
            assert future[-1] > asof

    def test_baseline_runs_over_the_same_windows(self, filled):
        cfg, panel, cache = filled
        dates = cache.dates(SYMBOLS[0])
        cached = gather_cached(cache, panel, SYMBOLS)
        base = gather_baseline(EwmaGaussianForecaster(), panel, cfg, dates, SYMBOLS)
        for sym in SYMBOLS:
            assert base[sym][0].shape == cached[sym][0].shape
            # Identical windows means identical truth.
            assert np.allclose(base[sym][1], cached[sym][1])

    def test_empty_cache_yields_nothing(self, tmp_path):
        cfg = RunConfig(symbols=("AAAUSDT",), lookback=64, horizon=8, root=tmp_path)
        panel = synthetic_panel(["AAAUSDT"], n_bars=300)
        assert gather_cached(ForecastCache.for_config(cfg), panel) == {}


class TestPool:
    def test_normalises_away_the_price_level(self):
        """A symbol trading at 60000 must not swamp one trading at 3."""
        cheap = (np.full((10, 8), 3.0), np.full(10, 3.0))
        dear = (np.full((10, 8), 60000.0), np.full(10, 60000.0))
        samples, truth = pool({"cheap": cheap, "dear": dear})
        assert samples.shape == (20, 8)
        assert np.allclose(truth, 1.0)
        assert np.allclose(samples, 1.0)

    def test_rejects_an_empty_input(self):
        with pytest.raises(ValueError, match="nothing to pool"):
            pool({})


class TestRunStudy:
    def test_produces_a_report_per_forecaster(self, filled):
        _, panel, cache = filled
        reports, table = run_study(cache, panel, SYMBOLS)
        assert len(reports) == 4
        assert len(table) == 4
        assert table["crps"].is_monotonic_increasing

    def test_marks_the_stub_so_results_cannot_be_mistaken(self, filled):
        _, panel, cache = filled
        reports, _ = run_study(cache, panel, SYMBOLS)
        assert "STUB" in reports[0].label

    def test_persistence_is_the_worst(self, filled):
        """Zero spread has to lose on a proper scoring rule when the truth moves."""
        _, panel, cache = filled
        _, table = run_study(cache, panel, SYMBOLS)
        assert table.iloc[-1]["label"] == "persistence"

    def test_correctly_specified_forecasters_beat_misspecified_ones(self, filled):
        """The synthetic panel is a GBM and the stub samples a GBM, so the stub is
        correctly specified here. So is the EWMA Gaussian, which estimates the same
        volatility from the window, so which of those two wins is noise at this many
        cases. What must hold is that both separate cleanly from the misspecified ones.

        This is the check that the study measures what it claims. If a correctly
        specified forecaster cannot be told apart from a flat line, the ranking means
        nothing when the answer is unknown.
        """
        _, panel, cache = filled
        _, table = run_study(cache, panel, SYMBOLS)
        scores = table.set_index("label")["crps"]
        stub = next(label for label in scores.index if "Stub" in label)

        assert scores[stub] < 0.8 * scores["persistence"], scores
        assert scores[stub] <= 1.05 * scores.min(), scores

    def test_baseline_selection_is_honoured(self, filled):
        _, panel, cache = filled
        _, table = run_study(cache, panel, SYMBOLS, baselines=["persistence"])
        assert set(table["label"]) == {"StubForecaster (STUB)", "persistence"}

    def test_no_baselines_leaves_only_the_cache(self, filled):
        _, panel, cache = filled
        _, table = run_study(cache, panel, SYMBOLS, baselines=[])
        assert len(table) == 1

    def test_errors_on_an_empty_cache(self, tmp_path):
        cfg = RunConfig(symbols=("AAAUSDT",), lookback=64, horizon=8, root=tmp_path)
        cache = ForecastCache.for_config(cfg)
        cache.write_manifest(forecaster=StubForecaster())
        panel = synthetic_panel(["AAAUSDT"], n_bars=300)
        with pytest.raises(ValueError, match="no usable forecasts"):
            run_study(cache, panel, ["AAAUSDT"])

    def test_per_symbol_table_covers_every_symbol(self, filled):
        _, panel, cache = filled
        table = per_symbol_table(gather_cached(cache, panel, SYMBOLS))
        assert set(table.index) == set(SYMBOLS)
        assert (table["n_cases"] > 0).all()


class TestCli:
    def test_calibrate_runs_and_ranks(self, tmp_path, capsys):
        common = [
            "--root", str(tmp_path),
            "--symbols", "AAAUSDT", "BBBUSDT",
            "--lookback", "128", "--horizon", "12", "--n-samples", "64",
            "--synthetic-bars", "900", "--stub",
        ]
        assert main([common[0], common[1], "forecast", *common[2:]]) == 0
        capsys.readouterr()

        assert main([common[0], common[1], "calibrate", *common[2:], "--per-symbol"]) == 0
        out = capsys.readouterr().out
        assert "ranked by CRPS" in out
        assert "persistence" in out
        assert "ewma-gaussian" in out
        assert "per symbol" in out
        assert "plumbing, not a result" in out
