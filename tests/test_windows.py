"""No-lookahead guarantees.

The one rule the whole project rests on: a forecast made at the close of bar t may use
no data after t, and may not be acted on before t+1. These tests are the enforcement.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kronosprior.config import RunConfig
from kronosprior.data import synthetic_panel
from kronosprior.forecast import rebalance_dates, window_for

SYMBOLS = ["AAAUSDT", "BBBUSDT"]


@pytest.fixture
def setup():
    cfg = RunConfig(symbols=tuple(SYMBOLS), lookback=64, horizon=8, max_context=512)
    panel = synthetic_panel(SYMBOLS, n_bars=400)
    return cfg, panel


class TestWindowing:
    def test_context_never_touches_the_future(self, setup):
        cfg, panel = setup
        for asof in rebalance_dates(panel.index, cfg):
            ctx, future = window_for(panel.index, asof, cfg)
            ctx_index = panel.index[ctx]
            assert ctx_index[-1] == asof, "context must end exactly at asof"
            assert len(ctx_index) == cfg.lookback
            assert len(future) == cfg.horizon
            assert ctx_index[-1] < future[0], "context overlaps the forecast window"
            assert future[0] > asof, "first predicted bar must be strictly after asof"

    def test_future_starts_one_bar_after_asof(self, setup):
        cfg, panel = setup
        asof = rebalance_dates(panel.index, cfg)[3]
        _, future = window_for(panel.index, asof, cfg)
        step = panel.index[1] - panel.index[0]
        assert future[0] - asof == step

    def test_rebalance_dates_are_spaced_by_the_horizon(self, setup):
        cfg, panel = setup
        dates = rebalance_dates(panel.index, cfg)
        step = panel.index[1] - panel.index[0]
        gaps = pd.Series(dates).diff().dropna().unique()
        assert len(gaps) == 1
        assert gaps[0] == step * cfg.horizon, "overlapping forecasts would leak across CV folds"

    def test_first_date_has_full_history_and_last_has_full_future(self, setup):
        cfg, panel = setup
        dates = rebalance_dates(panel.index, cfg)
        assert panel.index.get_loc(dates[0]) == cfg.lookback - 1
        assert panel.index.get_loc(dates[-1]) + cfg.horizon <= len(panel.index) - 1

    def test_rejects_a_window_without_enough_history(self, setup):
        cfg, panel = setup
        with pytest.raises(ValueError, match="insufficient history"):
            window_for(panel.index, panel.index[cfg.lookback - 2], cfg)

    def test_rejects_a_window_without_enough_future(self, setup):
        cfg, panel = setup
        with pytest.raises(ValueError, match="insufficient future"):
            window_for(panel.index, panel.index[-2], cfg)

    def test_rejects_a_panel_that_is_too_short(self):
        cfg = RunConfig(symbols=("AAAUSDT",), lookback=64, horizon=8)
        panel = synthetic_panel(["AAAUSDT"], n_bars=40)
        with pytest.raises(ValueError, match="not enough bars"):
            rebalance_dates(panel.index, cfg)


class TestConfigGuards:
    def test_lookback_may_not_exceed_context(self):
        with pytest.raises(ValueError, match="exceeds max_context"):
            RunConfig(lookback=1024, max_context=512)

    def test_rejects_duplicate_symbols(self):
        with pytest.raises(ValueError, match="duplicates"):
            RunConfig(symbols=("BTCUSDT", "BTCUSDT"))

    def test_rejects_unknown_market(self):
        with pytest.raises(ValueError, match="market"):
            RunConfig(market="coinm")

    def test_fingerprint_changes_with_any_result_bearing_field(self):
        base = RunConfig()
        assert base.fingerprint() == RunConfig().fingerprint()
        for field, value in [
            ("horizon", 48),
            ("seed", 1),
            ("n_samples", 256),
            ("temperature", 1.2),
            ("top_p", 0.95),
            ("model_id", "NeoQuasar/Kronos-base"),
            ("lookback", 256),
        ]:
            assert RunConfig(**{field: value}).fingerprint() != base.fingerprint(), field

    def test_fingerprint_ignores_the_data_directory(self, tmp_path):
        assert RunConfig(root=tmp_path).fingerprint() == RunConfig(root="elsewhere").fingerprint()

    def test_seeds_are_derived_per_key_so_resuming_is_safe(self):
        cfg = RunConfig()
        ts = pd.Timestamp("2024-06-01T00:00:00Z")
        other = pd.Timestamp("2024-06-02T00:00:00Z")
        assert cfg.seed_for("BTCUSDT", ts) == cfg.seed_for("BTCUSDT", ts)
        assert cfg.seed_for("BTCUSDT", ts) != cfg.seed_for("ETHUSDT", ts)
        assert cfg.seed_for("BTCUSDT", ts) != cfg.seed_for("BTCUSDT", other)
        assert cfg.seed_for("BTCUSDT", ts) != RunConfig(seed=1).seed_for("BTCUSDT", ts)
