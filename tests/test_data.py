"""Panel construction and the validator that guards cross-sectional work."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from kronosprior.data import (
    BAR_COLUMNS,
    build_panel,
    dump_url,
    load_panel,
    months,
    parse_kline_zip,
    save_panel,
    symbol_frame,
    synthetic_panel,
    validate_panel,
)

SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]


def _kline_zip(n_rows: int = 5, *, header: bool, unit: str) -> bytes:
    base_ms = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    step_ms = 3_600_000
    scale = 1000 if unit == "us" else 1
    rows = []
    for i in range(n_rows):
        ot = (base_ms + i * step_ms) * scale
        ct = ot + step_ms * scale - 1
        rows.append(
            f"{ot},100.{i},101.{i},99.{i},100.{i},10.5,{ct},1050.0,42,5.0,500.0,0"
        )
    body = "\n".join(rows) + "\n"
    if header:
        body = ",".join(
            [
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ]
        ) + "\n" + body
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("AAAUSDT-1h-2024-01.csv", body)
    return buf.getvalue()


class TestParsing:
    @pytest.mark.parametrize("header", [True, False])
    @pytest.mark.parametrize("unit", ["ms", "us"])
    def test_parses_both_header_styles_and_time_units(self, header, unit):
        df = parse_kline_zip(_kline_zip(header=header, unit=unit))
        assert list(df.columns) == BAR_COLUMNS
        assert str(df.index.tz) == "UTC"
        assert df.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
        assert df.index[1] - df.index[0] == pd.Timedelta(hours=1)

    def test_amount_uses_quote_volume_not_the_approximation(self):
        df = parse_kline_zip(_kline_zip(header=False, unit="ms"))
        # quote_volume in the fixture is 1050.0; volume * mean(OHLC) would be ~1050.x
        assert (df["amount"] == 1050.0).all()

    def test_month_range_is_inclusive(self):
        assert months("2024-01", "2024-03") == ["2024-01", "2024-02", "2024-03"]
        with pytest.raises(ValueError):
            months("2024-03", "2024-01")

    def test_dump_url_distinguishes_spot_from_perps(self):
        assert "/spot/monthly" in dump_url("BTCUSDT", "1h", "2024-01", "spot")
        assert "/futures/um/monthly" in dump_url("BTCUSDT", "1h", "2024-01", "um")


class TestValidator:
    def test_accepts_a_clean_panel(self):
        validate_panel(synthetic_panel(SYMBOLS, n_bars=200), "1h")

    def test_rejects_gaps(self):
        panel = synthetic_panel(SYMBOLS, n_bars=200)
        with pytest.raises(ValueError, match="gaps"):
            validate_panel(panel.drop(panel.index[50]), "1h")

    def test_rejects_naive_timestamps(self):
        panel = synthetic_panel(SYMBOLS, n_bars=50)
        panel.index = panel.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            validate_panel(panel, "1h")

    def test_rejects_unsorted(self):
        panel = synthetic_panel(SYMBOLS, n_bars=50)
        with pytest.raises(ValueError, match="not sorted"):
            validate_panel(panel.iloc[::-1], "1h")

    def test_rejects_duplicate_timestamps(self):
        panel = synthetic_panel(SYMBOLS, n_bars=50)
        with pytest.raises(ValueError, match="duplicate"):
            validate_panel(pd.concat([panel, panel.iloc[[10]]]).sort_index(), "1h")

    def test_rejects_nans(self):
        panel = synthetic_panel(SYMBOLS, n_bars=50)
        panel.iloc[10, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_panel(panel, "1h")

    def test_rejects_non_positive_prices(self):
        panel = synthetic_panel(SYMBOLS, n_bars=50)
        panel.loc[panel.index[5], ("AAAUSDT", "close")] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            validate_panel(panel, "1h")


class TestPanelRoundTrip:
    def test_parquet_preserves_shape_index_and_columns(self, tmp_path):
        panel = synthetic_panel(SYMBOLS, n_bars=300)
        path = tmp_path / "bars.parquet"
        save_panel(panel, path)
        back = load_panel(path)
        pd.testing.assert_frame_equal(panel, back, check_freq=False)
        assert str(back.index.tz) == "UTC"
        assert back.columns.names == ["symbol", "field"]

    def test_symbol_frame_is_kronos_shaped(self):
        panel = synthetic_panel(SYMBOLS, n_bars=100)
        frame = symbol_frame(panel, "BBBUSDT")
        assert list(frame.columns) == BAR_COLUMNS
        assert len(frame) == 100

    def test_build_panel_requires_raw_data(self, tmp_path):
        from kronosprior.config import RunConfig

        cfg = RunConfig(symbols=("AAAUSDT",), root=tmp_path)
        with pytest.raises(FileNotFoundError, match="no raw dumps"):
            build_panel(cfg)


class TestSyntheticPanel:
    def test_induces_the_requested_correlation(self):
        panel = synthetic_panel(SYMBOLS, n_bars=20_000, correlation=0.75, seed=3)
        rets = np.log(panel.xs("close", axis=1, level="field")).diff().dropna()
        corr = rets.corr().to_numpy()
        off = corr[~np.eye(len(SYMBOLS), dtype=bool)]
        # The fixture is the ground truth for Phase 2's coupling tests, so it has to
        # actually deliver the correlation it advertises.
        assert np.allclose(off, 0.75, atol=0.03), off

    def test_ohlc_bounds_hold(self):
        panel = synthetic_panel(SYMBOLS, n_bars=500)
        for sym in SYMBOLS:
            f = panel[sym]
            assert (f["high"] >= f[["open", "close"]].max(axis=1)).all()
            assert (f["low"] <= f[["open", "close"]].min(axis=1)).all()
            assert (f["close"] > 0).all()
