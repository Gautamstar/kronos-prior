"""Bar data: fetch, parse, validate, load.

Two hard rules this module enforces, because every downstream number depends on them:

1. Timestamps are tz-aware UTC and label the bar's OPEN. A bar stamped 14:00 covers
   [14:00, 15:00). Anything computed from it may only be acted on at 15:00.
2. The panel is strictly increasing, gap-free on the interval grid, and identical
   across symbols. Ragged panels are how cross-sectional correlation quietly breaks.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import RunConfig

BASE_URL = "https://data.binance.vision/data"

# Binance monthly kline dumps: 12 columns, sometimes with a header row, and the
# open_time unit changed from milliseconds to microseconds in 2025.
_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Columns Kronos itself consumes, in its expected order.
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]

_INTERVAL_TO_OFFSET = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------


def months(start: str, end: str) -> list[str]:
    """Inclusive YYYY-MM range."""
    rng = pd.period_range(start=start, end=end, freq="M")
    if len(rng) == 0:
        raise ValueError(f"empty month range: {start}..{end}")
    return [str(p) for p in rng]


def dump_url(symbol: str, interval: str, month: str, market: str = "spot") -> str:
    segment = "spot" if market == "spot" else "futures/um"
    return (
        f"{BASE_URL}/{segment}/monthly/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{month}.zip"
    )


def fetch_month(
    symbol: str, interval: str, month: str, dest_dir: Path, market: str = "spot"
) -> Path | None:
    """Download one monthly zip if not already present. Returns None if absent upstream.

    A missing month is normal at the edges of a listing's life and is not an error;
    the panel validator downstream is what decides whether the gap matters.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{symbol}-{interval}-{month}.zip"
    if out.exists() and out.stat().st_size > 0:
        return out

    url = dump_url(symbol, interval, month, market)
    try:
        with urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed, trusted host
            payload = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    tmp = out.with_suffix(".zip.part")
    tmp.write_bytes(payload)
    tmp.replace(out)
    return out


# --------------------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------------------


def _open_time_to_utc(raw: pd.Series) -> pd.DatetimeIndex:
    """Binance switched open_time from ms to us mid-2025. Detect by magnitude.

    A 2020 timestamp in microseconds and a year-50000 timestamp in milliseconds are
    both ~1e15, but the latter is not a plausible market date, so magnitude is a safe
    discriminator here.
    """
    v = pd.to_numeric(raw, errors="raise").astype("int64")
    unit = "us" if v.max() > 1e14 else "ms"
    return pd.to_datetime(v, unit=unit, utc=True)


def parse_kline_zip(path: Path | bytes) -> pd.DataFrame:
    """One monthly zip -> canonical bars for a single symbol."""
    data = Path(path).read_bytes() if isinstance(path, (str, Path)) else path
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = fh.readline()
        has_header = not head.split(b",")[0].strip().strip(b'"').isdigit()
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                header=0 if has_header else None,
                names=_KLINE_COLUMNS,
                usecols=range(len(_KLINE_COLUMNS)),
            )

    out = pd.DataFrame(index=_open_time_to_utc(df["open_time"]))
    out.index.name = "timestamp"
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy()
    # Kronos's sixth channel. quote_volume is the true traded notional; using it beats
    # the volume * mean(OHLC) approximation Kronos falls back to when it is absent.
    out["amount"] = pd.to_numeric(df["quote_volume"], errors="coerce").to_numpy()
    return out[BAR_COLUMNS].sort_index()


def _iter_symbol_frames(cfg: RunConfig) -> Iterator[tuple[str, pd.DataFrame]]:
    for symbol in cfg.symbols:
        frames = []
        for month in months(cfg.start_month, cfg.end_month):
            path = cfg.raw_dir / symbol / f"{symbol}-{cfg.interval}-{month}.zip"
            if path.exists():
                frames.append(parse_kline_zip(path))
        if not frames:
            raise FileNotFoundError(
                f"no raw dumps for {symbol} under {cfg.raw_dir}; run `kronosprior fetch` first"
            )
        yield symbol, pd.concat(frames).sort_index()


# --------------------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------------------


def build_panel(cfg: RunConfig) -> pd.DataFrame:
    """Assemble every symbol into one aligned panel.

    Returns a frame with a UTC DatetimeIndex and a (symbol, field) column MultiIndex.
    """
    parts = {}
    for symbol, frame in _iter_symbol_frames(cfg):
        frame = frame[~frame.index.duplicated(keep="last")]
        parts[symbol] = frame

    panel = pd.concat(parts, axis=1, join="inner")
    panel.columns.names = ["symbol", "field"]
    panel = panel.sort_index()
    validate_panel(panel, cfg.interval)
    return panel


def validate_panel(panel: pd.DataFrame, interval: str) -> None:
    """Fail loudly on the panel defects that corrupt cross-sectional work silently."""
    if panel.empty:
        raise ValueError("panel is empty")
    idx = panel.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")
    if idx.tz is None:
        raise ValueError("panel index must be tz-aware UTC")
    if str(idx.tz) not in ("UTC", "utc"):
        raise ValueError(f"panel index must be UTC, got {idx.tz}")
    if not idx.is_monotonic_increasing:
        raise ValueError("panel index is not sorted")
    if idx.has_duplicates:
        raise ValueError("panel index contains duplicate timestamps")

    offset = _INTERVAL_TO_OFFSET.get(interval)
    if offset is not None and len(idx) > 1:
        expected = pd.date_range(idx[0], idx[-1], freq=offset, tz="UTC")
        missing = expected.difference(idx)
        if len(missing):
            raise ValueError(
                f"panel has {len(missing)} gaps on the {interval} grid "
                f"(first: {missing[0]}, last: {missing[-1]})"
            )

    if panel.isnull().to_numpy().any():
        bad = panel.columns[panel.isnull().any()].tolist()
        raise ValueError(f"panel contains NaNs in: {bad[:6]}")

    closes = panel.xs("close", axis=1, level="field")
    if (closes <= 0).to_numpy().any():
        raise ValueError("panel contains non-positive close prices")


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = panel.copy()
    flat.columns = [f"{s}|{f}" for s, f in panel.columns]
    flat.to_parquet(path, index=True)


def load_panel(path: Path) -> pd.DataFrame:
    flat = pd.read_parquet(Path(path))
    flat.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("|", 1)) for c in flat.columns], names=["symbol", "field"]
    )
    if flat.index.tz is None:
        flat.index = flat.index.tz_localize("UTC")
    return flat.sort_index()


def symbol_frame(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """The Kronos-shaped frame for one symbol: open, high, low, close, volume, amount."""
    return panel[symbol][BAR_COLUMNS]


# --------------------------------------------------------------------------------------
# Synthetic data — used by the test suite so nothing in CI touches the network
# --------------------------------------------------------------------------------------


def synthetic_panel(
    symbols: list[str],
    n_bars: int = 900,
    interval: str = "1h",
    seed: int = 0,
    correlation: float = 0.75,
) -> pd.DataFrame:
    """A correlated GBM panel with the same shape and dtypes as the real thing.

    `correlation` is the pairwise correlation of the driving log-returns, which lets
    tests assert on dependence structure against a value they actually control.
    """
    rng = np.random.default_rng(seed)
    n = len(symbols)
    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    shocks = rng.multivariate_normal(np.zeros(n), corr, size=n_bars) * 0.01

    index = pd.date_range(
        "2024-01-01", periods=n_bars, freq=_INTERVAL_TO_OFFSET[interval], tz="UTC"
    )
    index.name = "timestamp"

    parts = {}
    for i, sym in enumerate(symbols):
        close = 100.0 * np.exp(np.cumsum(shocks[:, i]))
        openp = np.concatenate([[close[0]], close[:-1]])
        spread = np.abs(rng.normal(0, 0.002, n_bars)) * close
        high = np.maximum(openp, close) + spread
        low = np.minimum(openp, close) - spread
        volume = rng.lognormal(6, 0.4, n_bars)
        parts[sym] = pd.DataFrame(
            {
                "open": openp,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": volume * close,
            },
            index=index,
        )

    panel = pd.concat(parts, axis=1)
    panel.columns.names = ["symbol", "field"]
    return panel
