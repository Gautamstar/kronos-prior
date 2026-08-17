"""Assemble a calibration study from a forecast cache and a bar panel.

Turns the two artifacts Phase 0 produces into the (n_cases, n_samples) matrices the
diagnostics consume, and runs every baseline over the identical windows so the
comparison is like for like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import BASELINES, build_baseline, terminal_closes
from .cache import ForecastCache
from .calibration import CalibrationReport, assess, compare
from .config import RunConfig
from .data import symbol_frame
from .forecast import window_for


def realized_terminal(panel: pd.DataFrame, asof: pd.Timestamp, cfg: RunConfig,
                      symbols: list[str]) -> pd.Series:
    """Close at the end of the forecast horizon, which is the value being predicted."""
    _, future = window_for(panel.index, asof, cfg)
    closes = panel.xs("close", axis=1, level="field")
    return closes.loc[future[-1], symbols]


def gather_cached(cache: ForecastCache, panel: pd.DataFrame,
                  symbols: list[str] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pull every cached forecast into per-symbol (samples, realized) pairs.

    Only dates present in the cache are used, so a partially generated cache still
    produces a valid study over the subset it covers.
    """
    cfg = cache.cfg
    symbols = list(symbols or cfg.symbols)
    out = {}
    for symbol in symbols:
        rows, truth = [], []
        for asof in cache.dates(symbol):
            if asof not in panel.index:
                continue
            try:
                _, future = window_for(panel.index, asof, cfg)
            except ValueError:
                continue
            rows.append(terminal_closes(cache.get(symbol, asof)))
            truth.append(float(panel.loc[future[-1], (symbol, "close")]))
        if rows:
            out[symbol] = (np.vstack(rows), np.asarray(truth, dtype=float))
    return out


def gather_baseline(forecaster, panel: pd.DataFrame, cfg: RunConfig, dates,
                    symbols: list[str] | None = None
                    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run a baseline over the same windows, so the comparison is like for like."""
    symbols = list(symbols or cfg.symbols)
    out = {}
    for symbol in symbols:
        rows, truth = [], []
        for asof in dates:
            ctx, future = window_for(panel.index, asof, cfg)
            history = symbol_frame(panel, symbol).iloc[ctx]
            samples = forecaster.sample(
                history, future, cfg.n_samples, cfg.seed_for(symbol, asof)
            )
            rows.append(terminal_closes(samples))
            truth.append(float(panel.loc[future[-1], (symbol, "close")]))
        if rows:
            out[symbol] = (np.vstack(rows), np.asarray(truth, dtype=float))
    return out


def pool(per_symbol: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack every symbol's cases into one dataset.

    Prices differ by orders of magnitude across symbols, so the samples and the truth
    are divided by each symbol's own realized value first. Pooling raw prices would let
    the highest-priced asset dominate CRPS entirely.
    """
    if not per_symbol:
        raise ValueError("nothing to pool")
    samples, truth = [], []
    for x, y in per_symbol.values():
        scale = y[:, None]
        samples.append(x / scale)
        truth.append(y / y)
    return np.vstack(samples), np.concatenate(truth)


def run_study(cache: ForecastCache, panel: pd.DataFrame, symbols: list[str] | None = None,
              baselines: list[str] | None = None) -> tuple[list[CalibrationReport], pd.DataFrame]:
    """Assess the cached forecaster and every baseline over identical windows."""
    cfg = cache.cfg
    symbols = list(symbols or cfg.symbols)

    cached = gather_cached(cache, panel, symbols)
    if not cached:
        raise ValueError(f"no usable forecasts in {cache.root}")

    label = cache.read_manifest().get("forecaster", "cached")
    if cache.is_stub:
        label += " (STUB)"

    reports = [assess(*pool(cached), label=label)]

    dates = sorted({d for s in symbols for d in cache.dates(s) if d in panel.index})
    for name in baselines if baselines is not None else list(BASELINES):
        got = gather_baseline(build_baseline(name), panel, cfg, dates, symbols)
        reports.append(assess(*pool(got), label=name))

    return reports, compare(reports)


def per_symbol_table(per_symbol: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """One row per symbol, for spotting an asset the model handles badly."""
    rows = []
    for symbol, (samples, truth) in sorted(per_symbol.items()):
        scale = truth[:, None]
        rows.append(assess(samples / scale, truth / truth, label=symbol).summary())
    return pd.DataFrame(rows).set_index("label")
