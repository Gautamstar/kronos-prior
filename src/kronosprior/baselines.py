"""Reference forecasters for the calibration study.

A CRPS number on its own means nothing. These give it something to be measured against,
and they are ordered by how hard they are to beat:

* **Persistence** is the floor. Tomorrow equals today, with no uncertainty at all. Any
  probabilistic forecaster that loses to this is broken.
* **EWMA Gaussian** is the real bar. A random walk with exponentially weighted volatility
  captures the two things that actually matter in price data, which are the absence of
  drift and the clustering of volatility. Beating it requires the model to know something
  about shape.
* **Climatology** resamples the historical return distribution, so it gets fat tails for
  free while knowing nothing about the current regime.

All three implement `SampleForecaster`, so they flow through the same cache, the same
diagnostics and (later) the same prior as Kronos does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BAR_COLUMNS


def _log_returns(history: pd.DataFrame) -> np.ndarray:
    close = history["close"].to_numpy(dtype=float)
    return np.diff(np.log(close))


def _to_bars(closes: np.ndarray, anchor: float, volume: float) -> np.ndarray:
    """Turn simulated close paths into the six-column bar format.

    Open is the previous close, and the high and low bracket them. The optimizer only
    ever reads `close`, so the other columns exist to satisfy the shape contract.
    """
    n_samples, horizon = closes.shape
    opens = np.concatenate([np.full((n_samples, 1), anchor), closes[:, :-1]], axis=1)
    high = np.maximum(opens, closes)
    low = np.minimum(opens, closes)
    vol = np.full((n_samples, horizon), volume, dtype=float)
    return np.stack([opens, high, low, closes, vol, vol * closes], axis=-1).astype(np.float32)


class PersistenceForecaster:
    """Every path is a flat line at the last observed close.

    Zero spread by construction, so its CRPS reduces to mean absolute error and its
    coverage is zero at every level. That is the point. It marks the floor.
    """

    label = "persistence"

    def sample(self, history, future_index, n_samples: int, seed: int) -> np.ndarray:
        last = float(history["close"].to_numpy()[-1])
        closes = np.full((n_samples, len(future_index)), last, dtype=float)
        return _to_bars(closes, last, float(history["volume"].to_numpy()[-1]))


class EwmaGaussianForecaster:
    """Random walk with exponentially weighted volatility.

    `lam` is the RiskMetrics decay. 0.94 is the daily convention and works acceptably on
    hourly bars. Drift is fixed at zero, which is the honest choice over any horizon
    short enough that estimated drift is pure noise.
    """

    label = "ewma-gaussian"

    def __init__(self, lam: float = 0.94, min_obs: int = 32) -> None:
        if not 0 < lam < 1:
            raise ValueError(f"lam must be in (0, 1), got {lam}")
        self.lam = lam
        self.min_obs = min_obs

    def volatility(self, history: pd.DataFrame) -> float:
        rets = _log_returns(history)
        if len(rets) < self.min_obs:
            raise ValueError(f"need at least {self.min_obs} returns, got {len(rets)}")
        # Weight recent observations more heavily, normalised so the weights sum to 1.
        age = np.arange(len(rets) - 1, -1, -1)
        weights = (1 - self.lam) * self.lam**age
        weights /= weights.sum()
        return float(np.sqrt((weights * rets**2).sum()))

    def sample(self, history, future_index, n_samples: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sigma = self.volatility(history)
        last = float(history["close"].to_numpy()[-1])
        steps = rng.normal(0.0, sigma, size=(n_samples, len(future_index)))
        closes = last * np.exp(np.cumsum(steps, axis=1))
        return _to_bars(closes, last, float(history["volume"].to_numpy()[-1]))


class ClimatologyForecaster:
    """Bootstrap from the history's own returns, in blocks.

    Block resampling keeps short-run autocorrelation and volatility clustering that an
    independent draw would destroy. It knows nothing about the current regime, so it is
    the right control for whether a model is reacting to recent conditions at all.
    """

    label = "climatology"

    def __init__(self, block: int = 6) -> None:
        if block < 1:
            raise ValueError("block must be >= 1")
        self.block = block

    def sample(self, history, future_index, n_samples: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rets = _log_returns(history)
        horizon = len(future_index)
        if len(rets) < self.block:
            raise ValueError(f"need at least {self.block} returns, got {len(rets)}")

        n_blocks = int(np.ceil(horizon / self.block))
        starts = rng.integers(0, len(rets) - self.block + 1, size=(n_samples, n_blocks))
        offsets = np.arange(self.block)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_samples, -1)
        steps = rets[idx][:, :horizon]

        last = float(history["close"].to_numpy()[-1])
        closes = last * np.exp(np.cumsum(steps, axis=1))
        return _to_bars(closes, last, float(history["volume"].to_numpy()[-1]))


BASELINES = {
    "persistence": PersistenceForecaster,
    "ewma-gaussian": EwmaGaussianForecaster,
    "climatology": ClimatologyForecaster,
}


def build_baseline(name: str):
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    return BASELINES[name]()


def terminal_closes(samples: np.ndarray) -> np.ndarray:
    """Close price at the end of the horizon, one per sampled path."""
    return np.asarray(samples)[..., -1, BAR_COLUMNS.index("close")]
