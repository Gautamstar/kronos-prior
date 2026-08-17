"""Forecaster interface, the Kronos implementation, and a torch-free stub.

Everything downstream depends only on `SampleForecaster`, so the cache, the CLI and
(in Phase 2) the prior can all be exercised without weights or a GPU.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ._sampling import generate_samples, load_kronos
from .config import RunConfig
from .data import BAR_COLUMNS

# Kronos derives its time features from these calendar fields.
_TIME_FIELDS = ["minute", "hour", "weekday", "day", "month"]


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(index)
    return (
        pd.DataFrame(
            {
                "minute": s.dt.minute,
                "hour": s.dt.hour,
                "weekday": s.dt.weekday,
                "day": s.dt.day,
                "month": s.dt.month,
            }
        )
        .to_numpy()
        .astype(np.float32)
    )


@runtime_checkable
class SampleForecaster(Protocol):
    """Produces a predictive distribution, not a point forecast."""

    def sample(
        self,
        history: pd.DataFrame,
        future_index: pd.DatetimeIndex,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Return shape (n_samples, horizon, len(BAR_COLUMNS)) in price space."""
        ...


class KronosForecaster:
    """Wraps Kronos, returning every sampled path rather than their mean.

    Mirrors `KronosPredictor.predict`'s normalization exactly — per-window mean and
    std computed on the context only, then clipped — so our samples live in the same
    space as upstream's output and the parity test is meaningful.
    """

    def __init__(
        self,
        cfg: RunConfig,
        device: str | None = None,
        repo_path: str | None = None,
        clip: float = 5.0,
    ) -> None:
        self.cfg = cfg
        self.clip = clip  # matches KronosPredictor's default
        Kronos, KronosTokenizer, _, _ = load_kronos(repo_path)
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_id).to(device).eval()
        self.model = Kronos.from_pretrained(cfg.model_id).to(device).eval()

    def sample(
        self,
        history: pd.DataFrame,
        future_index: pd.DatetimeIndex,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        if list(history.columns) != BAR_COLUMNS:
            raise ValueError(
                f"history must have columns {BAR_COLUMNS}, got {list(history.columns)}"
            )
        if history.isnull().to_numpy().any():
            raise ValueError("history contains NaNs")

        x = history.to_numpy(dtype=np.float32)
        x_mean, x_std = x.mean(axis=0), x.std(axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        raw = generate_samples(
            self.tokenizer,
            self.model,
            x_norm[np.newaxis, :],
            _time_features(history.index)[np.newaxis, :],
            _time_features(future_index)[np.newaxis, :],
            max_context=self.cfg.max_context,
            pred_len=len(future_index),
            clip=self.clip,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            sample_count=n_samples,
            seed=seed,
        )
        # (1, S, H, F) -> (S, H, F), back to price space.
        return raw[0] * (x_std + 1e-5) + x_mean


class StubForecaster:
    """A seeded, torch-free stand-in with the right shape and rough dynamics.

    Exists so the cache, CLI, panel logic and (later) the prior are all testable in CI
    with no weights and no network. It is not a baseline and must never appear in a
    results table — `is_stub` marks it, and the cache manifest records it.
    """

    is_stub = True

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma

    def sample(
        self,
        history: pd.DataFrame,
        future_index: pd.DatetimeIndex,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        horizon = len(future_index)
        last = history["close"].to_numpy()[-1]
        steps = rng.normal(0.0, self.sigma, size=(n_samples, horizon))
        closes = last * np.exp(np.cumsum(steps, axis=1))

        opens = np.concatenate([np.full((n_samples, 1), last), closes[:, :-1]], axis=1)
        spread = np.abs(rng.normal(0, self.sigma / 4, size=(n_samples, horizon))) * closes
        volume = np.broadcast_to(
            history["volume"].to_numpy()[-1], (n_samples, horizon)
        ).astype(float)
        out = np.stack(
            [
                opens,
                np.maximum(opens, closes) + spread,
                np.minimum(opens, closes) - spread,
                closes,
                volume,
                volume * closes,
            ],
            axis=-1,
        )
        return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Windowing — the one place lookahead could enter
# --------------------------------------------------------------------------------------


def rebalance_dates(index: pd.DatetimeIndex, cfg: RunConfig) -> pd.DatetimeIndex:
    """Every timestamp at which a forecast can be made and acted on.

    A forecast at position i uses bars [i - lookback + 1 .. i] and predicts
    [i + 1 .. i + horizon]. Both ends must exist, so the first valid position is
    `lookback - 1` and the last is `len(index) - horizon - 1`.
    """
    first = cfg.lookback - 1
    last = len(index) - cfg.horizon - 1
    if last < first:
        raise ValueError(
            f"not enough bars: need at least {cfg.lookback + cfg.horizon}, have {len(index)}"
        )
    return index[first : last + 1 : cfg.horizon]


def window_for(
    panel_index: pd.DatetimeIndex, asof: pd.Timestamp, cfg: RunConfig
) -> tuple[slice, pd.DatetimeIndex]:
    """Context slice and future index for a forecast made at the close of `asof`.

    The returned slice ends AT `asof` inclusive; the future index starts one bar later.
    Nothing in this project may read a bar at or after the first future timestamp.
    """
    pos = panel_index.get_loc(asof)
    if isinstance(pos, slice):  # pragma: no cover - duplicate index guarded upstream
        raise ValueError(f"ambiguous timestamp {asof}")
    start = pos - cfg.lookback + 1
    if start < 0:
        raise ValueError(f"insufficient history before {asof}")
    future = panel_index[pos + 1 : pos + 1 + cfg.horizon]
    if len(future) < cfg.horizon:
        raise ValueError(f"insufficient future bars after {asof}")
    return slice(start, pos + 1), future
