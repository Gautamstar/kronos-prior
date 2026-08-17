"""Run configuration: the universe, the calendar, the seeds, and where things land.

Everything that could silently change a result lives here and gets written into
the cache manifest. If a number in the results moves, the diff on this module
should explain why.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
# Frozen on 2026-08-17 and deliberately never revised. Selection rule, applied once:
# the USDT-quoted pairs with continuous hourly history across the whole study window and
# top-of-book liquidity, excluding stablecoins and wrapped/staked derivatives of a name
# already in the list.
#
# Do NOT "refresh" this list to today's top coins. Picking today's survivors and running
# them backwards is survivorship bias, and it is the single easiest way to manufacture a
# backtest that cannot be reproduced out of sample.
UNIVERSE: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
)

UNIVERSE_FROZEN_ON = "2026-08-17"


@dataclass(frozen=True)
class RunConfig:
    """A complete, hashable description of one forecast run.

    The hash of this object names the cache directory, so two runs that differ in
    any field below cannot overwrite each other's forecasts.
    """

    # --- data ---
    symbols: tuple[str, ...] = UNIVERSE
    interval: str = "1h"
    market: str = "spot"  # "spot" or "um" (USD-M perpetuals)
    start_month: str = "2023-09"  # inclusive, YYYY-MM
    end_month: str = "2026-08"  # inclusive, YYYY-MM

    # --- model ---
    model_id: str = "NeoQuasar/Kronos-small"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    max_context: int = 512

    # --- forecasting ---
    lookback: int = 512  # bars of context fed to the model
    horizon: int = 24  # bars predicted ahead == rebalance period
    n_samples: int = 512  # paths per asset per date
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0

    # --- reproducibility ---
    seed: int = 20260817

    # --- paths ---
    root: Path = field(default=Path("data"), compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.market not in {"spot", "um"}:
            raise ValueError(f"market must be 'spot' or 'um', got {self.market!r}")
        if self.lookback > self.max_context:
            # Not fatal in Kronos (it rolls a buffer), but it means the model silently
            # never sees the oldest part of your window. Better to be explicit.
            raise ValueError(
                f"lookback ({self.lookback}) exceeds max_context ({self.max_context}); "
                "the extra history would be silently discarded"
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols contains duplicates")

    # -- identity -----------------------------------------------------------------

    def fingerprint(self) -> str:
        """Short stable hash over every field that can change a number."""
        payload = {k: v for k, v in asdict(self).items() if k != "root"}
        payload["symbols"] = list(payload["symbols"])
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def manifest(self) -> dict:
        payload = {k: v for k, v in asdict(self).items() if k != "root"}
        payload["symbols"] = list(payload["symbols"])
        payload["universe_frozen_on"] = UNIVERSE_FROZEN_ON
        payload["fingerprint"] = self.fingerprint()
        return payload

    # -- layout -------------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        """Downloaded Binance zips, kept so a re-parse never re-downloads."""
        return Path(self.root) / "raw" / self.market / self.interval

    @property
    def bars_path(self) -> Path:
        """The canonical bar panel, one parquet for the whole universe."""
        return Path(self.root) / "bars" / f"{self.market}_{self.interval}.parquet"

    @property
    def forecast_dir(self) -> Path:
        """Immutable, fingerprint-addressed forecast cache."""
        return Path(self.root) / "forecasts" / self.fingerprint()

    def seed_for(self, symbol: str, timestamp) -> int:
        """A per-(symbol, date) seed derived from the run seed.

        Derived rather than global so that forecasting one asset, or resuming a
        half-finished run, reproduces byte-identical output regardless of the order
        work was done in. A single global torch seed would not survive a resume.
        """
        key = f"{self.seed}|{symbol}|{int(getattr(timestamp, 'value', timestamp))}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
