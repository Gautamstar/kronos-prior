"""The forecast cache.

Generation is the only expensive step in the project, so it happens once and every
experiment afterwards reads from disk. Three properties make that safe:

* **Addressed by fingerprint.** The cache path is a hash of the full RunConfig. A
  changed horizon or seed writes to a new directory.
* **Append-only per (symbol, date).** Seeds are derived per key, so interrupting a run
  and resuming it produces the same bytes as running it start to finish.
* **Manifested.** Library versions, device and stub-ness are recorded next to the data.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .data import BAR_COLUMNS

MANIFEST_NAME = "manifest.json"


def _shard_path(root: Path, symbol: str, asof: pd.Timestamp) -> Path:
    return Path(root) / symbol / f"{asof.strftime('%Y%m%dT%H%M%SZ')}.npy"


@dataclass
class ForecastCache:
    """Stores sampled paths as (n_samples, horizon, n_fields) float32 shards."""

    root: Path
    cfg: RunConfig

    @classmethod
    def for_config(cls, cfg: RunConfig) -> ForecastCache:
        return cls(root=cfg.forecast_dir, cfg=cfg)

    # -- manifest -----------------------------------------------------------------

    def write_manifest(self, *, forecaster: object, device: str = "cpu") -> None:
        payload = self.cfg.manifest()
        payload["forecaster"] = type(forecaster).__name__
        payload["is_stub"] = bool(getattr(forecaster, "is_stub", False))
        payload["device"] = device
        payload["fields"] = list(BAR_COLUMNS)
        payload["python"] = platform.python_version()
        payload["versions"] = _versions()
        Path(self.root).mkdir(parents=True, exist_ok=True)
        (Path(self.root) / MANIFEST_NAME).write_text(json.dumps(payload, indent=2) + "\n")

    def read_manifest(self) -> dict:
        path = Path(self.root) / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"no manifest at {path}; the cache was never initialised")
        return json.loads(path.read_text())

    @property
    def is_stub(self) -> bool:
        """True if this cache was built with the test stub. Guard results on this."""
        try:
            return bool(self.read_manifest().get("is_stub", False))
        except FileNotFoundError:
            return False

    # -- shards -------------------------------------------------------------------

    def has(self, symbol: str, asof: pd.Timestamp) -> bool:
        return _shard_path(self.root, symbol, asof).exists()

    def put(self, symbol: str, asof: pd.Timestamp, samples: np.ndarray) -> Path:
        expected = (self.cfg.n_samples, self.cfg.horizon, len(BAR_COLUMNS))
        if samples.shape != expected:
            raise ValueError(f"expected samples of shape {expected}, got {samples.shape}")
        if not np.isfinite(samples).all():
            raise ValueError(f"non-finite samples for {symbol} @ {asof}")

        path = _shard_path(self.root, symbol, asof)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write through a file handle: np.save appends ".npy" to a *path* that lacks it,
        # which would silently land the bytes next to the name we then try to rename.
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as fh:
            np.save(fh, samples.astype(np.float32), allow_pickle=False)
        tmp.replace(path)
        return path

    def get(self, symbol: str, asof: pd.Timestamp) -> np.ndarray:
        path = _shard_path(self.root, symbol, asof)
        if not path.exists():
            raise KeyError(f"no forecast cached for {symbol} @ {asof}")
        return np.load(path, allow_pickle=False)

    def dates(self, symbol: str) -> list[pd.Timestamp]:
        d = Path(self.root) / symbol
        if not d.is_dir():
            return []
        return sorted(
            pd.to_datetime(p.stem, format="%Y%m%dT%H%M%SZ", utc=True) for p in d.glob("*.npy")
        )

    def coverage(self) -> pd.DataFrame:
        rows = [{"symbol": s, "n_dates": len(self.dates(s))} for s in self.cfg.symbols]
        return pd.DataFrame(rows).set_index("symbol")

    # -- the working set ------------------------------------------------------------

    def horizon_returns(
        self,
        asof: pd.Timestamp,
        anchor: pd.Series,
        symbols: list[str] | None = None,
    ) -> np.ndarray:
        """Simple returns over the full horizon, one row per sample.

        `anchor` maps symbol -> the realised close at `asof`, taken from the panel.
        The return is measured from that known price to the terminal sampled close, so
        the jump from the last observed bar into the first predicted bar is included.

        This is the (n_samples, n_assets) matrix the Phase 2 prior consumes. Columns are
        ordered exactly as `symbols`.

        NOTE: row k for BTC and row k for ETH are NOT a joint draw. Kronos samples each
        asset independently, so this matrix has zero cross-asset correlation. Coupling
        it is Phase 2's job.
        """
        symbols = list(symbols or self.cfg.symbols)
        missing = [s for s in symbols if s not in anchor.index]
        if missing:
            raise KeyError(f"anchor prices missing for {missing}")
        close = BAR_COLUMNS.index("close")
        cols = []
        for sym in symbols:
            terminal = self.get(sym, asof)[:, -1, close]  # (n_samples,)
            base = float(anchor[sym])
            if base <= 0:
                raise ValueError(f"non-positive anchor price for {sym} @ {asof}")
            cols.append(terminal / base - 1.0)
        return np.column_stack(cols)


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in ("numpy", "pandas", "pyarrow", "torch", "skfolio", "scikit-learn"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out
