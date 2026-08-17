"""kronos-prior: carry a Kronos forecast distribution into skfolio, dependence intact.

Phase 0 (this release) covers data, the un-averaged sampling path, and the forecast
cache. The prior itself lands in Phase 2.
"""

from .cache import ForecastCache
from .config import UNIVERSE, RunConfig
from .forecast import SampleForecaster, StubForecaster, rebalance_dates, window_for

__all__ = [
    "UNIVERSE",
    "ForecastCache",
    "RunConfig",
    "SampleForecaster",
    "StubForecaster",
    "rebalance_dates",
    "window_for",
]

__version__ = "0.0.1"
