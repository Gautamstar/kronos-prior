"""Calibration diagnostics for ensemble forecasts.

The question this answers: when the model says a 90% interval, does the truth land
inside it 90% of the time?

If the answer is no, every downstream number is built on sand. A prior that carries an
overconfident distribution into a tail-risk optimizer produces confidently wrong weights.
So this runs before the prior is built, and it is a deliverable on its own.

All functions take `samples` shaped (n_cases, n_samples) and `realized` shaped
(n_cases,). Everything is pure numpy.

The four diagnostics, and what each catches:

* **PIT / rank histogram** shows the shape of the miscalibration. Flat is calibrated.
  U-shaped means the ensemble is too narrow. Dome-shaped means too wide. A slope means
  a bias.
* **Interval coverage** puts a number on the same thing at specific levels.
* **Spread-skill ratio** is one number for whether the spread matches the error.
* **CRPS** is a proper scoring rule, so it rewards calibration and sharpness together.
  This is what baselines are compared on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_LEVELS = (0.5, 0.8, 0.95)


def _as_2d(samples, realized) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    realized = np.atleast_1d(np.asarray(realized, dtype=float))
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (n_cases, n_samples), got {samples.shape}")
    if samples.shape[0] != realized.shape[0]:
        raise ValueError(
            f"case count mismatch: samples has {samples.shape[0]}, realized has {realized.shape[0]}"
        )
    if not np.isfinite(samples).all() or not np.isfinite(realized).all():
        raise ValueError("samples and realized must be finite")
    return samples, realized


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def pit_values(samples, realized) -> np.ndarray:
    """Probability integral transform: where the truth fell in the predictive CDF.

    Returns one value per case in [0, 1]. Under a calibrated forecast these are uniform.
    Ties contribute half their mass, which matters for a degenerate ensemble where every
    sample is identical.
    """
    samples, realized = _as_2d(samples, realized)
    y = realized[:, None]
    below = (samples < y).sum(axis=1)
    equal = (samples == y).sum(axis=1)
    return (below + 0.5 * equal) / samples.shape[1]


def ranks(samples, realized, rng: np.random.Generator | None = None) -> np.ndarray:
    """Rank of the truth among the ensemble members, in 1..n_samples+1.

    Ties are broken at random, which is what keeps the histogram flat under a correct
    forecast of a discrete quantity. Pass a seeded `rng` for reproducibility.
    """
    samples, realized = _as_2d(samples, realized)
    rng = rng or np.random.default_rng(0)
    y = realized[:, None]
    below = (samples < y).sum(axis=1)
    equal = (samples == y).sum(axis=1)
    return below + rng.integers(0, equal + 1) + 1


def rank_histogram(samples, realized, n_bins: int = 20, rng=None) -> np.ndarray:
    """Rank counts collapsed into `n_bins` equal-width bins. Flat means calibrated.

    `n_bins` may not exceed n_samples + 1. More bins than possible ranks leaves some
    bins empty by construction, which reads as miscalibration when it is an artefact.
    """
    samples, _ = _as_2d(samples, realized)
    n = samples.shape[1]
    if n_bins > n + 1:
        raise ValueError(f"n_bins ({n_bins}) exceeds the {n + 1} possible ranks")
    r = ranks(samples, realized, rng)
    # ranks live in 1..n+1, so map onto [0, 1) before binning
    u = (r - 1) / n
    counts, _ = np.histogram(np.clip(u, 0, 1 - 1e-12), bins=n_bins, range=(0.0, 1.0))
    return counts


def flatness(counts: np.ndarray) -> tuple[float, float]:
    """Chi-square statistic and reliability index for a rank histogram.

    Chi-square has `len(counts) - 1` degrees of freedom under calibration. The
    reliability index is the total absolute deviation from uniform, in [0, 2), and is
    easier to compare across different case counts.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        raise ValueError("empty histogram")
    expected = total / len(counts)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    ri = float(np.abs(counts / total - 1.0 / len(counts)).sum())
    return chi2, ri


def coverage(samples, realized, levels=DEFAULT_LEVELS) -> pd.DataFrame:
    """Empirical coverage of central prediction intervals against nominal."""
    samples, realized = _as_2d(samples, realized)
    rows = []
    for level in levels:
        lo = np.quantile(samples, (1 - level) / 2, axis=1)
        hi = np.quantile(samples, (1 + level) / 2, axis=1)
        hit = float(((realized >= lo) & (realized <= hi)).mean())
        rows.append(
            {
                "nominal": level,
                "empirical": hit,
                "error": hit - level,
                "mean_width": float((hi - lo).mean()),
            }
        )
    return pd.DataFrame(rows)


def crps(samples, realized, fair: bool = True) -> np.ndarray:
    """Continuous ranked probability score, per case. Lower is better.

    Computed from samples via CRPS = E|X - y| - 0.5 E|X - X'|. The `fair` estimator
    divides the second term by n(n-1), which removes the finite-ensemble bias that
    otherwise rewards small ensembles.

    The pairwise term is evaluated by sorting, so this is O(n log n) per case.
    """
    samples, realized = _as_2d(samples, realized)
    n = samples.shape[1]
    if fair and n < 2:
        raise ValueError("fair CRPS needs at least 2 samples")

    absolute = np.abs(samples - realized[:, None]).mean(axis=1)
    ordered = np.sort(samples, axis=1)
    weights = 2 * np.arange(n) - n + 1
    pairwise = (ordered * weights).sum(axis=1) / (n * (n - 1) if fair else n * n)
    return absolute - pairwise


def spread_skill_ratio(samples, realized) -> float:
    """Ensemble spread divided by the error of the ensemble mean.

    Corrected for ensemble size, so 1.0 is calibrated regardless of n_samples. Below 1
    means the ensemble is too narrow for the errors it actually makes.
    """
    samples, realized = _as_2d(samples, realized)
    n = samples.shape[1]
    rmse = float(np.sqrt(((samples.mean(axis=1) - realized) ** 2).mean()))
    if rmse == 0:
        return float("inf")
    spread = float(np.sqrt(samples.var(axis=1, ddof=1).mean()))
    return spread * np.sqrt((n + 1) / n) / rmse


def bias(samples, realized) -> float:
    """Mean signed error of the ensemble mean. Detects a shifted forecast."""
    samples, realized = _as_2d(samples, realized)
    return float((samples.mean(axis=1) - realized).mean())


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Everything the calibration study needs about one forecaster on one dataset."""

    label: str
    n_cases: int
    n_samples: int
    crps: float
    spread_skill: float
    bias: float
    chi2: float
    reliability_index: float
    coverage: pd.DataFrame
    pit: np.ndarray = field(repr=False)
    rank_counts: np.ndarray = field(repr=False)

    @property
    def dispersion(self) -> str:
        """Plain reading of the spread-skill ratio."""
        if self.spread_skill < 0.9:
            return "under-dispersed (overconfident)"
        if self.spread_skill > 1.1:
            return "over-dispersed (underconfident)"
        return "calibrated"

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "label": self.label,
                "n_cases": self.n_cases,
                "n_samples": self.n_samples,
                "crps": self.crps,
                "spread_skill": self.spread_skill,
                "dispersion": self.dispersion,
                "bias": self.bias,
                "reliability_index": self.reliability_index,
            }
        )

    def __str__(self) -> str:
        lines = [
            f"{self.label}",
            f"  cases {self.n_cases}   samples {self.n_samples}",
            f"  CRPS  {self.crps:.6f}",
            f"  spread/skill {self.spread_skill:.3f}  ({self.dispersion})",
            f"  bias  {self.bias:+.6f}",
            f"  reliability index {self.reliability_index:.4f}   chi2 {self.chi2:.1f}",
            "  coverage:",
        ]
        for _, row in self.coverage.iterrows():
            lines.append(
                f"    {row['nominal']:.0%} nominal -> {row['empirical']:.1%} "
                f"({row['error']:+.1%})"
            )
        return "\n".join(lines)


def assess(samples, realized, label: str = "forecast", n_bins: int = 20,
           levels=DEFAULT_LEVELS, rng=None) -> CalibrationReport:
    """Run every diagnostic and package the result."""
    samples, realized = _as_2d(samples, realized)
    counts = rank_histogram(samples, realized, n_bins=n_bins, rng=rng)
    chi2, ri = flatness(counts)
    return CalibrationReport(
        label=label,
        n_cases=samples.shape[0],
        n_samples=samples.shape[1],
        crps=float(crps(samples, realized).mean()),
        spread_skill=spread_skill_ratio(samples, realized),
        bias=bias(samples, realized),
        chi2=chi2,
        reliability_index=ri,
        coverage=coverage(samples, realized, levels),
        pit=pit_values(samples, realized),
        rank_counts=counts,
    )


def compare(reports: list[CalibrationReport]) -> pd.DataFrame:
    """Side-by-side table, sorted by CRPS."""
    return (
        pd.DataFrame([r.summary() for r in reports])
        .sort_values("crps")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# Correction
# --------------------------------------------------------------------------------------


def rescale(samples, factor: float) -> np.ndarray:
    """Widen or narrow each ensemble around its own mean, leaving the mean fixed."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    centre = samples.mean(axis=1, keepdims=True)
    return centre + factor * (samples - centre)


def fit_rescale(samples, realized, grid=None) -> float:
    """Find the variance rescaling factor that minimises CRPS.

    A miscalibrated ensemble is often fixable with one scalar. If the fitted factor is
    far from 1, the model's uncertainty estimate is wrong by that much, and the number
    itself is a result worth reporting.
    """
    samples, realized = _as_2d(samples, realized)
    grid = np.asarray(grid if grid is not None else np.linspace(0.25, 4.0, 61), dtype=float)
    scores = [float(crps(rescale(samples, f), realized).mean()) for f in grid]
    coarse = grid[int(np.argmin(scores))]

    # Refine around the winner so the answer is not pinned to the grid spacing.
    fine = np.linspace(max(coarse - 0.1, 1e-3), coarse + 0.1, 41)
    fine_scores = [float(crps(rescale(samples, f), realized).mean()) for f in fine]
    return float(fine[int(np.argmin(fine_scores))])
