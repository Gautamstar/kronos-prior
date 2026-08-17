# kronos-prior

A scikit-learn prior that carries a **Kronos** forecast distribution into **skfolio**
portfolio optimization without destroying the cross-asset dependence structure.

> **Status: Phase 0.** Data, sampling and cache are in place. The prior itself lands in
> Phase 2. There are no performance claims here yet, and there may never be — see
> [What this is not](#what-this-is-not).

## The problem

[Kronos](https://github.com/shiyu-coder/Kronos) is a foundation model for candlesticks.
It produces a *distribution* over future price paths. [skfolio](https://skfolio.org) is a
portfolio optimizer whose tail-risk objectives (CVaR, EVaR) consume a **scenario matrix**
directly rather than reducing it to a mean and a covariance.

These two facts should fit together. Two things stop them.

**1. Kronos throws the distribution away.** The last three lines of its
`auto_regressive_inference` are:

```python
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)      # <- the predictive distribution dies here
```

The paths are generated in parallel as a batch dimension and then averaged away, so
`KronosPredictor.predict()` returns one path no matter what `sample_count` you pass.
`kronosprior/_sampling.py` reimplements that tail without the mean.

**2. Kronos has no cross-sectional channel.** Each asset is forecast independently, so
sample *k* for BTC and sample *k* for ETH are not a joint draw. Stack them into a scenario
matrix and you have asserted that crypto assets move independently — which is false, and
false in exactly the direction that makes a portfolio look safer than it is. Correlations
converge in drawdowns; that is when portfolio tail risk actually bites.

`tests/test_cache.py::test_independent_sampling_leaves_scenarios_uncorrelated` pins this
defect as a test, on synthetic data whose true correlation is 0.75.

## The intended fix (Phase 2)

Keep Kronos's marginals exactly — the shape, the tails, the vol clustering it learned
from 45+ exchanges — and impose the dependence structure from realised returns using
**Iman–Conover rank recombination**: draw a reference multivariate normal with the target
correlation, take its per-column ranks, reorder each asset's samples to match. Marginals
are preserved exactly; Spearman correlation lands on target.

The division of labour, which is the whole thesis:

> **Kronos knows what one asset's future looks like. History knows how assets move
> together. Take the shape from Kronos, take the relationships from history.**

Three prior variants ship, and the third exists to fail:

| Variant | μ | Σ | `returns` |
| :-- | :-- | :-- | :-- |
| Conservative | Kronos | Ledoit–Wolf on history | historical |
| **Coupled** | Kronos | of coupled scenarios | coupled scenarios |
| Uncoupled *(ablation)* | Kronos | of raw scenarios | raw scenarios |

The uncoupled run is built deliberately so the ablation has something to measure.
Demonstrating that the naive integration underestimates tail risk is the deliverable.

## Install

```bash
uv venv && uv pip install -e ".[dev]"          # core: numpy, pandas, pyarrow
uv pip install -e ".[kronos]"                   # + torch and the model deps
uv pip install -e ".[research]"                 # + skfolio, scipy, matplotlib
```

Kronos ships as a repository, not a wheel:

```bash
git clone https://github.com/shiyu-coder/Kronos ~/src/Kronos
export KRONOS_REPO=~/src/Kronos
```

## Use

```bash
kronosprior fetch                     # Binance monthly dumps -> data/raw
kronosprior build-panel               # parse, validate, write data/bars/*.parquet
kronosprior verify                    # the Phase 0 gate
kronosprior forecast                  # generate + cache sampled paths
```

Every command takes `--stub` to run the whole pipeline on synthetic data with a
torch-free forecaster — no weights, no network, no GPU:

```bash
kronosprior verify --stub --symbols AAAUSDT BBBUSDT --synthetic-bars 300
```

### The Phase 0 gate

`kronosprior verify` passes only if all three hold:

- the context window ends at `asof` and never overlaps the forecast window
- the same seed produces byte-identical samples across two calls
- the samples do not collapse — there is an actual distribution to carry

## Design notes

**The cache is the artifact.** Generation is the only expensive step, so it happens once
and every experiment reads from disk. The cache path is a hash of the full `RunConfig`,
so changing the horizon or the seed writes somewhere new rather than silently mixing with
an old run. A `manifest.json` records library versions, device, and whether the stub was
used — `cache.is_stub` guards results.

**Seeds are derived per `(symbol, timestamp)`**, not drawn from one global stream, so
interrupting a run and resuming it reproduces the same bytes as running it start to
finish.

**Determinism is per-device.** The same seed on the same device and torch build gives
identical samples. It is not guaranteed across CPU/GPU or torch versions, which is why
the manifest records both.

**The universe is frozen** in `config.py` and deliberately never revised. Re-picking
today's top coins and running them backwards is survivorship bias.

**Time convention.** Timestamps are tz-aware UTC and label the bar's *open*. A forecast
made at the close of bar *t* may use no data after *t* and may not be acted on before
*t*+1. `tests/test_windows.py` is the enforcement.

## What this is not

This is not a trading strategy and there is no expectation that it makes money. Kronos is
a public model, so any edge it carries is already crowded, and retail systematic trading
is negative expected value after costs for nearly everyone who attempts it.

The deliverable is the prior and the evidence. The most likely honest outcome is that
nothing here beats equal-weight after costs, and that estimation error in μ dominates —
which is why HRP and risk budgeting, which ignore μ entirely, are in the comparison set.
If they win, that is the finding.

## Licence

MIT. `_sampling.py` adapts Kronos's inference loop (MIT). skfolio is BSD-3-Clause.
