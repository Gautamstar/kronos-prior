# kronos-prior

Carries an AI forecasting model's full predictive distribution into portfolio
optimization.

Built on [Kronos](https://github.com/shiyu-coder/Kronos), a foundation model for
financial candlesticks, and [skfolio](https://skfolio.org), a portfolio optimizer with a
scikit-learn API.

> **Status: early.** The data and sampling layers are in place. The prior itself is next.

## The idea

A forecasting model like Kronos does not produce one guess about the future. It produces
hundreds of possible futures.

Most integrations average them into a single expected return and discard the range. But
the optimizers built for tail risk can consume the full set of scenarios directly.

This project keeps the distribution intact.

## The problem it solves

The model looks at one asset at a time. It never sees two assets side by side, so the
futures it imagines for one are unrelated to the futures it imagines for another.

Line those up and hand them to an optimizer and you have told it the assets move
independently. They do not. They fall together.

An optimizer that believes otherwise concludes a portfolio is safer than it is. Nothing
errors and the backtest looks good.

## The approach

Keep each asset's forecasts as they are. Reorder which forecasts pair together so the
pairings match how the assets have historically moved.

> **The model knows what one asset's future looks like. History knows how assets move
> together.**

Three variants ship:

| Variant | What it does |
| :-- | :-- |
| Conservative | Model's expected returns, risk from history |
| **Coupled** | Full scenario set, dependence restored from history |
| Uncoupled *(ablation)* | Full scenario set, dependence left broken |

The uncoupled version exists so the comparison has something to measure.

## Install

```bash
uv venv && uv pip install -e ".[dev]"      # core
uv pip install -e ".[kronos]"               # + the model
uv pip install -e ".[research]"             # + the optimizer and plotting
```

Kronos is not on PyPI. Clone it and point at it:

```bash
git clone https://github.com/shiyu-coder/Kronos ~/src/Kronos
export KRONOS_REPO=~/src/Kronos
```

## Use

```bash
kronosprior fetch          # download market data
kronosprior build-panel    # parse and validate it
kronosprior verify         # check reproducibility and that no future data leaks in
kronosprior forecast       # generate and cache the forecast distributions
```

Every command takes `--stub`, which runs the pipeline on synthetic data with a
stand-in forecaster. No model weights, no network, no GPU.

```bash
kronosprior verify --stub --symbols AAAUSDT BBBUSDT --synthetic-bars 300
```

Run `kronosprior <command> --help` for the options.

## What this is not

Not a trading strategy. The underlying model is public, so any edge it carries is already
crowded.

The deliverable is the tooling and the evidence. The likely outcome is that none of this
beats an equal-weight portfolio once trading costs are counted. That gets reported either
way.

## Licence

MIT. The sampling module adapts Kronos's inference code (MIT). skfolio is BSD-3-Clause.
