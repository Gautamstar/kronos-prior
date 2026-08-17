# kronos-prior

Carries an AI forecasting model's full predictive distribution into portfolio
optimization, instead of collapsing it to a single number first.

Built on [Kronos](https://github.com/shiyu-coder/Kronos), a foundation model for
financial candlesticks, and [skfolio](https://skfolio.org), a portfolio optimizer with a
scikit-learn API.

> **Status: early.** The data and sampling layers are in place. The prior itself is next.
> There are no performance claims here, and there may never be. See
> [What this is not](#what-this-is-not).

## The idea

A forecasting model like Kronos does not produce one guess about the future. It produces
hundreds of possible futures: some flat, some a rally, some a crash. That spread is the
most useful thing it knows.

Most integrations throw it away. They average those futures into a single expected
return, hand that to an optimizer, and discard everything about the range of outcomes.
But the optimizers built to protect against bad outcomes, the ones targeting tail risk
rather than average risk, can consume the full set of scenarios directly. Averaging first
throws away the only thing that distinguishes a generative model from a linear
regression.

This project keeps the distribution intact all the way through.

## The problem it solves

Connecting these two tools the obvious way introduces a subtle error that is easy to miss
and expensive to get wrong.

The forecasting model looks at **one asset at a time**. It never sees two assets side by
side, so the futures it imagines for one are unrelated to the futures it imagines for
another. Line them up and feed them to an optimizer and you have quietly told it that
these assets move independently.

They do not. They fall together. An optimizer that believes otherwise concludes a
portfolio is far safer than it really is, and it is wrong in exactly the moment that
matters most, which is the crash. Nothing errors, nothing looks unusual, and the backtest
comes out looking good.

## The approach

Keep the model's per-asset forecasts exactly as they are, then reorder which forecasts
get paired together so the pairings reflect how the assets have historically moved.
Nothing about any individual forecast changes, only which ones line up.

The division of labour is the whole point:

> **The model knows what one asset's future looks like, including the rare disasters.
> History knows how assets move together. Take the shape from the model, take the
> relationships from history.**

Each source does only what it is good at. The model has seen millions of charts but was
never shown assets side by side. History has the relationships but contains only the
handful of crashes that actually happened.

Three variants ship, and the third exists in order to fail:

| Variant | What it does |
| :-- | :-- |
| Conservative | Model's expected returns, risk estimated from history |
| **Coupled** | Full scenario set, dependence restored from history |
| Uncoupled *(ablation)* | Full scenario set, dependence left broken |

The uncoupled version is built deliberately so the comparison has something to measure.
Demonstrating that the naive integration underestimates risk is the deliverable.

## Install

```bash
uv venv && uv pip install -e ".[dev]"      # core
uv pip install -e ".[kronos]"               # + the model
uv pip install -e ".[research]"             # + the optimizer and plotting
```

## Use

```bash
kronosprior fetch          # download market data
kronosprior build-panel    # parse and validate it
kronosprior verify         # check reproducibility and that no future data leaks in
kronosprior forecast       # generate and cache the forecast distributions
```

Every command takes `--stub`, which runs the whole pipeline on synthetic data with a
lightweight stand-in forecaster: no model weights, no network, no GPU.

```bash
kronosprior verify --stub --symbols AAAUSDT BBBUSDT --synthetic-bars 300
```

Run `kronosprior <command> --help` for the options.

## What this is not

This is not a trading strategy, and there is no expectation that it makes money. The
underlying model is public, so any edge it carries is already crowded, and retail
systematic trading is negative expected value after costs for nearly everyone who
attempts it.

The deliverable is the tooling and the evidence. The most likely honest outcome is that
none of this beats a naive equal-weight portfolio once trading costs are counted, which
is why methods that ignore expected returns entirely are in the comparison set. If they
win, that is the finding, and it gets reported.

## Licence

MIT. The sampling module adapts Kronos's inference code (MIT). skfolio is BSD-3-Clause.
