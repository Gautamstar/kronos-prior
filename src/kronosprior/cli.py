"""Command line entry points for the Phase 0 pipeline.

    kronosprior fetch            download raw Binance dumps
    kronosprior build-panel      parse + validate + write the canonical panel
    kronosprior forecast         generate and cache sampled paths
    kronosprior verify           the Phase 0 gate: prove a run is reproducible
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import data as data_mod
from .cache import ForecastCache
from .config import RunConfig
from .forecast import StubForecaster, rebalance_dates, window_for


def _cfg(args: argparse.Namespace) -> RunConfig:
    kwargs: dict = {"root": Path(args.root)}
    for name in ("interval", "market", "start_month", "end_month", "horizon", "lookback", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    if getattr(args, "n_samples", None) is not None:
        kwargs["n_samples"] = args.n_samples
    if getattr(args, "symbols", None):
        kwargs["symbols"] = tuple(args.symbols)
    return RunConfig(**kwargs)


def _make_forecaster(args: argparse.Namespace, cfg: RunConfig):
    if args.stub:
        return StubForecaster(), "cpu"
    from .forecast import KronosForecaster

    fc = KronosForecaster(cfg, device=args.device, repo_path=args.kronos_repo)
    return fc, fc.device


# --------------------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    total = missing = 0
    for symbol in cfg.symbols:
        got = 0
        for month in data_mod.months(cfg.start_month, cfg.end_month):
            path = data_mod.fetch_month(
                symbol, cfg.interval, month, cfg.raw_dir / symbol, cfg.market
            )
            total += 1
            if path is None:
                missing += 1
            else:
                got += 1
        print(f"  {symbol:<10} {got} months")
    print(f"\n{total - missing}/{total} monthly files present under {cfg.raw_dir}")
    if missing:
        print(f"{missing} months returned 404 (normal near a listing's start).")
    return 0


def cmd_build_panel(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    panel = data_mod.build_panel(cfg)
    data_mod.save_panel(panel, cfg.bars_path)
    dates = rebalance_dates(panel.index, cfg)
    print(f"panel   {panel.shape[0]} bars x {len(cfg.symbols)} symbols -> {cfg.bars_path}")
    print(f"span    {panel.index[0]} .. {panel.index[-1]}")
    print(f"rebal   {len(dates)} dates at horizon={cfg.horizon}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if args.stub:
        panel = data_mod.synthetic_panel(list(cfg.symbols), n_bars=args.synthetic_bars)
    else:
        panel = data_mod.load_panel(cfg.bars_path)

    cache = ForecastCache.for_config(cfg)
    forecaster, device = _make_forecaster(args, cfg)
    cache.write_manifest(forecaster=forecaster, device=device)

    dates = rebalance_dates(panel.index, cfg)
    if args.limit:
        dates = dates[: args.limit]

    print(f"cache   {cache.root}")
    print(f"work    {len(dates)} dates x {len(cfg.symbols)} symbols")
    done = skipped = 0
    for asof in dates:
        ctx, future = window_for(panel.index, asof, cfg)
        for symbol in cfg.symbols:
            if cache.has(symbol, asof) and not args.overwrite:
                skipped += 1
                continue
            history = data_mod.symbol_frame(panel, symbol).iloc[ctx]
            samples = forecaster.sample(
                history, future, cfg.n_samples, cfg.seed_for(symbol, asof)
            )
            cache.put(symbol, asof, samples)
            done += 1
        print(f"  {asof}  written={done} skipped={skipped}", end="\r", flush=True)
    print(f"\nwrote {done} shards, skipped {skipped} already present")
    if getattr(forecaster, "is_stub", False):
        print("NOTE: stub forecaster — this cache is for plumbing tests only.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Phase 0 gate: the same command twice must produce byte-identical forecasts."""
    cfg = _cfg(args)
    panel = (
        data_mod.synthetic_panel(list(cfg.symbols), n_bars=args.synthetic_bars)
        if args.stub
        else data_mod.load_panel(cfg.bars_path)
    )
    forecaster, _ = _make_forecaster(args, cfg)

    asof = rebalance_dates(panel.index, cfg)[0]
    symbol = cfg.symbols[0]
    ctx, future = window_for(panel.index, asof, cfg)
    history = data_mod.symbol_frame(panel, symbol).iloc[ctx]

    print(f"symbol  {symbol} @ {asof}")
    print(f"context {history.index[0]} .. {history.index[-1]}  ({len(history)} bars)")
    print(f"future  {future[0]} .. {future[-1]}  ({len(future)} bars)")

    if history.index[-1] >= future[0]:
        print("FAIL: context overlaps the forecast window")
        return 1

    seed = cfg.seed_for(symbol, asof)
    a = forecaster.sample(history, future, cfg.n_samples, seed)
    b = forecaster.sample(history, future, cfg.n_samples, seed)

    identical = np.array_equal(a, b)
    print(f"shape   {a.shape}")
    print(f"spread  terminal close p05..p95 = "
          f"{np.percentile(a[:, -1, 3], 5):.4f} .. {np.percentile(a[:, -1, 3], 95):.4f}")
    print(f"\ndeterministic: {'PASS' if identical else 'FAIL'}")
    if not identical:
        print(f"  max abs diff {np.abs(a - b).max():.3e}")
        return 1

    if np.allclose(a.std(axis=0), 0):
        print("collapsed: FAIL — every path identical, no distribution to carry")
        return 1
    print("distribution: PASS — paths differ across samples")
    return 0


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kronosprior", description=__doc__)
    p.add_argument("--root", default="data", help="data directory (default: data)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, *, model=False):
        sp.add_argument("--interval", default=None)
        sp.add_argument("--market", default=None, choices=["spot", "um"])
        sp.add_argument("--start-month", dest="start_month", default=None)
        sp.add_argument("--end-month", dest="end_month", default=None)
        sp.add_argument("--symbols", nargs="+", default=None)
        if model:
            sp.add_argument("--horizon", type=int, default=None)
            sp.add_argument("--lookback", type=int, default=None)
            sp.add_argument("--n-samples", dest="n_samples", type=int, default=None)
            sp.add_argument("--seed", type=int, default=None)
            sp.add_argument("--stub", action="store_true", help="use the torch-free stub")
            sp.add_argument("--device", default=None)
            sp.add_argument("--kronos-repo", dest="kronos_repo", default=None)
            sp.add_argument("--synthetic-bars", dest="synthetic_bars", type=int, default=900)

    sp = sub.add_parser("fetch", help="download raw Binance monthly dumps")
    common(sp)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("build-panel", help="parse, validate and write the canonical panel")
    common(sp, model=True)
    sp.set_defaults(func=cmd_build_panel)

    sp = sub.add_parser("forecast", help="generate and cache sampled paths")
    common(sp, model=True)
    sp.add_argument("--limit", type=int, default=None, help="only the first N rebalance dates")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("verify", help="Phase 0 gate: reproducibility and no lookahead")
    common(sp, model=True)
    sp.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
