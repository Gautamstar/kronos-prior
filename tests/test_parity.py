"""Parity between our un-averaged sampling path and Kronos's own `predict()`.

These tests need real weights and torch, so they are skipped unless you opt in:

    pytest -m needs_weights --run-weights

The check that matters: if we average our samples, we must recover Kronos's own
output. `_sampling.generate_samples` is a copy of upstream's inference loop with the
final `np.mean` removed, so any divergence means the copy has drifted — either we
introduced a bug, or upstream changed and we need to re-sync.

Exact equality is not expected. Sampling is stochastic, and averaging N draws is a
Monte Carlo estimate of the same quantity upstream computes from its own N draws, so
the two agree in distribution rather than bitwise. The assertions below are therefore
on the mean path within a tolerance that scales with the sampled spread. Run with a
large `sample_count` — a tight tolerance on 8 samples will fail for honest reasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kronosprior.config import RunConfig
from kronosprior.data import BAR_COLUMNS, symbol_frame, synthetic_panel
from kronosprior.forecast import _time_features, rebalance_dates, window_for

pytestmark = pytest.mark.needs_weights

SAMPLE_COUNT = 256
CLOSE = BAR_COLUMNS.index("close")


@pytest.fixture(scope="module")
def kronos(request):
    if not request.config.getoption("--run-weights"):
        pytest.skip("needs Kronos weights; pass --run-weights to run")
    torch = pytest.importorskip("torch")
    from kronosprior._sampling import KronosNotAvailable, load_kronos

    try:
        Kronos, KronosTokenizer, KronosPredictor, _ = load_kronos()
    except KronosNotAvailable as exc:
        pytest.skip(str(exc))

    cfg = RunConfig(lookback=256, horizon=12, n_samples=SAMPLE_COUNT, max_context=512)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_id).to(device).eval()
    model = Kronos.from_pretrained(cfg.model_id).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    return cfg, tokenizer, model, predictor, device


@pytest.fixture(scope="module")
def window(kronos):
    cfg, *_ = kronos
    panel = synthetic_panel(["AAAUSDT"], n_bars=cfg.lookback + cfg.horizon + 40, seed=11)
    asof = rebalance_dates(panel.index, cfg)[0]
    ctx, future = window_for(panel.index, asof, cfg)
    return symbol_frame(panel, "AAAUSDT").iloc[ctx], future


def _our_samples(kronos, history, future, seed=0):
    from kronosprior._sampling import generate_samples

    cfg, tokenizer, model, _, _ = kronos
    x = history.to_numpy(dtype=np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    x_norm = np.clip((x - mean) / (std + 1e-5), -5, 5)

    raw = generate_samples(
        tokenizer,
        model,
        x_norm[np.newaxis, :],
        _time_features(history.index)[np.newaxis, :],
        _time_features(future)[np.newaxis, :],
        max_context=cfg.max_context,
        pred_len=len(future),
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        sample_count=SAMPLE_COUNT,
        seed=seed,
    )
    return raw[0] * (std + 1e-5) + mean


def test_upstream_predict_collapses_the_distribution(kronos, window):
    """The premise of this whole project, asserted against the real library.

    Two `predict()` calls with sample_count > 1 return a single path each, and that
    path is a mean over draws. If upstream ever exposes the samples, this test starts
    failing and `_sampling.py` can be deleted.
    """
    _, _, _, predictor, _ = kronos
    history, future = window
    out = predictor.predict(
        df=history.reset_index(drop=True),
        x_timestamp=pd.Series(history.index),
        y_timestamp=pd.Series(future),
        pred_len=len(future),
        sample_count=SAMPLE_COUNT,
        verbose=False,
    )
    assert out.shape == (len(future), len(BAR_COLUMNS)), (
        "predict() returned more than one path — upstream may now expose samples"
    )


def test_our_mean_matches_upstream_predict(kronos, window):
    """Averaging our samples reproduces upstream's answer, within Monte Carlo error."""
    _, _, _, predictor, _ = kronos
    history, future = window

    ours = _our_samples(kronos, history, future, seed=0)
    theirs = predictor.predict(
        df=history.reset_index(drop=True),
        x_timestamp=pd.Series(history.index),
        y_timestamp=pd.Series(future),
        pred_len=len(future),
        sample_count=SAMPLE_COUNT,
        verbose=False,
    ).to_numpy()

    our_mean = ours.mean(axis=0)
    # Standard error of the mean at each step, from our own draws.
    sem = ours.std(axis=0) / np.sqrt(SAMPLE_COUNT)
    diff = np.abs(our_mean[:, CLOSE] - theirs[:, CLOSE])
    tol = 4.0 * sem[:, CLOSE] + 1e-6 * np.abs(theirs[:, CLOSE])

    assert (diff <= tol).all(), (
        f"mean path diverges from upstream beyond Monte Carlo error:\n"
        f"  max |diff| = {diff.max():.6f}, at tol {tol[diff.argmax()]:.6f}\n"
        "  the vendored inference loop has drifted from Kronos's"
    )


def test_samples_are_a_real_distribution_not_a_repeated_path(kronos, window):
    history, future = window
    ours = _our_samples(kronos, history, future, seed=0)
    assert ours.shape == (SAMPLE_COUNT, len(future), len(BAR_COLUMNS))
    spread = ours[:, -1, CLOSE].std()
    assert spread > 0, "all sampled paths identical — nothing to carry into the prior"
    # Uncertainty must grow with horizon, or the model is not forecasting.
    per_step = ours[:, :, CLOSE].std(axis=0)
    assert per_step[-1] > per_step[0]


def test_seeded_generation_is_reproducible_on_this_device(kronos, window):
    history, future = window
    a = _our_samples(kronos, history, future, seed=1234)
    b = _our_samples(kronos, history, future, seed=1234)
    assert np.array_equal(a, b), "same seed, same device, different samples"


def test_different_seeds_diverge(kronos, window):
    history, future = window
    a = _our_samples(kronos, history, future, seed=1)
    b = _our_samples(kronos, history, future, seed=2)
    assert not np.array_equal(a, b)
