"""Un-averaged sampling from Kronos.

Why this module exists
----------------------
Kronos generates `sample_count` paths in parallel as a batch dimension and then
collapses them in the last line of `auto_regressive_inference`:

    z = z.reshape(-1, sample_count, z.size(1), z.size(2))
    preds = z.cpu().numpy()
    preds = np.mean(preds, axis=1)      # <- the predictive distribution dies here

So `KronosPredictor.predict()` cannot return a distribution: whatever you pass as
`sample_count`, you get one averaged path back. Since the entire point of this project
is to carry the distribution into portfolio construction, we reimplement the inference
tail without the mean.

The generation loop below is a direct adaptation of Kronos's own
`auto_regressive_inference` (MIT, github.com/shiyu-coder/Kronos). Keeping it here rather
than monkey-patching means upstream changes surface as a failing parity test rather than
as silently different numbers.

`test_parity_with_upstream` asserts that averaging our samples reproduces Kronos's
`predict()` output. That is the check that keeps this file honest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


class KronosNotAvailable(RuntimeError):
    """Raised when the Kronos source tree cannot be imported."""


def load_kronos(repo_path: str | Path | None = None):
    """Import Kronos's model package.

    Kronos ships as a repository, not a wheel, so it has to be put on the path. Pass
    `repo_path`, or set KRONOS_REPO, or have it importable already.
    """
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
        from model.kronos import sample_from_logits  # type: ignore

        return Kronos, KronosTokenizer, KronosPredictor, sample_from_logits
    except ImportError:
        pass

    candidate = repo_path or os.environ.get("KRONOS_REPO")
    if candidate is None:
        raise KronosNotAvailable(
            "Kronos is not importable. Clone it and point at it:\n"
            "  git clone https://github.com/shiyu-coder/Kronos ~/src/Kronos\n"
            "  export KRONOS_REPO=~/src/Kronos\n"
            "then install the model extra:  uv pip install -e '.[kronos]'"
        )

    candidate = Path(candidate).expanduser().resolve()
    if not (candidate / "model").is_dir():
        raise KronosNotAvailable(f"{candidate} does not look like the Kronos repo (no model/ dir)")
    sys.path.insert(0, str(candidate))
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
        from model.kronos import sample_from_logits  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local checkout
        raise KronosNotAvailable(f"failed to import Kronos from {candidate}: {exc}") from exc
    return Kronos, KronosTokenizer, KronosPredictor, sample_from_logits


def generate_samples(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context: int,
    pred_len: int,
    *,
    clip: float = 5.0,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    sample_count: int = 512,
    seed: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Autoregressive generation that keeps every sampled path.

    Args mirror Kronos's own inference function. `x`, `x_stamp`, `y_stamp` are
    normalized arrays shaped (batch, seq, features) exactly as `KronosPredictor.predict`
    prepares them.

    Returns
    -------
    ndarray, shape (batch, sample_count, pred_len, n_features)
        Still in normalized space. The caller denormalizes.

    Notes
    -----
    Determinism holds per device: the same seed on the same device and the same
    torch build reproduces byte-identical samples. It is not guaranteed across
    CPU/GPU or across torch versions, which is why the cache manifest records both.
    """
    import torch
    from model.kronos import sample_from_logits  # type: ignore

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        device = next(model.parameters()).device
        x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        x_stamp = torch.as_tensor(np.asarray(x_stamp, dtype=np.float32), device=device)
        y_stamp = torch.as_tensor(np.asarray(y_stamp, dtype=np.float32), device=device)

        x = torch.clip(x, -clip, clip)

        # Replicate each series `sample_count` times along the batch axis: the paths are
        # generated in one pass, which is why 512 samples costs far less than 512 runs.
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
        )

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start : start + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start : start + buffer_len]

        steps = range(pred_len)
        if verbose:  # pragma: no cover - cosmetic
            from tqdm import trange

            steps = trange(pred_len, desc="generate")

        for i in steps:
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(tokens[0], tokens[1], current_stamp)
            sample_pre = sample_from_logits(
                s1_logits[:, -1, :], temperature=temperature, top_k=top_k,
                top_p=top_p, sample_logits=True,
            )

            s2_logits = model.decode_s2(context, sample_pre)
            sample_post = sample_from_logits(
                s2_logits[:, -1, :], temperature=temperature, top_k=top_k,
                top_p=top_p, sample_logits=True,
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        decoded = tokenizer.decode(
            [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous(),
            ],
            half=True,
        )
        # (batch * sample_count, L, F) -> (batch, sample_count, L, F). The upstream
        # implementation averages axis 1 here. We keep it.
        decoded = decoded.reshape(-1, sample_count, decoded.size(1), decoded.size(2))
        return decoded[:, :, -pred_len:, :].cpu().numpy()
