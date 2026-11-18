"""Stage 4: is the cue used, not just present? (tier T2)

* Erasure: remove a concept's linear direction from every position of one layer's output with LEACE
  (Belrose et al., 2023) and measure accuracy on attacked and clean data. Because the eraser is applied
  at every position, it is fitted on token-level states sampled from every position (each token takes
  its article's concept label), not on the read-out state alone. Control: erase a random subspace of the
  same rank, centred on the same mean as LEACE.
* Controlled patching: restore the clean article's read-out state at one layer while the rest of the
  model runs on the rewrite, and report the share of lost margin recovered, minus the recovery from an
  unrelated same-label article (and a class-mean patch).

Layer l refers to ``hidden_states[l]``, i.e. the output of block l-1 (l >= 1). For models with a final
norm (GPT-2, Llama, Qwen, Mistral), ``hidden_states[n_layers]`` is reported after that norm, so layer
n_layers is edited at the norm's output.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..detectors.base import find_final_norm
from .behaviour import per_class


def edit_target(det, layer: int) -> torch.nn.Module:
    """The module whose output equals ``hidden_states[layer]``."""
    if layer < 1 or layer > det.n_layers:
        raise ValueError(f"layer must be in 1..{det.n_layers} (layer 0 is the embedding output)")
    if layer == det.n_layers:
        norm = find_final_norm(det.model)
        if norm is not None:
            return norm
    return det.layer_modules()[layer - 1]


@contextmanager
def edit_block_output(det, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Apply ``fn`` to the hidden-state tensor [b, T, d] produced by block ``layer - 1``."""
    block = edit_target(det, layer)

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------- erasure
def fit_leace(X: np.ndarray, z: np.ndarray):
    """LEACE from the concept-erasure package; the bundled port when it is not installed (Python < 3.10)."""
    try:
        from concept_erasure import LeaceEraser
    except ImportError:
        from ._leace import LeaceEraser
    return LeaceEraser.fit(torch.tensor(X, dtype=torch.float32), torch.tensor(z, dtype=torch.long))


def random_subspace_eraser(dim: int, rank: int, seed: int = 0, mean: Optional[np.ndarray] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Orthogonal removal of a random rank-``rank`` subspace, centred like LEACE: x - Q Q^T (x - mean)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=g))
    mu = torch.zeros(dim) if mean is None else torch.tensor(mean, dtype=torch.float32)

    def erase(x: torch.Tensor) -> torch.Tensor:
        Q, m = q.to(x.device, x.dtype), mu.to(x.device, x.dtype)
        return x - ((x - m) @ Q) @ Q.T
    return erase


@torch.no_grad()
def token_states(det, texts: Sequence[str], ids: Optional[Sequence[str]], layer: int, per_text: int = 32,
                 seed: int = 0, batch_size: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """States at ``layer`` from up to ``per_text`` random non-padding positions of each text.

    Returns (X [m, d], owner [m]) where owner[j] is the index of the text that row j came from.
    Inputs are built exactly as for scoring, so the positions match the ones the eraser edits.
    """
    rng = np.random.default_rng(seed)
    bs = batch_size or det.batch_size
    X, owner = [], []
    for i in range(0, len(texts), bs):
        bt = list(texts[i:i + bs])
        bi = list(ids[i:i + bs]) if ids is not None else None
        enc = det.batch_inputs(bt, bi)
        extra = det.forward_kwargs() if hasattr(det, "forward_kwargs") else {}
        h = det.model(**enc, output_hidden_states=True, **extra).hidden_states[layer].float()
        mask = enc["attention_mask"].bool()
        for k in range(h.shape[0]):
            pos = torch.nonzero(mask[k]).flatten().cpu().numpy()
            pick = rng.choice(pos, size=min(per_text, len(pos)), replace=False)
            X.append(h[k, torch.as_tensor(pick, device=h.device)].cpu().numpy())
            owner.extend([i + k] * len(pick))
    return np.concatenate(X), np.asarray(owner)


def _on_device(eraser, device: torch.device):
    """LeaceEraser is a dataclass of tensors without .to(); move its tensor fields."""
    import dataclasses

    if not dataclasses.is_dataclass(eraser):
        return eraser
    moved = {f.name: getattr(eraser, f.name).to(device) for f in dataclasses.fields(eraser)
             if isinstance(getattr(eraser, f.name), torch.Tensor)}
    return dataclasses.replace(eraser, **moved)


def _apply(eraser) -> Callable[[torch.Tensor], torch.Tensor]:
    cache: dict = {}

    def fn(h: torch.Tensor) -> torch.Tensor:
        if h.device not in cache:
            cache[h.device] = _on_device(eraser, h.device)
        shape = h.shape
        flat = h.reshape(-1, shape[-1]).float()
        return cache[h.device](flat).to(h.dtype).reshape(shape)
    return fn


def erasure_effect(det, layer: int, concept_X: np.ndarray, concept_z: np.ndarray,
                   clean_texts: Sequence[str], adv_texts: Sequence[str], y: np.ndarray,
                   ids: Optional[Sequence[str]] = None, seed: int = 0) -> dict:
    """Accuracy change from erasing the concept at ``layer``, with a random-subspace control of equal rank."""
    eraser = fit_leace(concept_X, concept_z)
    rank = max(1, len(np.unique(concept_z)) - 1)
    rand = random_subspace_eraser(concept_X.shape[1], rank, seed, mean=concept_X.mean(0))

    def acc(texts, fn=None):
        if fn is None:
            p = det.predict_proba(list(texts), ids)
        else:
            with edit_block_output(det, layer, _apply(fn)):
                p = det.predict_proba(list(texts), ids)
        return per_class(y, p)

    base_adv, base_clean = acc(adv_texts), acc(clean_texts)
    er_adv, er_clean = acc(adv_texts, eraser), acc(clean_texts, eraser)
    rd_adv, rd_clean = acc(adv_texts, rand), acc(clean_texts, rand)
    return {
        "layer": layer, "rank": rank,
        "adv_acc": base_adv["accuracy"], "adv_acc_erased": er_adv["accuracy"], "adv_acc_random": rd_adv["accuracy"],
        "clean_acc": base_clean["accuracy"], "clean_acc_erased": er_clean["accuracy"], "clean_acc_random": rd_clean["accuracy"],
        "erasure_gain": er_adv["accuracy"] - base_adv["accuracy"],
        "random_gain": rd_adv["accuracy"] - base_adv["accuracy"],
        "clean_cost": base_clean["accuracy"] - er_clean["accuracy"],
        "per_class_erased": er_adv,
    }


# ---------------------------------------------------------------------- patching
def _readout_positions(det, texts: Sequence[str]) -> list[int]:
    if hasattr(det, "decoder") and not det.decoder:
        return [0] * len(texts)
    if hasattr(det, "decoder"):  # decoder encoder-style detector, right padding
        return [min(len(det.tokenizer(t, truncation=True, max_length=det.max_len)["input_ids"]), det.max_len) - 1 for t in texts]
    return [-1] * len(texts)  # LLM detector, left padding: answer slot is the last position


def _margins(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, logits[:, 1] - logits[:, 0], logits[:, 0] - logits[:, 1])


def patched_logits(det, layer: int, adv_texts: Sequence[str], states: np.ndarray, ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """Logits on rewrites with the read-out state at ``layer`` replaced by ``states`` (one row per article)."""
    pos = _readout_positions(det, adv_texts)
    cursor = {"i": 0}
    src = torch.tensor(states, dtype=torch.float32)

    def fn(h: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        i0 = cursor["i"]
        h = h.clone()
        for k in range(b):
            h[k, pos[i0 + k]] = src[i0 + k].to(h.device, h.dtype)
        cursor["i"] += b
        return h

    with edit_block_output(det, layer, fn):
        return det.class_logits(list(adv_texts), ids)


def patching(det, layers: Sequence[int], clean_texts: Sequence[str], adv_texts: Sequence[str], y: np.ndarray,
             ids: Optional[Sequence[str]] = None, seed: int = 0) -> dict:
    """Recovered share of lost margin per layer: own state, unrelated same-label state, class-mean state."""
    y = np.asarray(y)
    m_clean = _margins(det.class_logits(list(clean_texts), ids), y)
    m_adv = _margins(det.class_logits(list(adv_texts), ids), y)
    damaged = m_clean > m_adv
    rng = np.random.default_rng(seed)
    unrelated = np.array([rng.choice(np.flatnonzero((y == y[i]) & (np.arange(len(y)) != i))) for i in range(len(y))])
    states = det.hidden_states(list(clean_texts), ids, layers=list(layers), pooling="readout")
    lost = float(np.sum((m_clean - m_adv)[damaged]))
    out = {"n_damaged": int(damaged.sum()), "layers": {}}
    for l in layers:
        if l < 1:
            continue
        S = states[l]
        class_mean = np.stack([S[y == c].mean(0) for c in (0, 1)])[y]
        rec = {}
        for name, src in (("own", S), ("unrelated", S[unrelated]), ("class_mean", class_mean)):
            m_p = _margins(patched_logits(det, l, adv_texts, src, ids), y)
            rec[name] = float(np.sum((m_p - m_adv)[damaged]) / lost) if lost > 0 else float("nan")
        rec["article_specific"] = rec["own"] - rec["unrelated"]
        out["layers"][l] = rec
    return out
