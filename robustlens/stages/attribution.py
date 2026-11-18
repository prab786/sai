"""Stage 3: which input cues does the decision rest on?

Attributions explain the fake-vs-real margin (logit_fake - logit_real), so their sign means
"pushes toward fake". Scores are aggregated from sub-word tokens to words of the article only
(prompt tokens of LLM detectors are excluded).

Explainers: Integrated Gradients and a second explainer for the agreement control:
embedding occlusion (default, no extra dependency) or SHAP's partition explainer (``pip install shap``).

Cue reliance = share of absolute attribution mass on cue words, minus the share on frequency-matched
random words (control). The cue is the set of words the rewrite changed (style reliance).
"""
from __future__ import annotations

import difflib
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from ..detectors.base import aggregate_to_words, tokens_to_words


# ---------------------------------------------------------------------- explainers
def _freeze(det) -> None:
    """Gradients are needed w.r.t. the input embeddings only; parameter gradients would double memory."""
    if not getattr(det, "_frozen", False):
        for p in det.model.parameters():
            p.requires_grad_(False)
        det._frozen = True


def _is_llm(det) -> bool:
    return hasattr(det, "build_prompt")


def _prepare(det, text: str, article_id: Optional[str]):
    inp = det.attribution_inputs(text, article_id)
    words, tok2word = tokens_to_words(inp["offsets"], inp["region"], inp["region_text"])
    ids, attn = inp["input_ids"], inp["attention_mask"]
    in_article = torch.tensor([w >= 0 for w in tok2word], device=ids.device)
    base_ids = ids.clone()
    base_ids[0, in_article] = det.tokenizer.pad_token_id
    return inp, words, tok2word, ids, attn, base_ids


def _margin_fn(det, attn, ctx):
    def f(embeds):
        logits = det.logits_from_embeds(embeds, attn.expand(embeds.shape[0], -1), ctx)
        return logits[:, 1] - logits[:, 0]
    return f


def integrated_gradients(det, text: str, article_id: Optional[str] = None, n_steps: int = 32, internal_batch: int = 8) -> dict:
    from captum.attr import IntegratedGradients

    _freeze(det)
    inp, words, tok2word, ids, attn, base_ids = _prepare(det, text, article_id)
    with torch.no_grad():
        emb, base = det.embed(ids).detach(), det.embed(base_ids).detach()
    f = _margin_fn(det, attn, inp["ctx"])
    with torch.no_grad():
        gap = float((f(emb) - f(base)).item())  # what the attributions must sum to (completeness)
    attr, delta = IntegratedGradients(f).attribute(emb, baselines=base, n_steps=n_steps,
                                                   internal_batch_size=internal_batch, return_convergence_delta=True)
    tok_scores = attr.sum(-1)[0].detach().float().cpu().numpy()
    d = float(delta.abs().max())
    return {"words": words, "scores": aggregate_to_words(tok_scores, tok2word, len(words)),
            # relative completeness error; the floor of 1 logit keeps tiny gaps (f(x) close to f(baseline)) from
            # inflating it, so below 1 logit it is the absolute error
            "convergence_delta": d, "gap": gap, "rel_delta": d / max(abs(gap), 1.0)}


@torch.no_grad()
def occlusion(det, text: str, article_id: Optional[str] = None, batch: int = 32) -> dict:
    """Margin drop when one word's token embeddings are replaced by the padding embedding."""
    inp, words, tok2word, ids, attn, base_ids = _prepare(det, text, article_id)
    emb, base = det.embed(ids), det.embed(base_ids)
    f = _margin_fn(det, attn, inp["ctx"])
    full = f(emb).item()
    t2w = np.array(tok2word)
    scores = np.zeros(len(words))
    for start in range(0, len(words), batch):
        ws = list(range(start, min(start + batch, len(words))))
        e = emb.repeat(len(ws), 1, 1)
        for k, w in enumerate(ws):
            m = torch.tensor(t2w == w, device=e.device)
            e[k, m] = base[0, m]
        scores[ws] = full - f(e).float().cpu().numpy()
    return {"words": words, "scores": scores}


def shap_values(det, text: str, article_id: Optional[str] = None, max_evals: int = 256) -> dict:
    """SHAP partition explainer over article words (optional dependency)."""
    import shap

    inp, words, tok2word, ids, attn, base_ids = _prepare(det, text, article_id)
    t2w = np.array(tok2word)
    with torch.no_grad():
        emb, base = det.embed(ids), det.embed(base_ids)
    f = _margin_fn(det, attn, inp["ctx"])

    def model_fn(masks: np.ndarray) -> np.ndarray:  # masks: [m, n_words], 1 = keep word
        out = []
        with torch.no_grad():
            for row in masks:
                e = emb.clone()
                drop = torch.tensor(np.isin(t2w, np.flatnonzero(row == 0)), device=e.device)
                e[0, drop] = base[0, drop]
                out.append(f(e).item())
        return np.array(out)

    explainer = shap.explainers.Partition(model_fn, shap.maskers.Partition(np.zeros((1, len(words)))))
    vals = explainer(np.ones((1, len(words))), max_evals=max_evals)
    return {"words": words, "scores": np.asarray(vals.values[0], dtype=float)}


def explain(det, text: str, article_id: Optional[str], method: str, cfg: dict) -> dict:
    llm = _is_llm(det)
    if method == "ig":
        batch = cfg.get("llm_ig_internal_batch", 1) if llm else cfg.get("ig_internal_batch", 8)
        steps = cfg.get("llm_ig_steps", cfg.get("ig_steps", 32)) if llm else cfg.get("ig_steps", 32)
        return integrated_gradients(det, text, article_id, steps, batch)
    if method == "occlusion":
        return occlusion(det, text, article_id, cfg.get("llm_occlusion_batch", 4) if llm else cfg.get("occlusion_batch", 32))
    if method == "shap":
        return shap_values(det, text, article_id, cfg.get("shap_evals", 256))
    raise ValueError(method)


# ---------------------------------------------------------------------- comparisons
def _word_totals(words: Sequence[str], scores: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for w, s in zip(words, scores):
        k = w.lower()
        out[k] = out.get(k, 0.0) + float(s)
    return out


def stability(a: dict, b: dict, k: int = 10) -> dict:
    """Agreement of two attributions over the words they share (clean vs attacked, or explainer vs explainer)."""
    ta, tb = _word_totals(a["words"], a["scores"]), _word_totals(b["words"], b["scores"])
    shared = sorted(set(ta) & set(tb))
    rho = float(spearmanr([ta[w] for w in shared], [tb[w] for w in shared]).statistic) if len(shared) >= 3 else float("nan")
    top_a = set(sorted(ta, key=lambda w: -abs(ta[w]))[:k])
    top_b = set(sorted(tb, key=lambda w: -abs(tb[w]))[:k])
    jac = len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else float("nan")
    return {"spearman": rho, "topk_jaccard": float(jac), "n_shared": len(shared)}


# ---------------------------------------------------------------------- cue reliance
def corpus_frequencies(texts: Sequence[str]) -> dict[str, int]:
    from collections import Counter

    from ..detectors.base import WORD_RE

    c: Counter = Counter()
    for t in texts:
        c.update(w.lower() for w in WORD_RE.findall(t))
    return dict(c)


def changed_word_mask(orig_words: Sequence[str], adv_words: Sequence[str]) -> np.ndarray:
    """Words of the attacked text that are not in a matching block with the original (style cue)."""
    sm = difflib.SequenceMatcher(None, [w.lower() for w in orig_words], [w.lower() for w in adv_words], autojunk=False)
    mask = np.ones(len(adv_words), dtype=bool)
    for blk in sm.get_matching_blocks():
        mask[blk.b:blk.b + blk.size] = False
    return mask


def cue_reliance(words: Sequence[str], scores: np.ndarray, cue_mask: np.ndarray, freq: dict[str, int],
                 rng: np.random.Generator, n_control: int = 20) -> dict:
    """Share of |attribution| on cue words minus the mean share on frequency-matched random non-cue words."""
    s = np.abs(np.asarray(scores, dtype=float))
    total = s.sum()
    cue_mask = np.asarray(cue_mask, dtype=bool)
    if total <= 0 or not cue_mask.any() or cue_mask.all():
        return {"share": float("nan"), "control": float("nan"), "reliance": float("nan"), "n_cue": int(cue_mask.sum())}
    bins = np.array([int(np.log2(freq.get(w.lower(), 0) + 1)) for w in words])
    cue_idx, other = np.flatnonzero(cue_mask), np.flatnonzero(~cue_mask)
    k_total = min(len(cue_idx), len(other))
    shares = []
    for _ in range(n_control):
        pick: list[int] = []
        for b in np.unique(bins[cue_idx]):
            k = int(np.sum(bins[cue_idx] == b))
            cand = np.setdiff1d(other[bins[other] == b], pick)
            pick += rng.choice(cand, size=min(k, len(cand)), replace=False).tolist()
        if len(pick) < k_total:  # not enough same-frequency words: fill from any non-cue word
            rest = np.setdiff1d(other, pick)
            pick += rng.choice(rest, size=k_total - len(pick), replace=False).tolist()
        shares.append(s[pick[:k_total]].sum() / total * (len(cue_idx) / max(k_total, 1)))
    share = float(s[cue_mask].sum() / total)
    ctrl = float(np.mean(shares))
    return {"share": share, "control": ctrl, "reliance": share - ctrl, "n_cue": int(cue_mask.sum())}
