"""Interface shared by every detector, plus helpers that map sub-word tokens back to words.

A detector scores P(fake) and exposes what the pipeline stages need:

* ``class_logits(texts, ids)``         -> [n, 2] logits ordered (real, fake)          Stage 1
* ``hidden_states(texts, ids, layers)`` -> {layer: [n, d]} read-out or mean states     Stage 2
* ``attribution_inputs`` / ``logits_from_embeds``  for gradient attribution             Stage 3

Layer indices follow Hugging Face ``hidden_states``: 0 is the embedding output, k is block k.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence

import numpy as np

WORD_RE = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]")


class Detector(Protocol):
    name: str

    def class_logits(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None) -> np.ndarray: ...
    def hidden_states(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None,
                      layers: Optional[Sequence[int]] = None, pooling: str = "readout") -> dict[int, np.ndarray]: ...
    @property
    def n_layers(self) -> int: ...


def softmax_fake(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def word_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in WORD_RE.finditer(text)]


def tokens_to_words(offsets: Sequence[tuple[int, int]], region: tuple[int, int], text_in_region: str) -> tuple[list[str], list[int]]:
    """Assign each token to a word of the region it falls in.

    ``offsets`` are character offsets of tokens in the full model input string; ``region`` is the
    (start, end) of the article inside that string. Returns the region's words and, per token, the
    index of its word or -1 for tokens outside the article (prompt, special tokens).
    """
    start, end = region
    spans = [(s + start, e + start) for s, e in word_spans(text_in_region)]
    words = [text_in_region[s - start:e - start] for s, e in spans]
    tok2word, w = [], 0
    for s, e in offsets:
        if e <= s or e <= start or s >= end:  # keep tokens that overlap the region (e.g. "Ġword" starting at the space before it)
            tok2word.append(-1)
            continue
        while w < len(spans) and spans[w][1] <= s:
            w += 1
        # a token starting in whitespace before the word (e.g. "Ġword") still belongs to it
        tok2word.append(w if w < len(spans) and spans[w][0] < e else -1)
    # keep only words the model actually saw (articles may be truncated)
    covered = sorted({w for w in tok2word if w >= 0})
    remap = {old: new for new, old in enumerate(covered)}
    return [words[i] for i in covered], [remap.get(w, -1) for w in tok2word]


def aggregate_to_words(token_scores: np.ndarray, tok2word: Sequence[int], n_words: int) -> np.ndarray:
    out = np.zeros(n_words, dtype=np.float64)
    for score, w in zip(token_scores, tok2word):
        if w >= 0:
            out[w] += score
    return out


def resolve_layers(spec, n_layers: int) -> list[int]:
    """'all' | 'every_2' | list of ints  ->  sorted layer indices in [0, n_layers]; always keeps the last layer."""
    if spec in (None, "all"):
        layers = list(range(n_layers + 1))
    elif spec == "every_2":
        layers = list(range(0, n_layers + 1, 2))
    else:
        layers = [int(x) for x in spec]
    if n_layers not in layers:
        layers.append(n_layers)
    return sorted(set(l for l in layers if 0 <= l <= n_layers))
