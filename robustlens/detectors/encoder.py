"""Fine-tuned encoder detectors (BERT, RoBERTa, GPT-2) and the three training strategies.

A  standard cross-entropy on original articles.
B  style augmentation: each epoch adds one reliable and one unreliable reframing of every article.
C  SheepDog consistency (Wu et al., 2024): cross-entropy on the original plus
   lambda * (0.5 KL(p_orig || p_reliable) + 0.5 KL(p_orig || p_unreliable)).
   SheepDog's auxiliary veracity-attribution head is not used, matching the thesis set-up.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data import RELIABLE_STYLES, UNRELIABLE_STYLES, Split
from ..utils import log, save_json, set_seed
from .base import find_block_list, softmax_fake

DECODER_TYPES = {"gpt2", "gpt_neo", "gptj", "llama", "mistral", "qwen2", "gpt_neox", "opt"}


def _device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderDetector:
    def __init__(self, model_name_or_path: str, max_len: int = 256, device: Optional[str] = None,
                 batch_size: int = 32, name: Optional[str] = None, model=None, tokenizer=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name or Path(str(model_name_or_path)).name
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = _device(device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = model or AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device).eval()
        self.decoder = self.model.config.model_type in DECODER_TYPES

    # ------------------------------------------------------------------ basics
    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def layer_modules(self) -> list[torch.nn.Module]:
        return list(find_block_list(self.model, self.n_layers))

    def encode(self, texts: Sequence[str], max_len: Optional[int] = None) -> dict:
        enc = self.tokenizer(list(texts), truncation=True, max_length=max_len or self.max_len,
                             padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def batch_inputs(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None) -> dict:
        """Model inputs exactly as class_logits builds them (used for token-level states in Stage 4)."""
        return self.encode(texts)

    def readout_positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """First token for encoders; last non-padding token for decoders (right padding)."""
        if self.decoder:
            return attention_mask.sum(dim=1) - 1
        return torch.zeros(attention_mask.shape[0], dtype=torch.long, device=attention_mask.device)

    @torch.no_grad()
    def class_logits(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None, max_len: Optional[int] = None) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.encode(texts[i:i + self.batch_size], max_len)
            out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))

    def predict_proba(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None) -> np.ndarray:
        return softmax_fake(self.class_logits(texts, ids))

    @torch.no_grad()
    def hidden_states(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None,
                      layers: Optional[Sequence[int]] = None, pooling: str = "readout",
                      max_len: Optional[int] = None) -> dict[int, np.ndarray]:
        layers = list(range(self.n_layers + 1)) if layers is None else list(layers)
        store: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        for i in range(0, len(texts), self.batch_size):
            enc = self.encode(texts[i:i + self.batch_size], max_len)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"]
            pos = self.readout_positions(mask)
            for l in layers:
                h = hs[l].float()
                if pooling == "readout":
                    v = h[torch.arange(h.shape[0], device=h.device), pos]
                elif pooling == "mean":
                    m = mask.unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    raise ValueError(pooling)
                store[l].append(v.cpu().numpy())
        return {l: np.concatenate(v) for l, v in store.items()}

    # ------------------------------------------------------------------ attribution interface
    def attribution_inputs(self, text: str, article_id: Optional[str] = None) -> dict:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_offsets_mapping=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "offsets": [tuple(o) for o in enc["offset_mapping"][0].tolist()],
            "region": (0, len(text)),
            "region_text": text,
            "ctx": None,
        }

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def baseline_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        # keep special tokens (CLS/SEP/BOS) so the baseline differs only in content
        special = torch.tensor(self.tokenizer.all_special_ids, device=input_ids.device)
        keep = torch.isin(input_ids, special)
        base[keep] = input_ids[keep]
        return base

    def logits_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, ctx=None) -> torch.Tensor:
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path, meta: Optional[dict] = None) -> None:
        path = Path(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_json({"name": self.name, "max_len": self.max_len, **(meta or {})}, path / "robustlens.json")

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None, batch_size: int = 32, max_len: Optional[int] = None) -> "EncoderDetector":
        from ..utils import load_json

        meta = load_json(Path(path) / "robustlens.json")
        return cls(str(path), max_len=max_len or meta["max_len"], device=device, batch_size=batch_size, name=meta["name"])


# ---------------------------------------------------------------------- training
def _sample_reframings(reframings: dict[str, list[str]], n: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    rel = [s for s in RELIABLE_STYLES if s in reframings]
    unrel = [s for s in UNRELIABLE_STYLES if s in reframings]
    if not rel or not unrel:
        raise ValueError("strategies B and C need at least one reliable and one unreliable reframing")
    r_choice = rng.integers(len(rel), size=n)
    u_choice = rng.integers(len(unrel), size=n)
    return ([reframings[rel[c]][i] for i, c in enumerate(r_choice)],
            [reframings[unrel[c]][i] for i, c in enumerate(u_choice)])


def train_encoder(model_name: str, strategy: str, train: Split, reframings: Optional[dict[str, list[str]]],
                  tcfg: dict, seed: int, device: Optional[str] = None, name: Optional[str] = None,
                  max_steps: Optional[int] = None) -> EncoderDetector:
    """Fine-tune one detector. ``max_steps`` caps optimisation steps (smoke tests only)."""
    from transformers import get_linear_schedule_with_warmup

    strategy = strategy.upper()
    assert strategy in {"A", "B", "C"}
    set_seed(seed)
    det = EncoderDetector(model_name, max_len=tcfg["max_len"], device=device, name=name)
    model = det.model
    model.train()
    rng = np.random.default_rng(seed)
    n, bs, accum = len(train), int(tcfg["batch_size"]), int(tcfg.get("grad_accum", 1))
    epochs = int(tcfg["epochs"])
    per_epoch = 3 * n if strategy == "B" else n
    total = max(1, epochs * math.ceil(per_epoch / bs) // accum)
    if max_steps:
        total = min(total, max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sched = get_linear_schedule_with_warmup(opt, int(float(tcfg["warmup_ratio"]) * total), total)
    lam = float(tcfg.get("consistency_lambda", 1.0))
    labels = torch.tensor(train.labels, dtype=torch.long)
    step = 0

    def ce_step(texts, y):
        enc = det.encode(texts)
        return F.cross_entropy(model(**enc).logits, y.to(det.device))

    for epoch in range(epochs):
        rel, unrel = _sample_reframings(reframings, n, rng) if strategy in {"B", "C"} else (None, None)
        if strategy == "B":
            pool_texts = list(train.texts) + rel + unrel
            pool_labels = torch.cat([labels, labels, labels])
        order = rng.permutation(len(pool_texts) if strategy == "B" else n)
        losses = []
        for b, start in enumerate(range(0, len(order), bs)):
            idx = order[start:start + bs]
            if strategy == "A":
                loss = ce_step([train.texts[i] for i in idx], labels[idx])
            elif strategy == "B":
                loss = ce_step([pool_texts[i] for i in idx], pool_labels[idx])
            else:
                k = len(idx)
                enc = det.encode([train.texts[i] for i in idx] + [rel[i] for i in idx] + [unrel[i] for i in idx])
                logits = model(**enc).logits
                lo, lr_, lu = logits[:k], logits[k:2 * k], logits[2 * k:]
                p = F.softmax(lo, dim=-1)
                kl = torch.nn.KLDivLoss(reduction="batchmean")
                cons = 0.5 * kl(F.log_softmax(lr_, dim=-1), p) + 0.5 * kl(F.log_softmax(lu, dim=-1), p)
                loss = F.cross_entropy(lo, labels[idx].to(det.device)) + lam * cons
            (loss / accum).backward()
            losses.append(loss.item())
            if (b + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if max_steps and step >= max_steps:
                    break
        log.info("%s strategy %s seed %d epoch %d: loss %.4f", det.name, strategy, seed, epoch, float(np.mean(losses)))
        if max_steps and step >= max_steps:
            break
    model.eval()
    return det
