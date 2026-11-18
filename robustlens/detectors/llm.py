"""Zero-shot instruction-tuned LLM detectors.

The article comes first and the question last, so the final prompt position is the answer slot:

    <article>

    Is the article above real or fake news?
    A. <Real|Fake>
    B. <Fake|Real>
    Answer:

The Real/Fake -> A/B mapping is randomised per article and fixed by a hash of (seed, article id), so an
article and all its rewrites share a layout and calibration is not driven by a positional "A" prior.
Scores are the next-token logits of the two answer letters, renormalised over {A, B}.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..utils import hf_dtype_kwargs, stable_hash
from .base import softmax_fake

QUESTION = "Is the article above real or fake news?"


class ZeroShotLLMDetector:
    def __init__(self, model_name: str, max_article_tokens: int = 1536, batch_size: int = 4,
                 device: Optional[str] = None, dtype: str = "bfloat16", mapping_seed: int = 0,
                 name: Optional[str] = None, model=None, tokenizer=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or model_name.split("/")[-1]
        self.max_article_tokens = max_article_tokens
        self.batch_size = batch_size
        self.mapping_seed = mapping_seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if model is None:
            torch_dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_name, **hf_dtype_kwargs(torch_dtype))
        self.model = model.to(self.device).eval()
        self.letter_ids = {L: self._letter_token_ids(L) for L in ("A", "B")}

    # ------------------------------------------------------------------ prompt
    def _letter_token_ids(self, letter: str) -> list[int]:
        ids = set()
        for variant in (letter, " " + letter):
            toks = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise ValueError(f"tokenizer has no single-token form of {letter!r}")
        return sorted(ids)

    def fake_is_a(self, article_id: str) -> bool:
        return stable_hash(self.mapping_seed, article_id) % 2 == 0

    def _truncate(self, text: str) -> str:
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        if len(toks) <= self.max_article_tokens:
            return text
        return self.tokenizer.decode(toks[: self.max_article_tokens])

    def build_prompt(self, text: str, article_id: str) -> tuple[str, tuple[int, int], str]:
        # strip: some chat templates (Llama-3) trim the message, which would move the article in the prompt
        article = self._truncate(text).strip()
        a, b = ("Fake", "Real") if self.fake_is_a(article_id) else ("Real", "Fake")
        user = f"{article}\n\n{QUESTION}\nA. {a}\nB. {b}\nAnswer:"
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = user
        start = prompt.find(article)
        if start < 0:
            raise ValueError("article not found verbatim in the formatted prompt; check the chat template")
        return prompt, (start, start + len(article)), article

    def _encode_prompts(self, prompts: list[str]) -> dict:
        add_special = not bool(getattr(self.tokenizer, "chat_template", None))
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special)
        return {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    @property
    def last_logits_kw(self) -> dict:
        """Ask for logits at the last position only (all we read). Without it the model returns, and keeps for
        the backward pass, [seq_len x vocab] logits: about 1,500 x 152k for Qwen per article. transformers >= 4.50
        calls the argument logits_to_keep, 4.45-4.49 num_logits_to_keep; older versions get nothing."""
        if not hasattr(self, "_last_logits_kw"):
            import inspect

            params = inspect.signature(self.model.forward).parameters
            key = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
            self._last_logits_kw = {key: 1} if key else {}
        return self._last_logits_kw

    def _letters(self, last_logits: torch.Tensor) -> torch.Tensor:
        """[b, V] -> [b, 2] logits for (A, B), taking the best of each letter's token variants."""
        cols = [last_logits[:, self.letter_ids[L]].max(dim=1).values for L in ("A", "B")]
        return torch.stack(cols, dim=1)

    def _to_real_fake(self, ab: torch.Tensor, ids: Sequence[str]) -> torch.Tensor:
        fake_a = torch.tensor([self.fake_is_a(i) for i in ids], device=ab.device)
        real = torch.where(fake_a, ab[:, 1], ab[:, 0])
        fake = torch.where(fake_a, ab[:, 0], ab[:, 1])
        return torch.stack([real, fake], dim=1)

    # ------------------------------------------------------------------ scoring
    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def _require_ids(self, texts, ids):
        if ids is None:
            raise ValueError("LLM detectors need article ids to fix the answer mapping")
        assert len(ids) == len(texts)

    @torch.no_grad()
    def class_logits(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None) -> np.ndarray:
        self._require_ids(texts, ids)
        out = []
        for i in range(0, len(texts), self.batch_size):
            bt, bi = texts[i:i + self.batch_size], ids[i:i + self.batch_size]
            enc = self._encode_prompts([self.build_prompt(t, a)[0] for t, a in zip(bt, bi)])
            logits = self.model(**enc, **self.last_logits_kw).logits[:, -1].float()
            out.append(self._to_real_fake(self._letters(logits), bi).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))

    def predict_proba(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None) -> np.ndarray:
        return softmax_fake(self.class_logits(texts, ids))

    @torch.no_grad()
    def hidden_states(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None,
                      layers: Optional[Sequence[int]] = None, pooling: str = "readout") -> dict[int, np.ndarray]:
        """Residual stream at the last prompt position (the answer slot)."""
        if pooling != "readout":
            raise ValueError("LLM detectors expose the answer-slot (read-out) state only")
        self._require_ids(texts, ids)
        layers = list(range(self.n_layers + 1)) if layers is None else list(layers)
        store: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        for i in range(0, len(texts), self.batch_size):
            bt, bi = texts[i:i + self.batch_size], ids[i:i + self.batch_size]
            enc = self._encode_prompts([self.build_prompt(t, a)[0] for t, a in zip(bt, bi)])
            hs = self.model(**enc, output_hidden_states=True, **self.last_logits_kw).hidden_states
            for l in layers:
                store[l].append(hs[l][:, -1].float().cpu().numpy())
        return {l: np.concatenate(v) for l, v in store.items()}

    # ------------------------------------------------------------------ attribution interface
    def attribution_inputs(self, text: str, article_id: str) -> dict:
        prompt, region, article = self.build_prompt(text, article_id)
        add_special = not bool(getattr(self.tokenizer, "chat_template", None))
        enc = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=add_special)
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "offsets": [tuple(o) for o in enc["offset_mapping"][0].tolist()],
            "region": region,
            "region_text": article,
            "ctx": article_id,
        }

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def baseline_ids(self, input_ids: torch.Tensor, offsets=None, region=None) -> torch.Tensor:
        return torch.full_like(input_ids, self.tokenizer.pad_token_id)

    def logits_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, ctx=None) -> torch.Tensor:
        logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **self.last_logits_kw).logits[:, -1].float()
        return self._to_real_fake(self._letters(logits), [ctx] * logits.shape[0])
