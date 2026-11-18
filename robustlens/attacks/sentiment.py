"""Held-out attack family: sentiment reframing generated with an open model (default Gemma-2-9B-Instruct).

Every test article is rewritten in a positive and a negative tone with greedy decoding. The adversarial
set pairs fake -> positive and real -> negative (as sets A-D raise fakes' and lower reals' credibility);
the opposite pairing is written as a control.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..data import Split, limit_split, load_split, pair_sentiment
from ..utils import log, read_jsonl, write_jsonl

PROMPT = (
    "Rewrite the following news article in a strongly {tone} tone. "
    "Keep every factual claim, name, number, date and quotation exactly as stated. "
    "Do not add or remove facts, and do not comment on whether the article is true. "
    "Return only the rewritten article.\n\nArticle:\n{text}"
)
TONES = {"positive": "positive, upbeat", "negative": "negative, alarmed"}
REFUSAL = re.compile(r"\b(I can(?:not|'t)|I'm sorry|As an AI)\b")


def truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def prefilter_ok(src: str, out: str, max_words: int) -> bool:
    """Cheap screen for truncated or refused generations; Stage 0 does the real validity check."""
    if len(out.split()) < 0.5 * min(len(src.split()), max_words):
        return False
    return not REFUSAL.search(out)


def build_conversations(texts: list[str], max_words: int) -> list[tuple[str, list[dict]]]:
    convs = []
    for tone, desc in TONES.items():
        for t in texts:
            # Gemma's chat template has no system role, so the instruction goes in the user turn
            convs.append((tone, [{"role": "user", "content": PROMPT.format(tone=desc, text=truncate_words(t, max_words))}]))
    return convs


VLLM_MIN = (0, 5, 1)  # first vLLM release that runs Gemma-2


def _chat_prompts(convs: list[list[dict]], tok) -> list[str]:
    if getattr(tok, "chat_template", None):
        return [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    return [c[0]["content"] for c in convs]


def generate_vllm(convs: list[list[dict]], model: str, max_tokens: int, seed: int) -> list[str]:
    """Greedy generation with vLLM. Prompts are built with the model's own chat template and passed to
    ``LLM.generate``, so this works on any vLLM that supports the model (no ``LLM.chat`` needed)."""
    import inspect
    import os

    from ..utils import version_tuple

    # Greedy decoding never uses FlashInfer's top-k/top-p sampler, but vLLM's start-up warm-up JIT-compiles it,
    # which needs nvcc (the CUDA toolkit). Turn it off unless the user chose otherwise. Set before importing
    # vLLM so the engine subprocess inherits it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vLLM is not installed. Install it (pip install 'vllm>=0.6.3'; on Python 3.8 use "
                           "'vllm==0.6.3.post1') or run with --backend hf.") from e
    if version_tuple(getattr(vllm, "__version__", "0")) < VLLM_MIN:
        raise RuntimeError(
            f"vLLM {vllm.__version__} is too old for {model} (Gemma-2 needs vLLM >= 0.5.1). Upgrade it "
            "(pip install 'vllm>=0.6.3'; on Python 3.8 the last release is 'vllm==0.6.3.post1'), or run "
            "this step with --backend hf (GEN_BACKEND=hf bash scripts/run_all.sh).")
    from transformers import AutoTokenizer

    prompts = _chat_prompts(convs, AutoTokenizer.from_pretrained(model))
    from vllm.engine.arg_utils import EngineArgs

    accepted = set(inspect.signature(EngineArgs.__init__).parameters)
    kwargs = {k: v for k, v in {"seed": seed, "max_model_len": 8192}.items() if k in accepted}
    try:
        llm = LLM(model=model, **kwargs)
    except RuntimeError as e:
        raise RuntimeError(
            f"vLLM could not start ({e}). The root cause is printed above the traceback. If it mentions nvcc or "
            "CUDA_HOME, a FlashInfer kernel needed the CUDA toolkit: install it (e.g. conda install -c nvidia cuda-nvcc) "
            "and set CUDA_HOME, or uninstall flashinfer-python. Otherwise rerun with --backend hf "
            "(GEN_BACKEND=hf bash scripts/run_all.sh).") from e
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    return [o.outputs[0].text.strip() for o in outs]


def generate_hf(convs: list[list[dict]], model: str, max_tokens: int, batch_size: int = 8, device: Optional[str] = None) -> list[str]:
    """Greedy generation with transformers (no vLLM). bf16 on GPU; prompts are length-sorted to reduce padding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..utils import hf_dtype_kwargs

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw = hf_dtype_kwargs(torch.bfloat16 if dev.type == "cuda" else torch.float32)
    if "gemma2" in model.lower().replace("-", ""):
        load_kw["attn_implementation"] = "eager"  # recommended for Gemma-2's logit soft-capping
    lm = AutoModelForCausalLM.from_pretrained(model, **load_kw).to(dev).eval()
    prompts = _chat_prompts(convs, tok)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outs: dict = {}
    has_template = bool(getattr(tok, "chat_template", None))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=not has_template).to(dev)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, text in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs[i] = text.strip()
        log.info("generated %d / %d", min(start + batch_size, len(order)), len(order))
    return [outs[i] for i in range(len(prompts))]


def generate_sentiment_sets(processed: str | Path, ds: str, model: str, max_words: int = 1200, max_tokens: int = 2048,
                            seed: int = 0, backend: str = "vllm", limit: Optional[int] = None, overwrite: bool = False,
                            batch_size: int = 8) -> dict:
    """Write attacks/sentiment_{pos,neg,adv,ctl}.jsonl for one dataset. ``limit`` caps articles (smoke tests).

    A dataset whose sets already exist for exactly these articles is skipped unless ``overwrite``, so a
    rerun after an interruption does not regenerate finished datasets.
    """
    test = load_split(processed, ds, "test")
    test = limit_split(test, limit)
    out_dir = Path(processed) / ds / "attacks"
    done = [out_dir / f"sentiment_{k}.jsonl" for k in ("pos", "neg", "adv", "ctl")]
    if not overwrite and all(f.exists() for f in done):
        if [r["id"] for r in read_jsonl(done[2])] == list(test.ids):
            log.info("sentiment sets for %s already exist: skipped (use --overwrite to regenerate)", ds)
            return {"skipped": True}
    pairs = build_conversations(test.texts, max_words)
    convs = [c for _, c in pairs]
    texts = (generate_vllm(convs, model, max_tokens, seed) if backend == "vllm"
             else generate_hf(convs, model, max_tokens, batch_size))
    n = len(test)
    sets, stats = {}, {}
    for k, tone in enumerate(TONES):
        gen = texts[k * n:(k + 1) * n]
        ok = [prefilter_ok(s, g, max_words) for s, g in zip(test.texts, gen)]
        tag = tone[:3]
        sets[tag] = Split(list(test.ids), gen, test.labels.copy(), f"sentiment_{tag}", {"prefilter_ok": ok})
        rows = [{**r, "prefilter_ok": o, "generator": model} for r, o in zip(sets[tag].rows(), ok)]
        write_jsonl(rows, out_dir / f"sentiment_{tag}.jsonl")
        stats[tag] = {"n": n, "prefilter_pass": float(sum(ok) / max(1, n))}
    for adversarial, name in ((True, "sentiment_adv"), (False, "sentiment_ctl")):
        write_jsonl(pair_sentiment(sets["pos"], sets["neg"], adversarial).rows(), out_dir / f"{name}.jsonl")
    log.info("sentiment sets for %s: %s", ds, stats)
    return stats
