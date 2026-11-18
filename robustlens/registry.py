"""Detector naming and result paths, so every script agrees on where things live.

Encoder detectors are named ``<model>-<strategy>-s<seed>`` (e.g. ``roberta-A-s0``); LLM detectors use
their config key (e.g. ``qwen25``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

ENC_RE = re.compile(r"^(?P<model>[a-z0-9]+)-(?P<strategy>[ABC])-s(?P<seed>\d+)$")


def encoder_names(cfg: dict) -> list[str]:
    e = cfg["encoders"]
    return [f"{m}-{s}-s{seed}" for m in e["models"] for s in e["strategies"] for seed in e["seeds"]]


def llm_names(cfg: dict) -> list[str]:
    return list(cfg.get("llms", {}).get("models", {}))


def all_detectors(cfg: dict) -> list[str]:
    return encoder_names(cfg) + llm_names(cfg)


def parse_encoder(name: str) -> Optional[dict]:
    m = ENC_RE.match(name)
    return {"model": m["model"], "strategy": m["strategy"], "seed": int(m["seed"])} if m else None


def is_llm(name: str, cfg: dict) -> bool:
    return name in llm_names(cfg)


def detector_family(name: str, cfg: dict) -> str:
    return "zero-shot LLM" if is_llm(name, cfg) else "fine-tuned encoder"


def checkpoint_dir(cfg: dict, ds: str, name: str) -> Path:
    return Path(cfg["paths"]["checkpoints"]) / ds / name


def load_detector(name: str, ds: str, cfg: dict, device: Optional[str] = None, batch_size: Optional[int] = None):
    if is_llm(name, cfg):
        from .detectors.llm import ZeroShotLLMDetector

        l = cfg["llms"]
        return ZeroShotLLMDetector(l["models"][name], max_article_tokens=l["max_article_tokens"],
                                   batch_size=batch_size or l["batch_size"], device=device, dtype=l["dtype"],
                                   mapping_seed=l["mapping_seed"], name=name)
    from .detectors.encoder import EncoderDetector

    # every stage uses the training length (max_len stored with the checkpoint)
    return EncoderDetector.load(checkpoint_dir(cfg, ds, name), device=device, batch_size=batch_size or 32)


def result_path(cfg: dict, kind: str, ds: str, name: str, item: str, ext: str) -> Path:
    """results/<kind>/<dataset>/<detector>/<item>.<ext>"""
    return Path(cfg["paths"]["results"]) / kind / ds / name / f"{item}.{ext}"


def control_name(attack: str) -> str:
    """Random-edit control matched to an attack set."""
    return f"random_{attack}"
