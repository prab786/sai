"""Small shared helpers: seeding, JSON / JSONL / NPZ I/O, config loading, stable hashing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

log = logging.getLogger("robustlens")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
for _noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the analysis-only parts
        pass


def version_tuple(v: str) -> tuple:
    """'0.6.3.post1' -> (0, 6, 3), '4.46.0rc1' -> (4, 46, 0): leading numeric parts only."""
    import re

    out = []
    for part in v.split("+")[0].split("."):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group()))
        if m.end() < len(part):  # e.g. "0rc1": stop after the number
            break
    return tuple(out)


def hf_dtype_kwargs(torch_dtype) -> dict:
    """transformers >= 4.56 names the argument ``dtype``; older versions (e.g. 4.46 on Python 3.8) ``torch_dtype``."""
    import transformers

    key = "dtype" if version_tuple(transformers.__version__) >= (4, 56) else "torch_dtype"
    return {key: torch_dtype}


def stable_hash(*parts: Any) -> int:
    """Process-independent hash (Python's hash() is salted per process)."""
    h = hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()
    return int(h[:15], 16)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_states(states: dict[int, np.ndarray], path: str | os.PathLike, ids: list[str] | None = None) -> None:
    """Store per-layer hidden states as float16 to keep LLM states manageable.

    A layer whose values exceed float16's range (LLM residual streams can) is kept as float32 instead
    of overflowing to inf.
    """
    ensure_dir(Path(path).parent)
    f16_max = float(np.finfo(np.float16).max)
    arrays = {f"layer_{k}": (v.astype(np.float16) if np.abs(v).max(initial=0.0) < f16_max else v.astype(np.float32))
              for k, v in states.items()}
    if ids is not None:
        arrays["ids"] = np.array(ids)
    np.savez_compressed(path, **arrays)


def load_states(path: str | os.PathLike) -> tuple[dict[int, np.ndarray], list[str] | None]:
    z = np.load(path, allow_pickle=False)
    states = {int(k.split("_")[1]): z[k].astype(np.float32) for k in z.files if k.startswith("layer_")}
    ids = z["ids"].tolist() if "ids" in z.files else None
    return states, ids
