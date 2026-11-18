"""Data layer.

Reads the SheepDog (Wu et al., 2024) release, which is pickled dicts of lists, and writes one canonical JSONL format:

    processed/<dataset>/train.jsonl, test.jsonl          {"id", "text", "label"}   label 0 = real, 1 = fake
    processed/<dataset>/reframings/<style>.jsonl          aligned with train ids
    processed/<dataset>/attacks/<attack>.jsonl            aligned with test ids

Every attack file has exactly the test ids in test order, so original and attacked versions
of an article can be paired by position.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import ensure_dir, log, read_jsonl, save_json, write_jsonl

DATASETS = ("politifact", "gossipcop", "lun")
REAL, FAKE = 0, 1
RELIABLE_STYLES = ("objective", "neutral")
UNRELIABLE_STYLES = ("emotionally_triggering", "sensational")
PUBLISHER_SETS = ("A", "B", "C", "D")


# --------------------------------------------------------------------------- safe unpickling
class _SafeUnpickler(pickle.Unpickler):
    """The releases contain only builtins and numpy arrays; refuse anything else."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def safe_load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


# --------------------------------------------------------------------------- containers
@dataclass
class Split:
    ids: list[str]
    texts: list[str]
    labels: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=int)
        assert len(self.ids) == len(self.texts) == len(self.labels), "misaligned split"

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, idx) -> "Split":
        idx = np.asarray(idx, dtype=int)
        return Split([self.ids[i] for i in idx], [self.texts[i] for i in idx], self.labels[idx], self.name, dict(self.meta))

    def select_ids(self, ids: list[str]) -> "Split":
        pos = {i: k for k, i in enumerate(self.ids)}
        return self.take([pos[i] for i in ids])

    def rows(self) -> list[dict]:
        return [{"id": i, "text": t, "label": int(y)} for i, t, y in zip(self.ids, self.texts, self.labels)]


def _split_from_rows(rows: list[dict], name: str) -> Split:
    return Split([r["id"] for r in rows], [r["text"] for r in rows], [r["label"] for r in rows], name)


# --------------------------------------------------------------------------- preparation
def prepare(sheepdog_dir: str | Path, out_dir: str | Path, datasets=DATASETS) -> dict:
    """Convert the pickled releases to canonical JSONL. Returns a manifest of what was written."""
    sd, out = Path(sheepdog_dir), ensure_dir(out_dir)
    manifest: dict = {}
    for ds in datasets:
        dsdir = ensure_dir(out / ds)
        entry: dict = {}
        splits = {}
        for split in ("train", "test"):
            d = safe_load_pickle(sd / "news_articles" / f"{ds}_{split}.pkl")
            ids = [f"{ds}-{split}-{i:05d}" for i in range(len(d["news"]))]
            s = Split(ids, [str(x) for x in d["news"]], d["labels"], split)
            write_jsonl(s.rows(), dsdir / f"{split}.jsonl")
            splits[split] = s
            entry[split] = {"n": len(s), "fake": int(s.labels.sum())}

        for style in RELIABLE_STYLES + UNRELIABLE_STYLES:
            p = sd / "reframings" / f"{ds}_train_{style}.pkl"
            if not p.exists():
                continue
            d = safe_load_pickle(p)
            _check_aligned(d["labels"], splits["train"].labels, p)
            rows = [{"id": i, "text": str(t), "label": int(y)} for i, t, y in zip(splits["train"].ids, d["rewritten"], d["labels"])]
            write_jsonl(rows, dsdir / "reframings" / f"{style}.jsonl")
            entry.setdefault("reframings", []).append(style)

        for s in PUBLISHER_SETS:
            p = sd / "adversarial_test" / f"{ds}_test_adv_{s}.pkl"
            if p.exists():
                _write_attack(safe_load_pickle(p), splits["test"], dsdir / "attacks" / f"publisher_{s}.jsonl", p)
                entry.setdefault("attacks", []).append(f"publisher_{s}")

        manifest[ds] = entry
        log.info("prepared %s: %s", ds, {k: v for k, v in entry.items() if k in ("train", "test")})
    save_json(manifest, out / "manifest.json")
    return manifest


def _check_aligned(labels, ref_labels, path) -> None:
    if list(map(int, labels)) != list(map(int, ref_labels)):
        raise ValueError(f"labels in {path} are not aligned with the original split")


def _write_attack(d: dict, test: Split, path: Path, src: Path) -> None:
    _check_aligned(d["labels"], test.labels, src)
    rows = [{"id": i, "text": str(t), "label": int(y)} for i, t, y in zip(test.ids, d["news"], d["labels"])]
    write_jsonl(rows, path)


# --------------------------------------------------------------------------- loading
def load_split(processed: str | Path, ds: str, split: str) -> Split:
    return _split_from_rows(read_jsonl(Path(processed) / ds / f"{split}.jsonl"), split)


def load_reframing(processed: str | Path, ds: str, style: str) -> list[str]:
    return [r["text"] for r in read_jsonl(Path(processed) / ds / "reframings" / f"{style}.jsonl")]


def load_attack(processed: str | Path, ds: str, attack: str) -> Split:
    s = _split_from_rows(read_jsonl(Path(processed) / ds / "attacks" / f"{attack}.jsonl"), attack)
    test = load_split(processed, ds, "test")
    if s.ids != test.ids:
        raise ValueError(f"attack {attack} for {ds} is not aligned with the test split")
    return s


def available_attacks(processed: str | Path, ds: str) -> list[str]:
    d = Path(processed) / ds / "attacks"
    return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []


def pair_sentiment(pos: Split, neg: Split, adversarial: bool = True) -> Split:
    """Build a sentiment attack set from tone-wide rewrites.

    adversarial=True mirrors sets A-D: fake articles take the positive (credible-sounding) rewrite and
    real articles the negative (alarming) one. adversarial=False is the opposite pairing (control).
    """
    assert pos.ids == neg.ids and np.array_equal(pos.labels, neg.labels)
    texts = []
    for tp, tn, y in zip(pos.texts, neg.texts, pos.labels):
        fake_gets_pos = adversarial
        texts.append(tp if (y == FAKE) == fake_gets_pos else tn)
    return Split(list(pos.ids), texts, pos.labels.copy(), "sentiment_adv" if adversarial else "sentiment_ctl")


def stratified_subset_ids(split: Split, n: int = 500, seed: int = 0) -> list[str]:
    """Fixed class-stratified subset; returns all ids if the split is smaller than n."""
    if len(split) <= n:
        return list(split.ids)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in (REAL, FAKE):
        idx = np.flatnonzero(split.labels == c)
        k = int(round(n * len(idx) / len(split)))
        chosen.extend(rng.choice(idx, size=k, replace=False).tolist())
    return [split.ids[i] for i in sorted(chosen)]


def limit_split(split: Split, n: Optional[int], seed: int = 0) -> Split:
    """Class-stratified cap for quick runs. Never take the first N: some releases are sorted by label."""
    if not n or n >= len(split):
        return split
    return split.select_ids(stratified_subset_ids(split, n, seed))


def family_sets(cfg: dict, include_controls: bool = False) -> dict[str, list[str]]:
    fams = {k: list(v) for k, v in cfg["families"].items()}
    if include_controls:
        for k, v in cfg.get("controls", {}).items():
            fams[f"{k}_control"] = list(v)
    return fams
