"""Stage 0: does the rewrite keep the false claim?

A rewrite passes the gate if
  1. an NLI model finds original and rewrite entail each other (chunked, both directions), and
  2. a fact-anchor check finds most numbers, dates and named entities of the original in the rewrite.

The anchor check is a lightweight proxy for the paper's proposition check (subject, relation, object,
modifiers); the human audit measures how precise the gate is, and that precision scales VASR.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..data import Split
from ..utils import ensure_dir

NUMBER_RE = re.compile(r"\b\d[\d,.]*%?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b", re.I)
MONTH_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")
STOP_CAPS = {"The", "A", "An", "In", "On", "At", "He", "She", "It", "They", "We", "I", "This", "That", "But", "And",
             "Or", "If", "As", "For", "With", "By", "From", "To", "Of", "His", "Her", "Its", "Their", "Our", "Mr", "Mrs", "Ms"}


def chunk_words(text: str, n: int = 200) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, max(1, len(w)), n)] or [""]


SENT_START = re.compile(r"(?:^|[.!?:;\"“”]\s*|\n\s*)$")


def fact_anchors(text: str) -> set[str]:
    anchors = {m.group().lower().strip(".,") for m in NUMBER_RE.finditer(text)}
    anchors |= {m.group().lower() for m in MONTH_RE.finditer(text)}
    for m in ENTITY_RE.finditer(text):
        words = [w for w in m.group().split() if w not in STOP_CAPS]
        # a single capitalised word at a sentence start is usually not a name
        if len(words) == 1 and len(m.group().split()) == 1 and SENT_START.search(text[max(0, m.start() - 3):m.start()]):
            continue
        if words and not (len(words) == 1 and words[0].isupper() and len(words[0]) > 5):  # skip shouting caps
            anchors.add(" ".join(words).lower())
    return {a for a in anchors if a}


def _contains(text_lower: str, anchor: str) -> bool:
    """Whole-token match, so "3" is not found inside "2023" and "ion" not inside "nation"."""
    return re.search(r"(?<!\w)" + re.escape(anchor) + r"(?!\w)", text_lower) is not None


def anchor_recall(original: str, rewrite: str) -> float:
    a = fact_anchors(original)
    if not a:
        return 1.0
    low = rewrite.lower()
    return float(np.mean([_contains(low, x) for x in a]))


class NLIScorer:
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 16, max_len: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        ent = [i for i, l in labels.items() if "entail" in l and "not" not in l]
        if not ent:
            raise ValueError(f"no entailment label in {labels}")
        self.ent_idx = ent[0]
        self.batch_size, self.max_len = batch_size, max_len

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(list(premises[i:i + self.batch_size]), list(hypotheses[i:i + self.batch_size]),
                           truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.model(**enc).logits.float().softmax(-1)[:, self.ent_idx]
            out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def directional(self, premise: str, hypothesis: str, chunk: int = 200) -> float:
        """Mean over hypothesis chunks of the best entailment from any premise chunk."""
        pc, hc = chunk_words(premise, chunk), chunk_words(hypothesis, chunk)
        pairs = [(p, h) for h in hc for p in pc]
        probs = self.entailment([p for p, _ in pairs], [h for _, h in pairs]).reshape(len(hc), len(pc))
        return float(probs.max(axis=1).mean())


def run_gate(original: Split, attacked: Split, scorer: Optional[NLIScorer], nli_threshold: float = 0.5,
             anchor_threshold: float = 0.8, chunk: int = 200, source_max_words: Optional[int] = None) -> list[dict]:
    """``source_max_words``: the generator saw only this many words of each article, so the rewrite is
    compared with that prefix rather than penalised for facts it was never shown."""
    assert original.ids == attacked.ids
    rows = []
    for i, o, a, y in zip(original.ids, original.texts, attacked.texts, original.labels):
        if source_max_words:
            o = " ".join(o.split()[:source_max_words])
        fwd = scorer.directional(o, a, chunk) if scorer else np.nan
        bwd = scorer.directional(a, o, chunk) if scorer else np.nan
        rec = anchor_recall(o, a)
        nli_ok = (fwd >= nli_threshold and bwd >= nli_threshold) if scorer else True
        rows.append({"id": i, "label": int(y), "entail_orig_to_rewrite": fwd, "entail_rewrite_to_orig": bwd,
                     "anchor_recall": rec, "passed": bool(nli_ok and rec >= anchor_threshold)})
    return rows


# ---------------------------------------------------------------------- human audit
def export_audit_sample(gate_rows: list[dict], original: Split, attacked: Split, path: str | Path, n: int = 100, seed: int = 0) -> list[str]:
    """Write a CSV for two annotators: equal numbers of real and fake gate-passing rewrites."""
    rng = np.random.default_rng(seed)
    passed = [r for r in gate_rows if r["passed"]]
    chosen = []
    for lab in (0, 1):
        pool = [r["id"] for r in passed if r["label"] == lab]
        chosen += rng.choice(pool, size=min(n // 2, len(pool)), replace=False).tolist()
    text_o, text_a = dict(zip(original.ids, original.texts)), dict(zip(attacked.ids, attacked.texts))
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "original", "rewrite", "claims_preserved_annotator_1", "claims_preserved_annotator_2",
                    "change_type_if_not", "resolved"])
        for i in chosen:
            w.writerow([i, text_o[i], text_a[i], "", "", "", ""])
    return chosen


def read_audit(path: str | Path) -> dict:
    """Read a completed audit CSV (1/0 judgements). Returns kappa and the audited precision."""
    from sklearn.metrics import cohen_kappa_score

    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["claims_preserved_annotator_1"] != ""]
    a1 = [int(r["claims_preserved_annotator_1"]) for r in rows]
    a2 = [int(r["claims_preserved_annotator_2"]) for r in rows]
    resolved = [int(r["resolved"]) if r["resolved"] != "" else (x if x == y else None) for r, x, y in zip(rows, a1, a2)]
    if any(v is None for v in resolved):
        raise ValueError("fill the 'resolved' column for every disagreement")
    return {"n": len(rows), "kappa": float(cohen_kappa_score(a1, a2)) if len(set(a1 + a2)) > 1 else float("nan"),
            "precision": float(np.mean(resolved)) if rows else float("nan")}


def vasr(y: np.ndarray, pred_orig: np.ndarray, pred_adv: np.ndarray, passed: np.ndarray, precision: float = 1.0) -> dict:
    """Validity-aware attack success: flips among originally-correct, gate-passing items, x audited precision."""
    y, po, pa, ok = map(np.asarray, (y, pred_orig, pred_adv, passed))
    correct = po == y
    raw = float(np.mean(pa[correct] != y[correct])) if correct.any() else float("nan")
    m = correct & ok
    gated = float(np.mean(pa[m] != y[m])) if m.any() else float("nan")
    return {"asr": raw, "asr_gated": gated, "vasr": gated * precision, "gate_pass_rate": float(ok.mean())}
