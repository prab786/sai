"""Recompute Stage 3 measures from saved word attributions, without running IG again.

Use after adding the NRC lexicon, changing the cue definitions, or filtering by IG completeness. Reads
results/stage3/<ds>/<detector>/attributions/*.jsonl.gz (written by 08_stage3_attribution.py) and rewrites
clean.jsonl, <attack>.jsonl and summary.json next to them. Flip labels and explainer agreement are kept.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.data import load_split  # noqa: E402
from robustlens.stages import attribution as A  # noqa: E402
from robustlens.utils import load_config, log, read_jsonl, save_json, write_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--datasets", nargs="*")
ap.add_argument("--detectors", nargs="*")
args = ap.parse_args()
cfg = load_config(args.config)
P, AC = cfg["paths"], cfg["attribution"]
nrc_path = Path(P["nrc_lexicon"])
if not nrc_path.exists():
    sys.exit(f"NRC lexicon not found at {nrc_path}")
nrc = A.load_nrc(nrc_path)


def read_attr(f: Path) -> dict:
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for k in [k for k in rows[0] if k not in ("id", "label")]:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and r.get(k) is not None]
        out[k] = float(np.nanmean(vals)) if vals else float("nan")
    rel = [r["ig_rel_delta"] for r in rows if r.get("ig_rel_delta") is not None]
    if rel:
        out["ig_rel_delta_median"] = float(np.median(rel))
        out["ig_rel_delta_share_gt_half"] = float(np.mean(np.asarray(rel) > 0.5))
    return out


for ds in args.datasets or cfg["datasets"]:
    freq = A.corpus_frequencies(load_split(P["processed"], ds, "train").texts)
    for det_dir in sorted((Path(P["results"]) / "stage3" / ds).glob("*")):
        if args.detectors and det_dir.name not in args.detectors:
            continue
        adir = det_dir / "attributions"
        if not (adir / "clean.jsonl.gz").exists():
            log.warning("%s %s: no saved attributions (rerun 08 to save them)", ds, det_dir.name)
            continue
        rng = np.random.default_rng(0)
        clean = read_attr(adir / "clean.jsonl.gz")
        old = {r["id"]: r for r in read_jsonl(det_dir / "clean.jsonl")}
        rows = []
        for i, a in clean.items():
            r = {k: v for k, v in old.get(i, {"id": i}).items() if not k.startswith("nrc_")}
            rel = A.cue_reliance(a["words"], np.asarray(a["ig"]), A.nrc_mask(a["words"], nrc), freq, rng)
            r.update({f"nrc_{k}": v for k, v in rel.items()})
            rows.append(r)
        write_jsonl(rows, det_dir / "clean.jsonl")
        summary = {"clean": summarise(rows)}
        for f in sorted(adir.glob("*.jsonl.gz")):
            attack = f.name[: -len(".jsonl.gz")]
            if attack == "clean":
                continue
            old = {r["id"]: r for r in read_jsonl(det_dir / f"{attack}.jsonl")}
            rows = []
            for i, a in read_attr(f).items():
                c = clean[i]
                ca, aa = {"words": c["words"], "scores": np.asarray(c["ig"])}, {"words": a["words"], "scores": np.asarray(a["ig"])}
                r = {k: v for k, v in old.get(i, {"id": i}).items()
                     if not k.startswith(("nrc_adv_", "style_", "stab_", "ctrl_stab_"))}
                r.update({f"stab_{k}": v for k, v in A.stability(ca, aa, AC["topk"]).items()})
                if "ctrl_ig" in a:
                    r.update({f"ctrl_stab_{k}": v for k, v in
                              A.stability(ca, {"words": a["ctrl_words"], "scores": np.asarray(a["ctrl_ig"])}, AC["topk"]).items()})
                mask = A.changed_word_mask(c["words"], a["words"])
                r.update({f"style_{k}": v for k, v in A.cue_reliance(a["words"], aa["scores"], mask, freq, rng).items()})
                r.update({f"nrc_adv_{k}": v for k, v in
                          A.cue_reliance(a["words"], aa["scores"], A.nrc_mask(a["words"], nrc), freq, rng).items()})
                rows.append(r)
            write_jsonl(rows, det_dir / f"{attack}.jsonl")
            summary[attack] = summarise(rows)
        save_json(summary, det_dir / "summary.json")
        log.info("%s %s: rescored %d sets", ds, det_dir.name, len(summary) - 1)
