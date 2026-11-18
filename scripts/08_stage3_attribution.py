"""Stage 3 on the fixed subset: IG (+ occlusion or SHAP), stability, cue reliance.

Per article (clean):    agreement between the two explainers, IG completeness error.
Per article x attack:   attribution stability clean vs attacked, the same for the matched random edit
                        (control), style reliance on words the rewrite changed, and whether the label flipped.
Writes results/stage3/<ds>/<detector>/{clean.jsonl, <attack>.jsonl, summary.json} and the word attributions
themselves in attributions/<set>.jsonl.gz, so new measures can be computed later without running IG again.

--resume keeps finished sets: it reloads the clean attributions and skips attack sets whose output exists.
IG completeness: ``ig_rel_delta`` = |sum of attributions - (f(x) - f(baseline))| / max(|f(x) - f(baseline)|, 1 logit).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.data import family_sets, limit_split, load_attack, load_split  # noqa: E402
from robustlens.registry import control_name, load_detector, result_path  # noqa: E402
from robustlens.stages import attribution as A  # noqa: E402
from robustlens.utils import ensure_dir, load_config, load_json, log, read_jsonl, save_json, write_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--dataset", required=True)
ap.add_argument("--detector", required=True)
ap.add_argument("--sets", nargs="*", help="attack sets (default: every set of every family)")
ap.add_argument("--limit", type=int, help="first N subset articles (smoke tests)")
ap.add_argument("--device")
ap.add_argument("--resume", action="store_true", help="reuse saved clean attributions and skip finished sets")
ap.add_argument("--no-save-attributions", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
P, AC, ds, name = cfg["paths"], cfg["attribution"], args.dataset, args.detector

det = load_detector(name, ds, cfg, args.device)
rng = np.random.default_rng(0)
ids = load_json(Path(P["processed"]) / ds / "subset_ids.json")
test = limit_split(load_split(P["processed"], ds, "test").select_ids(ids), args.limit)
ids = test.ids
freq = A.corpus_frequencies(load_split(P["processed"], ds, "train").texts)
second = AC["second_explainer"]
out_dir = Path(P["results"]) / "stage3" / ds / name
attr_dir = out_dir / "attributions"
save_attr = not args.no_save_attributions


def _pack(a: dict) -> dict:
    return {"words": a["words"], "scores": [float(f"{x:.5g}") for x in np.asarray(a["scores"], dtype=float)]}


def write_attr(rows: list[dict], set_name: str) -> None:
    if save_attr:
        ensure_dir(attr_dir)
        with gzip.open(attr_dir / f"{set_name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def read_attr(set_name: str) -> dict:
    with gzip.open(attr_dir / f"{set_name}.jsonl.gz", "rt", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


# ---- clean
clean_ig: dict = {}
clean_file = out_dir / "clean.jsonl"
if args.resume and clean_file.exists() and (attr_dir / "clean.jsonl.gz").exists():
    saved = read_attr("clean")
    clean_ig = {i: {"words": saved[i]["words"], "scores": np.asarray(saved[i]["ig"])} for i in ids if i in saved}
    clean_rows = [r for r in read_jsonl(clean_file) if r["id"] in clean_ig]
    log.info("resumed clean attributions for %d articles", len(clean_ig))
if len(clean_ig) < len(ids):
    clean_ig, clean_rows, attr_rows = {}, [], []
    for i, t, y in zip(test.ids, test.texts, test.labels):
        ig = A.explain(det, t, i, "ig", AC)
        alt = A.explain(det, t, i, second, AC)
        clean_ig[i] = ig
        row = {"id": i, "label": int(y), "ig_convergence_delta": ig.get("convergence_delta"),
               "ig_rel_delta": ig.get("rel_delta"),
               **{f"agree_{k}": v for k, v in A.stability(ig, alt, AC["topk"]).items()}}
        clean_rows.append(row)
        attr_rows.append({"id": i, "words": ig["words"], "ig": _pack(ig)["scores"], second: _pack(alt)["scores"],
                          "ig_rel_delta": ig.get("rel_delta")})
    write_jsonl(clean_rows, clean_file)
    write_attr(attr_rows, "clean")


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("id", "label")]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and r.get(k) is not None]
        out[k] = float(np.nanmean(vals)) if vals else float("nan")
    rel = [r["ig_rel_delta"] for r in rows if r.get("ig_rel_delta") is not None]
    if rel:
        out["ig_rel_delta_median"] = float(np.median(rel))
        out["ig_rel_delta_share_gt_half"] = float(np.mean(np.asarray(rel) > 0.5))
    return out


summary = {"clean": summarise(clean_rows)}

# ---- attacks
sets = args.sets or [a for v in family_sets(cfg, include_controls=True).values() for a in v]
for attack in sets:
    pred_f = result_path(cfg, "predictions", ds, name, attack, "npz")
    if not pred_f.exists():
        log.warning("no predictions for %s; run 05_score.py first", attack)
        continue
    if args.resume and (out_dir / f"{attack}.jsonl").exists():
        rows = read_jsonl(out_dir / f"{attack}.jsonl")
        summary[attack] = summarise(rows)
        log.info("%s %s %s: kept from an earlier run (--resume)", ds, name, attack)
        continue
    adv = load_attack(P["processed"], ds, attack).select_ids(ids)
    ctrl_rows = {r["id"]: r["text"] for r in read_jsonl(Path(P["processed"]) / ds / "controls" / f"{control_name(attack)}.jsonl")}
    pz, pc = np.load(pred_f), np.load(result_path(cfg, "predictions", ds, name, "test", "npz"))
    pos = {k: n for n, k in enumerate(pz["ids"].tolist())}
    rows, attr_rows = [], []
    for i, t, y in zip(adv.ids, adv.texts, adv.labels):
        ig_a = A.explain(det, t, i, "ig", AC)
        ig_c = clean_ig[i]
        k = pos.get(i)  # None if this article was not scored (only in --limit runs)
        flipped = None if k is None else bool((pz["logits"][k, 1] > pz["logits"][k, 0]) != (pc["logits"][k, 1] > pc["logits"][k, 0]))
        row = {"id": i, "label": int(y), "flipped": flipped, "ig_rel_delta": ig_a.get("rel_delta"),
               **{f"stab_{kk}": v for kk, v in A.stability(ig_c, ig_a, AC["topk"]).items()}}
        rec = {"id": i, "words": ig_a["words"], "ig": _pack(ig_a)["scores"], "ig_rel_delta": ig_a.get("rel_delta")}
        if i in ctrl_rows:
            ig_r = A.explain(det, ctrl_rows[i], i, "ig", AC)
            row.update({f"ctrl_stab_{kk}": v for kk, v in A.stability(ig_c, ig_r, AC["topk"]).items()})
            row["ctrl_ig_rel_delta"] = ig_r.get("rel_delta")
            rec.update(ctrl_words=ig_r["words"], ctrl_ig=_pack(ig_r)["scores"])
        mask = A.changed_word_mask(ig_c["words"], ig_a["words"])
        row.update({f"style_{kk}": v for kk, v in A.cue_reliance(ig_a["words"], ig_a["scores"], mask, freq, rng).items()})
        rows.append(row)
        attr_rows.append(rec)
    write_jsonl(rows, out_dir / f"{attack}.jsonl")
    write_attr(attr_rows, attack)
    summary[attack] = summarise(rows)
    log.info("%s %s %s: stability %.2f (control %.2f), style reliance %.3f, IG rel. error median %.2f", ds, name, attack,
             summary[attack].get("stab_spearman", np.nan), summary[attack].get("ctrl_stab_spearman", np.nan),
             summary[attack].get("style_reliance", np.nan), summary[attack].get("ig_rel_delta_median", np.nan))
save_json(summary, out_dir / "summary.json")
