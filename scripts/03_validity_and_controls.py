"""Stage 0 validity gate for every attack set, plus random-edit controls matched to each set.

Writes
  processed/<ds>/controls/random_<attack>.jsonl     random edits matched to each article's edit ratio
  results/gate/<ds>/<attack>.jsonl                   per-article gate scores
  results/gate/<ds>/summary.json                     pass rates
  results/audit/<family>.csv                         sample for the two annotators (fill, then run with --read-audit)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.attacks.validity import NLIScorer, anchor_recall, export_audit_sample, read_audit, run_gate  # noqa: E402
from robustlens.data import Split, available_attacks, family_sets, limit_split, load_attack, load_split  # noqa: E402
from robustlens.registry import control_name  # noqa: E402
from robustlens.stages.representation import matched_random_edits  # noqa: E402
from robustlens.utils import load_config, log, read_jsonl, save_json, write_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--datasets", nargs="*")
ap.add_argument("--no-nli", action="store_true", help="anchor check only (fast; for smoke tests)")
ap.add_argument("--limit", type=int, help="only the first N articles per set (smoke tests)")
ap.add_argument("--read-audit", action="store_true", help="read completed audit CSVs and report kappa / precision")
args = ap.parse_args()
cfg = load_config(args.config)
P, V = cfg["paths"], cfg["validity"]
results = Path(P["results"])

if args.read_audit:
    out = {}
    for f in sorted((results / "audit").glob("*.csv")):
        out[f.stem] = read_audit(f)
        print(f.stem, out[f.stem])
    save_json(out, results / "audit" / "audit_summary.json")
    sys.exit(0)

fams = family_sets(cfg, include_controls=True)
scorer = None if args.no_nli else NLIScorer(V["nli_model"])
for ds in args.datasets or cfg["datasets"]:
    test = load_split(P["processed"], ds, "test")
    have = set(available_attacks(P["processed"], ds))
    summary = {}
    for fam, sets in fams.items():
        for attack in sets:
            if attack not in have:
                log.warning("%s: attack set %s not found, skipped", ds, attack)
                continue
            adv = load_attack(P["processed"], ds, attack)
            orig = test
            if args.limit:
                orig = limit_split(test, args.limit)
                adv = adv.select_ids(orig.ids)
            ctrl_texts, ratios = matched_random_edits(orig.texts, adv.texts, seed=cfg["representation"]["random_edit_seed"])
            write_jsonl([{**r, "edit_ratio": float(e)} for r, e in zip(Split(orig.ids, ctrl_texts, orig.labels).rows(), ratios)],
                        Path(P["processed"]) / ds / "controls" / f"{control_name(attack)}.jsonl")
            # our generator saw at most generation.max_words words per article (AdSent and SheepDog sets: full text)
            own = attack.startswith("sentiment_") and not attack.startswith("sentiment_adsent")
            rows = run_gate(orig, adv, scorer, V["nli_threshold"], V["anchor_threshold"], V["chunk_words"],
                            source_max_words=cfg["generation"]["max_words"] if own else None)
            write_jsonl(rows, results / "gate" / ds / f"{attack}.jsonl")
            # alignment check: a rewrite should share far more anchors with its own original than with a neighbour's
            shifted = float(np.median([anchor_recall(o, a) for o, a in zip(orig.texts, adv.texts[1:] + adv.texts[:1])]))
            own_med = float(np.median([r["anchor_recall"] for r in rows]))
            if len(rows) > 1 and own_med <= 2 * shifted:
                log.warning("%s %s: rewrites look misaligned with the originals (anchor recall %.2f vs %.2f shifted)",
                            ds, attack, own_med, shifted)
            summary[attack] = {"n": len(rows), "pass_rate": float(np.mean([r["passed"] for r in rows])),
                               "anchor_recall_median": own_med, "anchor_recall_shifted_median": shifted,
                               "edit_ratio_median": float(np.median(ratios))}
            log.info("%s %s: %s", ds, attack, summary[attack])
    save_json(summary, results / "gate" / ds / "summary.json")

# audit sample per family, spread over datasets and sets
for fam, sets in family_sets(cfg).items():
    per = max(2, V["audit_per_family"] // max(1, len(sets) * len(args.datasets or cfg["datasets"])))
    rows_all = []
    for ds in args.datasets or cfg["datasets"]:
        for attack in sets:
            g = results / "gate" / ds / f"{attack}.jsonl"
            if g.exists():
                gate_rows = read_jsonl(g)
                ids = [r["id"] for r in gate_rows]
                test = load_split(P["processed"], ds, "test").select_ids(ids)
                adv = load_attack(P["processed"], ds, attack).select_ids(ids)
                tmp = results / "audit" / "parts" / f"{fam}_{ds}_{attack}.csv"
                export_audit_sample(gate_rows, test, adv, tmp, n=per)
                rows_all.append(tmp)
    if rows_all:
        out = results / "audit" / f"{fam}.csv"
        with open(out, "w", encoding="utf-8") as f:
            for k, part in enumerate(rows_all):
                lines = open(part, encoding="utf-8").read().splitlines(keepends=True)
                f.writelines(lines if k == 0 else lines[1:])
        log.info("audit sample for %s written to %s", fam, out)
