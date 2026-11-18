"""Assemble all stage outputs into cells and produce the paper's tables, the diagnosis map and cards.

Outputs (results/analysis/):
  cells_by_set.csv      one row per dataset x detector x attack set (map, RQ2, cards)
  cells_by_family.csv   seed- and set-averaged rows per dataset x detector x family (RQ1)
  table3_rq1.csv, table4_rq2_cases.csv, rq2_zone_counts.csv, table5_rq3.csv, table6_rq4.csv  (+ .md)
  figure5_diagnosis_map.png, diagnosis_cards.jsonl
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.analysis import rq  # noqa: E402
from robustlens.data import family_sets  # noqa: E402
from robustlens.diagnosis import DiagnosisCard, plot_map, representation_confirms_erasure  # noqa: E402
from robustlens.registry import detector_family, parse_encoder  # noqa: E402
from robustlens.utils import ensure_dir, load_config, load_json, read_jsonl, write_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--permutations", type=int)
ap.add_argument("--max-ig-rel-delta", type=float, default=None,
                help="RQ4: drop articles whose IG completeness error (relative) exceeds this, e.g. 0.5")
args = ap.parse_args()
cfg = load_config(args.config)
res = Path(cfg["paths"]["results"])
out = ensure_dir(res / "analysis")
D = cfg["diagnosis"]
set_family = {a: f for f, sets in family_sets(cfg).items() for a in sets}


def md(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- cells
rows = []
for f in sorted((res / "stage1").glob("*/*/*.json")):
    ds, det, attack = f.parts[-3], f.parts[-2], f.stem
    if attack not in set_family:
        continue  # controls and random edits are not cells
    s1 = load_json(f)
    enc = parse_encoder(det)
    row = {"dataset": ds, "detector": det, "detector_base": f"{enc['model']}-{enc['strategy']}" if enc else det,
           "model": enc["model"] if enc else det, "strategy": enc["strategy"] if enc else "zero-shot",
           "seed": enc["seed"] if enc else 0, "detector_family": detector_family(det, cfg),
           "family": set_family[attack], "attack": attack, "S": s1["S"], "G": s1["G"], "zone": s1["zone"],
           "f1_drop": s1["f1_drop"], "clean_f1": s1["clean"]["macro_f1"], "adv_f1": s1["adv"]["macro_f1"],
           "conf_margin_clean": s1["conf_margin_clean"], "delta_ece": s1["delta_ece"],
           "acc_real_adv": s1["adv"]["acc_real"], "acc_fake_adv": s1["adv"]["acc_fake"],
           "auroc_clean": s1["clean"]["auroc"], "auroc_adv": s1["adv"]["auroc"], "auroc_ci_high": s1["adv"]["auroc_ci"][1]}
    s2 = res / "stage2" / ds / det / f"{attack}.json"
    if s2.exists():
        j = load_json(s2)
        row.update({k: j.get(k) for k in ("probe_retention", "crossing_rate", "probe_margin_clean", "divergence_depth")})
    s3 = res / "stage3" / ds / det / "summary.json"
    if s3.exists():
        j = load_json(s3)
        row["nrc_reliance_clean"] = j["clean"].get("nrc_reliance")
        row["explainer_agreement"] = j["clean"].get("agree_spearman")
        if attack in j:
            row.update({"style_reliance": j[attack].get("style_reliance"), "stab_spearman": j[attack].get("stab_spearman"),
                        "ctrl_stab_spearman": j[attack].get("ctrl_stab_spearman")})
    s4 = res / "stage4" / ds / det / f"{attack}.json"
    if s4.exists():
        e = load_json(s4)["erasure"]
        row.update({"erasure_gain": e["erasure_gain"], "random_gain": e["random_gain"], "clean_cost": e["clean_cost"],
                    "erase_layer": e["layer"], "concept": load_json(s4)["concept"]})
    rows.append(row)

if not rows:
    sys.exit("no Stage 1 results found: run scripts 05 and 06 first")
cells = pd.DataFrame(rows)
cells.to_csv(out / "cells_by_set.csv", index=False)

num = [c for c in cells.columns if pd.api.types.is_numeric_dtype(cells[c]) and c != "seed"]
fam_cells = cells.groupby(["dataset", "detector_base", "detector_family", "model", "strategy", "family"], as_index=False)[num].mean()
fam_cells.to_csv(out / "cells_by_family.csv", index=False)

# ---------------------------------------------------------------------- RQ1
if fam_cells["family"].nunique() == 2:
    internal = [c for c in rq.INTERNAL if c in fam_cells and fam_cells[c].notna().any()]
    t3 = rq.rq1_table(fam_cells, n_perm=args.permutations or cfg["analysis"]["permutations"], internal=internal)
    md(t3, out / "table3_rq1")
else:
    print("RQ1 skipped: needs exactly two attack families with results")

# ---------------------------------------------------------------------- RQ2
md(rq.rq2_zone_counts(cells).reset_index(), out / "rq2_zone_counts")
cases = [
    {"label": "RoBERTa A, LUN, publisher style", "detector": "roberta-A", "dataset": "lun", "attack": "publisher", "expected": ["signal_erasure", "mixed"]},
    {"label": "BERT A, LUN, publisher style", "detector": "bert-A", "dataset": "lun", "attack": "publisher", "expected": ["signal_erasure", "mixed"]},
    *[{"label": f"Qwen2.5-7B, publisher set A, {d}", "detector": "qwen25", "dataset": d, "attack": "publisher_A", "expected": ["inversion"]}
      for d in cfg["datasets"]],
    *[{"label": f"{m} {s}, publisher style, {d}", "detector": f"{m}-{s}", "dataset": d, "attack": "publisher",
       "expected": ["robust", "threshold_shift"]} for m in ("bert", "roberta") for s in ("B", "C") for d in cfg["datasets"]],
]
md(rq.rq2_known_cases(cells, cases), out / "table4_rq2_cases")

# ---------------------------------------------------------------------- RQ3
enc = cells[cells["detector_family"] == "fine-tuned encoder"]
if {"style_reliance", "nrc_reliance_clean"} <= set(enc.columns) and enc["family"].nunique() == 2:
    pub = enc[enc["family"] == "publisher"].groupby(["model", "strategy"]).agg(
        style_reliance=("style_reliance", "mean"), nrc_reliance=("nrc_reliance_clean", "mean"),
        nrc_control_band=("nrc_reliance_clean", "std"), f1_publisher=("adv_f1", "mean"))
    sen = enc[enc["family"] == "sentiment"].groupby(["model", "strategy"]).agg(f1_sentiment=("adv_f1", "mean"))
    rel = pub.join(sen).reset_index().fillna({"nrc_control_band": 0.0})
    md(rq.rq3_cue_substitution(rel), out / "table5_rq3")

# ---------------------------------------------------------------------- RQ4
measures = [c for c in ("S", "G", "f1_drop", "probe_retention", "crossing_rate", "stab_spearman", "style_reliance",
                        "nrc_reliance_clean", "erasure_gain") if c in enc.columns]
if not enc.empty and enc["seed"].nunique() > 1:
    t6 = rq.seed_stability(enc, measures)
    if "explainer_agreement" in cells:
        t6.loc[len(t6)] = {"measure": "explainer agreement (mean Spearman)", "seed_stability_rho": cells["explainer_agreement"].mean(), "n_pairs": np.nan}
    stab_a, stab_r = [], []
    for f in (res / "stage3").glob("*/*/*.jsonl"):
        if f.stem in set_family:
            clean_rel = {}
            if args.max_ig_rel_delta is not None and (f.parent / "clean.jsonl").exists():
                clean_rel = {c["id"]: c.get("ig_rel_delta") for c in read_jsonl(f.parent / "clean.jsonl")}
            for r in read_jsonl(f):
                if args.max_ig_rel_delta is not None:
                    vals = [r.get("ig_rel_delta"), clean_rel.get(r["id"])]
                    if any(v is None or v > args.max_ig_rel_delta for v in vals):
                        continue  # attribution too inaccurate (or completeness not recorded) to compare
                if r.get("flipped") is False and r.get("stab_spearman") is not None:  # None = not scored
                    stab_a.append(r["stab_spearman"])
                if r.get("ctrl_stab_spearman") is not None:
                    stab_r.append(r["ctrl_stab_spearman"])
    nf = rq.nonflip_change(stab_a, stab_r)
    t6.loc[len(t6)] = {"measure": f"non-flipping attacks vs random edits (median stability {nf['median_attack']:.2f} vs {nf['median_random']:.2f}; p = {nf['p']:.3g})",
                       "seed_stability_rho": np.nan, "n_pairs": np.nan}
    md(t6, out / "table6_rq4")

# ---------------------------------------------------------------------- map and cards
plot_cells = cells.dropna(subset=["S", "G"]).assign(family_detector=lambda d: d["detector_family"])
if not plot_cells.empty:
    plot_map(plot_cells, str(out / "figure5_diagnosis_map.png"), D["s_cut"], D["g_cut"])
cards = []
for r in cells.itertuples(index=False):
    r = r._asdict()
    gain, rnd = r.get("erasure_gain"), r.get("random_gain")
    cards.append(DiagnosisCard(
        detector=r["detector"], dataset=r["dataset"], attack=r["attack"], zone=r["zone"], S=r["S"], G=r["G"],
        erasure_confirmed=representation_confirms_erasure(r.get("probe_retention"), D["pr_cut"]),
        damaged_from_layer=None if pd.isna(r.get("divergence_depth", np.nan)) else int(r["divergence_depth"]),
        candidate_cue=r.get("concept") if isinstance(r.get("concept"), str) else None,
        cue_confirmed=None if gain is None or pd.isna(gain) else bool(gain > rnd + 0.02),
    ).to_dict())
write_jsonl(cards, out / "diagnosis_cards.jsonl")
print(f"{len(cells)} cells; outputs in {out}")
