"""Stage 1 for every scored detector x attack set: per-class accuracy, flips, calibration, S, G and zone.

--gate none (default): every rewrite. --gate nli: only rewrites that pass the NLI check both ways.
--gate full: only rewrites that pass NLI and the fact-anchor check. Gated reports go to results/stage1_<gate>/
and record n and the share of fake articles kept, because the gate can change the class mix. Random-edit
controls inherit the gate of their attack set; sets with fewer than --min-n articles (or one class) are skipped.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.detectors.base import softmax_fake  # noqa: E402
from robustlens.diagnosis import zone  # noqa: E402
from robustlens.stages.behaviour import report  # noqa: E402
from robustlens.utils import load_config, log, read_jsonl, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--datasets", nargs="*")
ap.add_argument("--detectors", nargs="*")
ap.add_argument("--gate", choices=["none", "nli", "full"], default="none")
ap.add_argument("--min-n", type=int, default=20)
args = ap.parse_args()
cfg = load_config(args.config)
B, D = cfg["behaviour"], cfg["diagnosis"]
root = Path(cfg["paths"]["results"]) / "predictions"
out_root = Path(cfg["paths"]["results"]) / ("stage1" if args.gate == "none" else f"stage1_{args.gate}")
nli_t = cfg["validity"]["nli_threshold"]


def gate_mask(ds: str, attack: str, ids) -> "np.ndarray | None":
    if args.gate == "none":
        return np.ones(len(ids), bool)
    base = attack[len("random_"):] if attack.startswith("random_") else attack
    f = Path(cfg["paths"]["results"]) / "gate" / ds / f"{base}.jsonl"
    if not f.exists():
        return None
    g = {r["id"]: r for r in read_jsonl(f)}
    if args.gate == "full":
        return np.array([bool(g.get(i, {}).get("passed")) for i in ids])
    ok = lambda r: r.get("entail_orig_to_rewrite") is not None and min(r["entail_orig_to_rewrite"], r["entail_rewrite_to_orig"]) >= nli_t
    return np.array([ok(g[i]) if i in g else False for i in ids])

for ds in args.datasets or cfg["datasets"]:
    for det_dir in sorted((root / ds).glob("*")):
        if args.detectors and det_dir.name not in args.detectors:
            continue
        clean = np.load(det_dir / "test.npz")
        y, p_clean = clean["labels"], softmax_fake(clean["logits"])
        for f in sorted(det_dir.glob("*.npz")):
            if f.stem == "test":
                continue
            adv = np.load(f)
            assert (adv["ids"] == clean["ids"]).all(), f"{f} not aligned with test"
            m = gate_mask(ds, f.stem, clean["ids"].tolist())
            if m is None or m.sum() < args.min_n or len(np.unique(y[m])) < 2:
                continue
            r = report(y[m], p_clean[m], softmax_fake(adv["logits"])[m], B["ece_bins"], B["bootstrap"])
            r["zone"] = zone(r["S"], r["G"], r["adv"]["auroc_ci"][1], D["s_cut"], D["g_cut"])
            r["gate"], r["n"], r["frac_fake"] = args.gate, int(m.sum()), float(y[m].mean())
            save_json(r, out_root / ds / det_dir.name / f"{f.stem}.json")
            log.info("%s %s %s: F1 drop %.3f, S %.2f, G %.3f -> %s", ds, det_dir.name, f.stem, r["f1_drop"], r["S"], r["G"], r["zone"])
