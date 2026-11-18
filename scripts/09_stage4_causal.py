"""Stage 4 on the fixed subset: LEACE erasure of attack-free concepts, and controlled patching.

Concepts are fitted on training data only, never on the attacked test articles:
  style      original training articles vs their unreliable (sensational / emotional) reframings
  sentiment  training articles in the top vs bottom third of NRC polarity ((#positive - #negative) / #words)
Each concept is fitted on token-level states (up to --tokens-per-text positions per article, every token
labelled with its article's concept), because the eraser is applied at every position of the layer.
Erasure layer (causal.erase_layer): "two_thirds" (default) = two thirds of depth; an integer = that layer;
"auto" = the Stage 2 divergence depth (descriptive only: on the full run it is almost always the first layer).
Writes results/stage4/<ds>/<detector>/<attack>.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.data import UNRELIABLE_STYLES, family_sets, limit_split, load_attack, load_reframing, load_split, stratified_subset_ids  # noqa: E402
from robustlens.detectors.base import WORD_RE, resolve_layers  # noqa: E402
from robustlens.registry import is_llm, load_detector  # noqa: E402
from robustlens.stages.attribution import load_nrc  # noqa: E402
from robustlens.stages.causal import erasure_effect, patching, token_states  # noqa: E402
from robustlens.utils import load_config, load_json, log, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--dataset", required=True)
ap.add_argument("--detector", required=True)
ap.add_argument("--n-concept", type=int, default=1000, help="training articles used to fit each concept")
ap.add_argument("--tokens-per-text", type=int, default=32, help="positions sampled per article to fit the eraser")
ap.add_argument("--limit", type=int)
ap.add_argument("--no-patching", action="store_true")
ap.add_argument("--device")
args = ap.parse_args()
cfg = load_config(args.config)
P, ds, name = cfg["paths"], args.dataset, args.detector
det = load_detector(name, ds, cfg, args.device)
llm = is_llm(name, cfg)
res = Path(P["results"])

ids = load_json(Path(P["processed"]) / ds / "subset_ids.json")
test = limit_split(load_split(P["processed"], ds, "test").select_ids(ids), args.limit)
ids = test.ids
train = load_split(P["processed"], ds, "train")
cidx = [train.ids.index(i) for i in stratified_subset_ids(train, args.n_concept, seed=1)]


def erase_layer(first_attack: str) -> int:
    spec = cfg["causal"].get("erase_layer", "two_thirds")
    two_thirds = max(1, round(2 * det.n_layers / 3))
    if spec in ("two_thirds", None):
        return two_thirds
    if spec == "auto":
        s2 = res / "stage2" / ds / name / f"{first_attack}.json"
        depth = load_json(s2).get("divergence_depth") if s2.exists() else None
        return int(depth) if depth else two_thirds
    return int(spec)


def concept(kind: str, layer: int):
    if kind == "style":
        unrel = {s: load_reframing(P["processed"], ds, s) for s in UNRELIABLE_STYLES}
        rng = np.random.default_rng(0)
        texts = [train.texts[i] for i in cidx] + [unrel[UNRELIABLE_STYLES[rng.integers(2)]][i] for i in cidx]
        cids = [train.ids[i] for i in cidx] * 2
        z = np.array([0] * len(cidx) + [1] * len(cidx))
    else:
        nrc_pos, nrc_neg = load_nrc(P["nrc_lexicon"], {"positive"}), load_nrc(P["nrc_lexicon"], {"negative"})
        pol = []
        for i in cidx:
            w = [x.lower() for x in WORD_RE.findall(train.texts[i])]
            pol.append((sum(x in nrc_pos for x in w) - sum(x in nrc_neg for x in w)) / max(1, len(w)))
        pol = np.array(pol)
        lo, hi = np.quantile(pol, [1 / 3, 2 / 3])
        keep = np.flatnonzero((pol <= lo) | (pol >= hi))
        texts = [train.texts[cidx[k]] for k in keep]
        cids = [train.ids[cidx[k]] for k in keep]
        z = (pol[keep] >= hi).astype(int)
    X, owner = token_states(det, texts, cids, layer, per_text=args.tokens_per_text, seed=0)
    return X, z[owner]


for fam, sets in family_sets(cfg).items():
    kind = "sentiment" if fam.startswith("sentiment") else "style"
    if kind == "sentiment" and not Path(P["nrc_lexicon"]).exists():
        log.warning("NRC lexicon missing: sentiment erasure skipped")
        continue
    layer = erase_layer(sets[0])
    X, z = concept(kind, layer)
    for attack in sets:
        try:
            adv = load_attack(P["processed"], ds, attack).select_ids(ids)
        except FileNotFoundError:
            continue
        out = {"family": fam, "concept": kind,
               "erasure": erasure_effect(det, layer, X, z, test.texts, adv.texts, test.labels, test.ids)}
        if not args.no_patching:
            players = resolve_layers("every_2" if llm else "all", det.n_layers)
            out["patching"] = patching(det, [l for l in players if l >= 1], test.texts, adv.texts, test.labels, test.ids)
        save_json(out, res / "stage4" / ds / name / f"{attack}.json")
        e = out["erasure"]
        log.info("%s %s %s: erasing %s at layer %d: %+.3f (random %+.3f), clean cost %.3f",
                 ds, name, attack, kind, layer, e["erasure_gain"], e["random_gain"], e["clean_cost"])
