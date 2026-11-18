"""Score one detector on clean, attacked and control sets, and store hidden states for Stage 2.

Writes results/predictions/<ds>/<detector>/<set>.npz   (ids, logits[real, fake])
       results/states/<ds>/<detector>/<set>.npz        (read-out state per stored layer)
Sets: test, every attack set of every family (+ controls), and the matched random-edit controls.
Training states (for the probes) use a class-stratified sample of at most --probe-train articles.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.data import Split, available_attacks, family_sets, limit_split, load_attack, load_split, stratified_subset_ids  # noqa: E402
from robustlens.detectors.base import resolve_layers  # noqa: E402
from robustlens.registry import control_name, is_llm, load_detector, result_path  # noqa: E402
from robustlens.utils import load_config, log, read_jsonl, save_states  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--dataset", required=True)
ap.add_argument("--detector", required=True)
ap.add_argument("--probe-train", type=int, default=2000)
ap.add_argument("--limit", type=int, help="first N test articles only (smoke tests)")
ap.add_argument("--no-states", action="store_true")
ap.add_argument("--device")
ap.add_argument("--batch-size", type=int)
args = ap.parse_args()
cfg = load_config(args.config)
P, ds, name = cfg["paths"], args.dataset, args.detector
det = load_detector(name, ds, cfg, args.device, args.batch_size)
llm = is_llm(name, cfg)
layers = resolve_layers(cfg["llms"]["layers"] if llm else "all", det.n_layers)
# states use the same input length as the logits (encoders: the training length), so Stages 1-4 see one input
state_kw = {}

test = load_split(P["processed"], ds, "test")
test = limit_split(test, args.limit)
sets = {"test": test}  # set name -> Split
have = set(available_attacks(P["processed"], ds))
for fam, names in family_sets(cfg, include_controls=True).items():
    for a in names:
        if a in have:
            sets[a] = load_attack(P["processed"], ds, a).select_ids(test.ids)
        ctrl = Path(P["processed"]) / ds / "controls" / f"{control_name(a)}.jsonl"
        if ctrl.exists():
            rows = {r["id"]: r for r in read_jsonl(ctrl)}
            if all(i in rows for i in test.ids):
                sets[control_name(a)] = Split(test.ids, [rows[i]["text"] for i in test.ids], test.labels, control_name(a))

for set_name, split in sets.items():
    logits = det.class_logits(split.texts, split.ids)
    out = result_path(cfg, "predictions", ds, name, set_name, "npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(split.ids), logits=logits, labels=split.labels)
    acc = float(np.mean((logits[:, 1] > logits[:, 0]).astype(int) == split.labels))
    log.info("%s %s %s: accuracy %.3f", ds, name, set_name, acc)
    if not args.no_states:
        st = det.hidden_states(split.texts, split.ids, layers=layers, pooling="readout", **state_kw)
        save_states(st, result_path(cfg, "states", ds, name, set_name, "npz"), split.ids)

if not args.no_states:
    train = load_split(P["processed"], ds, "train")
    train = train.select_ids(stratified_subset_ids(train, args.probe_train, seed=0))
    st = det.hidden_states(train.texts, train.ids, layers=layers, pooling="readout", **state_kw)
    save_states(st, result_path(cfg, "states", ds, name, "train", "npz"), train.ids)
    np.save(result_path(cfg, "states", ds, name, "train_labels", "npy"), train.labels)
