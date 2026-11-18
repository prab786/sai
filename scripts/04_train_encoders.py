"""Fine-tune one encoder detector (model x strategy x seed) on one dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.data import RELIABLE_STYLES, UNRELIABLE_STYLES, load_reframing, load_split, stratified_subset_ids  # noqa: E402
from robustlens.detectors.encoder import train_encoder  # noqa: E402
from robustlens.registry import checkpoint_dir  # noqa: E402
from robustlens.utils import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--dataset", required=True)
ap.add_argument("--model", required=True, help="key in encoders.models, e.g. bert / roberta / gpt2")
ap.add_argument("--strategy", required=True, choices=["A", "B", "C"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--hf-model", help="override the Hugging Face model id (e.g. a tiny model for smoke tests)")
ap.add_argument("--max-steps", type=int, help="cap optimisation steps (smoke tests)")
ap.add_argument("--epochs", type=int)
ap.add_argument("--train-limit", type=int, help="class-stratified sample of N training articles (smoke tests)")
ap.add_argument("--device")
args = ap.parse_args()
cfg = load_config(args.config)
P = cfg["paths"]
tcfg = dict(cfg["encoders"]["train"])
if args.epochs:
    tcfg["epochs"] = args.epochs

train = load_split(P["processed"], args.dataset, "train")
refr = None
if args.strategy in {"B", "C"}:
    refr = {s: load_reframing(P["processed"], args.dataset, s) for s in RELIABLE_STYLES + UNRELIABLE_STYLES}
if args.train_limit:  # never the first N: the releases can be sorted by label
    keep = [train.ids.index(i) for i in stratified_subset_ids(train, args.train_limit, seed=args.seed)]
    train = train.take(keep)
    refr = {k: [v[i] for i in keep] for k, v in refr.items()} if refr else None

name = f"{args.model}-{args.strategy}-s{args.seed}"
det = train_encoder(args.hf_model or cfg["encoders"]["models"][args.model], args.strategy, train, refr, tcfg,
                    args.seed, args.device, name=name, max_steps=args.max_steps)
out = checkpoint_dir(cfg, args.dataset, name)
det.save(out, meta={"dataset": args.dataset, "strategy": args.strategy, "seed": args.seed, "train": tcfg})
print("saved", out)
