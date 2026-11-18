"""Generate the held-out sentiment-reframing family (Gemma-2-9B-Instruct by default)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.attacks.sentiment import generate_sentiment_sets  # noqa: E402
from robustlens.utils import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--datasets", nargs="*", help="default: all datasets in the config")
ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
ap.add_argument("--model", help="override the generator model")
ap.add_argument("--limit", type=int, help="only the first N test articles (smoke tests)")
ap.add_argument("--max-tokens", type=int)
ap.add_argument("--batch-size", type=int, default=8, help="hf backend: prompts per generate() call")
ap.add_argument("--overwrite", action="store_true", help="regenerate datasets whose sets already exist")
args = ap.parse_args()
cfg = load_config(args.config)
g = cfg["generation"]
for ds in args.datasets or cfg["datasets"]:
    generate_sentiment_sets(cfg["paths"]["processed"], ds, args.model or g["model"], g["max_words"],
                            args.max_tokens or g["max_tokens"], g["seed"], args.backend, args.limit,
                            overwrite=args.overwrite, batch_size=args.batch_size)
