"""Convert the SheepDog release to canonical JSONL and fix the Stage 3 subset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.data import load_split, prepare, stratified_subset_ids  # noqa: E402
from robustlens.utils import load_config, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
args = ap.parse_args()
cfg = load_config(args.config)
P = cfg["paths"]
prepare(P["sheepdog_data"], P["processed"], cfg["datasets"])
for ds in cfg["datasets"]:
    test = load_split(P["processed"], ds, "test")
    ids = stratified_subset_ids(test, cfg["subset"]["size"], cfg["subset"]["seed"])
    save_json(ids, Path(P["processed"]) / ds / "subset_ids.json")
    print(f"{ds}: subset of {len(ids)} test articles for Stage 3")
