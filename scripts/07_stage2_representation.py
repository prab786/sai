"""Stage 2 for every detector with stored states: probe retention, crossing rate, layer drift vs random edits."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustlens.registry import control_name  # noqa: E402
from robustlens.stages.representation import report  # noqa: E402
from robustlens.utils import load_config, load_states, log, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--datasets", nargs="*")
ap.add_argument("--detectors", nargs="*")
args = ap.parse_args()
cfg = load_config(args.config)
R = cfg["representation"]
root = Path(cfg["paths"]["results"]) / "states"

for ds in args.datasets or cfg["datasets"]:
    for det_dir in sorted((root / ds).glob("*")):
        if args.detectors and det_dir.name not in args.detectors:
            continue
        if not (det_dir / "train.npz").exists():
            continue
        tr, _ = load_states(det_dir / "train.npz")
        ytr = np.load(det_dir / "train_labels.npy")
        te, te_ids = load_states(det_dir / "test.npz")
        yte = np.load(Path(cfg["paths"]["results"]) / "predictions" / ds / det_dir.name / "test.npz")["labels"]
        for f in sorted(det_dir.glob("*.npz")):
            if f.stem in ("train", "test") or f.stem.startswith("random_"):
                continue
            adv, adv_ids = load_states(f)
            assert adv_ids == te_ids
            ctrl_f = det_dir / f"{control_name(f.stem)}.npz"
            ctrl = load_states(ctrl_f)[0] if ctrl_f.exists() else None
            r = report(tr, ytr, te, adv, yte, ctrl, R["probe_C"], R["min_clean_acc"], seed=0)
            save_json(r, Path(cfg["paths"]["results"]) / "stage2" / ds / det_dir.name / f"{f.stem}.json")
            log.info("%s %s %s: PR %.2f, crossing %.2f, depth %s", ds, det_dir.name, f.stem,
                     r["probe_retention"], r["crossing_rate"], r["divergence_depth"])
