#!/usr/bin/env bash
# Full core study. Each step can be run on its own; steps 04-09 loop over datasets and detectors.
# Usage: bash scripts/run_all.sh [config]      (GPU strongly recommended for steps 02, 04, 05, 08, 09)
#   GEN_BACKEND=hf bash scripts/run_all.sh      generate the sentiment family with transformers instead of vLLM
# Step 02 skips datasets whose sentiment sets already exist, so the script can be rerun after an interruption.
set -euo pipefail
CFG=${1:-configs/default.yaml}
export TOKENIZERS_PARALLELISM=false

DATASETS=$(python -c "import yaml,sys; print(' '.join(yaml.safe_load(open('$CFG'))['datasets']))")
ENCODERS=$(python -c "import yaml; c=yaml.safe_load(open('$CFG'))['encoders']; print(' '.join(f'{m}-{s}-s{x}' for m in c['models'] for s in c['strategies'] for x in c['seeds']))")
LLMS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$CFG')).get('llms',{}).get('models',{})))")

python scripts/01_prepare_data.py --config "$CFG"
python scripts/02_generate_sentiment.py --config "$CFG" --backend "${GEN_BACKEND:-vllm}"
python scripts/03_validity_and_controls.py --config "$CFG"

for ds in $DATASETS; do
  for det in $ENCODERS; do
    IFS=- read -r model strategy seed <<< "$det"
    python scripts/04_train_encoders.py --config "$CFG" --dataset "$ds" --model "$model" --strategy "$strategy" --seed "${seed#s}"
    python scripts/05_score.py --config "$CFG" --dataset "$ds" --detector "$det"
  done
  for det in $LLMS; do
    python scripts/05_score.py --config "$CFG" --dataset "$ds" --detector "$det"
  done
done

python scripts/06_stage1_behaviour.py --config "$CFG"
python scripts/07_stage2_representation.py --config "$CFG"

for ds in $DATASETS; do
  for det in $ENCODERS $LLMS; do
    python scripts/08_stage3_attribution.py --config "$CFG" --dataset "$ds" --detector "$det"
    python scripts/09_stage4_causal.py --config "$CFG" --dataset "$ds" --detector "$det"
  done
done

python scripts/10_analysis.py --config "$CFG"
echo "done: see results/analysis/"
