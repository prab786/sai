#!/usr/bin/env bash
# End-to-end smoke test on PolitiFact with tiny models (CPU, ~1-2 h on 2 cores).
# Needs ../SheepDog-main (see README).
set -euo pipefail
C=tests/smoke.yaml
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

python scripts/01_prepare_data.py --config $C
# tone rewrites with a tiny random model (output is gibberish; checks the code path only)
python scripts/02_generate_sentiment.py --config $C --datasets politifact --backend hf \
  --model hf-internal-testing/tiny-random-Gemma2ForCausalLM --limit 90 --max-tokens 16
python scripts/03_validity_and_controls.py --config $C --no-nli
for m in bert roberta gpt2; do for s in A B C; do for seed in 0 1; do
  python scripts/04_train_encoders.py --config $C --dataset politifact --model $m --strategy $s --seed $seed
  python scripts/05_score.py --config $C --dataset politifact --detector $m-$s-s$seed --probe-train 360
done; done; done
python scripts/05_score.py --config $C --dataset politifact --detector qwen25 --limit 20 --probe-train 40
python scripts/06_stage1_behaviour.py --config $C
python scripts/07_stage2_representation.py --config $C
for d in bert-A-s0 bert-B-s0 gpt2-A-s0 roberta-A-s0; do
  python scripts/08_stage3_attribution.py --config $C --dataset politifact --detector $d --limit 10 \
    --sets publisher_A publisher_B
done
echo "smoke test finished: see smoke/results/"
