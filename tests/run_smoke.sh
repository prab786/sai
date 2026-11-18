#!/usr/bin/env bash
# End-to-end smoke test on PolitiFact with tiny models (CPU, ~1-2 h on 2 cores).
# Needs ../SheepDog-main, ../AdSent-main and lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt (see README).
set -euo pipefail
C=tests/smoke.yaml
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore

python scripts/01_prepare_data.py --config $C
python scripts/03_validity_and_controls.py --config $C --no-nli
for m in bert roberta gpt2; do for s in A B C; do for seed in 0 1; do
  python scripts/04_train_encoders.py --config $C --dataset politifact --model $m --strategy $s --seed $seed
  python scripts/05_score.py --config $C --dataset politifact --detector $m-$s-s$seed --probe-train 360
done; done; done
python scripts/05_score.py --config $C --dataset politifact --detector qwen25 --limit 20 --probe-train 40
python scripts/06_stage1_behaviour.py --config $C
python scripts/07_stage2_representation.py --config $C
for d in bert-A-s0 bert-B-s0 bert-C-s0 bert-A-s1 bert-B-s1 bert-C-s1 gpt2-A-s0 roberta-A-s0; do
  python scripts/08_stage3_attribution.py --config $C --dataset politifact --detector $d --limit 10 --sets publisher_A sentiment_adsent_adv
done
python scripts/08_stage3_attribution.py --config $C --dataset politifact --detector qwen25 --limit 4 --sets publisher_A
for d in bert-A-s0 bert-B-s0; do
  python scripts/09_stage4_causal.py --config $C --dataset politifact --detector $d --limit 20 --n-concept 200
done
python scripts/09_stage4_causal.py --config $C --dataset politifact --detector qwen25 --limit 4 --n-concept 20
python scripts/10_analysis.py --config $C --permutations 200
# generator path with a tiny random model (output is gibberish; checks the code only)
python scripts/02_generate_sentiment.py --config $C --datasets politifact --backend hf \
  --model hf-internal-testing/tiny-random-Gemma2ForCausalLM --limit 2 --max-tokens 16
echo "smoke test finished: see smoke/results/analysis/"
