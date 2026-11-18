# When Real News Looks Fake

Code and data for *When Real News Looks Fake: LLM Style Attacks Can Reverse the Predictions of Fake News
Detectors*.

* **`paper_data/`**: the results behind every number, table and figure, and a script that recomputes them in about
  a minute without a GPU. See [`paper_data/README.md`](paper_data/README.md).
* **`robustlens/`, `scripts/`**: the pipeline that produced those results (data preparation, tone rewrites,
  validity checks, training, scoring, probes, attributions).

## Reproduce the paper's numbers

```bash
pip install numpy pandas scipy scikit-learn
python paper_data/reproduce_numbers.py      # compare with paper_data/expected_output.txt
```

## Run the full pipeline

### Install

```bash
pip install -r requirements.txt          # Python >= 3.8 (3.10+ recommended); a GPU is needed for training and scoring
pip install "vllm>=0.6.3"                # optional, faster tone rewrites (Python 3.8: "vllm==0.6.3.post1")
```

Put the SheepDog release (Wu et al., KDD 2024) next to this folder, or edit `paths` in `configs/default.yaml`:

```
SheepDog-main/data/{news_articles,reframings,adversarial_test}/
```

Llama 3 and Gemma 2 are gated models and need `huggingface-cli login`.

### Steps

`bash scripts/run_all.sh` runs every step in order. Each step can also be run on its own.

| Script | What it does | Output | Paper |
| --- | --- | --- | --- |
| `01_prepare_data.py` | SheepDog pickles → JSONL (label 0 = real, 1 = fake); fixed class-balanced subset of 500 test articles per dataset for the attributions | `data/processed/` | §2 |
| `02_generate_sentiment.py` | Tone rewrites with Gemma-2-9B-Instruct (greedy). `sentiment_adv` = tone attack (fake → positive, real → negative); `sentiment_ctl` = opposite pairing | `data/processed/<ds>/attacks/` | §2, §3.6 |
| `03_validity_and_controls.py` | Validity check of every rewrite (NLI in both directions, share of numbers, dates and named entities kept) and random-edit controls that change the same share of words | `results/gate/`, `data/processed/<ds>/controls/` | §3.5, §3.7 |
| `04_train_encoders.py` | One encoder detector: BERT, RoBERTa or GPT-2 × strategy A (FT), B or C (FT+Aug) × seed | `checkpoints/` | §2 |
| `05_score.py` | Logits on the original, attacked and control sets for encoders and zero-shot LLMs; hidden states for the probes | `results/predictions/`, `results/states/` | §3.1–3.2 |
| `06_stage1_behaviour.py` | AUROC with bootstrap interval, accuracy per class, share predicted fake, balanced accuracy at the model's own and the best threshold; `zone = "inversion"` marks a reversed case | `results/stage1/` | §3.1, §3.2, §3.6, §3.7 |
| `07_stage2_representation.py` | Linear probe per layer on original and attacked articles, shuffled-label control | `results/stage2/` | §3.3 |
| `08_stage3_attribution.py` | Integrated Gradients and occlusion for the encoders on publisher sets A/B: completeness error, attribution stability, style reliance | `results/stage3/` | §3.4 |

`06_stage1_behaviour.py --gate nli|full` repeats Stage 1 on the rewrites that pass the NLI check or the full check
(§3.5); `paper_data/reproduce_numbers.py` computes the same numbers from the predictions and the gate files.

File names use `publisher_A`–`publisher_D` for the SheepDog test sets A–D and `sentiment_adv` / `sentiment_ctl` for
the tone attack and its control pairing.

### Smoke test (CPU)

`bash tests/run_smoke.sh` runs every step on PolitiFact with tiny models (2-layer BERT, tiny random RoBERTa, GPT-2
and Gemma-2, Qwen2.5-0.5B-Instruct on 20 articles). It checks the code paths, not the results. Unit tests:
`pytest tests/test_units.py`.

## Implementation notes

* **Publisher sets A–D are a 2×2 design.** Real articles are rewritten in the style of the National Enquirer (A, B)
  or The Sun (C, D); fake articles in the style of CNN (A, C) or The New York Times (B, D). The same rewritten real
  article appears in A and B, so the four sets are not independent attacks.
* **LLM detectors** receive the article (up to 1,536 tokens) and two options, A and B; the order of "real" and
  "fake" is fixed per article id. The score is the logit of each option letter, normalised over {A, B}. Logits are
  computed at the last position only (`logits_to_keep`), which saves memory.
* **Probes** read the `[CLS]` token (BERT, RoBERTa) or the last token (GPT-2, LLMs). Layer index l means
  `hidden_states[l]` (0 = embeddings). LLMs are probed at every second layer.
* **Strategy C** uses SheepDog's consistency loss (λ = 1) without its auxiliary veracity-attribution head.
* **One input length.** Encoder scores, hidden states and attributions all use the training length
  (`encoders.train.max_len` = 256).
* **Style reliance** is the share of absolute attribution on the words the rewrite changed, minus the share on the
  same number of unchanged words with similar corpus frequency.
* **IG completeness.** Each article records `ig_rel_delta`, the completeness error of Integrated Gradients relative
  to the logit difference (floor of 1 logit). The paper keeps articles with `ig_rel_delta < 0.25`.
* **Tone rewrites are compared with the first `generation.max_words` words** of each article, the part the
  generator saw.
* **Sampling.** Every `--limit N` option takes a class-stratified sample: the SheepDog test files are sorted by
  label.
* **Python 3.8.** Everything runs on 3.8 with torch 2.4 and transformers 4.46. For tone rewrites, either install
  `vllm==0.6.3.post1` in a separate environment or use transformers: `GEN_BACKEND=hf bash scripts/run_all.sh`. A
  vLLM older than 0.5.1 cannot run Gemma-2 and stops with a message saying so.
* **vLLM without the CUDA toolkit.** Recent vLLM compiles FlashInfer's sampling kernel at start-up, which needs
  `nvcc`. Step 02 decodes greedily and does not need that sampler, so it sets `VLLM_USE_FLASHINFER_SAMPLER=0`
  (export it as 1 to override).

## Compute (rough, single A100)

27 encoders × 3 datasets; GossipCop and LUN runs take about 20–40 minutes each (strategy C about 3× strategy A).
LLM scoring with hidden states: 1–2 hours per model and dataset. Stage 3: minutes per encoder.
