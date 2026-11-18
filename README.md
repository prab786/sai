# robustlens

Code and data for *When Real News Looks Fake: LLM Style Attacks Can Reverse the Predictions of Fake News Detectors*.

**To reproduce the paper's numbers**, see [`paper_data/`](paper_data/README.md): `python paper_data/reproduce_numbers.py`.

The code below is the full analysis pipeline (training, scoring, probes, attributions, causal tests).

The framework runs five stages on every detector x dataset x attack set and places each cell on a diagnosis map:

| Stage | Question | Module | Access |
| --- | --- | --- | --- |
| 0 Validity | Does the rewrite keep the false claim? | `attacks/validity.py` | labels |
| 1 Behaviour | How does the output fail, and for which class? (per-class accuracy, flips, ECE, AUROC, **S**, **G**) | `stages/behaviour.py` | scores |
| 2 Representation | Where is usable separation lost? (probe retention, crossing rate, CKA drift vs random edits) | `stages/representation.py` | weights |
| 3 Attribution | Which cues carry the decision? (Integrated Gradients + occlusion/SHAP, stability, cue reliance) | `stages/attribution.py` | scores/weights |
| 4 Causal | Does removing the cue change the decision? (LEACE erasure, controlled patching) | `stages/causal.py` | weights |

Zones (`diagnosis.py`): robust, threshold shift, signal erasure, mixed, inversion, from
S = (AUROC_adv − 0.5)/(AUROC_clean − 0.5) and G = BA(oracle threshold) − BA(model threshold).

## Install

```bash
pip install -r requirements.txt          # Python >= 3.8 (3.10+ recommended); GPU recommended
pip install "vllm>=0.6.3"                # optional, fast sentiment generation (Python 3.8: "vllm==0.6.3.post1")
```

**Python 3.8.** Everything runs on 3.8 with torch 2.4 and transformers 4.46 (the last releases for 3.8).
Two things differ: `concept-erasure` needs Python 3.10, so Stage 4 uses the bundled LEACE port
(`robustlens/stages/_leace.py`, checked against the package); and the newest vLLM is out of reach, so either
install `vllm==0.6.3.post1` in a separate environment (it pins its own torch) or generate with transformers:
`GEN_BACKEND=hf bash scripts/run_all.sh`. A vLLM older than 0.5.1 cannot run Gemma-2 and stops with a message
saying so. A fresh Python 3.10/3.11 environment avoids all of this.

**vLLM without the CUDA toolkit.** Recent vLLM JIT-compiles FlashInfer's sampling kernel at start-up, which needs
`nvcc`. Step 02 decodes greedily and never uses that sampler, so it sets `VLLM_USE_FLASHINFER_SAMPLER=0` itself
(export it as 1 to override).

Put the two data releases next to this folder (or edit `paths` in `configs/default.yaml`):

```
SheepDog-main/data/{news_articles,reframings,adversarial_test}/   # Wu et al. (2024)
AdSent-main/data/reformed/                                         # optional, Tahmasebi et al. (2026)
```

Download the NRC Emotion Lexicon (free for research; not redistributed here) and place
`NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` in `lexicons/`:
<https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>

Gated models (Llama 3, Gemma 2) need `huggingface-cli login`.

## Run

`bash scripts/run_all.sh` runs everything. The steps, in order:

| Script | What it does | Output |
| --- | --- | --- |
| `01_prepare_data.py` | Pickles → canonical JSONL (label 0 = real, 1 = fake); fixed 500-article subsets | `data/processed/` |
| `02_generate_sentiment.py` | Gemma-2-9B-Instruct positive/negative rewrites; adversarial pairing fake→positive, real→negative; control pairing | `processed/<ds>/attacks/sentiment_*.jsonl` |
| `03_validity_and_controls.py` | Stage 0 gate (bidirectional NLI + fact-anchor check), random-edit controls matched to each article's edit size, audit CSVs | `results/gate/`, `results/audit/` |
| `04_train_encoders.py` | One detector: BERT / RoBERTa / GPT-2 × strategy A, B, C × seed | `checkpoints/<ds>/<model>-<strategy>-s<seed>/` |
| `05_score.py` | Logits on clean, attacked, control and random-edit sets; read-out hidden states | `results/predictions/`, `results/states/` |
| `06_stage1_behaviour.py` | Stage 1 + zone per cell | `results/stage1/` |
| `07_stage2_representation.py` | Stage 2 per layer | `results/stage2/` |
| `08_stage3_attribution.py` | Stage 3 on the subset | `results/stage3/` |
| `09_stage4_causal.py` | Stage 4 on the subset | `results/stage4/` |
| `10_analysis.py` | Paper tables 3–6, Figure 5, diagnosis cards | `results/analysis/` |

After the annotators fill `results/audit/<family>.csv` (1 = claims preserved, 0 = not; `resolved` for disagreements),
`python scripts/03_validity_and_controls.py --read-audit` reports Cohen's κ and the audited precision used in VASR.

### Mapping to the paper draft

| Paper item | Produced by |
| --- | --- |
| Table 2 (validity gate) | `results/gate/<ds>/summary.json`, `results/audit/audit_summary.json` |
| Table 3 (RQ1) | `results/analysis/table3_rq1.csv` |
| Figure 5, Table 4 (RQ2) | `figure5_diagnosis_map.png`, `table4_rq2_cases.csv`, `rq2_zone_counts.csv` |
| Table 5 (RQ3) | `table5_rq3.csv` |
| Table 6 (RQ4) | `table6_rq4.csv` |
| Appendix D | `cells_by_set.csv`, `diagnosis_cards.jsonl` |

## Smoke test (CPU, about an hour)

`tests/smoke.yaml` runs the whole pipeline on PolitiFact with tiny models (2-layer BERT, random tiny RoBERTa/GPT-2,
Qwen2.5-0.5B-Instruct on 20 articles) and AdSent's Llama-3.1 sentiment sets standing in for the Gemma family.
Edit its absolute paths first. Unit tests: `pytest tests/test_units.py`.

Every `--limit N` option takes a class-stratified sample, never the first N articles: the SheepDog test files are
sorted by label (PolitiFact's first 45 test articles are all real).

## Design decisions to know about

* **Generators.** Publisher-style sets come from GPT-3.5 (SheepDog). The held-out sentiment family is generated with
  Gemma-2-9B-Instruct, which is neither GPT-3.5 nor one of the LLM detectors. AdSent's sets (Llama-3.1-8B-Instruct)
  are converted too (`sentiment_adsent_adv/ctl`, PolitiFact and GossipCop only) but are close to the Llama-3-8B
  detector, so they are an optional second generator, not the main family.
* **Validity gate.** The "proposition check" is implemented as a fact-anchor check (numbers, dates, named entities
  kept). It is strict: on PolitiFact the median anchor recall of the publisher-style and AdSent rewrites is about
  0.5 (anchors matched as whole tokens), so at the default threshold 0.8 only a minority pass. Rewrites of our own
  generator are compared with the first `generation.max_words` words, the part the generator saw. Calibrate `validity.anchor_threshold` with the human audit.
* **RQ1 predictors are attack-free.** Probe margin and NRC sentiment reliance are measured on clean articles. Style
  reliance needs the rewrite (words it changed), so it is used for RQ3 but not as an RQ1 predictor.
* **Cue reliance** is the excess attribution mass on cue words over an equal number of frequency-matched non-cue words
  in the same article; negative values mean cue words carry less attribution per word than other words.
* **Strategy C** uses SheepDog's consistency loss (λ = 1) without its auxiliary veracity-attribution head.
* **LEACE concepts** are fitted on training data only (originals vs unreliable reframings; NRC polarity terciles).
  The eraser is applied at every position of the chosen layer, so it is fitted on token-level states sampled from
  every position (`--tokens-per-text`, default 32), each token labelled with its article's concept. The random
  control removes a subspace of the same rank, centred on the same mean.
* **One input length.** Encoder scores, hidden states, attributions, erasure and patching all use the training
  length (`encoders.train.max_len`), so every stage looks at the same input.
* **LLM detectors** use the two-option prompt with the Real/Fake → A/B mapping fixed per article id; scores are the
  answer-letter logits renormalised over {A, B}.
* Layer index l means `hidden_states[l]` (0 = embeddings); hooks edit the output of block l−1. For the last layer
  of GPT-2 and the LLMs, Hugging Face reports the state after the final norm, so the hook goes on that norm instead.
* **Gated Stage 1.** `06_stage1_behaviour.py --gate nli|full` scores only rewrites that pass the NLI check or the full
  gate and writes `results/stage1_<gate>/`, with `n` and `frac_fake` in each report (the gate can change the class mix).
* **Erasure layer.** `causal.erase_layer` defaults to two thirds of depth. The Stage 2 divergence depth (`auto`) is kept
  as a descriptive measure only: on the full run it is the first layer for nearly every attack, including the sentiment
  control. Layer 0 drift is reported as undefined (the read-out position has the same embedding for every article).
* **Publisher sets A-D are a 2x2 design** (real-article rewrite shared by A/B and by C/D; fake-article rewrite shared
  by A/C and by B/D), so they are not four independent attacks.
* **Stage 3 needs the NRC lexicon** and stops without it (`--allow-no-nrc` to override). It saves the word
  attributions (`attributions/*.jsonl.gz`), so `08b_rescore_attributions.py` can recompute every measure without
  running IG again, and `--resume` continues an interrupted run. Each article records `ig_rel_delta`, IG's completeness
  error relative to its margin gap (floor 1 logit); `10_analysis.py --max-ig-rel-delta 0.25` drops articles above it.
  The first full run had a median error of 10–27% of the margin for encoders, about 40% for Llama-3 and Mistral and
  about 100% for Qwen2.5; more steps did not help in our test.
* **LLM logits** are computed at the last position only (`logits_to_keep`), which cuts memory and time in every stage.
* **RQ1 permutation test** shuffles the clean-data predictors between detectors within a dataset, keeping each
  detector's two family cells together.
* **Alignment check.** `03_validity_and_controls.py` warns if an attack set's anchor recall against its own
  originals is not well above the recall against shifted originals.

## Compute (rough, single A100)

Encoders: ~54 runs × 3 datasets; GossipCop/LUN runs take ~20–40 min each (strategy C is ~3× strategy A).
LLM scoring with states: ~1–2 h per model per dataset. Stage 3 on 500 articles: minutes per encoder, hours per LLM;
restrict with `--sets` if needed.
