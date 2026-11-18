# Data for "When Real News Looks Fake: LLM Style Attacks Can Reverse the Predictions of Fake News Detectors"

This folder contains the results behind every number, table and figure in the paper, and a script that
recomputes them. It contains no article text: the rewritten articles belong to the SheepDog release
(Wu et al., KDD 2024), and our tone rewrites are summarised only by counts.

## Reproduce the paper's numbers

```bash
pip install numpy pandas scipy scikit-learn
python reproduce_numbers.py          # about a minute; output grouped by paper section
```

`expected_output.txt` is the output we obtained. Each line maps to a statement in the paper
(section numbers are given in the headers).

## Contents

| Path | What it is | Used in |
|---|---|---|
| `results/stage1/<dataset>/<detector>/<set>.json` | Prediction-level metrics for each detector and test set: macro-F1, AUROC with 95% bootstrap interval, accuracy per class, share predicted fake, balanced accuracy at the model's own and the best threshold | §3.1, §3.2, §3.6, §3.7, Tables 1–3, Fig. 1a |
| `results/stage2/<dataset>/<detector>/<set>.json` | Linear-probe accuracy per layer on original and attacked articles (sets A–D), shuffled-label control | §3.3, Fig. 1b |
| `results/predictions/<dataset>/<detector>/<set>.npz` | Per-article logits (`ids`, `labels`, `logits`; column 1 = fake) for the original test set, sets A–D and the tone attack | §3.5, §3.6 (results under the validity checks) |
| `results/gate/<dataset>/<set>.jsonl` | Per-article validity checks for sets A–D and the tone attack: NLI entailment in both directions, share of numbers, dates and named entities kept, pass/fail | §3.5, §3.6 |
| `results/gate/<dataset>/summary.json` | Pass rates, median share of anchors kept, median share of words changed per set | §3.5, §3.6 |
| `results/stage3/<dataset>/<encoder>/` | Encoder attributions (Integrated Gradients): per-article completeness error and IG–occlusion agreement (`clean.jsonl`), and per attacked article the attribution stability, style reliance and whether the prediction changed (`publisher_A.jsonl`, `publisher_B.jsonl`) | §3.4 |
| `derived/style_markers.json` | Exclamation marks, capitalised words and question marks per 100 words, by dataset, test set and class | §2 |
| `derived/tone_markers.csv` | The same counts for 96 tone rewrites that passed the validity check (no text) | §2 |
| `derived/layer_curves.json` | Probing accuracy by relative layer depth, averaged per detector group (the data of Fig. 1b) | Fig. 1b |
| `compute_style_markers.py` | Recomputes `style_markers.json` from the public SheepDog data | §2 |

## Naming

* Datasets: `politifact`, `gossipcop`, `lun`.
* Detectors: `<encoder>-<strategy>-s<seed>` with encoder `bert`, `roberta`, `gpt2`; strategy `A` = standard
  fine-tuning (FT), `B` = with SheepDog style rewrites, `C` = with rewrites and SheepDog's consistency loss
  (B and C are reported together as FT+Aug); seeds 0–2. Zero-shot LLMs: `llama3` (Llama-3-8B-Instruct),
  `mistral` (Mistral-7B-Instruct-v0.3), `qwen25` (Qwen2.5-7B-Instruct).
* Test sets: `test` (original), `publisher_A`–`publisher_D` (SheepDog style attacks: real articles in the
  style of the National Enquirer in A/B and The Sun in C/D; fake articles in the style of CNN in A/C and The
  New York Times in B/D), `sentiment_adv` (our tone attack: fake positive, real negative). The Stage 1 folder
  also holds `sentiment_ctl` (the opposite pairing) and `random_publisher_*` (random-edit controls that replace
  the same share of words with other words from the same article), used in §3.6 and §3.7.
* In the Stage 1 files, `zone = "inversion"` marks a reversed case (upper end of the AUROC interval below 0.5).

## Notes

* The pipeline that produced these files (training, scoring, probes, attributions) is in the parent folder;
  see `../README.md`.
* Stage 3 files cover the encoders on sets A/B, as in the paper. The paper keeps the 62% of encoder
  articles whose IG completeness error is below a quarter of their logit margin; `reproduce_numbers.py`
  applies the same filter.
* The bootstrap interval in §3.5 is recomputed with 1,000 resamples and seed 0.
