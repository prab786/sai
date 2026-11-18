"""Unit tests for the measures and rules (no models needed). Run: pytest tests/test_units.py"""
import numpy as np
import pytest

from robustlens.attacks.validity import anchor_recall, fact_anchors, vasr
from robustlens.data import Split, pair_sentiment, stratified_subset_ids
from robustlens.detectors.base import aggregate_to_words, resolve_layers, tokens_to_words
from robustlens.diagnosis import zone
from robustlens.stages import attribution as A
from robustlens.stages import behaviour as B
from robustlens.stages import representation as R


def test_per_class_and_flips():
    y = np.array([0, 0, 1, 1])
    p_clean = np.array([0.1, 0.2, 0.9, 0.8])
    p_adv = np.array([0.9, 0.2, 0.9, 0.8])  # one real article flips toward fake
    pc = B.per_class(y, p_adv)
    assert pc["acc_real"] == 0.5 and pc["acc_fake"] == 1.0 and pc["p_pred_fake"] == 0.75
    f = B.flip_table(y, p_clean, p_adv)
    assert f["toward_fake"] == 1 and f["toward_real"] == 0 and f["real_toward_fake"] == 1


def test_ece_perfect_and_overconfident():
    y = np.array([1] * 10 + [0] * 10)
    assert B.ece(y, np.array([1.0] * 10 + [0.0] * 10)) == pytest.approx(0.0)
    assert B.ece(y, np.array([0.0] * 10 + [1.0] * 10)) == pytest.approx(1.0)


def test_signal_retained_and_gap():
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 200)
    p_clean = np.clip(np.r_[rng.normal(0.3, 0.1, 200), rng.normal(0.7, 0.1, 200)], 0, 1)
    p_shift = np.clip(p_clean + 0.35, 0, 1)  # same ranking, threshold moved
    p_inv = 1 - p_clean                      # ranking reversed
    r_shift = B.report(y, p_clean, p_shift, n_boot=0)
    assert r_shift["S"] == pytest.approx(1.0, abs=1e-6) and r_shift["G"] > 0.2
    r_inv = B.report(y, p_clean, p_inv, n_boot=50)
    assert r_inv["S"] < 0 and r_inv["ba_oracle_signflip"] > 0.9


def test_signal_retained_undefined_near_chance():
    assert np.isnan(B.signal_retained(0.3, 0.52))
    assert B.signal_retained(0.75, 1.0) == pytest.approx(0.5)


def test_zone_rules():
    assert zone(1.0, 0.0) == "robust"
    assert zone(0.95, 0.2) == "threshold_shift"
    assert zone(0.4, 0.01) == "signal_erasure"
    assert zone(0.4, 0.1) == "mixed"
    assert zone(-0.5, 0.0, auroc_ci_high=0.3) == "inversion"
    assert zone(-0.1, 0.0, auroc_ci_high=0.55) == "signal_erasure"  # interval includes chance: not an inversion
    assert zone(float("nan"), 0.1) == "undefined"


def test_probe_retention_bounds():
    assert R.probe_retention(0.9, 0.9) == pytest.approx(1.0)
    assert R.probe_retention(0.9, 0.5) == pytest.approx(0.0)
    assert np.isnan(R.probe_retention(0.52, 0.5))


def test_crossing_rate_and_cka():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-1, 0.3, (50, 5)), rng.normal(1, 0.3, (50, 5))]
    y = np.array([0] * 50 + [1] * 50)
    cr, cf = R.class_centroids(X, y)
    same = R.crossing_rate(X, X, y, cr, cf)
    assert same["crossing_rate"] == 0 and same["flip_rate_geom"] == 0
    swapped = R.crossing_rate(X, -X, y, cr, cf)
    assert swapped["crossing_rate"] == pytest.approx(1.0)
    assert R.linear_cka(X, X) == pytest.approx(1.0)
    assert R.linear_cka(X, 3 * X + 1) == pytest.approx(1.0)


def test_random_edit_matches_ratio():
    rng = np.random.default_rng(0)
    text = " ".join(f"w{i}" for i in range(200))
    edited = R.random_edit(text, 0.3, rng)
    assert 0.15 < R.edit_ratio(text, edited) < 0.45


def test_tokens_to_words_with_prompt_region():
    full = "PROMPT: Hello world, 2024. END"
    region = (8, 26)  # "Hello world, 2024."
    offsets = [(0, 6), (6, 7), (7, 13), (13, 19), (19, 20), (20, 25), (25, 26), (26, 30)]
    words, t2w = tokens_to_words(offsets, region, full[8:26])
    assert words == ["Hello", "world", ",", "2024", "."]
    assert t2w == [-1, -1, 0, 1, 2, 3, 4, -1]
    assert aggregate_to_words(np.ones(8), t2w, len(words)).tolist() == [1, 1, 1, 1, 1]


def test_tokens_to_words_drops_truncated_words():
    text = "alpha beta gamma delta"
    words, t2w = tokens_to_words([(0, 0), (0, 5), (6, 10), (0, 0)], (0, len(text)), text)  # truncated after "beta"
    assert words == ["alpha", "beta"] and t2w == [-1, 0, 1, -1]


def test_resolve_layers():
    assert resolve_layers("every_2", 5) == [0, 2, 4, 5]
    assert resolve_layers("all", 2) == [0, 1, 2]


def test_pair_sentiment_and_subset():
    ids = [f"t{i}" for i in range(6)]
    y = np.array([0, 1, 0, 1, 0, 1])
    pos = Split(ids, [f"pos{i}" for i in range(6)], y)
    neg = Split(ids, [f"neg{i}" for i in range(6)], y)
    adv = pair_sentiment(pos, neg, adversarial=True)
    assert adv.texts == ["neg0", "pos1", "neg2", "pos3", "neg4", "pos5"]  # fake -> positive, real -> negative
    ctl = pair_sentiment(pos, neg, adversarial=False)
    assert ctl.texts[1] == "neg1"
    big = Split([f"x{i}" for i in range(1000)], ["t"] * 1000, [0] * 600 + [1] * 400)
    sub = big.select_ids(stratified_subset_ids(big, 500, 0))
    assert len(sub) == 500 and sub.labels.sum() == 200


def test_anchors_and_vasr():
    a = fact_anchors("On 3 March 2020, Senator Jane Doe said taxes rose 12%. However, critics disagreed.")
    assert {"3", "march", "2020", "12%", "senator jane doe"} <= a and "however" not in a
    assert anchor_recall("Jane Doe paid 5 dollars.", "SHOCKING: Jane Doe paid 5 dollars!") == 1.0
    r = vasr(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 1]), np.array([0, 1, 1, 1]), np.array([True, True, True, False]), 0.8)
    assert r["asr"] == pytest.approx(2 / 3) and r["vasr"] == pytest.approx(2 / 3 * 0.8)


def test_cue_reliance_and_changed_words():
    words = ["the", "terrible", "senator", "said", "great", "things", "today", "here"]
    scores = np.array([0.0, 5.0, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1])
    mask = A.nrc_mask(words, {"terrible", "great"})
    r = A.cue_reliance(words, scores, mask, {}, np.random.default_rng(0))
    assert r["share"] > 0.9 and r["reliance"] > 0.8
    changed = A.changed_word_mask(["the", "senator", "said"], ["SHOCKING", "the", "senator", "said"])
    assert changed.tolist() == [True, False, False, False]
    st = A.stability({"words": words, "scores": scores}, {"words": words, "scores": scores})
    assert st["spearman"] == pytest.approx(1.0) and st["topk_jaccard"] == 1.0


# ---------------------------------------------------------------------- regression tests for the recheck fixes
def test_anchor_whole_token_match():
    assert anchor_recall("Taxes rose 3 points.", "Taxes rose 3 points!") == 1.0
    assert anchor_recall("Taxes rose 3 points.", "In 2023 taxes rose.") == 0.0  # "3" must not match inside "2023"


def test_gate_compares_with_generator_prefix():
    from robustlens.attacks.validity import run_gate

    orig = Split(["a"], ["Jane Doe spoke. " * 3 + "Later, John Roe arrived in Paris."], [1])
    adv = Split(["a"], ["Jane Doe spoke. " * 3], [1])
    full = run_gate(orig, adv, None)[0]["anchor_recall"]
    prefix = run_gate(orig, adv, None, source_max_words=9)[0]["anchor_recall"]
    assert full < 1.0 and prefix == 1.0


def test_save_states_keeps_large_values(tmp_path):
    from robustlens.utils import load_states, save_states

    big = {3: np.array([[1e5, -2.0]], dtype=np.float32), 4: np.array([[0.5, 1.0]], dtype=np.float32)}
    save_states(big, tmp_path / "s.npz", ["x"])
    st, ids = load_states(tmp_path / "s.npz")
    assert np.isfinite(st[3]).all() and st[3][0, 0] == pytest.approx(1e5) and ids == ["x"]


def test_permutation_keeps_detector_cells_together():
    import pandas as pd
    from robustlens.analysis import rq

    rng = np.random.default_rng(0)
    rows = []
    for ds in ("d1", "d2"):
        for u in range(12):
            margin = rng.normal()
            for fam in ("publisher", "sentiment"):
                rows.append({"dataset": ds, "detector_base": f"m{u}", "family": fam, "clean_f1": 0.8 + 0.01 * rng.normal(),
                             "conf_margin_clean": 0.3 + 0.01 * rng.normal(), "probe_margin_clean": margin,
                             "f1_drop": 0.5 * margin + 0.05 * rng.normal()})
    cells = pd.DataFrame(rows)
    p_signal = rq.permutation_p(cells, rq.OUTPUT_ONLY, ["probe_margin_clean"], "f1_drop", "family", 199, 0)
    assert p_signal < 0.05
    cells["probe_margin_clean"] = rng.normal(size=len(cells))
    cells["probe_margin_clean"] = cells.groupby(["dataset", "detector_base"])["probe_margin_clean"].transform("first")
    p_noise = rq.permutation_p(cells, rq.OUTPUT_ONLY, ["probe_margin_clean"], "f1_drop", "family", 199, 0)
    assert p_noise > 0.05


def test_random_eraser_is_centred():
    import torch
    from robustlens.stages.causal import random_subspace_eraser

    mu = np.full(8, 5.0, dtype=np.float32)
    erase = random_subspace_eraser(8, 1, seed=0, mean=mu)
    x = torch.tensor(mu).unsqueeze(0)
    assert torch.allclose(erase(x), x, atol=1e-5)  # the concept mean is a fixed point, as with LEACE


def _tiny(name):
    from robustlens.detectors.encoder import EncoderDetector

    try:
        return EncoderDetector(name, max_len=32, device="cpu", batch_size=2)
    except Exception as e:  # no network and no cache
        pytest.skip(f"{name} unavailable: {e}")


@pytest.mark.parametrize("name", ["sshleifer/tiny-gpt2", "google/bert_uncased_L-2_H-128_A-2"])
def test_edit_target_matches_hidden_states(name):
    import torch
    from robustlens.stages.causal import edit_block_output

    det = _tiny(name)
    texts = ["The senator said taxes rose.", "A short one."]
    enc = det.batch_inputs(texts)
    with torch.no_grad():
        hs = det.model(**enc, output_hidden_states=True).hidden_states
    for layer in range(1, det.n_layers + 1):
        seen = {}

        def grab(h, seen=seen):
            seen["h"] = h.detach().clone()
            return h
        with edit_block_output(det, layer, grab), torch.no_grad():
            det.model(**enc)
        assert torch.allclose(seen["h"], hs[layer], atol=1e-5), f"layer {layer}"


def test_token_states_cover_all_positions():
    from robustlens.stages.causal import token_states

    det = _tiny("google/bert_uncased_L-2_H-128_A-2")
    texts = ["one two three four five six", "seven eight"]
    X, owner = token_states(det, texts, None, layer=1, per_text=100)
    lengths = det.batch_inputs(texts)["attention_mask"].sum(1).tolist()
    assert X.shape == (sum(lengths), det.model.config.hidden_size)
    assert (owner == 0).sum() == lengths[0] and (owner == 1).sum() == lengths[1]


def test_llm_prompt_finds_stripped_article():
    from robustlens.detectors.llm import ZeroShotLLMDetector

    class Tok:  # template that trims, like Llama-3
        chat_template = "x"
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<user>" + msgs[0]["content"].strip() + "<assistant>"
        def encode(self, t, add_special_tokens=False):
            return t.split()
    det = ZeroShotLLMDetector.__new__(ZeroShotLLMDetector)
    det.tokenizer, det.max_article_tokens, det.mapping_seed = Tok(), 10_000, 0
    prompt, (s, e), art = det.build_prompt("\n  Jane Doe said taxes rose.  \n", "id1")
    assert prompt[s:e] == art == "Jane Doe said taxes rose."


def test_bundled_leace_matches_package():
    import torch
    from robustlens.stages import _leace

    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 16, generator=g)
    z = (x[:, 0] + 0.5 * x[:, 3] + 0.3 * torch.randn(400, generator=g) > 0).long()
    x[:, 5] += 3 * z  # a strong concept direction
    mine = _leace.LeaceEraser.fit(x, z)
    xe = mine(x)
    # the erased features carry no linear information about z: class means coincide
    assert torch.allclose(xe[z == 0].mean(0), xe[z == 1].mean(0), atol=1e-3)
    try:
        from concept_erasure import LeaceEraser
    except ImportError:
        return
    ref = LeaceEraser.fit(x, z)
    assert torch.allclose(ref(x), xe, atol=1e-4)


def test_version_tuple():
    from robustlens.utils import version_tuple

    assert version_tuple("0.6.3.post1") == (0, 6, 3)
    assert version_tuple("4.46.0rc1") == (4, 46, 0)
    assert version_tuple("0.1.7") < (0, 5, 1) <= version_tuple("0.5.1")


def _fake_vllm(monkeypatch, version, engine_params):
    import sys
    import types

    calls = {}
    vllm = types.ModuleType("vllm")
    vllm.__version__ = version

    class SamplingParams:
        def __init__(self, **kw):
            calls["sampling"] = kw

    class Out:
        def __init__(self, t):
            self.outputs = [types.SimpleNamespace(text=" " + t + " ")]

    class LLM:
        def __init__(self, model, **kw):
            calls["llm"] = kw

        def generate(self, prompts, params):
            calls["prompts"] = prompts
            return [Out(f"gen{i}") for i in range(len(prompts))]

    exec("def __init__(self, " + ", ".join(f"{p}=None" for p in ["model"] + engine_params) + "): pass", ns := {})
    engine = types.ModuleType("vllm.engine.arg_utils")
    engine.EngineArgs = type("EngineArgs", (), {"__init__": ns["__init__"]})
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", types.ModuleType("vllm.engine"))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", engine)
    return calls


def test_old_vllm_gets_a_clear_message(monkeypatch):
    from robustlens.attacks import sentiment

    _fake_vllm(monkeypatch, "0.1.7", ["seed"])
    with pytest.raises(RuntimeError, match="too old.*--backend hf"):
        sentiment.generate_vllm([[{"role": "user", "content": "x"}]], "google/gemma-2-9b-it", 16, 0)


def test_vllm_path_uses_generate_and_filters_engine_args(monkeypatch):
    import transformers
    from robustlens.attacks import sentiment

    calls = _fake_vllm(monkeypatch, "0.6.3.post1", ["seed"])  # no max_model_len in this EngineArgs
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)

    class Tok:
        chat_template = "x"
        def apply_chat_template(self, c, tokenize=False, add_generation_prompt=True):
            return "<u>" + c[0]["content"] + "<a>"
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    out = sentiment.generate_vllm([[{"role": "user", "content": "hi"}]] * 2, "m", 16, 0)
    assert out == ["gen0", "gen1"] and calls["llm"] == {"seed": 0} and calls["prompts"][0] == "<u>hi<a>"
    assert calls["sampling"]["temperature"] == 0.0
    import os
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # no nvcc needed for greedy decoding


def test_cka_undefined_for_constant_features():
    X = np.ones((50, 4))
    Y = np.random.default_rng(0).normal(size=(50, 4))
    assert np.isnan(R.linear_cka(X, Y))
    d, u = R.cka_drift_band(X, Y, n_boot=10)
    assert np.isnan(d) and np.isnan(u)
    assert R.divergence_depth({0: float("nan"), 1: 0.2, 2: 0.6}, {0: float("nan"), 1: 0.3, 2: 0.4}) == 2


def test_ig_records_relative_completeness():
    det = _tiny("google/bert_uncased_L-2_H-128_A-2")
    r = A.integrated_gradients(det, "The senator said taxes rose sharply last year.", None, n_steps=64, internal_batch=8)
    assert {"gap", "rel_delta", "convergence_delta"} <= set(r)
    assert r["rel_delta"] == pytest.approx(r["convergence_delta"] / max(abs(r["gap"]), 1.0))
    assert abs(sum(r["scores"]) - r["gap"]) <= r["convergence_delta"] + 1e-4  # word scores keep the token sum
