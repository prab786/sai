"""Recompute every number reported in the paper from the files in results/ and derived/.

Usage:  python reproduce_numbers.py            (needs numpy, pandas, scipy, scikit-learn)
Output is grouped by paper section; expected_output.txt holds the output we obtained.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
DER = os.path.join(HERE, "derived")
LLMS = ("llama3", "qwen25", "mistral")
DATASETS = ("politifact", "gossipcop", "lun")
N_TEST = {"politifact": 90, "gossipcop": 1584, "lun": 1500}


def head(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def rng_(a, b, d=2):
    return f"{a:.{d}f}-{b:.{d}f}"


# ---------------------------------------------------------------- load Stage 1 (all cases, all rewrites)
rows = []
for f in glob.glob(os.path.join(R, "stage1", "*", "*", "*.json")):
    ds, det, atk = f.split(os.sep)[-3], f.split(os.sep)[-2], os.path.basename(f)[:-5]
    j = json.load(open(f))
    llm = det in LLMS
    rows.append(dict(dataset=ds, detector=det, attack=atk, llm=llm,
                     arch=None if llm else det.split("-")[0], strategy=None if llm else det.split("-")[1],
                     clean_f1=j["clean"]["macro_f1"], adv_f1=j["adv"]["macro_f1"], f1_drop=j["f1_drop"],
                     clean_auroc=j["clean"]["auroc"], adv_auroc=j["adv"]["auroc"], auroc_hi=j["adv"]["auroc_ci"][1],
                     acc_real_c=j["clean"]["acc_real"], acc_fake_c=j["clean"]["acc_fake"],
                     acc_real_a=j["adv"]["acc_real"], acc_fake_a=j["adv"]["acc_fake"],
                     p_fake_c=j["clean"]["p_pred_fake"], p_fake_a=j["adv"]["p_pred_fake"],
                     ba_own=j["ba_own"], ba_oracle=j["ba_oracle"], clean_ba=j["clean"]["balanced_accuracy"],
                     reversed=j["zone"] == "inversion"))
df = pd.DataFrame(rows)
df["group"] = np.where(df.llm, df.detector, np.where(df.strategy == "A", "FT", "FT+Aug"))
pub = df[df.attack.str.match(r"^publisher_[ABCD]$")].copy()
pub["pair"] = np.where(pub.attack.isin(["publisher_A", "publisher_B"]), "A/B", "C/D")
llm_ab = pub[pub.llm & (pub.pair == "A/B")]

# ---------------------------------------------------------------- Section 2
head("Section 2: Experimental setup")
sm = json.load(open(os.path.join(DER, "style_markers.json")))


def sm_mean(s, cls, key):
    return np.mean([sm[f"{d}|{s}|{cls}"][key] for d in DATASETS])


print(f"NE rewrites of real articles: exclamation marks {sm_mean('A','real','excl_o'):.2f} -> {sm_mean('A','real','excl_r'):.2f} per 100 words")
print(f"Sun rewrites of real articles: exclamation marks -> {sm_mean('C','real','excl_r'):.2f}")
print(f"Rewrites of fake articles: exclamation marks {sm_mean('A','fake','excl_o'):.2f} -> "
      f"{min(sm_mean(s,'fake','excl_r') for s in 'AB'):.2f}-{max(sm_mean(s,'fake','excl_r') for s in 'AB'):.2f}")
tm = pd.read_csv(os.path.join(DER, "tone_markers.csv"))
for y, name in [(0, "real"), (1, "fake")]:
    t = tm[tm.label == y]
    print(f"Tone rewrites ({name}, n={len(t)}): exclamation {t.orig_excl_per100.mean():.2f} -> {t.rew_excl_per100.mean():.2f}; "
          f"capitalised {t.orig_caps_per100.mean():.2f} -> {t.rew_caps_per100.mean():.2f}")
enc = pub[~pub.llm]
print("Encoder macro-F1 drop on sets A-D by strategy (A=FT, B=+rewrites, C=+consistency):",
      {k: round(v, 3) for k, v in enc.groupby("strategy").f1_drop.mean().items()})
sh = [v["control_shuffled_acc"] for f in glob.glob(os.path.join(R, "stage2", "*", "*", "publisher_*.json"))
      for k, v in json.load(open(f))["layers"].items()
      if k == str(json.load(open(f))["final_layer"]) and v.get("control_shuffled_acc") is not None]
print(f"Shuffled-label probe accuracy (last layer): mean {np.mean(sh):.3f}")

# attribution quality filter (encoders): IG error below a quarter of the article's |logit margin|
good = set()
for f in glob.glob(os.path.join(R, "stage3", "*", "*", "clean.jsonl")):
    ds, det = f.split(os.sep)[-3], f.split(os.sep)[-2]
    p = np.load(os.path.join(R, "predictions", ds, det, "test.npz"))
    marg = dict(zip(p["ids"].tolist(), np.abs(p["logits"][:, 1] - p["logits"][:, 0])))
    n = 0
    for line in open(f):
        r = json.loads(line)
        n += 1
        m = marg.get(r["id"])
        if m is not None and r["ig_convergence_delta"] / max(m, 1.0) <= 0.25:
            good.add((ds, det, r["id"]))
n_all = sum(sum(1 for _ in open(f)) for f in glob.glob(os.path.join(R, "stage3", "*", "*", "clean.jsonl")))
print(f"Encoder articles kept for attributions: {len(good)/n_all:.2f}")
print(f"Cases: {len(pub)} style-attack, {len(df[df.attack.isin(['sentiment_adv','sentiment_ctl'])])} tone")

# ---------------------------------------------------------------- Section 3.1
head("Section 3.1: Style attacks reverse the predictions (Table 1)")
t1 = []
for g in ["FT+Aug", "FT", "llama3", "mistral", "qwen25"]:
    r = {"detector": g}
    for ds in DATASETS:
        x = pub[(pub.group == g) & (pub.dataset == ds)]
        r[f"{ds[:2]}_orig"] = x.clean_auroc.mean()
        r[f"{ds[:2]}_AB"] = x[x.pair == "A/B"].adv_auroc.mean()
        r[f"{ds[:2]}_CD"] = x[x.pair == "C/D"].adv_auroc.mean()
    t1.append(r)
print(pd.DataFrame(t1).set_index("detector").round(2).to_string())
pairs = llm_ab.groupby(["detector", "dataset"]).reversed.all()
print(f"LLM-dataset pairs reversed under both A and B: {int(pairs.sum())} of {len(pairs)}")
print(f"LLM AUROC on A/B: {llm_ab.clean_auroc.mean():.2f} -> {llm_ab.adv_auroc.mean():.2f}; "
      f"lowest {llm_ab.adv_auroc.min():.2f} ({llm_ab.loc[llm_ab.adv_auroc.idxmin(), ['detector','dataset']].tolist()})")
print(f"Additional reversed LLM cases on C/D: {int(pub[pub.llm & (pub.pair == 'C/D')].reversed.sum())}")
ft_lun = pub[(pub.group == "FT") & (pub.dataset == "lun") & (pub.pair == "A/B")]
print(f"FT on LUN A/B: AUROC {ft_lun.clean_auroc.mean():.2f} -> {ft_lun.adv_auroc.mean():.2f}; reversed {int(ft_lun.reversed.sum())} of {len(ft_lun)}; "
      f"by architecture {ft_lun[ft_lun.reversed].arch.value_counts().to_dict()}")
aug = pub[pub.group == "FT+Aug"]
print(f"FT+Aug: reversed {int(aug.reversed.sum())} of {len(aug)}; lowest AUROC {aug.adv_auroc.min():.2f}; mean loss {(aug.clean_auroc-aug.adv_auroc).mean():.2f}")
br = pub[(pub.group == "FT") & pub.arch.isin(["bert", "roberta"])]
gc = br[br.dataset == "gossipcop"]
lun = br[(br.dataset == "lun") & (br.pair == "A/B")]
print(f"Reproduction (BERT/RoBERTa, FT): GossipCop macro-F1 original {100*gc.clean_f1.mean():.1f}, A/B "
      f"{rng_(100*gc[gc.pair=='A/B'].groupby('arch').adv_f1.mean().min(), 100*gc[gc.pair=='A/B'].groupby('arch').adv_f1.mean().max(), 0)}; "
      f"LUN A/B {rng_(100*lun.groupby('arch').adv_f1.mean().min(), 100*lun.groupby('arch').adv_f1.mean().max(), 0)}")
print(f"LLM balanced accuracy on A/B: own threshold {llm_ab.ba_own.mean():.2f}; best threshold {rng_(llm_ab.ba_oracle.min(), llm_ab.ba_oracle.max())}")
flip = 1 - llm_ab.adv_auroc.mean()
print(f"Flipped LLM scores: AUROC {flip:.2f} vs original {llm_ab.clean_auroc.mean():.2f} "
      f"= {100*(flip-0.5)/(llm_ab.clean_auroc.mean()-0.5):.0f}% of the original performance above random")
la = df[df.llm & (df.attack == "publisher_A")]
print(f"LLMs on original data: predicted fake {rng_(la.p_fake_c.min(), la.p_fake_c.max())}; accuracy real {la.acc_real_c.mean():.2f}, fake {la.acc_fake_c.mean():.2f}")
sd = enc.groupby(["dataset", "detector"]).size()  # noqa: F841
enc2 = enc.assign(base=enc.arch + "-" + enc.strategy)
s = enc2.groupby(["dataset", "base", "attack"]).adv_auroc.std()
print(f"Seed standard deviation of attacked AUROC: mean {s.mean():.3f}, max {s.max():.3f}")

# ---------------------------------------------------------------- Section 3.2
head("Section 3.2: Rewriting real news causes the reversal (Table 2)")
pub["real_rw"] = pub.attack.str[-1].map({"A": "NE", "B": "NE", "C": "Sun", "D": "Sun"})
pub["fake_rw"] = pub.attack.str[-1].map({"A": "CNN", "B": "NYT", "C": "CNN", "D": "NYT"})
pub["grp2"] = np.where(pub.llm, "Zero-shot LLMs", pub.group)
t2 = []
for g in ["FT+Aug", "FT", "Zero-shot LLMs"]:
    x = pub[pub.grp2 == g]
    t2.append(dict(detector=g, real_orig=x.acc_real_c.mean(), real_NE=x[x.real_rw == "NE"].acc_real_a.mean(),
                   real_Sun=x[x.real_rw == "Sun"].acc_real_a.mean(), fake_orig=x.acc_fake_c.mean(),
                   fake_CNN=x[x.fake_rw == "CNN"].acc_fake_a.mean(), fake_NYT=x[x.fake_rw == "NYT"].acc_fake_a.mean()))
print(pd.DataFrame(t2).set_index("detector").round(2).to_string())

# ---------------------------------------------------------------- Section 3.3
head("Section 3.3: Hidden layers (Figure 1b)")
s2 = []
for f in glob.glob(os.path.join(R, "stage2", "*", "*", "publisher_*.json")):
    ds, det, atk = f.split(os.sep)[-3], f.split(os.sep)[-2], os.path.basename(f)[:-5]
    if atk.startswith("random"):
        continue
    j = json.load(open(f))
    s2.append(dict(dataset=ds, detector=det, attack=atk, probe_clean=j["probe_acc_clean"], probe_adv=j["probe_acc_adv"]))
s2 = pd.DataFrame(s2).merge(df[["dataset", "detector", "attack", "adv_auroc"]], on=["dataset", "detector", "attack"])
lab = s2[s2.detector.isin(LLMS) & s2.attack.isin(["publisher_A", "publisher_B"])]
print(f"LLM last-layer probe on original articles: {lab.probe_clean.mean():.2f}")
cur = json.load(open(os.path.join(DER, "layer_curves.json")))
for key, name in [("LLM", "LLMs"), ("enc-A", "FT"), ("enc-BC", "FT+Aug")]:
    a = list(cur[f"{key}|AB|adv"].values())
    c = list(cur[f"{key}|AB|clean"].values())
    print(f"{name:7s} probing accuracy under A/B across depth: {min(a):.2f}-{max(a):.2f} (first {a[0]:.2f}, last {a[-1]:.2f}; original last {c[-1]:.2f})")
pl = lab[lab.dataset != "gossipcop"]
g_ = lab[lab.dataset == "gossipcop"]
print(f"PolitiFact/LUN last-layer probe under A/B: {rng_(pl.probe_adv.min(), pl.probe_adv.max())}")
print(f"GossipCop: probe {rng_(g_.probe_adv.min(), g_.probe_adv.max())} vs original {rng_(g_.probe_clean.min(), g_.probe_clean.max())}; "
      f"prediction AUROC {rng_(g_.adv_auroc.min(), g_.adv_auroc.max())}")
below = lab[lab.probe_adv < 0.5]
sig = [binomtest(int(round(r.probe_adv * N_TEST[r.dataset])), N_TEST[r.dataset], 0.5, alternative="less").pvalue < 0.05
       for _, r in lab.iterrows()]
print(f"LLM cases with probe accuracy below 0.5: {len(below)} of {len(lab)}; significantly (p<0.05): {sum(sig)} "
      f"{lab[sig][['detector','dataset','probe_adv']].round(2).values.tolist()}")

# ---------------------------------------------------------------- Section 3.4
head("Section 3.4: Attributions (encoders, articles with reliable IG)")
a = []
for f in glob.glob(os.path.join(R, "stage3", "*", "*", "publisher_*.jsonl")):
    ds, det, atk = f.split(os.sep)[-3], f.split(os.sep)[-2], os.path.basename(f)[:-6]
    if atk not in ("publisher_A", "publisher_B"):
        continue
    for line in open(f):
        r = json.loads(line)
        if (ds, det, r["id"]) in good:
            a.append(dict(dataset=ds, detector=det, grp="FT" if det.split("-")[1] == "A" else "FT+Aug",
                          flipped=r.get("flipped"), stab=r.get("stab_spearman"), sr=r.get("style_reliance")))
a = pd.DataFrame(a)
cell = a.groupby(["dataset", "detector", "grp"]).sr.mean().reset_index()  # one value per detector and dataset
for ds in ("gossipcop", "lun", "politifact"):
    x = cell[cell.dataset == ds]
    print(f"{ds:10s} style reliance (mean over detectors): FT {x[x.grp=='FT'].sr.mean():+.3f}, FT+Aug {x[x.grp=='FT+Aug'].sr.mean():+.3f}")
c2 = cell[cell.dataset.isin(["gossipcop", "lun"]) & (cell.grp == "FT+Aug")]
print(f"FT+Aug detector-dataset cases (GossipCop, LUN) with negative style reliance: {int((c2.sr < 0).sum())} of {len(c2)}")
ft = a[(a.grp == "FT") & a.flipped.notna()]
print(f"FT attribution stability: flipped {ft[ft.flipped == True].stab.mean():.2f} vs not flipped {ft[ft.flipped == False].stab.mean():.2f}")  # noqa: E712
fl = a[a.flipped.notna()].groupby(["dataset", "detector", "flipped"]).stab.mean().unstack().dropna()
print(f"Cases where flipped articles are less stable: {int((fl[True] < fl[False]).sum())} of {len(fl)}")
ag = [json.loads(line)["agree_spearman"] for f in glob.glob(os.path.join(R, "stage3", "*", "*", "clean.jsonl")) for line in open(f)]
print(f"IG vs occlusion agreement (encoders): mean Spearman {np.nanmean([x for x in ag if x is not None]):.2f}")

# ---------------------------------------------------------------- Sections 3.5 and 3.6: validity checks, recomputed from predictions
head("Sections 3.5-3.6: Validity checks and tone attack")


def auroc_ci(y, p, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    a0 = roc_auc_score(y, p)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], p[i]))
    return a0, np.quantile(bs, 0.975)


def p_fake(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


def gated(ds, atk, det, gate):
    g = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(R, "gate", ds, f"{atk}.jsonl"))}
    c = np.load(os.path.join(R, "predictions", ds, det, "test.npz"))
    a_ = np.load(os.path.join(R, "predictions", ds, det, f"{atk}.npz"))
    ids, y = c["ids"].tolist(), c["labels"]
    if gate == "nli":
        m = np.array([min(g[i]["entail_orig_to_rewrite"], g[i]["entail_rewrite_to_orig"]) >= 0.5 for i in ids])
    else:
        m = np.array([bool(g[i]["passed"]) for i in ids])
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return None
    return auroc_ci(y[m], p_fake(a_["logits"])[m])


for gate in ("nli", "full"):
    res = {}
    for ds in DATASETS:
        for det in LLMS:
            for atk in ("publisher_A", "publisher_B"):
                out = gated(ds, atk, det, gate)
                if out is not None:
                    res[(ds, det, atk)] = out
    rev = {k: v for k, v in res.items() if v[1] < 0.5}
    per_ds = {ds: f"{sum(1 for k in rev if k[0]==ds)}/{sum(1 for k in res if k[0]==ds)}" for ds in DATASETS}
    print(f"{gate.upper():4s} check, LLM A/B cases reversed: {len(rev)} of {len(res)} {per_ds}")
    if gate == "full":
        lun_auc = [v[0] for k, v in res.items() if k[0] == "lun"]
        if lun_auc:
            print(f"     LUN AUROC under strict check: {rng_(min(lun_auc), max(lun_auc))}")
pr = []
for ds in DATASETS:
    for s in "ABCD":
        rows_ = [json.loads(line) for line in open(os.path.join(R, "gate", ds, f"publisher_{s}.jsonl"))]
        pr.append((np.mean([min(r["entail_orig_to_rewrite"], r["entail_rewrite_to_orig"]) >= 0.5 for r in rows_]),
                   np.mean([r["passed"] for r in rows_])))
print(f"Style rewrites passing NLI: {rng_(min(x for x,_ in pr)*100, max(x for x,_ in pr)*100, 0)}%; strict check: {rng_(min(y for _,y in pr)*100, max(y for _,y in pr)*100, 0)}%")
tn, ta, er_t, er_p = [], [], [], []
for ds in DATASETS:
    rows_ = [json.loads(line) for line in open(os.path.join(R, "gate", ds, "sentiment_adv.jsonl"))]
    tn.append(np.mean([min(r["entail_orig_to_rewrite"], r["entail_rewrite_to_orig"]) >= 0.5 for r in rows_]))
    sm_ = json.load(open(os.path.join(R, "gate", ds, "summary.json")))
    ta.append(sm_["sentiment_adv"]["anchor_recall_median"])
    er_t.append(sm_["sentiment_adv"]["edit_ratio_median"])
    er_p += [sm_[f"publisher_{s}"]["edit_ratio_median"] for s in "ABCD"]
print(f"Tone rewrites: NLI pass {rng_(100*min(tn),100*max(tn),0)}%; anchors kept (median) {rng_(100*min(ta),100*max(ta),0)}%")
print(f"Share of words changed (median): tone {rng_(100*min(er_t),100*max(er_t),0)}%, style {rng_(100*min(er_p),100*max(er_p),0)}%")
for gate in ("nli", "full"):
    res = {(ds, det): gated(ds, "sentiment_adv", det, gate)
           for ds in DATASETS for det in sorted(os.listdir(os.path.join(R, "predictions", ds)))}
    res = {k: v for k, v in res.items() if v is not None}
    print(f"Tone attack under the {gate.upper()} check: reversed {sum(1 for v in res.values() if v[1] < 0.5)} of {len(res)}")
for atk in ("sentiment_adv", "sentiment_ctl"):
    x = df[df.attack == atk]
    print(f"{atk}: " + ", ".join(f"{g} {x[x.group==g].clean_auroc.mean():.2f}->{x[x.group==g].adv_auroc.mean():.2f}"
                                 for g in ["FT", "FT+Aug"]) +
          f", LLMs {x[x.llm].clean_auroc.mean():.2f}->{x[x.llm].adv_auroc.mean():.2f}; reversed {int(x.reversed.sum())} of {len(x)}; lowest AUROC {x.adv_auroc.min():.2f}")

# ---------------------------------------------------------------- Section 3.7
head("Section 3.7: Share of articles predicted as fake (Table 3)")
df["rise"] = df.p_fake_a - df.p_fake_c
pub_ = df[df.attack.str.match(r"^publisher_[ABCD]$")]
cats = [("Reversed (style attack)", pub_[pub_.reversed]), ("Other style-attack cases", pub_[~pub_.reversed]),
        ("Random-edit controls", df[df.attack.str.startswith("random_publisher")]),
        ("Tone attack", df[df.attack == "sentiment_adv"]), ("Tone control pairing", df[df.attack == "sentiment_ctl"])]
for name, x in cats:
    print(f"{name:26s} n={len(x):3d}  mean change {x.rise.mean():+.3f}  increase >0.1: {100*(x.rise > 0.1).mean():.0f}%")
print(f"Smallest increase among reversed cases: {pub_[pub_.reversed].rise.min():.3f}")
