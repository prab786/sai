"""Recompute derived/style_markers.json (Section 2) from the public SheepDog data release.

Usage:  python compute_style_markers.py /path/to/SheepDog-main/data
Counts exclamation marks, fully capitalised words (3+ letters) and question marks per 100 words in the
original test articles and in each adversarial test set (A-D), separately for real and fake articles.
"""
import json
import os
import pickle
import re
import sys

import numpy as np

WORD = re.compile(r"[A-Za-z']+")


def feats(t):
    t = str(t)
    w = WORD.findall(t)
    n = max(len(w), 1)
    return np.array([100 * t.count("!") / n, 100 * sum(1 for x in w if len(x) > 2 and x.isupper()) / n,
                     100 * t.count("?") / n, n])


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # only run on the official SheepDog release


def texts_of(d):
    return d["news"] if "news" in d else d[[k for k in d if k != "labels"][0]]


data = sys.argv[1]
out = {}
for ds in ["politifact", "gossipcop", "lun"]:
    orig = load(os.path.join(data, "news_articles", f"{ds}_test.pkl"))
    y = np.asarray(orig["labels"])
    fo = np.array([feats(t) for t in texts_of(orig)])
    for s in "ABCD":
        fa = np.array([feats(t) for t in texts_of(load(os.path.join(data, "adversarial_test", f"{ds}_test_adv_{s}.pkl")))])
        for lab, name in [(0, "real"), (1, "fake")]:
            m = y == lab
            out[f"{ds}|{s}|{name}"] = dict(excl_o=fo[m, 0].mean(), excl_r=fa[m, 0].mean(), caps_o=fo[m, 1].mean(),
                                           caps_r=fa[m, 1].mean(), q_o=fo[m, 2].mean(), q_r=fa[m, 2].mean(),
                                           len_ratio=float(np.median(fa[m, 3] / fo[m, 3])))
json.dump(out, open("style_markers.json", "w"), indent=1)
print("wrote style_markers.json")
