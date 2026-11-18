"""Stage 2: where in the network is usable separation lost? (tier T2)

* Probe retention: a linear probe trained on clean training states, evaluated on clean and attacked test
  states; PR = (acc_adv - 0.5) / (acc_clean - 0.5). Control: the same probe trained on shuffled labels.
* Crossing rate: share of correctly-sided articles whose rewrite lands on the wrong side of the midpoint
  between the class centroids; the label-free flip rate counts any change of side.
* Layer drift: 1 - linear CKA between clean and attacked states, against random edits matched for edit size.
"""
from __future__ import annotations

import difflib
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------- probes
def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed)).fit(X, y)


def probe_retention(acc_clean: float, acc_adv: float, min_clean: float = 0.55) -> float:
    if not np.isfinite(acc_clean) or acc_clean < min_clean:
        return float("nan")
    return float((acc_adv - 0.5) / (acc_clean - 0.5))


def probe_layer(Xtr, ytr, Xte, Xadv, yte, C: float = 1.0, min_clean: float = 0.55, seed: int = 0) -> dict:
    probe = fit_probe(Xtr, ytr, C, seed)
    acc_c = float(probe.score(Xte, yte))
    acc_a = float(probe.score(Xadv, yte))
    sign = 2 * np.asarray(yte) - 1
    margin = float(np.mean(probe.decision_function(Xte) * sign))
    rng = np.random.default_rng(seed)
    shuffled = fit_probe(Xtr, rng.permutation(ytr), C, seed)
    return {"probe_acc_clean": acc_c, "probe_acc_adv": acc_a, "probe_retention": probe_retention(acc_c, acc_a, min_clean),
            "probe_margin_clean": margin, "control_shuffled_acc": float(shuffled.score(Xte, yte))}


# ---------------------------------------------------------------------- geometry
def class_centroids(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[y == 0].mean(0), X[y == 1].mean(0)


def crossing_rate(X_orig: np.ndarray, X_adv: np.ndarray, y: np.ndarray, c_real: np.ndarray, c_fake: np.ndarray) -> dict:
    axis = c_fake - c_real
    mid = (c_real + c_fake) / 2
    side_o = np.sign((X_orig - mid) @ axis)  # +1 = fake side
    side_a = np.sign((X_adv - mid) @ axis)
    want = 2 * np.asarray(y) - 1
    correct = side_o == want
    return {
        "crossing_rate": float(np.mean(side_a[correct] != want[correct])) if correct.any() else float("nan"),
        "flip_rate_geom": float(np.mean(side_o != side_a)),
        "shift_to_margin": float(np.mean(np.linalg.norm(X_adv - X_orig, axis=1)) / max(np.linalg.norm(axis), 1e-12)),
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA. NaN when either side has (almost) no variance across articles, e.g. layer 0 at the CLS or
    answer position, which is the same embedding for every article: drift is undefined there, not 1."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    nx, ny = np.linalg.norm(X.T @ X, "fro"), np.linalg.norm(Y.T @ Y, "fro")
    scale = max(np.abs(X).max(initial=0.0), np.abs(Y).max(initial=0.0), 1.0)
    if min(nx, ny) <= 1e-8 * scale ** 2:
        return float("nan")
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(hsic / (nx * ny))


def cka_drift_band(X: np.ndarray, Y: np.ndarray, n_boot: int = 200, seed: int = 0, q: float = 0.975) -> tuple[float, float]:
    """1 - CKA and its bootstrap upper quantile (resampling articles)."""
    rng = np.random.default_rng(seed)
    point = 1 - linear_cka(X, Y)
    if not np.isfinite(point):
        return point, float("nan")
    boots = [1 - linear_cka(X[i], Y[i]) for i in (rng.integers(0, len(X), len(X)) for _ in range(n_boot))]
    return point, float(np.nanquantile(boots, q))


def divergence_depth(drift: dict[int, float], control_upper: dict[int, float]) -> Optional[int]:
    """First layer whose drift exceeds the random-edit control's upper band (undefined layers skipped).

    Descriptive only. On the full run this lands on the first layer for nearly every attack, including the
    non-adversarial sentiment control, because rewrites restructure the text while random edits keep word
    positions; it is therefore not used to choose the erasure layer (see ``causal.erase_layer``).
    """
    for l in sorted(drift):
        d, u = drift[l], control_upper.get(l, np.inf)
        if np.isfinite(d) and np.isfinite(u) and d > u:
            return l
    return None


# ---------------------------------------------------------------------- random-edit control
def edit_ratio(a: str, b: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def random_edit(text: str, ratio: float, rng: np.random.Generator) -> str:
    """Replace about ``ratio`` of the words with other words from the same article (keeps vocabulary, breaks content)."""
    words = text.split()
    if len(words) < 2 or ratio <= 0:
        return text
    k = min(len(words), max(1, int(round(ratio * len(words)))))
    pos = rng.choice(len(words), size=k, replace=False)
    pool = np.array(words)
    for p in pos:
        words[p] = pool[rng.integers(len(pool))]
    return " ".join(words)


def matched_random_edits(originals: Sequence[str], attacked: Sequence[str], seed: int = 0) -> tuple[list[str], np.ndarray]:
    """One random-edit control per article, matched to that article's attack edit ratio."""
    rng = np.random.default_rng(seed)
    ratios = np.array([edit_ratio(o, a) for o, a in zip(originals, attacked)])
    return [random_edit(o, r, rng) for o, r in zip(originals, ratios)], ratios


# ---------------------------------------------------------------------- full report
def report(train_states: dict[int, np.ndarray], y_train: np.ndarray, test_states: dict[int, np.ndarray],
           adv_states: dict[int, np.ndarray], y_test: np.ndarray, ctrl_states: Optional[dict[int, np.ndarray]] = None,
           C: float = 1.0, min_clean: float = 0.55, seed: int = 0) -> dict:
    layers = sorted(set(train_states) & set(test_states) & set(adv_states))
    per_layer, drift, ctrl_upper = {}, {}, {}
    for l in layers:
        row = probe_layer(train_states[l], y_train, test_states[l], adv_states[l], y_test, C, min_clean, seed)
        cr, cf = class_centroids(train_states[l], y_train)
        row.update(crossing_rate(test_states[l], adv_states[l], y_test, cr, cf))
        row["drift"] = 1 - linear_cka(test_states[l], adv_states[l])
        if ctrl_states is not None and l in ctrl_states:
            row["control_drift"], row["control_drift_upper"] = cka_drift_band(test_states[l], ctrl_states[l], seed=seed)
            ctrl_upper[l] = row["control_drift_upper"]
        drift[l] = row["drift"]
        per_layer[l] = row
    last = max(layers)
    return {"layers": per_layer, "final_layer": last, "divergence_depth": divergence_depth(drift, ctrl_upper) if ctrl_upper else None,
            **{k: per_layer[last][k] for k in ("probe_retention", "probe_acc_clean", "probe_acc_adv", "probe_margin_clean", "crossing_rate", "flip_rate_geom")}}
