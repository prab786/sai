"""Stage 1: how does the output fail, and for which class?

Everything here needs only the detector scores and the labels.
Probabilities are P(fake); label 0 = real, 1 = fake.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def predict(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(p) >= threshold).astype(int)


def per_class(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y, yhat = np.asarray(y), predict(p, threshold)
    out = {"accuracy": float(np.mean(yhat == y)), "p_pred_fake": float(np.mean(yhat == 1))}
    for c, name in ((0, "real"), (1, "fake")):
        m = y == c
        out[f"acc_{name}"] = float(np.mean(yhat[m] == c)) if m.any() else float("nan")
    out["balanced_accuracy"] = float(np.nanmean([out["acc_real"], out["acc_fake"]]))
    return out


def macro_f1(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, predict(p, threshold), average="macro", zero_division=0))


def flip_table(y: np.ndarray, p_orig: np.ndarray, p_adv: np.ndarray, threshold: float = 0.5) -> dict:
    """Transitions between each article's prediction before and after the rewrite."""
    y, a, b = np.asarray(y), predict(p_orig, threshold), predict(p_adv, threshold)
    out = {"toward_real": int(np.sum((a == 1) & (b == 0))), "toward_fake": int(np.sum((a == 0) & (b == 1)))}
    for c, name in ((0, "real"), (1, "fake")):
        m = y == c
        out[f"{name}_toward_real"] = int(np.sum(m & (a == 1) & (b == 0)))
        out[f"{name}_toward_fake"] = int(np.sum(m & (a == 0) & (b == 1)))
    out["flip_rate"] = float(np.mean(a != b))
    return out


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the predicted class's confidence."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat == 1, p, 1 - p)
    correct = (yhat == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def auroc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Stratified bootstrap percentile interval."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def balanced_accuracy_at(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return per_class(y, p, threshold)["balanced_accuracy"]


def oracle_balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Best balanced accuracy over all thresholds, with the readout's own direction."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max((tpr + 1 - fpr) / 2))


def signal_retained(auc_adv: float, auc_clean: float, min_clean: float = 0.55) -> float:
    """S = (AUROC_adv - 0.5) / (AUROC_clean - 0.5); undefined when the clean detector is near chance
    (same convention as probe retention), because the ratio then explodes."""
    if not np.isfinite(auc_clean) or auc_clean < min_clean:
        return float("nan")
    return float((auc_adv - 0.5) / (auc_clean - 0.5))


def threshold_gap(y: np.ndarray, p_adv: np.ndarray, threshold: float = 0.5) -> float:
    """G = BA at the oracle threshold - BA at the model's own threshold, on the attacked set."""
    return oracle_balanced_accuracy(y, p_adv) - balanced_accuracy_at(y, p_adv, threshold)


def report(y: np.ndarray, p_clean: np.ndarray, p_adv: np.ndarray, n_bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """All Stage 1 measures for one detector x attack set."""
    auc_c, auc_a = auroc(y, p_clean), auroc(y, p_adv)
    lo, hi = auroc_ci(y, p_adv, n_boot, seed) if n_boot else (float("nan"), float("nan"))
    clean, adv = per_class(y, p_clean), per_class(y, p_adv)
    return {
        "clean": {**clean, "macro_f1": macro_f1(y, p_clean), "ece": ece(y, p_clean, n_bins), "auroc": auc_c},
        "adv": {**adv, "macro_f1": macro_f1(y, p_adv), "ece": ece(y, p_adv, n_bins), "auroc": auc_a,
                "auroc_ci": [lo, hi]},
        "delta_ece": ece(y, p_adv, n_bins) - ece(y, p_clean, n_bins),
        "acc_cr": adv["accuracy"] / clean["accuracy"] if clean["accuracy"] > 0 else float("nan"),
        "f1_drop": macro_f1(y, p_clean) - macro_f1(y, p_adv),
        "flips": flip_table(y, p_clean, p_adv),
        "S": signal_retained(auc_a, auc_c),
        "G": threshold_gap(y, p_adv),
        "ba_own": balanced_accuracy_at(y, p_adv),
        "ba_oracle": oracle_balanced_accuracy(y, p_adv),
        "ba_oracle_signflip": oracle_balanced_accuracy(y, 1 - np.asarray(p_adv)),
        "conf_margin_clean": float(np.mean(np.abs(np.asarray(p_clean) - 0.5))),
    }
