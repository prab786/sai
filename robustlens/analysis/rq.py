"""Analyses behind the four research questions.

RQ1  Do internal measures computed on clean data predict the F1 drop on an attack family never used to
     fit the model? Cross-family regression (fit on one family's cells, test on the other's, both ways),
     internal vs output-only predictors, with a within-dataset permutation test of the R^2 gain.
     Style reliance needs the rewrite, so it is not an RQ1 predictor; NRC sentiment reliance and probe
     margin are attack-free.
RQ2  Zone counts and agreement with the cases established in prior work.
RQ3  Cue substitution: does a defence (B, C) lower style reliance but raise sentiment reliance, and does
     F1 on the unseen family stay below F1 on the defended family?
RQ4  Reliability: seed-to-seed stability of each measure, explainer agreement, and attribution change
     under non-flipping attacks vs. random edits.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

OUTPUT_ONLY = ["clean_f1", "conf_margin_clean"]
INTERNAL = ["probe_margin_clean", "nrc_reliance_clean"]


# ---------------------------------------------------------------------- RQ1
def _cross_r2(df: pd.DataFrame, predictors: Sequence[str], outcome: str, fam_col: str) -> dict:
    fams = sorted(df[fam_col].unique())
    if len(fams) != 2:
        raise ValueError(f"cross-family prediction needs exactly two families, got {fams}")
    out = {}
    for a, b in ((fams[0], fams[1]), (fams[1], fams[0])):
        tr, te = df[df[fam_col] == a].dropna(subset=list(predictors) + [outcome]), df[df[fam_col] == b].dropna(subset=list(predictors) + [outcome])
        if len(tr) < len(predictors) + 2 or len(te) < 2:
            out[f"{a}->{b}"] = float("nan")
            continue
        model = LinearRegression().fit(tr[list(predictors)], tr[outcome])
        out[f"{a}->{b}"] = float(r2_score(te[outcome], model.predict(te[list(predictors)])))
    return out


def rq1_table(cells: pd.DataFrame, outcome: str = "f1_drop", fam_col: str = "family", n_perm: int = 10000,
              seed: int = 0, output_only: Sequence[str] = OUTPUT_ONLY, internal: Sequence[str] = INTERNAL) -> pd.DataFrame:
    sets = [("output-only", list(output_only))]
    sets += [(f"+ {p}", list(output_only) + [p]) for p in internal]
    sets.append(("all internal", list(output_only) + list(internal)))
    base = _cross_r2(cells, output_only, outcome, fam_col)
    rows = []
    for name, preds in sets:
        r2 = _cross_r2(cells, preds, outcome, fam_col)
        row = {"predictors": name, **{f"R2 {k}": v for k, v in r2.items()}}
        if name != "output-only":
            extra = [p for p in preds if p not in output_only]
            row["p_perm"] = permutation_p(cells, output_only, extra, outcome, fam_col, n_perm, seed, base)
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_p(cells: pd.DataFrame, output_only: Sequence[str], extra: Sequence[str], outcome: str, fam_col: str,
                  n_perm: int, seed: int, base: dict | None = None, ds_col: str = "dataset",
                  unit_col: str = "detector_base") -> float:
    """p for the mean held-out R^2 gain of the extra (internal) predictors.

    The internal predictors are clean-data measures of a detector, so they are shared by that detector's
    cells in both families. The null shuffles them between detectors (units) within each dataset and
    keeps each detector's two family cells together; shuffling cell by cell would give one detector
    different clean values in its two families.
    """
    base = base or _cross_r2(cells, output_only, outcome, fam_col)

    def gain(df):
        r = _cross_r2(df, list(output_only) + list(extra), outcome, fam_col)
        return np.nanmean([r[k] - base[k] for k in r])

    obs = gain(cells)
    if not np.isfinite(obs):
        return float("nan")
    extra = list(extra)
    if unit_col not in cells:
        raise ValueError(f"permutation needs a {unit_col!r} column identifying the detector of each cell")
    units = cells.groupby([ds_col, unit_col], as_index=False)[extra].first()  # one row of clean predictors per detector
    groups = [np.flatnonzero((units[ds_col] == d).to_numpy()) for d in units[ds_col].unique()]
    rest = cells.drop(columns=extra)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        vals = units[extra].to_numpy().copy()
        for g in groups:
            vals[g] = vals[g][rng.permutation(len(g))]
        perm = units[[ds_col, unit_col]].assign(**{c: vals[:, k] for k, c in enumerate(extra)})
        df = rest.merge(perm, on=[ds_col, unit_col], how="left")
        hits += gain(df) >= obs
    return float((1 + hits) / (1 + n_perm))


# ---------------------------------------------------------------------- RQ2
def rq2_zone_counts(cells: pd.DataFrame, by: Sequence[str] = ("detector_family", "family")) -> pd.DataFrame:
    return cells.groupby(list(by))["zone"].value_counts().unstack(fill_value=0)


def rq2_known_cases(cells: pd.DataFrame, cases: Sequence[dict]) -> pd.DataFrame:
    """``cases``: dicts with detector, dataset, attack (substring match allowed) and expected zone(s)."""
    rows = []
    for c in cases:
        m = (cells["detector"].str.contains(c["detector"])) & (cells["dataset"] == c["dataset"]) & (cells["attack"].str.contains(c["attack"]))
        sub = cells[m]
        observed = sub["zone"].value_counts().idxmax() if len(sub) else "missing"
        rows.append({"case": c.get("label", f'{c["detector"]} / {c["dataset"]} / {c["attack"]}'),
                     "expected": "/".join(c["expected"]), "observed": observed,
                     "S": sub["S"].mean() if len(sub) else np.nan, "G": sub["G"].mean() if len(sub) else np.nan,
                     "agrees": observed in c["expected"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- RQ3
def rq3_cue_substitution(rel: pd.DataFrame, defended_family: str = "publisher", unseen_family: str = "sentiment") -> pd.DataFrame:
    """``rel`` has one row per model x strategy (seed-averaged) with style_reliance, nrc_reliance,
    nrc_control_band, f1_<family> columns. Strategy A is the reference for B and C."""
    rows = []
    for model, g in rel.groupby("model"):
        ref = g[g["strategy"] == "A"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, r in g.iterrows():
            row = {"model": model, "strategy": r["strategy"], "style_reliance": r["style_reliance"],
                   "sentiment_reliance": r["nrc_reliance"], f"f1_{defended_family}": r[f"f1_{defended_family}"],
                   f"f1_{unseen_family}": r[f"f1_{unseen_family}"]}
            if r["strategy"] != "A":
                style_down = r["style_reliance"] < ref["style_reliance"]
                sent_up = (r["nrc_reliance"] - ref["nrc_reliance"]) > r.get("nrc_control_band", 0.0)
                unseen_lower = r[f"f1_{unseen_family}"] < r[f"f1_{defended_family}"]
                row["substitution"] = bool(style_down and sent_up and unseen_lower)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- RQ4
def seed_stability(measures: pd.DataFrame, measure_cols: Sequence[str], unit_cols: Sequence[str] = ("model", "strategy", "dataset", "attack"),
                   seed_col: str = "seed") -> pd.DataFrame:
    """Mean Spearman correlation of each measure across cells between every pair of seeds."""
    rows = []
    seeds = sorted(measures[seed_col].unique())
    for col in measure_cols:
        piv = measures.pivot_table(index=list(unit_cols), columns=seed_col, values=col)
        rhos = [spearmanr(piv[a], piv[b], nan_policy="omit").statistic for a, b in combinations(seeds, 2) if a in piv and b in piv]
        rows.append({"measure": col, "seed_stability_rho": float(np.nanmean(rhos)) if rhos else np.nan, "n_pairs": len(rhos)})
    return pd.DataFrame(rows)


def nonflip_change(stab_attack: Sequence[float], stab_random: Sequence[float]) -> dict:
    """Attribution stability (clean vs rewrite) on non-flipping attacked items vs random edits: lower = more change."""
    a, r = np.asarray(stab_attack, float), np.asarray(stab_random, float)
    a, r = a[np.isfinite(a)], r[np.isfinite(r)]
    if len(a) < 3 or len(r) < 3:
        return {"median_attack": np.nan, "median_random": np.nan, "p": np.nan}
    return {"median_attack": float(np.median(a)), "median_random": float(np.median(r)),
            "p": float(mannwhitneyu(a, r, alternative="less").pvalue)}
