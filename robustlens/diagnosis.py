"""Diagnosis map, decision rules and diagnosis cards (paper Section 3.2-3.4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

ZONES = ("robust", "threshold_shift", "signal_erasure", "mixed", "inversion")


def zone(S: float, G: float, auroc_ci_high: Optional[float] = None, s_cut: float = 0.9, g_cut: float = 0.05) -> str:
    """Decision rules on signal retained S and threshold gap G.

    Inversion needs S < 0 and, if given, an AUROC interval whose upper end is below 0.5.
    """
    if not (np.isfinite(S) and np.isfinite(G)):
        return "undefined"
    if S < 0 and (auroc_ci_high is None or auroc_ci_high < 0.5):
        return "inversion"
    if S >= s_cut:
        return "robust" if G < g_cut else "threshold_shift"
    return "signal_erasure" if G < g_cut else "mixed"


def representation_confirms_erasure(probe_retention: Optional[float], pr_cut: float = 0.5) -> Optional[bool]:
    if probe_retention is None or not np.isfinite(probe_retention):
        return None
    return bool(probe_retention < pr_cut)


@dataclass
class DiagnosisCard:
    detector: str
    dataset: str
    attack: str
    zone: str
    S: float
    G: float
    erasure_confirmed: Optional[bool] = None
    damaged_from_layer: Optional[int] = None
    candidate_cue: Optional[str] = None
    cue_confirmed: Optional[bool] = None
    defence_verdict: Optional[str] = None  # "removed" | "substituted" | None
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def plot_map(cells, path: str, s_cut: float = 0.9, g_cut: float = 0.05, color_col: str = "family_detector",
             marker_col: str = "family", title: str = "Detector x dataset x attack cells on the diagnosis map") -> None:
    """Figure 5: every cell placed by (S, G). ``cells`` is a DataFrame with columns S, G and the two style columns."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    s = cells["S"].clip(-1.2, 1.4)
    xmin, xmax = min(-0.3, s.min() - 0.05), max(1.2, s.max() + 0.05)
    ymax = max(0.25, cells["G"].max() + 0.02)
    ax.axvspan(xmin, 0, color="#d62728", alpha=0.07, lw=0)
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvline(s_cut, color="grey", lw=0.8, ls="--")
    ax.axhline(g_cut, color="grey", lw=0.8, ls="--", xmin=(0 - xmin) / (xmax - xmin))
    for x, y, label in ((xmin + 0.02, ymax * 0.95, "inversion"), (0.02, ymax * 0.95, "mixed"), (0.02, g_cut * 0.4, "signal erasure"),
                        (s_cut + 0.02, ymax * 0.95, "threshold shift"), (s_cut + 0.02, g_cut * 0.4, "robust")):
        ax.text(x, y, label, fontsize=8, color="grey", va="top")
    markers = dict(zip(sorted(cells[marker_col].unique()), "osD^vP"))
    for (c, m), g in cells.groupby([color_col, marker_col]):
        ax.scatter(g["S"].clip(-1.2, 1.4), g["G"], marker=markers[m], s=28, alpha=0.8, label=f"{c} / {m}")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Signal retained S")
    ax.set_ylabel("Threshold gap G")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
