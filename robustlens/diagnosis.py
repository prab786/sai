"""Decision rule that labels each detector x test set (a reversed case is zone "inversion")."""
from __future__ import annotations

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
