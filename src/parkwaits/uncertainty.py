"""How much of a measured lead could be luck? A day-block bootstrap for the AUC gap between two forecasts.

Snapshots of one ride a few minutes apart are nearly the same, so resampling rows would make the lead look far
more certain than it is. Whole days are resampled instead: each draw rebuilds the test window from days picked
at random (with replacement), and the gap is re-measured. The spread of those gaps is the interval.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def day_bootstrap_gap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    days: np.ndarray,
    *,
    draws: int = 300,
    seed: int = 0,
    level: float = 0.95,
) -> dict | None:
    """AUC(a) - AUC(b) with a day-block bootstrap interval. None when there are fewer than 3 usable days.

    Returns {"gap", "low", "high", "level", "days", "draws", "share_ahead"}; share_ahead is the fraction of
    draws in which `a` beat `b`."""
    y, score_a, score_b, days = (np.asarray(v) for v in (y, score_a, score_b, days))
    labels = np.unique(days)
    if len(labels) < 3 or len(np.unique(y)) < 2:
        return None
    index = {d: np.flatnonzero(days == d) for d in labels}
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        pick = rng.choice(len(labels), size=len(labels), replace=True)
        rows = np.concatenate([index[labels[i]] for i in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        gaps.append(roc_auc_score(yy, score_a[rows]) - roc_auc_score(yy, score_b[rows]))
    if len(gaps) < draws // 2:
        return None
    gaps_arr = np.asarray(gaps)
    lo, hi = np.quantile(gaps_arr, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "gap": float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)),
        "low": float(lo),
        "high": float(hi),
        "level": level,
        "days": int(len(labels)),
        "draws": int(len(gaps_arr)),
        "share_ahead": float((gaps_arr > 0).mean()),
    }
