"""Threshold-free open-set detection metrics.

Both treat a per-image "known-ness" score (higher = more confidently a known
class) and ask how well it separates known-class images from images of
unseen generators, without committing to a single threshold.
"""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz


def auroc(score_known_high: np.ndarray, is_known: np.ndarray) -> float:
    """AUROC for "known" as the positive class (rank-based, tie-safe)."""
    pos = score_known_high[is_known == 1]
    neg = score_known_high[is_known == 0]
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(n_p + n_n)
    ranks[order] = np.arange(1, n_p + n_n + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def au_oscr(score_known_high: np.ndarray, correct_known: np.ndarray, is_known: np.ndarray) -> float:
    """Area under the Open-Set Classification Rate curve: correct-and-accepted
    rate on knowns (CCR) vs. false-positive rate of accepting unknowns (FPR),
    swept over every threshold on `score_known_high`.
    """
    n_k = max(1, int((is_known == 1).sum()))
    n_u = max(1, int((is_known == 0).sum()))
    order = np.argsort(-score_known_high)
    ccr = np.cumsum(correct_known[order] == 1) / n_k
    fpr = np.cumsum(is_known[order] == 0) / n_u
    keep = np.r_[True, np.diff(fpr) > 0]
    x, y = fpr[keep], ccr[keep]
    if x[0] > 0:
        x, y = np.r_[0, x], np.r_[0, y]
    if x[-1] < 1:
        x, y = np.r_[x, 1], np.r_[y, y[-1]]
    return float(_trapezoid(y, x))
