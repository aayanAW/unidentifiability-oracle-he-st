"""Statistics for the unidentifiability oracle gate: Moran's I, BH-FDR, AUROC.

Plumbing in service of the pre-registered feasibility test (feasibility.md). Pure numpy/scipy/sklearn.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns boolean mask of rejected (significant) hypotheses."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    cutoff = ranked[kmax]
    return p <= cutoff


def grid_weights(coords: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """Binary spatial weight matrix: neighbours within `radius` (row-standardised)."""
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff**2, axis=-1)
    w = (d2 <= radius**2).astype(float)
    np.fill_diagonal(w, 0.0)
    rs = w.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return w / rs


def morans_i(values: np.ndarray, w: np.ndarray) -> float:
    """Moran's I spatial autocorrelation of `values` under row-standardised weights `w`."""
    x = np.asarray(values, dtype=float)
    z = x - x.mean()
    denom = np.sum(z**2)
    if denom == 0:
        return 0.0
    num = np.sum(w * np.outer(z, z))
    s0 = np.sum(w)
    return float((len(x) / s0) * (num / denom))


def morans_i_pvalue(
    values: np.ndarray, w: np.ndarray, n_perm: int = 199, seed: int = 0
) -> tuple[float, float]:
    """Permutation p-value (one-sided, positive autocorrelation) for Moran's I."""
    rng = np.random.default_rng(seed)
    obs = morans_i(values, w)
    perm = np.empty(n_perm)
    v = np.asarray(values, dtype=float)
    for i in range(n_perm):
        perm[i] = morans_i(rng.permutation(v), w)
    p = (1.0 + np.sum(perm >= obs)) / (1.0 + n_perm)
    return obs, float(p)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC; returns 0.5 if a class is missing."""
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return 0.5
    return float(roc_auc_score(labels, scores))
