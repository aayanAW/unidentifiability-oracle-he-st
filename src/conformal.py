"""Post-hoc conformal layer for the oracle (architecture.md PDC; preregistration.md H1).

Turns the predictor's residuals into coverage-guaranteed prediction intervals, and exposes WHERE the
guarantee is hard to keep. Two estimators:

  - split conformal (marginal): one global quantile of the calibration nonconformity scores. Gives
    distribution-free marginal coverage >= 1-alpha under exchangeability.
  - spatial-Mondrian (group-conditional): a separate quantile per spatial block. PARTIALLY improves per-block
    coverage when spots are spatially autocorrelated and naive (pooled) conformal under-covers some
    regions -- the pre-registered H1 phenomenon (Foygel-Barber: exact conditional coverage is impossible,
    so we target group-conditional coverage over spatial Mondrian groups, not a per-spot guarantee).

Nonconformity scores can be raw |residual| or |residual|/sigma using the dual-head variance (locally
adaptive intervals -- narrower where the model is confident, while keeping coverage). This is where the
trained variance head earns utility even when its U ranking is weak: calibrated interval WIDTH.

Pure numpy -- plumbing in service of the H1 coverage test; no torch, runs anywhere.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The (ceil((n+1)(1-alpha))/n)-th empirical quantile of calibration scores (Vovk split-conformal).

    Returns +inf when the calibration set is too small to certify 1-alpha (k>n) -- the honest behavior
    (the interval becomes everything; coverage holds trivially) rather than a silently invalid finite q.
    """
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = s.size
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


def nonconformity_scores(
    resid_abs: np.ndarray, sigma: np.ndarray | None = None
) -> np.ndarray:
    """|residual| (marginal) or |residual|/sigma (locally adaptive, uses the dual-head variance head)."""
    resid_abs = np.asarray(resid_abs, dtype=float)
    if sigma is None:
        return resid_abs
    return resid_abs / (np.asarray(sigma, dtype=float) + 1e-8)


def conformal_intervals(
    pred: np.ndarray, q: float | np.ndarray, sigma: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Prediction intervals [pred-h, pred+h]. h = q (marginal) or q*sigma (locally adaptive)."""
    pred = np.asarray(pred, dtype=float)
    half = (
        np.asarray(q, dtype=float)
        if sigma is None
        else np.asarray(q, dtype=float) * (np.asarray(sigma, dtype=float) + 1e-8)
    )
    return pred - half, pred + half


def empirical_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of targets falling inside their two-sided interval."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(((y >= lo) & (y <= hi)).mean())


def mondrian_quantiles(
    cal_scores: np.ndarray, cal_groups: np.ndarray, alpha: float
) -> dict:
    """One conformal quantile per group (spatial Mondrian block)."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_groups = np.asarray(cal_groups)
    return {
        g: conformal_quantile(cal_scores[cal_groups == g], alpha)
        for g in np.unique(cal_groups)
    }


def mondrian_q_for(test_groups: np.ndarray, qmap: dict, fallback: float) -> np.ndarray:
    """Per-test-point quantile from the group map; `fallback` covers groups unseen in calibration."""
    return np.array(
        [qmap.get(g, fallback) for g in np.asarray(test_groups)], dtype=float
    )


def per_group_coverage(
    scores: np.ndarray, q: float | np.ndarray, groups: np.ndarray
) -> dict:
    """One-sided coverage per group on nonconformity scores: covered iff score <= q (q scalar or per-point).

    This is the form used for the H1 diagnostic: a point is covered iff its nonconformity score is within
    the calibrated quantile, so coverage = mean(scores <= q) per spatial group.
    """
    scores = np.asarray(scores, dtype=float)
    q = np.asarray(q, dtype=float)
    groups = np.asarray(groups)
    covered = scores <= q
    return {g: float(covered[groups == g].mean()) for g in np.unique(groups)}


def selective_conformal_sweep(
    U: np.ndarray,
    abs_resid: np.ndarray,
    alpha: float = 0.1,
    grid: int = 10,
    cal_frac: float = 0.5,
    seed: int = 0,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Coverage-guaranteed selective prediction = the oracle's deferral + conformal coverage, assembled.

    Abstain on the highest-U genes (keep the lowest-U, most identifiable first), then split-conformal on the
    retained (niche, gene) cells: calibrate the quantile on a held-out half of the NICHES and measure
    coverage + interval width on the rest. As more high-U genes are abstained, the retained scores shrink so
    the same 1-alpha guarantee buys TIGHTER intervals -- the headline utility of pairing U with conformal.

    `abs_resid` (n_niche, n_gene) = |OOF residual| nonconformity scores; `sigma` (same shape, optional) =
    the dual-head variance head for locally adaptive (|resid|/sigma) intervals. Returns (grid, 3):
    [retained_gene_fraction, marginal_coverage, mean_interval_width].
    """
    U = np.asarray(U, dtype=float)
    abs_resid = np.asarray(abs_resid, dtype=float)
    n_spot, n_gene = abs_resid.shape
    order = np.argsort(U)  # keep low-U (most identifiable) genes first

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_spot)
    n_cal = max(1, int(cal_frac * n_spot))
    cal_spots, test_spots = perm[:n_cal], perm[n_cal:]
    if len(test_spots) == 0:
        raise ValueError(
            f"selective_conformal_sweep: n_spot={n_spot} with cal_frac={cal_frac} leaves no test "
            "niches; need at least 2 niches."
        )

    rows = []
    for frac in np.linspace(1.0, 0.1, grid):
        k = max(1, int(round(frac * n_gene)))
        keep = order[:k]
        cal_sig = None if sigma is None else sigma[np.ix_(cal_spots, keep)]
        test_sig = None if sigma is None else sigma[np.ix_(test_spots, keep)]
        scores_cal = nonconformity_scores(
            abs_resid[np.ix_(cal_spots, keep)], cal_sig
        ).ravel()
        scores_test = nonconformity_scores(
            abs_resid[np.ix_(test_spots, keep)], test_sig
        ).ravel()
        q = conformal_quantile(scores_cal, alpha)
        coverage = float((scores_test <= q).mean())
        width = 2.0 * q if sigma is None else 2.0 * q * float(np.mean(test_sig))
        rows.append((k / n_gene, coverage, width))
    return np.array(rows)
