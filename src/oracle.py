"""The unidentifiability oracle: noise floor, ensemble aleatoric estimate, U, and selective-risk layer.

Implements the pre-registered estimand (preregistration.md sec 2 + sec 11 amendments):

    U(g) = aleatoric_residual(g)  -  sigma2_xen(g)  -  sigma2_reg(g)

where aleatoric is the out-of-fold residual of an ensemble of predictors with the *epistemic* (ensemble
disagreement) component removed -- so U is the irreducible part NOT explained by morphology, beyond the
Xenium technical noise floor. f is a stand-in (Ridge ensemble here; UNI/Virchow2 + MLP on real data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .metrics import bh_fdr, grid_weights, morans_i_pvalue, safe_auroc
from .simulator import TriadData


def noise_floor(rep1: np.ndarray, rep2: np.ndarray) -> np.ndarray:
    """Unbiased per-gene Xenium technical variance from two replicates: Var(r1-r2)=2*sigma2 -> sigma2."""
    return 0.5 * np.mean((rep1 - rep2) ** 2, axis=0)


def _ensemble_oof(X: np.ndarray, Y: np.ndarray, n_models: int, n_folds: int, seed: int):
    """Out-of-fold ensemble predictions. Returns (pbar, epistemic) each (n_spots, n_genes)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    pbar = np.zeros_like(Y)
    epi = np.zeros_like(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        preds = np.empty((n_models, te.size, Y.shape[1]))
        for m in range(n_models):
            rows = rng.choice(tr, size=tr.size, replace=True)  # bootstrap rows
            feats = rng.choice(
                d, size=max(2, int(0.75 * d)), replace=False
            )  # feature subsample
            model = Ridge(alpha=1.0, random_state=seed + m)
            model.fit(X[np.ix_(rows, feats)], Y[rows])
            preds[m] = model.predict(X[np.ix_(te, feats)])
        pbar[te] = preds.mean(0)
        epi[te] = preds.var(0)
    return pbar, epi


@dataclass
class OracleResult:
    U: np.ndarray  # (n_genes,) intrinsic unidentifiability per gene (>=0)
    pvalue: np.ndarray  # (n_genes,) bootstrap p that U<=0
    flagged: np.ndarray  # (n_genes,) bool: BH-FDR significant AND U>0
    resid2: np.ndarray  # (n_spots, n_genes) squared OOF residual
    epistemic: np.ndarray  # (n_spots, n_genes) ensemble variance
    s2_xen: np.ndarray  # (n_genes,) noise floor
    dropout_score: np.ndarray  # (n_genes,) Visium zero-fraction


def run_oracle(
    triad: TriadData,
    seed: int = 0,
    n_models: int = 8,
    n_folds: int = 4,
    n_boot: int = 200,
    fdr_alpha: float = 0.05,
    sigma2_reg: np.ndarray | None = None,
) -> OracleResult:
    z_est = 0.5 * (triad.xen_rep1 + triad.xen_rep2)  # denoised cleaner-reference target
    s2_xen = noise_floor(triad.xen_rep1, triad.xen_rep2)
    if sigma2_reg is None:
        sigma2_reg = np.zeros_like(s2_xen)

    pbar, epi = _ensemble_oof(triad.morph, z_est, n_models, n_folds, seed)
    resid2 = (z_est - pbar) ** 2
    aleatoric = resid2 - epi  # remove epistemic (ensemble disagreement)

    U = np.clip(aleatoric.mean(0) - s2_xen - sigma2_reg, 0.0, None)

    # bootstrap over spots for a per-gene p-value that U <= 0
    rng = np.random.default_rng(seed + 999)
    n = triad.morph.shape[0]
    leq0 = np.zeros(U.size)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Ub = aleatoric[idx].mean(0) - s2_xen - sigma2_reg
        leq0 += (Ub <= 0).astype(float)
    pvalue = (1.0 + leq0) / (1.0 + n_boot)
    flagged = bh_fdr(pvalue, fdr_alpha) & (U > 0)

    dropout_score = (triad.visium == 0).mean(0)
    return OracleResult(U, pvalue, flagged, resid2, epi, s2_xen, dropout_score)


def separation_auroc(
    res: OracleResult, is_unident: np.ndarray, is_dropout: np.ndarray
) -> dict:
    """Pre-registered separation metric: deferral(=U) tracks unidentifiability, not dropout."""
    auroc_unid = safe_auroc(is_unident, res.U)
    auroc_drop = safe_auroc(is_dropout, res.U)
    return {
        "auroc_unid": auroc_unid,
        "auroc_dropout": auroc_drop,
        "gap": auroc_unid - auroc_drop,
    }


def spatial_structure(triad: TriadData, res: OracleResult, seed: int = 0) -> dict:
    """Moran's I of the per-spot aleatoric map over flagged genes (is U spatially structured?)."""
    if res.flagged.sum() == 0:
        return {"morans_i": 0.0, "pvalue": 1.0}
    a_spot = np.clip(
        res.resid2[:, res.flagged] - res.epistemic[:, res.flagged], 0, None
    ).mean(1)
    w = grid_weights(triad.coords)
    mi, p = morans_i_pvalue(a_spot, w, seed=seed)
    return {"morans_i": mi, "pvalue": p}


def selective_risk_curve(
    res: OracleResult, error: np.ndarray, grid: int = 20
) -> np.ndarray:
    """Retained-fraction vs selective-risk: abstain on highest-U genes first.

    Returns (grid, 2): columns = [retained_fraction, mean_error_on_retained]. `error` is per-gene loss.
    """
    order = np.argsort(res.U)  # keep low-U (most identifiable) first
    out = []
    for frac in np.linspace(1.0, 0.05, grid):
        k = max(1, int(frac * res.U.size))
        keep = order[:k]
        out.append((k / res.U.size, float(np.mean(error[keep]))))
    return np.array(out)
