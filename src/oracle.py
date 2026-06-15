"""The unidentifiability oracle: noise floor, ensemble aleatoric estimate, U, and selective-risk layer.

Implements the pre-registered estimand (preregistration.md sec 2 + sec 11 amendments):

    U(g) = aleatoric_residual(g)  -  target_noise(g)  -  sigma2_reg(g)

where aleatoric is the out-of-fold residual of an ensemble of predictors with the *epistemic* (ensemble
disagreement) component removed. Audit fixes folded in (rollout 9 -> 10):

  - C1: the substrate `f` is a NONLINEAR ensemble (RandomForest) by default, not linear Ridge -- a linear
        f leaves nonlinear-but-learnable structure in the residual and inflates U (the negative-control
        test exploits this). `predictor` selects the family.
  - C3: the independent anti-circularity arm `f'` must pass a DIFFERENT architecture (e.g. 'knn'), not the
        same model reseeded.
  - SA1: the prediction target z_est = mean(rep1,rep2) has HALF the single-replicate technical variance,
         so the subtracted floor is `0.5 * sigma2_xen`, not the full single-replicate estimate.
  - H5: out-of-fold CV uses SPATIAL-BLOCK folds (not random KFold) to block spatial leakage.
  - C2: `sigma2_reg` is explicit. It is 0.0 ONLY when there is no registration step (synthetic). Real
        data MUST pass `registration_sigma2(...)`; a silent zero on real data is a pre-registered kill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from .metrics import bh_fdr, grid_weights, morans_i_pvalue, safe_auroc
from .simulator import TriadData


def _make_predictor(kind: str, seed: int):
    """Predictor factory. 'rf' = nonlinear substrate (default f); 'knn' = different architecture (f');
    'ridge' = linear (negative control only -- under-fits the nonlinear map)."""
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if kind == "knn":
        return KNeighborsRegressor(n_neighbors=15, weights="distance")
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    raise ValueError(f"unknown predictor kind: {kind!r}")


def noise_floor(rep1: np.ndarray, rep2: np.ndarray) -> np.ndarray:
    """Unbiased per-gene Xenium single-replicate technical variance: Var(r1-r2)=2*sigma2 -> sigma2."""
    return 0.5 * np.mean((rep1 - rep2) ** 2, axis=0)


def _spatial_block_folds(coords: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each spot to a fold by SPATIAL BLOCK so held-out spots are spatially separated (audit H5)."""
    rng = np.random.default_rng(seed)
    nb = max(n_folds, int(np.ceil(np.sqrt(4 * n_folds))))  # more blocks than folds
    mins = coords.min(0)
    span = np.ptp(coords, axis=0) + 1e-9
    bij = np.clip(np.floor((coords - mins) / span * nb).astype(int), 0, nb - 1)
    block = bij[:, 0] * nb + bij[:, 1]
    ublocks = np.unique(block)
    rng.shuffle(ublocks)
    fold_of = {b: i % n_folds for i, b in enumerate(ublocks)}
    return np.array([fold_of[b] for b in block])


def _ensemble_oof(
    X: np.ndarray,
    Y: np.ndarray,
    coords: np.ndarray,
    predictor: str,
    n_models: int,
    n_folds: int,
    seed: int,
):
    """Out-of-fold bootstrap ensemble with spatial-block CV. Returns (pbar, epistemic), each (n,n_genes)."""
    rng = np.random.default_rng(seed)
    pbar = np.zeros_like(Y)
    epi = np.zeros_like(Y)
    fold = _spatial_block_folds(coords, n_folds, seed)
    for f in range(n_folds):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if te.size == 0 or tr.size == 0:
            continue
        preds = np.empty((n_models, te.size, Y.shape[1]))
        for m in range(n_models):
            rows = rng.choice(
                tr, size=tr.size, replace=True
            )  # bootstrap for epistemic spread
            model = _make_predictor(predictor, seed + m)
            model.fit(X[rows], Y[rows])
            preds[m] = model.predict(X[te])
        pbar[te] = preds.mean(0)
        epi[te] = preds.var(0)
    return pbar, epi


def registration_sigma2(
    triad: TriadData, predictor: str = "rf", shift: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Estimate per-gene sigma2_reg from a +/-1-bin coordinate-perturbation of the morphology lookup.

    On real serial sections, registration error injects target variance the predictor cannot explain.
    We approximate it as the extra residual variance induced when morphology features are perturbed by a
    +/-`shift`-bin spatial roll (a stand-in for sub-bin misregistration). On the synthetic regular grid
    with no registration step this is ~0 (and callers may pass 0.0 directly). preregistration.md sec 11-D.
    """
    coords = triad.coords
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    rolled = order[
        (inv + int(shift)) % order.size
    ]  # neighbor-shifted morphology assignment
    z_est = 0.5 * (triad.xen_rep1 + triad.xen_rep2)
    base, _ = _ensemble_oof(triad.morph, z_est, coords, predictor, 3, 4, seed)
    pert, _ = _ensemble_oof(triad.morph[rolled], z_est, coords, predictor, 3, 4, seed)
    extra = ((z_est - pert) ** 2).mean(0) - ((z_est - base) ** 2).mean(0)
    return np.clip(extra, 0.0, None)


@dataclass
class OracleResult:
    U: np.ndarray  # (n_genes,) intrinsic unidentifiability per gene (>=0)
    pvalue: np.ndarray  # (n_genes,) bootstrap p that U<=0
    flagged: np.ndarray  # (n_genes,) bool: BH-FDR significant AND U>0
    resid2: np.ndarray  # (n_spots, n_genes) squared OOF residual
    epistemic: np.ndarray  # (n_spots, n_genes) ensemble variance
    s2_xen: np.ndarray  # (n_genes,) single-replicate noise floor
    target_noise: (
        np.ndarray
    )  # (n_genes,) noise floor of the averaged target (0.5*s2_xen)
    dropout_score: np.ndarray  # (n_genes,) Visium zero-fraction


def run_oracle(
    triad: TriadData,
    sigma2_reg,
    predictor: str = "rf",
    seed: int = 0,
    n_models: int = 5,
    n_folds: int = 4,
    n_boot: int = 200,
    fdr_alpha: float = 0.05,
) -> OracleResult:
    """Compute U. `sigma2_reg` is REQUIRED and explicit (audit C2): pass 0.0 only when there is no
    registration step (synthetic); on real data pass registration_sigma2(...). `predictor` selects the
    substrate family ('rf' nonlinear default; 'knn' for the independent f' arm; 'ridge' = broken control).
    """
    z_est = 0.5 * (triad.xen_rep1 + triad.xen_rep2)  # denoised cleaner-reference target
    s2_xen = noise_floor(triad.xen_rep1, triad.xen_rep2)
    target_noise = (
        0.5 * s2_xen
    )  # audit SA1: averaged target carries half the single-rep variance
    s2_reg = _as_per_gene(sigma2_reg, s2_xen.shape)

    pbar, epi = _ensemble_oof(
        triad.morph, z_est, triad.coords, predictor, n_models, n_folds, seed
    )
    resid2 = (z_est - pbar) ** 2
    aleatoric = resid2 - epi  # remove epistemic (ensemble disagreement)

    U = np.clip(aleatoric.mean(0) - target_noise - s2_reg, 0.0, None)

    # bootstrap over spots for a per-gene p-value that U <= 0 (unclipped statistic)
    rng = np.random.default_rng(seed + 999)
    n = triad.morph.shape[0]
    leq0 = np.zeros(U.size)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Ub = aleatoric[idx].mean(0) - target_noise - s2_reg
        leq0 += (Ub <= 0).astype(float)
    pvalue = (1.0 + leq0) / (1.0 + n_boot)
    flagged = bh_fdr(pvalue, fdr_alpha) & (U > 0)

    dropout_score = (triad.visium == 0).mean(0)
    return OracleResult(
        U, pvalue, flagged, resid2, epi, s2_xen, target_noise, dropout_score
    )


def _as_per_gene(value, shape) -> np.ndarray:
    """Broadcast a scalar/array sigma2_reg to per-gene. Raises on None so the term is never silently 0."""
    if value is None:
        raise ValueError(
            "sigma2_reg must be explicit (audit C2). Pass 0.0 only for synthetic/no-registration data; "
            "on real data pass registration_sigma2(triad)."
        )
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"sigma2_reg shape {arr.shape} != per-gene {shape}")
    return arr


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


def top_u_genes(res: OracleResult, top_frac: float = 0.34) -> np.ndarray:
    """Indices of the highest-U genes (the ones the selective layer would defer first)."""
    k = max(1, int(round(top_frac * res.U.size)))
    return np.argsort(res.U)[::-1][:k]


def spatial_structure(
    triad: TriadData, res: OracleResult, seed: int = 0, top_frac: float = 0.34
) -> dict:
    """Moran's I of the per-spot aleatoric map over the highest-U genes (is the deferred U spatially
    structured?). Uses the top-U set rather than the BH flag: because U upper-bounds (it carries finite
    learner error), the flag can saturate, so the *ranking* is the meaningful selective-layer signal.

    Guard (audit B): if U has zero spread (e.g. fully floor-clipped to 0), the top-U set is an arbitrary
    argsort tie-order, so Moran's I would be a meaningless artifact -> return NaN, not a fake number."""
    if np.ptp(res.U) < 1e-12:
        return {"morans_i": float("nan"), "pvalue": float("nan")}
    top = top_u_genes(res, top_frac)
    a_spot = np.clip(res.resid2[:, top] - res.epistemic[:, top], 0, None).mean(1)
    w = grid_weights(triad.coords)
    mi, p = morans_i_pvalue(a_spot, w, seed=seed)
    return {"morans_i": mi, "pvalue": p}


def risk_coverage_curve(
    order_score: np.ndarray, error: np.ndarray, grid: int = 20
) -> np.ndarray:
    """Selective-risk-coverage curve: abstain on the highest-`order_score` items first.

    Returns (grid, 2): columns = [retained_fraction, mean_error_on_retained]. Keeping the lowest-score
    (most identifiable) items first, this is the real utility metric (preregistration.md sec 5): a better
    deferral signal pushes more error onto the abstained set, so risk on the retained set falls faster as
    coverage drops. Works for any score (raw-variance-U, the trained dual-head's U, an oracle, ...), so the
    baseline and the HYBRID head are compared on one footing.
    """
    order_score = np.asarray(order_score, dtype=float)
    error = np.asarray(error, dtype=float)
    order = np.argsort(order_score)  # keep low-score (most identifiable) first
    n = order_score.size
    out = []
    for frac in np.linspace(1.0, 0.05, grid):
        k = max(1, int(round(frac * n)))
        keep = order[:k]
        out.append((k / n, float(np.mean(error[keep]))))
    return np.array(out)


def aurc(curve: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower = more efficient deferral)."""
    cov, risk = curve[:, 0], curve[:, 1]
    o = np.argsort(cov)
    return float(np.trapz(risk[o], cov[o]))


def selective_risk_curve(
    res: OracleResult, error: np.ndarray, grid: int = 20
) -> np.ndarray:
    """Retained-fraction vs selective-risk for an OracleResult, abstaining on highest-U genes first."""
    return risk_coverage_curve(res.U, error, grid)
