"""Synthetic H&E->ST triad with PLANTED ground truth.

The swap point for real data is `src/loaders.py` (same TriadData interface). The simulator plants three
gene classes so the gate (tests/test_gate.py) can verify the oracle separates true biological
unidentifiability from technical dropout BEFORE any real download:

  - class A (id_nonlinear) : predictable from morphology via a NONLINEAR map, low dropout -> U ~ 0
                             (only if the oracle's predictor is nonlinear-capable; a linear substrate
                             under-fits and wrongly inflates U here -- this is what the negative-control
                             test in tests/test_gate.py exploits, addressing audit C1/C4).
  - class B (id_dropout)   : same nonlinear-identifiable signal but HEAVY Visium dropout -> U ~ 0 on the
                             Xenium target, yet a dropout-NAIVE oracle (one that regressed on Visium)
                             would flag it. The B-vs-C contrast is only meaningful because B is genuinely
                             identifiable in clean Xenium space (audit C4).
  - class C (unident)      : NOT predictable from morphology (field orthogonal to morph), low dropout ->
                             U > 0, spatially structured.

A correct oracle flags C (not B). A linear-substrate oracle additionally mis-flags A -> the negative
control catches a broken method, so the gate can actually fail (audit M2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriadData:
    coords: np.ndarray  # (n_spots, 2) spatial coordinates
    morph: (
        np.ndarray
    )  # (n_spots, n_feat) morphology features (stand-in for UNI embeddings)
    xen_rep1: np.ndarray  # (n_spots, n_genes) Xenium replicate 1 (cleaner reference)
    xen_rep2: (
        np.ndarray
    )  # (n_spots, n_genes) Xenium replicate 2 (gives the technical noise floor)
    visium: np.ndarray  # (n_spots, n_genes) Visium target (dropout-corrupted)
    gene_class: (
        np.ndarray
    )  # (n_genes,) one of {"A","B","C"} -- ground truth, unused by the oracle


def _smooth_field(rng, grid, n_cols, scale=2.0):
    """A spatially smooth random field over the grid via Gaussian blur of white noise."""
    from scipy.ndimage import gaussian_filter

    g = int(round(np.sqrt(grid.shape[0])))
    out = np.empty((grid.shape[0], n_cols))
    for c in range(n_cols):
        w = rng.standard_normal((g, g))
        w = gaussian_filter(w, sigma=scale, mode="wrap")
        out[:, c] = (w.reshape(-1) - w.mean()) / (w.std() + 1e-8)
    return out


def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def _nonlinear_basis(morph):
    """A SHARED set of nonlinear morphology features (pairwise products, centered squares, saturations).

    Each is orthogonal to any linear projection, so Ridge cannot form them and leaves ~full variance in
    its residual; a RandomForest recovers them. Identifiable genes are random LINEAR combinations of these
    shared bases, so a multi-output forest learns the (few) bases jointly and drives their residual -- and
    hence U -- to ~0, instead of failing to specialize per gene.
    """
    # all terms are pairwise products or CENTERED squares -> exactly zero linear correlation with morph,
    # so Ridge cannot partially "cheat" via a linear component (which would weaken the negative control).
    m = morph
    cols = [
        m[:, 0] * m[:, 1],
        m[:, 2] * m[:, 3],
        m[:, 4] * m[:, 5],
        m[:, 6] * m[:, 7],
        m[:, 0] ** 2 - 1.0,
        m[:, 3] ** 2 - 1.0,
        m[:, 1] * m[:, 4],
        m[:, 2] * m[:, 5],
    ]
    return np.stack([_zscore(c) for c in cols], axis=1)


def simulate_triad(
    grid_size: int = 30,
    n_feat: int = 8,
    n_genes_per_class: int = 40,
    xen_noise: float = 0.30,
    dropout_p_high: float = 0.6,
    seed: int = 0,
) -> TriadData:
    rng = np.random.default_rng(seed)
    g = grid_size
    coords = (
        np.stack(np.meshgrid(np.arange(g), np.arange(g)), axis=-1)
        .reshape(-1, 2)
        .astype(float)
    )

    # Morphology features are i.i.d. per spot (NOT a smooth spatial field): this isolates the machinery
    # test from spatial-extrapolation error under the spatial-block CV (audit H5) -- class A/B become
    # cleanly learnable out-of-fold, while the unidentifiable class C carries ALL the spatial structure.
    # Real H&E morphology is spatially smooth; that regime is exercised on real data, not in this gate.
    morph = rng.standard_normal((coords.shape[0], n_feat))
    phi = _nonlinear_basis(morph)  # shared nonlinear bases for the identifiable classes

    classes, z_cols = [], []
    for cls, count in (
        ("A", n_genes_per_class),
        ("B", n_genes_per_class),
        ("C", n_genes_per_class),
    ):
        for _ in range(count):
            if cls in ("A", "B"):
                z = phi @ rng.standard_normal(
                    phi.shape[1]
                )  # identifiable: nonlinear-in-morph, RF-learnable
            else:
                z = _smooth_field(rng, coords, 1, scale=2.5)[
                    :, 0
                ]  # unidentifiable: orthogonal field
            z_cols.append(_zscore(z))
            classes.append(cls)
    z_clean = np.stack(z_cols, axis=1)  # (n, n_genes)
    gene_class = np.array(classes)

    xen_rep1 = z_clean + rng.normal(0, xen_noise, z_clean.shape)
    xen_rep2 = z_clean + rng.normal(0, xen_noise, z_clean.shape)

    # Visium: clean signal + measurement noise + zero-inflation dropout (heavy on class B)
    visium = z_clean + rng.normal(0, xen_noise, z_clean.shape)
    p_drop = np.where(gene_class == "B", dropout_p_high, 0.05)
    mask = rng.random(z_clean.shape) < p_drop[None, :]
    visium = np.where(mask, 0.0, visium)

    return TriadData(coords, morph, xen_rep1, xen_rep2, visium, gene_class)
