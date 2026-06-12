"""Synthetic H&E->ST triad with PLANTED ground truth.

The swap point for real data is `src/loaders.py` (same TriadData interface). The simulator plants three
gene classes so the gate test (tests/test_gate.py) can verify the oracle separates true biological
unidentifiability from technical dropout BEFORE any real download:

  - class A (id_clean)   : predictable from morphology, low dropout    -> U should be ~0
  - class B (id_dropout) : predictable from morphology, HIGH dropout   -> U should be ~0 (the discriminator)
  - class C (unident)    : NOT predictable from morphology, low dropout -> U should be > 0, spatially structured

A correct oracle flags C (not B), proving it tracks biology, not measurement noise.
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
    )  # (n_genes,) one of {"A","B","C"}  -- ground truth, unused by the oracle


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


def simulate_triad(
    grid_size: int = 24,
    n_feat: int = 8,
    n_genes_per_class: int = 40,
    xen_noise: float = 0.35,
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
    n = coords.shape[0]

    morph = _smooth_field(rng, coords, n_feat, scale=2.5)  # smooth morphology field

    classes, z_cols = [], []
    for cls, count in (
        ("A", n_genes_per_class),
        ("B", n_genes_per_class),
        ("C", n_genes_per_class),
    ):
        for _ in range(count):
            if cls in ("A", "B"):
                w = rng.standard_normal(n_feat)
                z = morph @ w  # identifiable: linear in morphology
            else:
                z = _smooth_field(rng, coords, 1, scale=2.5)[
                    :, 0
                ]  # unidentifiable: field orthogonal to morph
            z = (z - z.mean()) / (z.std() + 1e-8)
            z_cols.append(z)
            classes.append(cls)
    z_clean = np.stack(z_cols, axis=1)  # (n, n_genes)
    gene_class = np.array(classes)

    s2 = xen_noise**2
    xen_rep1 = z_clean + rng.normal(0, xen_noise, z_clean.shape)
    xen_rep2 = z_clean + rng.normal(0, xen_noise, z_clean.shape)

    # Visium: clean signal + measurement noise + zero-inflation dropout (heavy on class B)
    visium = z_clean + rng.normal(0, xen_noise, z_clean.shape)
    p_drop = np.where(gene_class == "B", dropout_p_high, 0.05)
    mask = rng.random(z_clean.shape) < p_drop[None, :]
    visium = np.where(mask, 0.0, visium)

    return TriadData(coords, morph, xen_rep1, xen_rep2, visium, gene_class)
