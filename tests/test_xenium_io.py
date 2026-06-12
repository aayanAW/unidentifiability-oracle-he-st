"""Download-free tests for the real-loader machinery: niche binning + serial-section registration.

Builds two synthetic Xenium "sections" sharing a smooth spatial expression field, with section 2 rotated
+ translated (mimicking the unregistered serial-section frame) and given independent Poisson sampling
noise. A correct registration + niche matching must recover HIGH cross-replicate concordance (low
sigma2_xen) -- i.e. the noise floor must reflect sampling noise, not registration failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracle import noise_floor  # noqa: E402
from src.xenium_io import (  # noqa: E402
    XeniumSection,
    apply_affine,
    bin_pseudobulk,
    matched_bins,
    rigid_register,
)


def _tissue_mask(xy, span):
    """Anisotropic tissue outline: a 2:1 ellipse with a corner notch removed (breaks rotational symmetry,
    as real serial sections do -- this is what makes rigid registration well-posed)."""
    u, v = xy[:, 0] / span, xy[:, 1] / span
    in_ellipse = ((u - 0.5) / 0.45) ** 2 + ((v - 0.5) / 0.22) ** 2 <= 1.0
    notch = (u > 0.75) & (v > 0.55)  # asymmetric bite to fix the 180-deg flip
    return in_ellipse & ~notch


def _make_section(rng, n_cells=4000, span=2000.0, n_genes=30, rate=8.0):
    """Cells over an anisotropic tissue mask; gene rates = smooth spatial functions of (masked) position."""
    pts = rng.uniform(0, span, size=(n_cells * 3, 2))
    xy = pts[_tissue_mask(pts, span)][:n_cells]
    u, v = xy[:, 0] / span, xy[:, 1] / span
    fields = []
    for g in range(n_genes):
        a, b = (g % 5) + 1, (g % 3) + 1
        fields.append(0.5 + 0.5 * np.sin(a * np.pi * u) * np.cos(b * np.pi * v))
    lam = rate * np.stack(fields, 1)  # (n_cells, n_genes) intensity
    counts = rng.poisson(lam).astype(float)
    return xy, counts


def _rotate(xy, deg, t):
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ xy.T).T + t


def test_binning_shapes_and_counts_conserved():
    rng = np.random.default_rng(0)
    xy, counts = _make_section(rng)
    sec = XeniumSection(xy, counts, [f"g{i}" for i in range(counts.shape[1])])
    keys, centers, pb, ncell = bin_pseudobulk(sec, bin_um=100.0)
    assert keys.shape[0] == centers.shape[0] == pb.shape[0] == ncell.shape[0]
    assert pb.shape[1] == counts.shape[1]
    # pseudobulk conserves total counts and total cells
    assert np.isclose(pb.sum(), counts.sum())
    assert int(ncell.sum()) == counts.shape[0]


def test_rigid_register_recovers_known_transform():
    rng = np.random.default_rng(1)
    xy, _ = _make_section(rng)
    moved = _rotate(xy, 37.0, np.array([500.0, -300.0]))
    aff = rigid_register(moved, xy, bin_um=100.0)  # map moved -> original frame
    back = apply_affine(moved, aff)
    # median per-point alignment error well under one niche after registration
    err = np.median(np.linalg.norm(back - xy, axis=1))
    assert err < 50.0, f"registration error too high: {err:.1f} um"


def test_noise_floor_below_one_after_registration():
    """Two serial sections of the SAME field (different cells/frame) must show reproducible signal."""
    rng = np.random.default_rng(2)
    xy1, c1 = _make_section(rng, rate=12.0)
    xy2_raw, c2 = _make_section(
        rng, rate=12.0
    )  # independent cells, same field definition
    xy2 = _rotate(xy2_raw, 25.0, np.array([-400.0, 250.0]))  # unregistered serial frame
    genes = [f"g{i}" for i in range(c1.shape[1])]
    s1 = XeniumSection(xy1, c1, genes)
    s2 = XeniumSection(xy2, c2, genes)

    z1, z2, centers, gout = matched_bins(s1, s2, bin_um=100.0, min_cells=5)
    assert len(centers) > 20, "too few co-occupied niches -- registration likely failed"
    s2_xen = noise_floor(z1, z2)
    # most genes should clear the floor (reproducible spatial signal recovered across the two frames)
    cleared = (s2_xen < 0.7).mean()
    assert cleared > 0.5, (
        f"only {cleared:.2f} of genes clear the floor; registration/binning broken"
    )


if __name__ == "__main__":
    test_binning_shapes_and_counts_conserved()
    test_rigid_register_recovers_known_transform()
    test_noise_floor_below_one_after_registration()
    print("ALL xenium_io TESTS PASS")
