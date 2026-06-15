"""Download-free tests for the post-hoc conformal layer (Phase-3 PDC; preregistration.md H1).

Pins the two guarantees the paper leans on:
  - split-conformal marginal coverage hits 1-alpha on exchangeable scores;
  - the H1 phenomenon: under heteroscedastic spatial groups, NAIVE (pooled) conformal under-covers the
    hard group, and group-conditional (spatial-Mondrian) conformal restores per-group coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import (  # noqa: E402
    conformal_quantile,
    empirical_coverage,
    mondrian_q_for,
    mondrian_quantiles,
    per_group_coverage,
    selective_conformal_sweep,
)


def test_quantile_infinite_when_calibration_too_small_for_level():
    # (n+1)(1-alpha) > n  ->  cannot guarantee 1-alpha  ->  +inf (interval = everything)
    assert conformal_quantile(np.array([0.3]), alpha=0.1) == np.inf
    assert np.isfinite(conformal_quantile(np.arange(100.0), alpha=0.1))


def test_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    scores = np.abs(rng.standard_normal(5000))  # exchangeable |residual|
    cal, test = scores[:2500], scores[2500:]
    q = conformal_quantile(cal, alpha=0.1)
    cov = float((test <= q).mean())
    assert 0.88 <= cov <= 0.93, f"marginal coverage {cov:.3f} should sit near 0.90"


def test_mondrian_fixes_spatial_undercoverage():
    """Two groups with very different residual scales: naive pooled q under-covers the high-scale group;
    per-group (Mondrian) q restores ~1-alpha in BOTH (the pre-registered H1 fix)."""
    rng = np.random.default_rng(1)
    n = 4000
    groups = np.where(np.arange(n) < n // 2, "easy", "hard")
    scale = np.where(groups == "easy", 0.3, 3.0)
    scores = np.abs(rng.standard_normal(n)) * scale
    cal = np.arange(n) % 2 == 0
    test = ~cal
    alpha = 0.1

    q_naive = conformal_quantile(scores[cal], alpha)
    cov_naive = per_group_coverage(
        scores[test], q_naive * np.ones(test.sum()), groups[test]
    )

    qmap = mondrian_quantiles(scores[cal], groups[cal], alpha)
    q_test = mondrian_q_for(groups[test], qmap, fallback=q_naive)
    cov_mond = per_group_coverage(scores[test], q_test, groups[test])

    # naive under-covers the hard group well below 0.90; Mondrian brings BOTH groups back near 0.90
    assert cov_naive["hard"] < 0.85, (
        f"expected naive under-coverage on hard group, got {cov_naive['hard']:.3f}"
    )
    assert cov_mond["hard"] >= 0.86 and cov_mond["easy"] >= 0.86, (
        f"Mondrian must restore per-group coverage: {cov_mond}"
    )


def test_empirical_coverage_and_per_group():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    lo = np.array([-1.0, -1.0, 1.0, 1.0])
    hi = np.array([1.0, 1.0, 2.0, 2.0])
    assert empirical_coverage(y, lo, hi) == 0.5
    g = np.array(["a", "a", "b", "b"])
    pg = per_group_coverage_from_intervals(y, lo, hi, g)
    assert pg["a"] == 1.0 and pg["b"] == 0.0


def per_group_coverage_from_intervals(y, lo, hi, groups):
    out = {}
    for grp in np.unique(groups):
        m = groups == grp
        out[grp] = empirical_coverage(y[m], lo[m], hi[m])
    return out


def test_selective_conformal_sweep_keeps_coverage_and_tightens_width():
    """The headline product: abstaining on highest-U genes lets conformal give TIGHTER guaranteed intervals
    on the retained set, while marginal coverage stays ~1-alpha at every retained fraction."""
    rng = np.random.default_rng(0)
    n_spot, n_gene = 200, 60
    # SHUFFLE difficulty so gene INDEX != difficulty (audit H): otherwise a flat/uninformative U argsorts to
    # low-index=low-difficulty genes and "tightens" with no real signal. Decoupling forces U to be informative.
    difficulty = rng.permutation(np.linspace(0.2, 3.0, n_gene))
    U = difficulty + 0.05 * rng.standard_normal(n_gene)  # U tracks difficulty (noisy)
    abs_resid = np.abs(rng.standard_normal((n_spot, n_gene)) * difficulty[None, :])

    table = selective_conformal_sweep(U, abs_resid, alpha=0.1, grid=8, seed=0)
    assert table.shape == (8, 3)  # [retained_fraction, coverage, mean_width]
    frac, cov, width = table[:, 0], table[:, 1], table[:, 2]
    assert np.all((cov >= 0.85) & (cov <= 0.95)), f"coverage off target: {cov}"
    keep_small, keep_all = width[np.argmin(frac)], width[np.argmax(frac)]
    assert keep_small < 0.9 * keep_all, (
        f"informative U must clearly tighten: small={keep_small:.3f} all={keep_all:.3f}"
    )

    # NEGATIVE CONTROL: a flat (uninformative) U must NOT meaningfully tighten intervals
    flat = selective_conformal_sweep(
        np.ones(n_gene), abs_resid, alpha=0.1, grid=8, seed=0
    )
    fw_small, fw_all = flat[np.argmin(flat[:, 0]), 2], flat[np.argmax(flat[:, 0]), 2]
    assert fw_small >= 0.9 * fw_all, (
        f"flat U must not tighten intervals (got {fw_small:.3f} vs {fw_all:.3f})"
    )


def test_selective_conformal_sweep_sigma_path():
    """The locally-adaptive (|resid|/sigma) path must keep coverage + tighten width as we abstain."""
    rng = np.random.default_rng(7)
    n_spot, n_gene = 200, 60
    difficulty = np.linspace(0.2, 3.0, n_gene)
    U = difficulty + 0.05 * rng.standard_normal(n_gene)
    abs_resid = np.abs(rng.standard_normal((n_spot, n_gene)) * difficulty[None, :])
    sigma = np.clip(
        rng.random((n_spot, n_gene)) * difficulty[None, :] + 0.1, 0.05, None
    )

    table = selective_conformal_sweep(
        U, abs_resid, alpha=0.1, grid=6, seed=0, sigma=sigma
    )
    frac, cov, width = table[:, 0], table[:, 1], table[:, 2]
    assert table.shape == (6, 3)
    assert np.all((cov >= 0.84) & (cov <= 0.95)), f"adaptive coverage off: {cov}"
    assert np.all(width > 0)
    assert width[np.argmin(frac)] < width[np.argmax(frac)], (
        "adaptive abstention must tighten width"
    )


def test_selective_conformal_sweep_raises_on_too_few_niches():
    try:
        selective_conformal_sweep(np.array([0.1]), np.zeros((1, 5)), alpha=0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError when no test niches remain")


if __name__ == "__main__":
    test_quantile_infinite_when_calibration_too_small_for_level()
    test_split_conformal_marginal_coverage()
    test_mondrian_fixes_spatial_undercoverage()
    test_empirical_coverage_and_per_group()
    test_selective_conformal_sweep_keeps_coverage_and_tightens_width()
    test_selective_conformal_sweep_sigma_path()
    test_selective_conformal_sweep_raises_on_too_few_niches()
    print("ALL conformal TESTS PASS")
