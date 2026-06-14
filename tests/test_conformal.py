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


if __name__ == "__main__":
    test_quantile_infinite_when_calibration_too_small_for_level()
    test_split_conformal_marginal_coverage()
    test_mondrian_fixes_spatial_undercoverage()
    test_empirical_coverage_and_per_group()
    print("ALL conformal TESTS PASS")
