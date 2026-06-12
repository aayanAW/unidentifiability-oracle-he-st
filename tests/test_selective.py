"""Download-free tests for the selective-risk-coverage metric (the real utility metric, prereg sec 5).

A deferral score that ranks high-error items highest must produce a risk-coverage curve whose AURC is
below random, and an oracle score (= the true error) must be the best achievable. These pin the headline
metric so the trained-oracle-vs-RF comparison in experiments/trained_oracle_breast.py is meaningful.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracle import aurc, risk_coverage_curve  # noqa: E402


def test_curve_shape_and_coverage_monotone():
    rng = np.random.default_rng(0)
    err = rng.random(100)
    score = rng.random(100)
    curve = risk_coverage_curve(score, err, grid=20)
    assert curve.shape == (20, 2)
    cov = curve[:, 0]
    assert cov[0] >= cov[-1] and cov.max() <= 1.0 and cov.min() > 0.0
    assert np.isfinite(curve[:, 1]).all()


def test_oracle_score_beats_random_beats_nothing():
    rng = np.random.default_rng(1)
    err = rng.random(200) ** 2  # skewed errors so deferral has headroom
    # oracle score == the error itself: abstaining highest-error first minimizes retained risk
    aurc_oracle = aurc(risk_coverage_curve(err, err, grid=40))
    # random score: deferral uncorrelated with error -> retained risk ~ flat at the mean
    aurc_random = np.mean(
        [aurc(risk_coverage_curve(rng.random(200), err, grid=40)) for _ in range(20)]
    )
    full_risk = float(err.mean())
    assert aurc_oracle < aurc_random, "oracle ordering must beat random"
    # random deferral cannot do better than keeping everything (risk stays ~ the full-set mean)
    assert aurc_random <= full_risk + 1e-6


if __name__ == "__main__":
    test_curve_shape_and_coverage_monotone()
    test_oracle_score_beats_random_beats_nothing()
    print("ALL selective TESTS PASS")
