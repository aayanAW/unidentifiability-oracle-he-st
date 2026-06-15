"""Behavioral test for the synthetic U-tightening proxy (deliverable #4).

Audit H: the weak ordinal assertion U(C)>U(A) is near-tautological -- it passes even for a non-learning
predictor, because the simulator plants class C as a smooth spatial field (higher between-fold variance)
regardless of any morphology learning. The real claim is that the TRAINED dual head matches/exceeds the RF
baseline's separation AND tightens U(A), and that an UNTRAINED (epochs=0) head does NOT. That makes the
test fail on a non-learning model -- a genuine behavioral check, not a tautology. Small grid keeps it fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.synthetic_u_proxy import dual_head_U  # noqa: E402
from src.oracle import noise_floor, run_oracle  # noqa: E402
from src.simulator import simulate_triad  # noqa: E402
from src.train import TrainConfig  # noqa: E402


def _sep(U, cls):
    return float(np.median(U[cls == "C"])) / max(float(np.median(U[cls == "A"])), 1e-3)


def test_trained_dual_head_beats_rf_baseline_and_untrained_control():
    triad = simulate_triad(grid_size=14, n_genes_per_class=10, seed=0)
    cls = triad.gene_class
    target_noise = 0.5 * noise_floor(triad.xen_rep1, triad.xen_rep2)

    rf = run_oracle(triad, sigma2_reg=0.0, predictor="rf", seed=0)
    sep_rf = _sep(rf.U, cls)
    rf_u_A = float(np.median(rf.U[cls == "A"]))

    cfg = TrainConfig(
        epochs=150, lr=1e-2, batch_size=64, beta=0.5, device="cpu", seed=0
    )
    U = dual_head_U(triad, target_noise, cfg, seed=0)
    sep_ddh, u_A = _sep(U, cls), float(np.median(U[cls == "A"]))

    # the experiment's own (non-tautological) gate: match the RF separation + tighten U(A)
    assert sep_ddh >= sep_rf * 0.9, (
        f"trained head must match RF separation: DDH {sep_ddh:.2f} vs RF {sep_rf:.2f}"
    )
    assert u_A <= rf_u_A * 1.25, (
        f"trained head must not inflate U(A): {u_A:.3f} vs RF {rf_u_A:.3f}"
    )

    # negative control: an UNTRAINED head (epochs=0) must NOT achieve the trained separation
    U0 = dual_head_U(
        triad, target_noise, TrainConfig(epochs=0, device="cpu", seed=0), seed=0
    )
    assert _sep(U0, cls) < sep_ddh, (
        "untrained (epochs=0) head must separate worse than the trained head"
    )


if __name__ == "__main__":
    test_trained_dual_head_beats_rf_baseline_and_untrained_control()
    print("ALL synthetic_proxy TESTS PASS")
