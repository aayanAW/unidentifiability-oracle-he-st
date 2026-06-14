"""Characterization test for the synthetic U-tightening proxy (deliverable #4).

Locks the cluster-bet premise on PLANTED ground truth: the trained dual-head substrate must leave more
residual on intrinsically-unidentifiable genes (class C) than on nonlinearly-identifiable ones (class A),
i.e. rank U(C) > U(A). This characterizes existing code (predictor + train), so it is a research-claim
gate, not new-code TDD. Small grid + few epochs keeps it fast (CPU, seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.synthetic_u_proxy import dual_head_U  # noqa: E402
from src.oracle import noise_floor  # noqa: E402
from src.simulator import simulate_triad  # noqa: E402
from src.train import TrainConfig  # noqa: E402


def test_dual_head_ranks_unidentifiable_above_identifiable():
    triad = simulate_triad(grid_size=14, n_genes_per_class=10, seed=0)
    cls = triad.gene_class
    target_noise = 0.5 * noise_floor(triad.xen_rep1, triad.xen_rep2)
    cfg = TrainConfig(
        epochs=120, lr=1e-2, batch_size=64, beta=0.5, device="cpu", seed=0
    )
    U = dual_head_U(triad, target_noise, cfg, seed=0)
    u_A = float(np.median(U[cls == "A"]))
    u_C = float(np.median(U[cls == "C"]))
    assert u_C > u_A, (
        f"dual head must rank unidentifiable C above identifiable A: U(A)={u_A:.3f} U(C)={u_C:.3f}"
    )


if __name__ == "__main__":
    test_dual_head_ranks_unidentifiable_above_identifiable()
    print("ALL synthetic_proxy TESTS PASS")
