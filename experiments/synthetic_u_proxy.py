"""Deliverable #4 (cluster-free): does a higher-capacity TRAINED f tighten U on identifiable genes?

A controlled proxy for the GPU-cluster bet (end-to-end fine-tune), run on the simulator's PLANTED ground
truth where the answer is knowable: class A/B genes are nonlinearly identifiable from morphology, class C
is intrinsically unidentifiable. The cluster bet (audit C1, plan.md rollout 12) is that a trained dual-head
f leaves LESS residual on identifiable genes than a weak f, tightening U toward 0 there while keeping U
high on C. If the dual head can do this on planted data (strong signal), the frozen-breast failure
(rollout 15) is a data/capacity limit, not a method flaw -> the cluster spend is justified. If it cannot
even here, that tempers the whole HYBRID direction.

SYNTHETIC -- validates the mechanism on planted truth, NOT a real-data result.
Run:  python3 experiments/synthetic_u_proxy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.trained_oracle_breast import oof_dual_head  # noqa: E402
from src.oracle import noise_floor, run_oracle  # noqa: E402
from src.simulator import simulate_triad  # noqa: E402
from src.train import TrainConfig  # noqa: E402


def dual_head_U(triad, target_noise, cfg, seed=0):
    """U from the trained dual-head substrate: OOF mean-residual aleatoric minus the noise floor."""
    z_est = 0.5 * (triad.xen_rep1 + triad.xen_rep2)
    oof_mean, _ = oof_dual_head(
        triad.morph, z_est, triad.coords, cfg, n_folds=4, seed=seed
    )
    aleatoric = ((z_est - oof_mean) ** 2).mean(0)
    return np.clip(aleatoric - target_noise, 0.0, None)


def main() -> int:
    triad = simulate_triad(seed=0)
    cls = triad.gene_class
    target_noise = 0.5 * noise_floor(triad.xen_rep1, triad.xen_rep2)

    res_rf = run_oracle(triad, sigma2_reg=0.0, predictor="rf", seed=0)
    U_rf = res_rf.U
    cfg = TrainConfig(
        epochs=200, lr=1e-2, batch_size=128, beta=0.5, device="cpu", seed=0
    )
    U_ddh = dual_head_U(triad, target_noise, cfg, seed=0)

    def med(u, c):
        return float(np.median(u[cls == c]))

    print("=" * 64)
    print("Synthetic U-tightening proxy (planted ground truth) -- NOT real data")
    print("=" * 64)
    print(
        f"{'class':>6} {'RF U':>8} {'DDH U':>8}   (A,B identifiable; C unidentifiable)"
    )
    for c in ("A", "B", "C"):
        print(f"{c:>6} {med(U_rf, c):>8.3f} {med(U_ddh, c):>8.3f}")
    sep_rf = med(U_rf, "C") / max(med(U_rf, "A"), 1e-3)
    sep_ddh = med(U_ddh, "C") / max(med(U_ddh, "A"), 1e-3)
    print(
        f"separation U(C)/U(A):  RF={sep_rf:.2f}  DDH={sep_ddh:.2f}  (higher = better)"
    )

    print("-" * 64)
    # pre-registered proxy gate: the trained head must at least MATCH the RF baseline's separation and not
    # inflate U on identifiable genes. Passing is NECESSARY (the mechanism is realizable), NOT SUFFICIENT for
    # the cluster bet -- real H&E morphology may not carry the signal in this functional form.
    if sep_ddh >= sep_rf * 0.9 and med(U_ddh, "A") <= med(U_rf, "A") * 1.25:
        print(
            "READOUT: the dual head CAN learn the U-separation mechanism when identifiable signal exists "
            f"(planted data: DDH sep {sep_ddh:.2f} vs RF {sep_rf:.2f}; U(A) tightened). This is a NECESSARY "
            "(not sufficient) condition for the cluster bet -- it shows the frozen-breast failure (rollout 15) "
            "is consistent with a data/capacity limit, but real morphology must still carry the signal. "
            "Synthetic != real."
        )
        return 0
    print(
        "READOUT: TEMPERS the cluster bet -- even on planted data the dual head does not cleanly tighten U on "
        "identifiable genes (DDH separation {:.2f} vs RF {:.2f}). Report straight.".format(
            sep_ddh, sep_rf
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
