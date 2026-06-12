"""Gate script (the test suite, per CLAUDE.md).

Validates the oracle MACHINERY on synthetic data with planted ground truth, before any real download.
The pre-registered claim (H2) is that the selective layer's DEFERRALS (driven by U) concentrate on the
intrinsically unidentifiable genes (class C), not on dropout-corrupted-but-identifiable ones (class B) --
a RANKING claim. We test the ranking (separation AUROC, top-U precision, spatial structure of the top-U
set), not an absolute flag count: U is an UPPER BOUND that carries finite learner error
(preregistration.md sec 11-B), so identifiable genes keep a nonzero U and an absolute-count band would be
a reverse-engineered rubber stamp (audit M2).

Audit (rollout 9->10) hardening:
  - f = nonlinear RF substrate; f' = KNN (different ARCHITECTURE, audit C3).
  - sigma2_reg passed explicitly (0.0: synthetic has no registration step, audit C2).
  - NEGATIVE CONTROL: a broken LINEAR (ridge) oracle loses the separation -- the gate can FAIL
    (audit C1/C4/M2).

Run:  python tests/test_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracle import (  # noqa: E402
    run_oracle,
    separation_auroc,
    spatial_structure,
    top_u_genes,
)
from src.simulator import simulate_triad  # noqa: E402


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def _med(u, m):
    return float(np.median(u[m]))


def test_oracle_separates_unidentifiability_from_dropout(seed: int = 0):
    triad = simulate_triad(seed=seed)
    is_C = triad.gene_class == "C"
    is_B = triad.gene_class == "B"
    is_A = triad.gene_class == "A"

    res_f = run_oracle(triad, sigma2_reg=0.0, predictor="rf", seed=seed)
    res_fp = run_oracle(
        triad, sigma2_reg=0.0, predictor="knn", seed=seed + 101
    )  # different architecture

    gap_f = separation_auroc(res_f, is_C, is_B)["gap"]
    gap_fp = separation_auroc(res_fp, is_C, is_B)["gap"]
    ratio_A = _med(res_f.U, is_C) / max(_med(res_f.U, is_A), 1e-3)
    ratio_B = _med(res_f.U, is_C) / max(_med(res_f.U, is_B), 1e-3)
    top = top_u_genes(res_f, top_frac=is_C.mean())  # top 1/3 by U
    top_prec_C = float(is_C[top].mean())
    spatial = spatial_structure(triad, res_f, seed=seed)

    print(
        f"  gap_f={gap_f:.3f} gap_f'={gap_fp:.3f}  U med A={_med(res_f.U, is_A):.3f} "
        f"B={_med(res_f.U, is_B):.3f} C={_med(res_f.U, is_C):.3f}"
    )
    print(
        f"  C/A={ratio_A:.2f} C/B={ratio_B:.2f}  top-U precision(C)={top_prec_C:.3f}  "
        f"Moran's I={spatial['morans_i']:.3f} (p={spatial['pvalue']:.3f})"
    )

    _check(
        "deferral<->unidentifiability separation gap >= 0.20 under nonlinear f",
        gap_f >= 0.20,
    )
    _check(
        "separation gap >= 0.20 under independent-architecture f' (knn)", gap_fp >= 0.20
    )
    _check(
        "U(C) >= 1.5x U(A) and 1.5x U(B) (ranks unidentifiable above identifiable)",
        ratio_A >= 1.5 and ratio_B >= 1.5,
    )
    _check(
        "top-U deferral set is mostly class C (precision >= 0.65)", top_prec_C >= 0.65
    )
    _check(
        "top-U aleatoric map is spatially structured (Moran's I>0, p<0.05)",
        spatial["morans_i"] > 0 and spatial["pvalue"] < 0.05,
    )


def test_negative_control_broken_linear_oracle_fails(seed: int = 0):
    """A LINEAR substrate under-fits the nonlinear IDENTIFIABLE map (class A), inflating U_A toward the
    truly-unidentifiable level, so it can no longer tell identifiable-A from unidentifiable-C. The
    nonlinear substrate keeps U_A low and separates them. If the gate passed regardless of substrate it
    would be a rubber stamp -- this is its failable side (audit C1/C4/M2)."""
    triad = simulate_triad(seed=seed)
    is_A = triad.gene_class == "A"

    res_rf = run_oracle(triad, sigma2_reg=0.0, predictor="rf", seed=seed)
    res_ridge = run_oracle(triad, sigma2_reg=0.0, predictor="ridge", seed=seed)
    uA_rf = _med(res_rf.U, is_A)
    uA_ridge = _med(res_ridge.U, is_A)
    print(
        f"  U(identifiable class A)  rf(correct)={uA_rf:.3f}  ridge(broken)={uA_ridge:.3f}  "
        f"inflation={uA_ridge / max(uA_rf, 1e-3):.2f}x"
    )
    _check(
        "broken linear substrate inflates U on identifiable genes >= 1.5x vs the nonlinear substrate",
        uA_ridge >= 1.5 * uA_rf,
    )


def main():
    print("Gate: oracle separates biological unidentifiability from technical dropout")
    for s in (0, 1, 2):
        print(f"seed {s}:")
        test_oracle_separates_unidentifiability_from_dropout(s)
    print("negative control (broken linear oracle must lose the separation):")
    test_negative_control_broken_linear_oracle_fails(0)
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
