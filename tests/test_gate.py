"""Gate script (the test suite, per CLAUDE.md).

Validates the oracle MACHINERY on synthetic data with planted ground truth, before any real download:
the method must flag the unidentifiable genes (class C), ignore dropout-corrupted-but-identifiable genes
(class B), produce a spatially-structured U, and hold the separation under an independent f'.

Run:  python tests/test_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracle import run_oracle, separation_auroc, spatial_structure  # noqa: E402
from src.simulator import simulate_triad  # noqa: E402


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_oracle_separates_unidentifiability_from_dropout(seed: int = 0):
    triad = simulate_triad(seed=seed)
    is_C = triad.gene_class == "C"
    is_B = triad.gene_class == "B"

    res_f = run_oracle(triad, seed=seed)
    res_fp = run_oracle(triad, seed=seed + 101)

    frac = res_f.flagged.mean()
    sep_f = separation_auroc(res_f, is_C, is_B)
    sep_fp = separation_auroc(res_fp, is_C, is_B)
    spatial = spatial_structure(triad, res_f, seed=seed)
    prec = is_C[res_f.flagged].mean() if res_f.flagged.sum() else 0.0
    b_flagged = int(is_B[res_f.flagged].sum())

    print(
        f"  structured-U fraction={frac:.3f}  gap_f={sep_f['gap']:.3f}  gap_f'={sep_fp['gap']:.3f}"
    )
    print(
        f"  Moran's I={spatial['morans_i']:.3f} (p={spatial['pvalue']:.3f})  flag precision(C)={prec:.3f}  B flagged={b_flagged}"
    )

    _check(
        "structured-U fraction in [0.15, 0.45] (recovers ~the planted 1/3 class C)",
        0.15 <= frac <= 0.45,
    )
    _check("separation gap >= 0.10 under f", sep_f["gap"] >= 0.10)
    _check("separation gap >= 0.10 under independent f'", sep_fp["gap"] >= 0.10)
    _check(
        "U is spatially structured (Moran's I>0, p<0.05)",
        spatial["morans_i"] > 0 and spatial["pvalue"] < 0.05,
    )
    _check("flags are mostly true class C (precision>=0.8)", prec >= 0.8)
    _check("dropout class B is NOT over-flagged (<=2 genes)", b_flagged <= 2)


def main():
    print("Gate: oracle separates biological unidentifiability from technical dropout")
    # multiple seeds -> guards against a lucky single draw
    for s in (0, 1, 2):
        print(f"seed {s}:")
        test_oracle_separates_unidentifiability_from_dropout(s)
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
