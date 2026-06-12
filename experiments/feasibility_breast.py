"""Phase 1 -- the cheapest decisive experiment (feasibility.md).

Tests the load-bearing hypothesis H2: does a non-zero, spatially-structured intrinsic unidentifiability U
survive the Xenium-replicate noise floor, and does abstention (driven by U) concentrate on genuinely
unidentifiable genes rather than dropout-corrupted ones -- under BOTH f and an independent f'?

Default runs on the synthetic simulator (planted ground truth, no download). `--real data/` uses the
GSE243280 loader once wired. Prints the pre-registered CONFIRM / KILL / AMBIGUOUS verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracle import run_oracle, separation_auroc, spatial_structure  # noqa: E402

# pre-registered thresholds (feasibility.md)
CONFIRM_FRACTION = 0.20
KILL_FRACTION = 0.05
GAP_CONFIRM = 0.10
GAP_KILL = 0.05


def load(args):
    if args.real:
        from src.loaders import load_breast_triad

        return load_breast_triad(args.real)
    from src.simulator import simulate_triad

    return simulate_triad(seed=args.seed)


def verdict(frac, gap_f, gap_fp, morans_p):
    """Pre-registered decision logic, requiring the separation to hold under f AND f'."""
    gap_ok = (gap_f >= GAP_CONFIRM) and (gap_fp >= GAP_CONFIRM)
    spatial_ok = morans_p < 0.05
    if frac >= CONFIRM_FRACTION and gap_ok and spatial_ok:
        return "CONFIRM"
    if frac < KILL_FRACTION or gap_f < GAP_KILL or gap_fp < GAP_KILL or not spatial_ok:
        return "KILL"
    return "AMBIGUOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synthetic", action="store_true", help="(default) run on planted simulator"
    )
    ap.add_argument(
        "--real", type=str, default=None, help="path to real GSE243280 data dir"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    triad = load(args)
    n_genes = triad.xen_rep1.shape[1]
    is_unident = triad.gene_class == "C" if triad.gene_class is not None else None
    is_dropout = triad.gene_class == "B" if triad.gene_class is not None else None

    res_f = run_oracle(triad, seed=args.seed)  # f
    res_fp = run_oracle(
        triad, seed=args.seed + 101
    )  # independent f' (different bootstrap/feature draws)

    frac = res_f.flagged.mean()
    spatial = spatial_structure(triad, res_f, seed=args.seed)

    print("=" * 64)
    print("Phase 1 feasibility -- unidentifiability oracle (breast)")
    print("=" * 64)
    print(f"genes={n_genes}  spots={triad.morph.shape[0]}")
    print(
        f"structured-U fraction (FDR<0.05 & U>0): {frac:.3f}   [confirm>={CONFIRM_FRACTION}, kill<{KILL_FRACTION}]"
    )
    print(
        f"Moran's I of U map: {spatial['morans_i']:.3f}  (perm p={spatial['pvalue']:.3f})"
    )

    if is_unident is not None:
        sep_f = separation_auroc(res_f, is_unident, is_dropout)
        sep_fp = separation_auroc(res_fp, is_unident, is_dropout)
        gap_f, gap_fp = sep_f["gap"], sep_fp["gap"]
        print(
            f"separation AUROC  f : unid={sep_f['auroc_unid']:.3f} dropout={sep_f['auroc_dropout']:.3f} gap={gap_f:.3f}"
        )
        print(
            f"separation AUROC  f': unid={sep_fp['auroc_unid']:.3f} dropout={sep_fp['auroc_dropout']:.3f} gap={gap_fp:.3f}"
        )
        # precision of the flag against the planted unidentifiable class
        if res_f.flagged.sum():
            prec = is_unident[res_f.flagged].mean()
            print(
                f"flag precision vs planted class C: {prec:.3f}  (class-B-dropout flagged: {is_dropout[res_f.flagged].sum()})"
            )
        v = verdict(frac, gap_f, gap_fp, spatial["pvalue"])
        print("-" * 64)
        print(f"VERDICT: {v}")
        return 0 if v == "CONFIRM" else (2 if v == "KILL" else 1)

    print(
        "(real data: no ground-truth class labels; report structured fraction + Moran's + held-out coverage)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
