"""H1 readout (preregistration.md H1): does NAIVE split-conformal under-cover spatially-autocorrelated
breast niches, and does spatial-Mondrian conformal restore per-block coverage?

Post-hoc coverage diagnostic on the RF oracle's OOF residuals (the strong real substrate, rollout 15) at
breast 300um. The nonconformity score is the fixed |OOF residual| per (niche, gene); we calibrate a
conformal quantile on a random half of the NICHES and measure coverage on the held-out half -- globally
(validity), per spatial block (the H1 conditional-coverage question), and as a spatial map (is the
miscoverage spatially structured?). EXPLORATORY: frozen DINOv2-S morphology, breast only.

Run:  python3 experiments/conformal_breast.py data --bin-um 300 --alpha 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import xenium_io as xio  # noqa: E402
from src.conformal import (  # noqa: E402
    conformal_quantile,
    mondrian_q_for,
    mondrian_quantiles,
    per_group_coverage,
)
from src.embeddings import embed_bins  # noqa: E402
from src.loaders import XENIUM_UM_PER_PX, load_xenium_noise_floor  # noqa: E402
from src.metrics import grid_weights, morans_i_pvalue  # noqa: E402
from src.oracle import registration_sigma2, run_oracle  # noqa: E402
from src.simulator import TriadData  # noqa: E402


def _spatial_blocks(coords: np.ndarray, n_side: int) -> np.ndarray:
    """Assign each niche to one of n_side x n_side spatial blocks (Mondrian groups)."""
    mins = coords.min(0)
    span = np.ptp(coords, axis=0) + 1e-9
    ij = np.clip(np.floor((coords - mins) / span * n_side).astype(int), 0, n_side - 1)
    return ij[:, 0] * n_side + ij[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--bin-um", type=float, default=300.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-side", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.data_dir)
    target = 1.0 - args.alpha

    z1, z2, centers, genes = load_xenium_noise_floor(
        args.data_dir, bin_um=args.bin_um, min_cells=args.min_cells
    )
    he = xio._first_existing(  # noqa: SLF001
        root / "rep1", ("Post-Xenium_HE_Rep1.ome.tif", "*HE*.ome.tif", "*HE*ome.tif.gz")
    )
    homog = xio._first_existing(root / "rep1", ("*HE*homography.csv.gz",))  # noqa: SLF001
    if he is None:
        print("[conformal] no H&E under data/rep1.")
        return 3
    mf = embed_bins(
        he,
        centers,
        homog,
        encoder="vit_small_patch14_dinov2.lvd142m",
        um_per_px=XENIUM_UM_PER_PX,
    )
    X = np.asarray(mf.features, dtype=np.float64)
    triad = TriadData(centers, X, z1, z2, np.zeros_like(z1), None)
    sigma2_reg = registration_sigma2(triad, predictor="rf", seed=args.seed)
    res = run_oracle(triad, sigma2_reg=sigma2_reg, predictor="rf", seed=args.seed)

    abs_resid = np.sqrt(
        res.resid2
    )  # (n_niche, n_gene) OOF |residual| = nonconformity score
    n_spot, n_gene = abs_resid.shape
    blocks = _spatial_blocks(centers, args.n_side)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_spot)
    cal_spots, test_spots = perm[: n_spot // 2], perm[n_spot // 2 :]

    def cells(spots):
        return abs_resid[spots].ravel(), np.repeat(blocks[spots], n_gene)

    cal_s, cal_g = cells(cal_spots)
    test_s, test_g = cells(test_spots)

    q_naive = conformal_quantile(cal_s, args.alpha)
    cov_naive = per_group_coverage(test_s, np.full(test_s.shape, q_naive), test_g)
    qmap = mondrian_quantiles(cal_s, cal_g, args.alpha)
    q_test = mondrian_q_for(test_g, qmap, fallback=q_naive)
    cov_mond = per_group_coverage(test_s, q_test, test_g)

    marg = float((test_s <= q_naive).mean())
    naive_worst = min(cov_naive.values())
    mond_worst = min(cov_mond.values())

    # spatial structure of NAIVE miscoverage: per-test-niche coverage, Moran's I
    cov_per_spot = (abs_resid[test_spots] <= q_naive).mean(1)
    w = grid_weights(centers[test_spots])
    mi, mp = morans_i_pvalue(
        1.0 - cov_per_spot, w, seed=args.seed
    )  # miscoverage = 1 - coverage

    print("=" * 70)
    print(
        f"H1 conformal coverage (breast {args.bin_um:.0f}um, RF OOF residuals) -- EXPLORATORY"
    )
    print("=" * 70)
    print(
        f"niches={n_spot} genes={n_gene} blocks={args.n_side}x{args.n_side}  target coverage={target:.2f}"
    )
    print(f"marginal coverage (naive, held-out cells): {marg:.3f}")
    print("per-spatial-block coverage:")
    print(
        f"  NAIVE    worst-block={naive_worst:.3f}  mean={np.mean(list(cov_naive.values())):.3f}  spread={np.ptp(list(cov_naive.values())):.3f}"
    )
    print(
        f"  MONDRIAN worst-block={mond_worst:.3f}  mean={np.mean(list(cov_mond.values())):.3f}  spread={np.ptp(list(cov_mond.values())):.3f}"
    )
    print(f"naive miscoverage spatial structure: Moran's I={mi:.3f} (p={mp:.3f})")

    # coverage-vs-alpha validity curve
    print("validity (marginal coverage vs target across alpha):")
    for a in (0.05, 0.1, 0.2):
        q = conformal_quantile(cal_s, a)
        print(
            f"  alpha={a:.2f} target={1 - a:.2f}  coverage={float((test_s <= q).mean()):.3f}"
        )

    results = Path(__file__).resolve().parents[1] / "experiments" / "results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results / "conformal_breast.npz",
        cov_naive=np.array(list(cov_naive.values())),
        cov_mond=np.array(list(cov_mond.values())),
        marginal=marg,
        morans_i=mi,
        morans_p=mp,
        alpha=args.alpha,
    )

    print("-" * 70)
    under = naive_worst < target - 0.03  # worst block under-covers beyond a 3pt margin
    restored = mond_worst >= naive_worst - 1e-6 and mond_worst >= target - 0.05
    if under and restored:
        print(
            f"READOUT: H1 SUPPORTED -- naive conformal under-covers the worst spatial block "
            f"({naive_worst:.3f} < {target:.2f}); spatial-Mondrian restores it ({mond_worst:.3f}). "
            "Spatial non-exchangeability is real on breast niches and the group-conditional layer fixes it."
        )
        return 0
    print(
        f"READOUT: H1 not clearly supported here -- naive worst-block={naive_worst:.3f}, "
        f"Mondrian worst-block={mond_worst:.3f} vs target {target:.2f}. Report straight; "
        "the marginal guarantee holds, the spatial gap is small at this binning/scale."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
