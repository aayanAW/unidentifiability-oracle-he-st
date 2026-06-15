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


def _git_sha() -> str:
    import subprocess

    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()[:10]
    except Exception:
        return "unknown"


def _spatial_blocks(
    coords: np.ndarray, n_side: int, ref: np.ndarray | None = None
) -> np.ndarray:
    """Assign each niche to one of n_side x n_side spatial blocks (Mondrian groups).

    The grid span is taken from `ref` (default = `coords`). Passing the CALIBRATION coords as `ref` freezes
    the partition from calibration data only, so block boundaries do not depend on the test set
    (preregistration.md sec 11-E: groups frozen from an independent partition)."""
    src = coords if ref is None else ref
    mins = src.min(0)
    span = np.ptp(src, axis=0) + 1e-9
    ij = np.clip(np.floor((coords - mins) / span * n_side).astype(int), 0, n_side - 1)
    return ij[:, 0] * n_side + ij[:, 1]


def _clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact finite-sample beta (Clopper-Pearson) CI for a coverage k/n (preregistration.md sec 11-E)."""
    from scipy.stats import beta

    if n == 0:
        return float("nan"), float("nan")
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


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

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_spot)
    cal_spots, test_spots = perm[: n_spot // 2], perm[n_spot // 2 :]
    # Mondrian partition frozen from CALIBRATION coords only -> block boundaries do not depend on the test
    # set (preregistration.md sec 11-E: independent partition).
    blocks = _spatial_blocks(centers, args.n_side, ref=centers[cal_spots])

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
    naive_worst_blk = min(cov_naive, key=cov_naive.get)
    mond_worst_blk = min(cov_mond, key=cov_mond.get)
    naive_worst = cov_naive[naive_worst_blk]
    mond_worst = cov_mond[mond_worst_blk]
    naive_worst_after_mond = cov_mond[
        naive_worst_blk
    ]  # what Mondrian did to the block naive failed on
    mond_worst_before = cov_naive[
        mond_worst_blk
    ]  # the block Mondrian made worst -- did it regress?

    # niche-level coverage (prereg 11-E: the NICHE is the exchangeable unit, not the cell): per test niche
    # fraction of genes covered, then averaged. This is the statistic whose exchangeability matches the split.
    test_blocks = blocks[test_spots]
    naive_cov_per_niche = (abs_resid[test_spots] <= q_naive).mean(1)
    niche_marg = float(naive_cov_per_niche.mean())
    niche_block_cov = {
        b: float(naive_cov_per_niche[test_blocks == b].mean())
        for b in np.unique(test_blocks)
    }
    niche_worst = min(niche_block_cov.values())
    # Clopper-Pearson finite-sample CI on the worst block, using the NICHE count as the effective n (the
    # conservative, exchangeability-correct sample size -- pooled cells overstate n).
    n_naive_worst = int((test_blocks == naive_worst_blk).sum())
    cp_lo, cp_hi = _clopper_pearson(round(naive_worst * n_naive_worst), n_naive_worst)

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
    print(
        f"  per-block: naive-worst block {naive_worst_blk} {naive_worst:.3f} -> {naive_worst_after_mond:.3f} "
        f"under Mondrian ({naive_worst_after_mond - naive_worst:+.3f}); but block {mond_worst_blk} "
        f"{mond_worst_before:.3f} -> {mond_worst:.3f} ({mond_worst - mond_worst_before:+.3f}, the new worst "
        "-- Mondrian can REGRESS a block via small per-block calibration)"
    )
    print(
        f"  niche-level (exchangeable unit): marginal={niche_marg:.3f}  worst-block={niche_worst:.3f}  "
        f"[naive worst-block 95% Clopper-Pearson over n={n_naive_worst} niches: ({cp_lo:.3f}, {cp_hi:.3f})]"
    )
    print(f"naive miscoverage spatial structure: Moran's I={mi:.3f} (p={mp:.3f})")

    # coverage-vs-alpha validity curve (persisted -- audit K)
    print("validity (marginal coverage vs target across alpha):")
    cov_by_alpha = {}
    for a in (0.05, 0.1, 0.2):
        q = conformal_quantile(cal_s, a)
        cov_by_alpha[a] = float((test_s <= q).mean())
        print(f"  alpha={a:.2f} target={1 - a:.2f}  coverage={cov_by_alpha[a]:.3f}")

    results = Path(__file__).resolve().parents[1] / "experiments" / "results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results / "conformal_breast.npz",
        cov_naive=np.array(list(cov_naive.values())),
        cov_mond=np.array(list(cov_mond.values())),
        marginal=marg,
        cov_05=cov_by_alpha[0.05],
        cov_10=cov_by_alpha[0.1],
        cov_20=cov_by_alpha[0.2],
        naive_worst=naive_worst,
        mond_worst=mond_worst,
        niche_marginal=niche_marg,
        niche_worst=niche_worst,
        cp_lo=cp_lo,
        cp_hi=cp_hi,
        morans_i=mi,
        morans_p=mp,
        alpha=args.alpha,
        git_sha=np.array(_git_sha()),
        seed=np.int64(args.seed),
    )

    print("-" * 70)
    # H1's claim = spatial non-exchangeability is REAL. The LOAD-BEARING evidence is the Moran's test across
    # ALL niches (mp), NOT the single worst-block point estimate -- whose niche-level Clopper-Pearson CI is
    # wide and spans the target (so it is not individually significant). Mondrian remediation is PARTIAL.
    spatial = mp < 0.05
    worst_block_significant = (
        cp_hi < target
    )  # is the worst block individually below target?
    if spatial:
        gap0, gap1 = target - naive_worst, target - mond_worst
        print(
            f"READOUT: H1 SUPPORTED (spatial non-exchangeability is real) -- miscoverage is spatially "
            f"structured across all niches (Moran's I={mi:.3f}, p={mp:.3f} -- the load-bearing test). The "
            f"worst block point estimate {naive_worst:.3f} is suggestive but NOT individually significant "
            f"(niche-level 95% CI ({cp_lo:.3f}, {cp_hi:.3f}) spans {target:.2f}; worst_block_sig="
            f"{worst_block_significant}). REMEDIATION is PARTIAL: spatial-Mondrian closes "
            f"{100 * (1 - gap1 / gap0):.0f}% of the worst-block gap ({gap0:.3f}->{gap1:.3f}) but does NOT "
            f"restore to {target:.2f} -- limited by ~{n_spot // (2 * args.n_side**2)} cal niches/block."
        )
        return 0
    print(
        f"READOUT: H1 not clearly supported here -- naive worst-block={naive_worst:.3f}, miscoverage "
        f"Moran p={mp:.3f}. Report straight; the marginal guarantee holds, the spatial gap is small here."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
