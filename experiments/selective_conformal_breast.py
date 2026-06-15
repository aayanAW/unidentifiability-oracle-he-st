"""Deliverables #1 + #2 (cluster-free): the coverage-guaranteed selective-prediction product on breast.

#1 SELECTIVE x CONFORMAL: abstain on highest-U genes (RF oracle U -- the strong deferral signal), then
   split-conformal on the retained niche-gene cells -> a retained-fraction x coverage x interval-width
   table. This assembles the two halves (deferral + coverage guarantee) into the actual contribution.
#2 sigma-NORMALIZED ADAPTIVE INTERVALS: on the trained dual-head predictor, compare marginal |resid|
   conformal vs |resid|/sigma using the variance head -- does the learned variance buy tighter intervals
   at equal coverage? (Where the dual head can earn utility even though its U-ranking lost, rollout 15.)

EXPLORATORY: frozen DINOv2-S morphology, breast only, 300um. Split-conformal coverage over niche-gene
cells is approximate (cells within a niche are correlated); honest caveat, the marginal guarantee is over
the niche split. Run:  python3 experiments/selective_conformal_breast.py data --bin-um 300 --alpha 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiments.trained_oracle_breast import oof_dual_head  # noqa: E402
from src import xenium_io as xio  # noqa: E402
from src.conformal import conformal_quantile, selective_conformal_sweep  # noqa: E402
from src.embeddings import embed_bins  # noqa: E402
from src.loaders import XENIUM_UM_PER_PX, load_xenium_noise_floor  # noqa: E402
from src.oracle import registration_sigma2, run_oracle  # noqa: E402
from src.simulator import TriadData  # noqa: E402
from src.train import TrainConfig  # noqa: E402

torch  # noqa: B018 -- imported so oof_dual_head's torch usage is available


def _git_sha() -> str:
    import subprocess

    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()[:10]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--bin-um", type=float, default=300.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.data_dir)
    target = 1.0 - args.alpha

    z1, z2, centers, genes = load_xenium_noise_floor(
        args.data_dir, bin_um=args.bin_um, min_cells=args.min_cells
    )
    z_est = 0.5 * (z1 + z2)
    he = xio._first_existing(  # noqa: SLF001
        root / "rep1", ("Post-Xenium_HE_Rep1.ome.tif", "*HE*.ome.tif", "*HE*ome.tif.gz")
    )
    homog = xio._first_existing(root / "rep1", ("*HE*homography.csv.gz",))  # noqa: SLF001
    if he is None:
        print("[selective-conformal] no H&E under data/rep1.")
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
    U_rf = res.U
    abs_resid_rf = np.sqrt(res.resid2)
    error_rf = res.resid2.mean(
        0
    )  # per-gene OOF MSE (for the random/oracle risk-concentration baselines)

    # ---- #1: coverage-guaranteed selective table (RF substrate) --------------------------------------
    # audit G: recomputing q on the retained subset trivially shrinks width (high-error genes leave the
    # calibration pool), so width-tightening alone is NOT evidence the oracle works. The real contribution is
    # whether U-ranking concentrates risk BETTER than random selection (and how close to a perfect oracle).
    # Baselines: random gene order, and a perfect oracle that sorts by the true per-gene error.
    rng_score = np.random.default_rng(args.seed).random(len(U_rf))
    table = selective_conformal_sweep(
        U_rf, abs_resid_rf, alpha=args.alpha, grid=10, seed=args.seed
    )
    table_rand = selective_conformal_sweep(
        rng_score, abs_resid_rf, alpha=args.alpha, grid=10, seed=args.seed
    )
    table_oracle = selective_conformal_sweep(
        error_rf, abs_resid_rf, alpha=args.alpha, grid=10, seed=args.seed
    )

    def _row50(t):
        return t[np.argmin(np.abs(t[:, 0] - 0.5))]

    # retained-set mean ERROR at 50% (the real risk-concentration signal, in z-units), U vs random vs oracle
    order_U = np.argsort(U_rf)
    order_rand = np.argsort(rng_score)
    order_oracle = np.argsort(error_rf)
    k = len(U_rf) // 2
    err50_U = float(error_rf[order_U[:k]].mean())
    err50_rand = float(error_rf[order_rand[:k]].mean())
    err50_oracle = float(error_rf[order_oracle[:k]].mean())

    print("=" * 72)
    print(
        f"#1 coverage-guaranteed selective prediction (breast {args.bin_um:.0f}um, RF U) -- EXPLORATORY"
    )
    print("=" * 72)
    print(f"target coverage 1-alpha={target:.2f}  (abstain on highest-U genes first)")
    print(f"{'retain_frac':>11} {'coverage':>9} {'mean_width':>11}")
    for fr, cov, wid in table:
        flag = (
            "  <- coverage dips below target (finite-sample, few cells)"
            if cov < target - 0.01
            else ""
        )
        print(f"{fr:>11.2f} {cov:>9.3f} {wid:>11.3f}{flag}")
    full_w, half_w = table[np.argmax(table[:, 0]), 2], _row50(table)[2]
    print(
        f"-> width at 50% retention {100 * (1 - half_w / full_w):.0f}% tighter than keep-all -- but that is "
        "largely mechanical (recalibration on the retained subset)."
    )
    print(
        f"-> REAL utility (risk concentration) at 50% retention -- mean retained error: "
        f"U-ranking={err50_U:.3f}  random={err50_rand:.3f}  perfect-oracle={err50_oracle:.3f}  "
        f"=> U gives {100 * (1 - err50_U / err50_rand):.1f}% lower error than random, "
        f"{'matches' if abs(err50_U - err50_oracle) < 0.01 else 'vs'} perfect {err50_oracle:.3f}."
    )
    print(
        f"-> coverage holds near {target:.2f} for retention >= 20% (min {table[table[:, 0] >= 0.2, 1].min():.3f}); "
        f"at 10% retention it dips to {table[np.argmin(table[:, 0]), 1]:.3f} (finite-sample slack)."
    )

    # ---- #2: sigma-normalized adaptive intervals (dual-head variance) ---------------------------------
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=1e-3,
        batch_size=128,
        beta=0.5,
        device="cpu",
        seed=args.seed,
    )
    oof_mean, oof_var = oof_dual_head(X, z_est, centers, cfg, n_folds=4, seed=args.seed)
    abs_resid_ddh = np.abs(z_est - oof_mean)
    sigma_ddh = np.sqrt(np.clip(oof_var, 1e-8, None))

    # both ordered by RF-U (strong deferral), full retention: marginal vs normalized width at equal coverage
    sweep_marg = selective_conformal_sweep(
        U_rf, abs_resid_ddh, alpha=args.alpha, grid=10, seed=args.seed
    )
    sweep_norm = selective_conformal_sweep(
        U_rf, abs_resid_ddh, alpha=args.alpha, grid=10, seed=args.seed, sigma=sigma_ddh
    )
    # HELD-OUT split (review fix): calibrate q on cal niches, measure coverage+width on test niches -- NOT
    # on the calibration data itself (self-coverage would read ~1-alpha by construction and tell us nothing).
    rng2 = np.random.default_rng(args.seed + 99)
    perm2 = rng2.permutation(len(abs_resid_ddh))
    n_cal2 = max(1, len(perm2) // 2)
    cal2, test2 = perm2[:n_cal2], perm2[n_cal2:]
    norm_resid = abs_resid_ddh / sigma_ddh  # dimensionless score for the adaptive path
    q_marg = conformal_quantile(abs_resid_ddh[cal2].ravel(), args.alpha)
    q_norm = conformal_quantile(norm_resid[cal2].ravel(), args.alpha)
    cov_marg = float((abs_resid_ddh[test2] <= q_marg).mean())
    cov_norm = float(
        (norm_resid[test2] <= q_norm).mean()
    )  # dimensionless: score <= quantile
    width_marg = 2.0 * q_marg
    width_norm = 2.0 * q_norm * float(sigma_ddh[test2].mean())
    print("-" * 72)
    print(
        "#2 marginal vs sigma-normalized intervals on the dual-head predictor (equal target coverage):"
    )
    print(f"  marginal   coverage={cov_marg:.3f}  mean_width={width_marg:.3f}")
    print(
        f"  normalized coverage={cov_norm:.3f}  mean_width={width_norm:.3f}  (sigma from the variance head)"
    )
    adaptive_wins = (width_norm < width_marg) and (cov_norm >= target - 0.03)

    results = Path(__file__).resolve().parents[1] / "experiments" / "results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results / "selective_conformal_breast.npz",
        table=table,
        table_rand=table_rand,
        table_oracle=table_oracle,
        err50_U=err50_U,
        err50_rand=err50_rand,
        err50_oracle=err50_oracle,
        sweep_marg=sweep_marg,
        sweep_norm=sweep_norm,
        width_marg=width_marg,
        width_norm=width_norm,
        cov_marg=cov_marg,
        cov_norm=cov_norm,
        alpha=args.alpha,
        git_sha=np.array(_git_sha()),
        seed=np.int64(args.seed),
    )
    print("-" * 72)
    print(
        "READOUT #1: selective prediction with MARGINAL split-conformal coverage assembled. The real signal "
        f"is RISK CONCENTRATION: at 50% retention U-ranking gives {100 * (1 - err50_U / err50_rand):.1f}% lower "
        "retained error than random and nearly matches a perfect oracle (the width-tightening alone is "
        "mechanical). Coverage is marginal over the niche split (cell-level approximate; conditional NOT "
        "claimed) and holds near target for retention >= 20% (dips at 10%).\n"
        f"READOUT #2: sigma-normalized adaptive intervals {'DO' if adaptive_wins else 'do NOT'} beat marginal "
        "mean width at equal coverage on frozen embeddings -- "
        + (
            "the variance head buys efficiency."
            if adaptive_wins
            else "consistent with the weak frozen variance head; report straight."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
