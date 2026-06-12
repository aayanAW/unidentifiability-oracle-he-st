"""FIRST trained-oracle result: train the dual-head on FROZEN DINOv2 embeddings of breast @300um, then
compare its unidentifiability U and its selective-risk-coverage efficiency against the frozen-RF baseline.

This is the Tier-2 HYBRID result that is CLUSTER-FREE (plan.md "Direction update"): the backbone stays a
frozen, ungated DINOv2-S (the morphology features are exploratory per preregistration.md sec 10 -- NOT
UNI, NOT an end-to-end fine-tune), but the dual-head oracle ON TOP is genuinely TRAINED (beta-NLL,
spatial-block OOF). It answers two questions the frozen-RF feasibility vehicle could not:

  1. Does a trained, higher-capacity f leave LESS residual on identifiable genes, tightening U toward true
     unidentifiability?  (audit C1: a low-capacity f under-fits and inflates U.)
  2. Does the LEARNED variance head defer more EFFICIENTLY than raw ensemble variance on the real
     selective-risk-coverage curve (the utility metric, prereg sec 5)?  (plan.md Phase-3 gate.)

Honesty: frozen ungated DINOv2-S embeddings + breast only + 300um niches => EXPLORATORY. The end-to-end
backbone fine-tune (+ UNI, + 5 organs, + Xenium-5K) is the cluster step (scripts/train_dualhead.sbatch).
Run:  python3 experiments/trained_oracle_breast.py data --bin-um 300 --epochs 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.embeddings import embed_bins  # noqa: E402
from src.loaders import XENIUM_UM_PER_PX, load_xenium_noise_floor  # noqa: E402
from src.oracle import (  # noqa: E402
    aurc,
    noise_floor,
    registration_sigma2,
    risk_coverage_curve,
    run_oracle,
    spatial_structure,
)
from src.predictor import DualHeadOracle  # noqa: E402
from src.simulator import TriadData  # noqa: E402
from src.train import TrainConfig, fit_dual_head  # noqa: E402
from src import xenium_io as xio  # noqa: E402


def _stamp(root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()[:10]
        dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet"]) != 0
    except Exception:
        sha, dirty = "unknown", False
    return f"git_sha={sha}{'+dirty' if dirty else ''}"


def oof_dual_head(
    X: np.ndarray,
    Y: np.ndarray,
    coords: np.ndarray,
    cfg: TrainConfig,
    n_folds: int = 4,
    seed: int = 0,
    trunk: tuple[int, ...] = (256, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial-block out-of-fold dual-head predictions (mean, variance), matching the RF path's CV (H5).

    Per fold: standardize embeddings on the train rows (no leakage), train the dual head on the train
    blocks, predict held-out mean+variance. Returns (oof_mean (n,g), oof_var (n,g))."""
    from src.oracle import _spatial_block_folds  # noqa: SLF001 -- shared CV splitter

    fold = _spatial_block_folds(coords, n_folds, seed)
    n, g = Y.shape
    oof_mean = np.zeros((n, g), dtype=np.float64)
    oof_var = np.zeros((n, g), dtype=np.float64)
    for f in range(n_folds):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if te.size == 0 or tr.size == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        model = DualHeadOracle(
            n_genes=g, in_dim=X.shape[1], trunk_dims=trunk, dropout=0.1
        )
        model, _ = fit_dual_head(model, Xs, Y, tr, te, cfg)
        dev = next(
            model.parameters()
        ).device  # model may be on mps/cuda; from_numpy is CPU (review fix)
        mean, var = model.predict(torch.from_numpy(Xs[te].astype(np.float32)).to(dev))
        oof_mean[te] = mean.cpu().numpy()
        oof_var[te] = var.cpu().numpy()
    return oof_mean, oof_var


def _efficiency(order_score: np.ndarray, error: np.ndarray) -> dict:
    """Normalized selective efficiency in [0,1]: 1 = oracle deferral, 0 = random. Comparable across systems
    with different base error scales (the fair "selective-risk efficiency" gain, plan.md Phase-3 gate)."""
    curve = risk_coverage_curve(order_score, error)
    a = aurc(curve)
    a_oracle = aurc(risk_coverage_curve(error, error))  # defer true-highest-error first
    # the random-ordering AURC is the constant mean-risk integrated over the SAME (truncated) coverage grid,
    # not error.mean() (which is ~5% larger); using the curve's own coverage points keeps eff in [0,1].
    cov = curve[:, 0]
    a_random = aurc(np.column_stack([cov, np.full_like(cov, error.mean())]))
    denom = a_random - a_oracle
    eff = (a_random - a) / denom if denom > 1e-12 else 0.0
    # risk on the retained set at 50% coverage
    risk50 = float(curve[np.argmin(np.abs(curve[:, 0] - 0.5)), 1])
    return {
        "aurc": a,
        "aurc_oracle": a_oracle,
        "full_risk": a_random,
        "efficiency": eff,
        "risk50": risk50,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--bin-um", type=float, default=300.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--encoder", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.data_dir)
    repo = Path(__file__).resolve().parents[1]

    # ---- data: matched-niche cleaner-reference target + frozen morphology embeddings -------------------
    z1, z2, centers, genes = load_xenium_noise_floor(
        args.data_dir, bin_um=args.bin_um, min_cells=args.min_cells
    )
    z_est = 0.5 * (z1 + z2)
    he = xio._first_existing(  # noqa: SLF001
        root / "rep1", ("Post-Xenium_HE_Rep1.ome.tif", "*HE*.ome.tif", "*HE*ome.tif.gz")
    )
    homog = xio._first_existing(root / "rep1", ("*HE*homography.csv.gz",))  # noqa: SLF001
    if he is None:
        print("[trained-oracle] no H&E under data/rep1 -- run the H&E download first.")
        return 3
    mf = embed_bins(
        he, centers, homog, encoder=args.encoder, um_per_px=XENIUM_UM_PER_PX
    )
    X = np.asarray(mf.features, dtype=np.float64)
    print(f"[{_stamp(repo)}]")
    print(
        f"niches={len(centers)} genes={len(genes)} bin={args.bin_um:.0f}um "
        f"encoder={mf.encoder_name} confirmatory={mf.is_confirmatory} feat_dim={X.shape[1]}"
    )

    triad = TriadData(
        coords=centers,
        morph=X,
        xen_rep1=z1,
        xen_rep2=z2,
        visium=np.zeros_like(z1),
        gene_class=None,
    )
    s2_xen = noise_floor(z1, z2)
    target_noise = 0.5 * s2_xen
    sigma2_reg = registration_sigma2(triad, predictor="rf", seed=args.seed)
    print(
        f"sigma2_reg median={np.median(sigma2_reg):.4f}  target_noise median={np.median(target_noise):.4f}"
    )

    # ---- baseline: frozen-RF raw-variance oracle ------------------------------------------------------
    res_rf = run_oracle(triad, sigma2_reg=sigma2_reg, predictor="rf", seed=args.seed)
    U_rf = res_rf.U
    error_rf = res_rf.resid2.mean(0)  # per-gene OOF MSE on the cleaner target

    # ---- trained: dual-head oracle on the SAME frozen embeddings + splits ----------------------------
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=1e-3,
        batch_size=128,
        beta=0.5,
        device=args.device,
        seed=args.seed,
    )
    oof_mean, oof_var = oof_dual_head(X, z_est, centers, cfg, n_folds=4, seed=args.seed)
    error_ddh = ((z_est - oof_mean) ** 2).mean(0)
    rawvar_ddh = oof_var.mean(
        0
    )  # the head's raw per-gene predictive variance (pre floor-subtraction)
    U_ddh = np.clip(rawvar_ddh - target_noise - sigma2_reg, 0.0, None)

    # ---- the real utility metric: selective-risk-coverage efficiency ---------------------------------
    eff_rf = _efficiency(U_rf, error_rf)
    eff_ddh = _efficiency(U_ddh, error_ddh)
    # ranking-quality isolation: defer on the TRAINED predictor's own errors, rank by RF-U vs DDH-U
    eff_rankRF = _efficiency(U_rf, error_ddh)
    eff_rankDDH = _efficiency(U_ddh, error_ddh)
    # DIAGNOSTIC (exploratory, labeled): does the head's RAW variance (before the noise-floor subtraction
    # that clips U's spread) defer better? Isolates floor-clipping vs a genuinely weak variance head.
    eff_rawDDH = _efficiency(rawvar_ddh, error_ddh)

    # audit-C1 check: on the BASELINE-identifiable genes (low RF error -- a symmetric, non-self-referential
    # set so the RF side is not assessed only on DDH-easy genes), did the trained head shrink U vs RF?
    pred = error_rf <= np.percentile(error_rf, 33)
    c1 = (float(np.median(U_rf[pred])), float(np.median(U_ddh[pred])))
    sp_rf = spearmanr(U_rf, error_rf).correlation
    sp_ddh = spearmanr(U_ddh, error_ddh).correlation
    sp_rawvar = spearmanr(rawvar_ddh, error_ddh).correlation
    # epistemic=zeros: the DDH is single-seed, so spatial_structure tests the RAW residual variance on the
    # top-U genes (an UPPER BOUND on aleatoric; the RF path subtracts a bootstrap-ensemble epistemic). Any
    # structure found here is therefore conservative -- compare against the RF Moran's I with that caveat.
    ddh_struct = spatial_structure(
        TriadData(centers, X, z1, z2, np.zeros_like(z1), None),
        type(
            "R",
            (),
            {
                "resid2": (z_est - oof_mean) ** 2,
                "epistemic": np.zeros_like(oof_var),
                "U": U_ddh,
            },
        )(),
        seed=args.seed,
    )

    r2_ddh = 1.0 - error_ddh / (z_est.var(0) + 1e-9)
    print("=" * 72)
    print(
        "FIRST TRAINED-ORACLE RESULT (breast, frozen DINOv2-S embeddings) -- EXPLORATORY"
    )
    print("=" * 72)
    print(
        f"morph->ST R^2 (trained mean head): max={r2_ddh.max():.3f} p90={np.percentile(r2_ddh, 90):.3f} median={np.median(r2_ddh):.3f}"
    )
    print(f"  genes R^2>0.05: {(r2_ddh > 0.05).sum()}/{len(genes)}")
    print(
        f"U median  RF(raw-var)={np.median(U_rf):.3f}  DDH(trained)={np.median(U_ddh):.3f}"
    )
    print(
        f"audit-C1 (U on identifiable genes, lower=better):  RF={c1[0]:.3f}  DDH={c1[1]:.3f}"
    )
    print(
        f"deferral calibration Spearman(score,error):  RF-U={sp_rf:.3f}  DDH-U={sp_ddh:.3f}  "
        f"DDH-rawvar={sp_rawvar:.3f}"
    )
    print(
        f"DDH top-U spatial structure: Moran's I={ddh_struct['morans_i']:.3f} (p={ddh_struct['pvalue']:.3f})"
    )
    print("-" * 72)
    print("selective-risk-coverage efficiency (1=oracle deferral, 0=random):")
    print(
        "  NOTE: the two self-normalized rows below use DIFFERENT error denominators (each system's own "
        "OOF error) and are NOT a head-to-head comparison; the 'ranking on DDH error' row IS (same error)."
    )
    print(
        f"  self-normalized   RF : eff={eff_rf['efficiency']:.3f}  AURC={eff_rf['aurc']:.4f} risk@50%={eff_rf['risk50']:.4f}"
    )
    print(
        f"  self-normalized   DDH: eff={eff_ddh['efficiency']:.3f}  AURC={eff_ddh['aurc']:.4f} risk@50%={eff_ddh['risk50']:.4f}"
    )
    print(
        f"  HEAD-TO-HEAD (rank on DDH error): RF-U eff={eff_rankRF['efficiency']:.3f}  vs  DDH-U eff={eff_rankDDH['efficiency']:.3f}"
    )
    print(
        f"  DIAGNOSTIC: DDH RAW-variance (no floor subtraction) eff={eff_rawDDH['efficiency']:.3f}  "
        "(vs DDH-U above -- isolates floor-clipping from a weak variance head)"
    )

    # ---- persist results + the cluster training input -------------------------------------------------
    cache = repo / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache / "dualhead_breast.npz",
        X=X.astype(np.float32),
        Y=z_est.astype(np.float32),
        coords=centers,
        # label the artifact so a downstream/cluster load can never mistake exploratory frozen DINOv2-S
        # embeddings for a confirmatory UNI/end-to-end run (preregistration.md sec 10; embeddings.py H7).
        encoder_name=np.array(mf.encoder_name),
        is_confirmatory=np.bool_(mf.is_confirmatory),
    )
    results = repo / "experiments" / "results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results / "trained_oracle_breast.npz",
        genes=np.array(genes),
        U_rf=U_rf,
        U_ddh=U_ddh,
        error_rf=error_rf,
        error_ddh=error_ddh,
        rawvar_ddh=rawvar_ddh,
        curve_rf=risk_coverage_curve(U_rf, error_rf),
        curve_ddh=risk_coverage_curve(U_ddh, error_ddh),
        curve_rawvar_ddh=risk_coverage_curve(rawvar_ddh, error_ddh),
        curve_oracle_ddh=risk_coverage_curve(error_ddh, error_ddh),
    )

    # ---- honest verdict -------------------------------------------------------------------------------
    tightened = c1[1] <= c1[0] + 1e-6
    more_efficient = eff_rankDDH["efficiency"] >= eff_rankRF["efficiency"]
    structured = ddh_struct["pvalue"] < 0.05
    print("-" * 72)
    if tightened and more_efficient and structured:
        print(
            "READOUT: GREEN -- training tightened U on identifiable genes AND the learned variance head "
            "defers at least as efficiently as raw ensemble variance, with spatially-structured U. The "
            "end-to-end backbone fine-tune (cluster) is the next lever to push R^2 + retained utility."
        )
        return 0
    print(
        "READOUT: MIXED -- report straight. "
        f"U tightened={tightened}, DDH-defers-better={more_efficient}, U-structured={structured}. "
        "On frozen DINOv2 embeddings the trained head may not yet beat the RF baseline; the end-to-end "
        "fine-tune (a stronger f) is the pre-registered bet. Not a kill -- a frozen-substrate lower bound."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
