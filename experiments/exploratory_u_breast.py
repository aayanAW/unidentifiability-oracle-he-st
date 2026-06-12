"""EXPLORATORY real-data U gate on breast (rollout 12+, Tier-1, no cluster, no UNI).

Cheap feasibility check BEFORE the GPU HYBRID build (honors the build prompt: feasibility before infra).
Computes morphology-conditioned U on the REAL GSE243280 breast niches using FROZEN, UNGATED DINOv2
embeddings of the post-Xenium H&E. This is EXPLORATORY, not confirmatory (preregistration.md sec 10):
the frozen ungated encoder is weaker than the pre-registered UNI/trained-f, so a positive signal here is
a lower bound and justifies the cluster build; a null here is a cheap early warning.

The decisive readouts (no planted ground truth on real data):
  - per-gene morph->ST predictability R^2 (does H&E predict expression at all, and is there a SPREAD?);
  - structured-U fraction + Moran's I of the top-U map (is the unidentifiable signal spatially real?).

Memory-safe: the breast WSI is multi-GB; we gunzip once and read only the 224x224 patch windows lazily
via tifffile's zarr store (never the whole image into RAM).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import xenium_io as xio  # noqa: E402
from src.embeddings import _run_encoder  # noqa: E402
from src.loaders import load_xenium_noise_floor  # noqa: E402
from src.oracle import run_oracle, spatial_structure, top_u_genes  # noqa: E402
from src.simulator import TriadData  # noqa: E402


def _gunzip_once(gz: Path) -> Path:
    out = gz.with_suffix("")  # drop .gz
    if not out.exists():
        print(f"[he] gunzip {gz.name} -> {out.name} (one-time) ...")
        with gzip.open(gz, "rb") as f, open(out, "wb") as g:
            shutil.copyfileobj(f, g, length=1 << 22)
    return out


def _load_homography(csv_gz: Path | None) -> np.ndarray | None:
    if csv_gz is None or not csv_gz.exists():
        return None
    import pandas as pd

    vals = pd.read_csv(csv_gz, header=None).to_numpy(dtype=float)
    return vals.reshape(3, 3) if vals.size == 9 else None


def _patches_lazy(he_tif: Path, centers_um, H, patch_px=224, um_per_px=0.2125):
    """Read patch_px windows at homography-mapped centers from the full-res level via a lazy zarr store."""
    import tifffile
    import zarr

    store = tifffile.imread(he_tif, aszarr=True)
    z = zarr.open(store, mode="r")
    arr = z[0] if hasattr(z, "shape") is False else z  # series may be a group
    arr = z[0] if isinstance(z, zarr.hierarchy.Group) else z
    h, w = arr.shape[0], arr.shape[1]
    if H is not None:
        pts = np.hstack([centers_um, np.ones((len(centers_um), 1))])
        proj = (H @ pts.T).T
        px = proj[:, :2] / proj[:, 2:3]
    else:
        px = centers_um / um_per_px
    half = patch_px // 2
    out = np.zeros((len(centers_um), patch_px, patch_px, 3), np.uint8)
    inb = 0
    for k, (cx, cy) in enumerate(px):
        x0, y0 = int(cx) - half, int(cy) - half
        xs, ys, xe, ye = (
            max(0, x0),
            max(0, y0),
            min(w, x0 + patch_px),
            min(h, y0 + patch_px),
        )
        if xe <= xs or ye <= ys:
            continue
        tile = np.asarray(arr[ys:ye, xs:xe])
        if tile.ndim == 2:
            tile = np.stack([tile] * 3, -1)
        out[k, ys - y0 : ye - y0, xs - x0 : xe - x0] = tile[..., :3]
        inb += 1
    print(
        f"[he] {inb}/{len(centers_um)} niche patches in-bounds ({'homography' if H is not None else 'um/px'} map)"
    )
    return out, inb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--bin-um", type=float, default=300.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument(
        "--encoder", default="dinov2_vits14", help="ungated; UNI falls back to this"
    )
    args = ap.parse_args()

    root = Path(args.data_dir)
    z1, z2, centers, genes = load_xenium_noise_floor(
        args.data_dir, bin_um=args.bin_um, min_cells=args.min_cells
    )
    z_est = 0.5 * (z1 + z2)
    print(
        f"niches={len(centers)}  genes={len(genes)}  bin={args.bin_um:.0f}um  encoder={args.encoder}"
    )

    he_gz = xio._first_existing(
        root / "rep1", ("*Post-Xenium_HE_Rep1.ome.tif.gz", "*HE*ome.tif.gz")
    )  # noqa: SLF001
    if he_gz is None:
        print("[he] no H&E .ome.tif.gz under data/rep1 -- run the H&E download first.")
        return 3
    homog = _load_homography(
        xio._first_existing(root / "rep1", ("*HE*homography.csv.gz",))
    )  # noqa: SLF001
    patches, inb = _patches_lazy(_gunzip_once(he_gz), centers, homog, um_per_px=0.2125)
    if inb < 0.5 * len(centers) and homog is not None:
        print(
            "[he] <50% in-bounds with homography; retrying with plain um/px scaling ..."
        )
        patches, inb = _patches_lazy(
            _gunzip_once(he_gz), centers, None, um_per_px=0.2125
        )

    feats, enc_name, confirmatory = _run_encoder(patches, args.encoder)
    print(
        f"[morph] encoder={enc_name} confirmatory={confirmatory}  feat_dim={feats.shape[1]}"
    )

    triad = TriadData(
        coords=centers,
        morph=feats,
        xen_rep1=z1,
        xen_rep2=z2,
        visium=np.zeros_like(z1),
        gene_class=None,
    )
    res = run_oracle(triad, sigma2_reg=0.0, predictor="rf", seed=0)

    # morph->ST predictability per gene: R^2 = 1 - mean(resid2)/Var(z_est); z_est ~ unit var (z-scored)
    var = z_est.var(0) + 1e-9
    r2 = 1.0 - res.resid2.mean(0) / var
    spatial = spatial_structure(triad, res, seed=0)
    print("=" * 66)
    print("EXPLORATORY real-data U (breast) -- ungated DINOv2, NOT confirmatory")
    print("=" * 66)
    print(
        f"morph->ST R^2:  max={r2.max():.3f}  p90={np.percentile(r2, 90):.3f}  median={np.median(r2):.3f}"
    )
    print(
        f"  genes with R^2>0.05 (morphology predicts): {(r2 > 0.05).sum()}/{len(genes)}"
    )
    print(
        f"U: structured-fraction(FDR)={res.flagged.mean():.3f}  "
        f"top-U Moran's I={spatial['morans_i']:.3f} (p={spatial['pvalue']:.3f})"
    )
    top = top_u_genes(res, top_frac=0.2)
    pred = np.argsort(r2)[::-1][: max(1, len(genes) // 5)]
    print(f"  most-predictable genes (low U expected): {[genes[i] for i in pred[:8]]}")
    print(
        f"  highest-U genes (unidentifiable candidates): {[genes[i] for i in top[:8]]}"
    )
    print("-" * 66)
    spread = r2.max() - np.median(r2)
    if (r2 > 0.05).sum() >= 5 and spread > 0.08 and spatial["pvalue"] < 0.05:
        print(
            "READOUT: SIGNAL -- morphology predicts a subset of genes, there is a predictability SPREAD, "
            "and high-U is spatially structured. Cluster HYBRID build is justified (trained f should sharpen it)."
        )
        return 0
    print(
        "READOUT: WEAK/NULL -- frozen ungated morphology shows little predictability spread or no spatial "
        "U structure here. A stronger trained f MIGHT recover it, but this is a yellow flag before cluster spend."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
